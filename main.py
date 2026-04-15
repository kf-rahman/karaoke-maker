import os
import re
import uuid
import shutil
import subprocess
import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI(docs_url=None, redoc_url=None)

# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory, per-IP)
# ---------------------------------------------------------------------------
import time
from collections import defaultdict

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 3       # max requests per window

# ---------------------------------------------------------------------------
# Concurrency guard – avoid OOM from many Demucs jobs running at once
# ---------------------------------------------------------------------------
MAX_CONCURRENT_JOBS = 1  # CPU can only usefully run one Demucs job at a time
_job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Temp directory for processing
WORK_DIR = Path("/tmp/karaoke_work")
WORK_DIR.mkdir(exist_ok=True)

# Maximum allowed audio duration in seconds (10 minutes)
MAX_DURATION_SECONDS = 600

# YouTube URL pattern
_YT_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=[\w\-]{11}|youtu\.be/[\w\-]{11}|youtube\.com/shorts/[\w\-]{11})"
)


class ProcessRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("No URL provided")
        if not _YT_RE.match(v):
            raise ValueError(
                "Please provide a valid YouTube URL "
                "(e.g. https://www.youtube.com/watch?v=XXXXXXXXXXX)"
            )
        return v


def _check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    timestamps = _rate_limit_store[client_ip]
    # Prune old entries
    _rate_limit_store[client_ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait before trying again.",
        )
    _rate_limit_store[client_ip].append(now)


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/api/process")
async def process_video(req: ProcessRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    url = req.url  # already stripped/validated by Pydantic

    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    input_file = job_dir / "input.wav"

    logging.info("[%s] New job started for URL: %s", job_id, url)

    try:
        # Acquire semaphore so we don't run too many heavy jobs at once
        try:
            logging.info("[%s] Waiting for processing slot...", job_id)
            await asyncio.wait_for(_job_semaphore.acquire(), timeout=900)
        except asyncio.TimeoutError:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=503,
                detail="Server is busy. Please try again in a few minutes.",
            )

        song_duration: float | None = None

        try:
            # Step 0: Check video duration before downloading
            logging.info("[%s] Probing video duration...", job_id)
            probe_result = await asyncio.to_thread(
                subprocess.run,
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--print", "duration",
                    "--skip-download",
                    "--js-runtimes", "nodejs",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if probe_result.returncode == 0:
                raw_duration = probe_result.stdout.strip()
                try:
                    song_duration = float(raw_duration)
                    logging.info("[%s] Video duration: %ds", job_id, int(song_duration))
                    if song_duration > MAX_DURATION_SECONDS:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Video is too long ({int(song_duration)}s). Maximum is {MAX_DURATION_SECONDS // 60} minutes.",
                        )
                except ValueError:
                    pass  # duration unavailable – fall back to fixed timeout

            # Step 1: Download audio from YouTube
            logging.info("[%s] Downloading audio...", job_id)
            dl_result = await asyncio.to_thread(
                subprocess.run,
                [
                    "yt-dlp",
                    "-x",
                    "--audio-format", "wav",
                    "--no-playlist",
                    "--max-filesize", "50M",
                    "--no-exec",
                    "--no-batch-file",
                    "--no-config",
                    "--js-runtimes", "nodejs",
                    "-o", str(input_file),
                    "--", url,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if dl_result.returncode != 0:
                # Sanitize stderr – never leak server paths
                stderr_safe = dl_result.stderr[:200] if dl_result.stderr else "Unknown error"
                stderr_safe = stderr_safe.replace(str(WORK_DIR), "[workdir]")
                logging.error("[%s] Download failed: %s", job_id, stderr_safe)
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download audio: {stderr_safe}",
                )

            # Find the actual downloaded file (yt-dlp may add extension)
            actual_files = list(job_dir.glob("input*"))
            if not actual_files:
                raise HTTPException(status_code=500, detail="Download succeeded but file not found")
            actual_input = actual_files[0]
            logging.info("[%s] Download complete: %s (%.1fMB)", job_id, actual_input.name, actual_input.stat().st_size / 1e6)

            # Step 2: Run Demucs to separate vocals
            # Timeout = 3x song duration (floor 5 min, ceiling 20 min).
            # Falls back to 15 min if duration probe failed.
            demucs_timeout = int(max(300, min(song_duration * 3, 1200))) if song_duration else 900
            logging.info("[%s] Starting vocal separation (timeout: %ds)...", job_id, demucs_timeout)
            t0 = time.time()
            demucs_result = await asyncio.to_thread(
                subprocess.run,
                [
                    "python", "-m", "demucs",
                    "-n", "mdx_extra_q",
                    "--two-stems", "vocals",
                    "-o", str(job_dir / "output"),
                    "--mp3",
                    str(actual_input),
                ],
                capture_output=True,
                text=True,
                timeout=demucs_timeout,
            )
            elapsed = time.time() - t0

            if demucs_result.returncode != 0:
                combined = (demucs_result.stderr or "") + (demucs_result.stdout or "")
                combined = combined[:400] if combined else "Unknown error"
                combined = combined.replace(str(WORK_DIR), "[workdir]")
                logging.error("[%s] Demucs failed after %.1fs: %s", job_id, elapsed, combined)
                raise HTTPException(
                    status_code=500,
                    detail=f"Vocal separation failed: {combined}",
                )

            logging.info("[%s] Vocal separation complete in %.1fs", job_id, elapsed)

            # Step 3: Find the instrumental (no_vocals) track
            output_dir = job_dir / "output" / "mdx_extra_q"
            if not output_dir.exists():
                raise HTTPException(status_code=500, detail="Vocal separation completed but output directory not found")

            no_vocals = None
            for d in output_dir.iterdir():
                if not d.is_dir():
                    continue
                for ext in ("mp3", "wav"):
                    candidate = d / f"no_vocals.{ext}"
                    if candidate.exists():
                        no_vocals = candidate
                        break
                if no_vocals:
                    break

            if no_vocals is None:
                raise HTTPException(status_code=500, detail="Vocal separation completed but output file not found")

            # Move to a servable location
            serve_name = f"{job_id}_karaoke{no_vocals.suffix}"
            serve_path = WORK_DIR / serve_name
            shutil.copy2(no_vocals, serve_path)

        finally:
            _job_semaphore.release()

        # Cleanup job directory
        shutil.rmtree(job_dir, ignore_errors=True)

        logging.info("[%s] Job complete. Serving: %s", job_id, serve_name)
        return JSONResponse({
            "status": "success",
            "audio_url": f"/api/audio/{serve_name}",
        })

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except subprocess.TimeoutExpired:
        logging.error("[%s] Job timed out", job_id)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=504, detail="Processing timed out. Try a shorter song.")
    except Exception:
        logging.exception("[%s] Unexpected error", job_id)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")


@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    # Strict allowlist: job_id (hex) + _karaoke + .mp3/.wav
    if not re.fullmatch(r"[a-f0-9]{8}_karaoke\.(mp3|wav)", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = (WORK_DIR / filename).resolve()
    # Ensure resolved path is still under WORK_DIR
    if not str(file_path).startswith(str(WORK_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    media_type = "audio/mpeg" if file_path.suffix == ".mp3" else "audio/wav"
    return FileResponse(
        file_path,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Periodic cleanup of stale files (older than 1 hour)
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _start_cleanup_task():
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            try:
                cutoff = time.time() - 3600
                for p in WORK_DIR.iterdir():
                    try:
                        if p.stat().st_mtime < cutoff:
                            if p.is_dir():
                                shutil.rmtree(p, ignore_errors=True)
                            else:
                                p.unlink(missing_ok=True)
                    except OSError:
                        pass
            except Exception:
                pass

    asyncio.create_task(_cleanup_loop())
