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

import time
from collections import defaultdict

app = FastAPI(docs_url=None, redoc_url=None)

# ---------------------------------------------------------------------------
# YouTube cookies — loaded from YOUTUBE_COOKIES env var (HF Space secret).
# Written to a temp file once at startup so yt-dlp can use them.
# ---------------------------------------------------------------------------
COOKIES_FILE = Path("/tmp/yt_cookies.txt")

_cookies_content = os.environ.get("YOUTUBE_COOKIES", "").strip()
if _cookies_content:
    COOKIES_FILE.write_text(_cookies_content)
    logging.info("YouTube cookies loaded from environment.")
else:
    logging.warning("YOUTUBE_COOKIES not set — downloads may fail on server IPs.")

# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory, per-IP)
# ---------------------------------------------------------------------------
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 3       # max requests per window

# ---------------------------------------------------------------------------
# Concurrency guard – one Demucs job at a time on CPU
# ---------------------------------------------------------------------------
MAX_CONCURRENT_JOBS = 1
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


def _yt_dlp_base_args() -> list[str]:
    """Common yt-dlp flags used in every call."""
    args = [
        "yt-dlp",
        "--no-playlist",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
    ]
    if COOKIES_FILE.exists():
        args += ["--cookies", str(COOKIES_FILE)]
    return args


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
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
    ]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait before trying again.",
        )
    _rate_limit_store[client_ip].append(now)


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/api/process")
async def process_video(req: ProcessRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    url = req.url
    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    input_file = job_dir / "input.wav"

    logging.info("[%s] New job: %s", job_id, url)

    try:
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
            # Step 0: Probe duration
            logging.info("[%s] Probing video duration...", job_id)
            probe = await asyncio.to_thread(
                subprocess.run,
                _yt_dlp_base_args() + ["--print", "duration", "--skip-download", url],
                capture_output=True, text=True, timeout=30,
            )
            if probe.returncode == 0:
                try:
                    song_duration = float(probe.stdout.strip())
                    logging.info("[%s] Duration: %ds", job_id, int(song_duration))
                    if song_duration > MAX_DURATION_SECONDS:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Video is too long ({int(song_duration)}s). Maximum is {MAX_DURATION_SECONDS // 60} minutes.",
                        )
                except ValueError:
                    pass  # duration unavailable – fall back to fixed timeout

            # Step 1: Download audio
            logging.info("[%s] Downloading audio...", job_id)
            dl = await asyncio.to_thread(
                subprocess.run,
                _yt_dlp_base_args() + [
                    "-x", "--audio-format", "wav",
                    "--max-filesize", "50M",
                    "--no-exec", "--no-batch-file", "--no-config",
                    "-o", str(input_file),
                    "--", url,
                ],
                capture_output=True, text=True, timeout=120,
            )

            if dl.returncode != 0:
                err = (dl.stderr or "")[:200].replace(str(WORK_DIR), "[workdir]")
                logging.error("[%s] Download failed: %s", job_id, err)
                raise HTTPException(status_code=400, detail=f"Failed to download audio: {err}")

            actual_files = list(job_dir.glob("input*"))
            if not actual_files:
                raise HTTPException(status_code=500, detail="Download succeeded but file not found")
            actual_input = actual_files[0]
            logging.info("[%s] Download complete (%.1fMB)", job_id, actual_input.stat().st_size / 1e6)

            # Step 2: Vocal separation
            demucs_timeout = int(max(300, min(song_duration * 3, 1200))) if song_duration else 900
            logging.info("[%s] Starting vocal separation (timeout: %ds)...", job_id, demucs_timeout)
            t0 = time.time()
            demucs = await asyncio.to_thread(
                subprocess.run,
                [
                    "python", "-m", "demucs",
                    "-n", "mdx_extra_q",
                    "--two-stems", "vocals",
                    "-o", str(job_dir / "output"),
                    "--mp3",
                    str(actual_input),
                ],
                capture_output=True, text=True, timeout=demucs_timeout,
            )
            elapsed = time.time() - t0

            if demucs.returncode != 0:
                combined = ((demucs.stderr or "") + (demucs.stdout or ""))
                combined = combined.replace(str(WORK_DIR), "[workdir]")
                logging.error("[%s] Demucs failed after %.1fs:\n%s", job_id, elapsed, combined)
                raise HTTPException(status_code=500, detail=f"Vocal separation failed: {combined[-800:]}")

            logging.info("[%s] Vocal separation complete in %.1fs", job_id, elapsed)

            # Step 3: Find output file
            output_dir = job_dir / "output" / "mdx_extra_q"
            if not output_dir.exists():
                raise HTTPException(status_code=500, detail="Output directory not found")

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
                raise HTTPException(status_code=500, detail="Output file not found")

            serve_name = f"{job_id}_karaoke{no_vocals.suffix}"
            shutil.copy2(no_vocals, WORK_DIR / serve_name)

        finally:
            _job_semaphore.release()

        shutil.rmtree(job_dir, ignore_errors=True)
        logging.info("[%s] Job complete. Serving: %s", job_id, serve_name)
        return JSONResponse({"status": "success", "audio_url": f"/api/audio/{serve_name}"})

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
    if not re.fullmatch(r"[a-f0-9]{8}_karaoke\.(mp3|wav)", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = (WORK_DIR / filename).resolve()
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
            await asyncio.sleep(300)
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
