import re
import uuid
import json
import shutil
import subprocess
import asyncio
import logging
import urllib.request
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

# Max download size in bytes (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# YouTube URL + video ID patterns
_YT_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=[\w\-]{11}|youtu\.be/[\w\-]{11}|youtube\.com/shorts/[\w\-]{11})"
)
_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([\w\-]{11})")

# Piped API instances (tried in order, falls back if one is down)
_PIPED_INSTANCES = [
    "https://pipedapi.kavin.rocks",
    "https://pipedapi.adminforge.de",
    "https://piped-api.privacy.com.de",
]


def _get_piped_audio(video_id: str) -> tuple[str, float | None]:
    """
    Fetch audio stream URL and duration from Piped API.
    Tries multiple instances and returns the first success.
    Returns (stream_url, duration_seconds).
    """
    last_err: Exception | None = None
    for instance in _PIPED_INSTANCES:
        try:
            req = urllib.request.Request(
                f"{instance}/streams/{video_id}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            duration = float(data["duration"]) if data.get("duration") else None
            streams = data.get("audioStreams", [])
            if not streams:
                raise ValueError("No audio streams returned")

            # Pick highest-bitrate stream
            best = max(streams, key=lambda s: s.get("bitrate", 0))
            return best["url"], duration

        except Exception as e:
            logging.warning("Piped instance %s failed: %s", instance, e)
            last_err = e
            continue

    raise RuntimeError(f"All Piped instances failed. Last error: {last_err}")


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
    video_id_match = _YT_ID_RE.search(url)
    if not video_id_match:
        raise HTTPException(status_code=400, detail="Could not extract video ID from URL")
    video_id = video_id_match.group(1)

    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    input_file = job_dir / "input.wav"

    logging.info("[%s] New job — video ID: %s", job_id, video_id)

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

        try:
            # Step 1: Get audio stream URL from Piped API
            logging.info("[%s] Fetching stream URL from Piped API...", job_id)
            try:
                stream_url, song_duration = await asyncio.to_thread(
                    _get_piped_audio, video_id
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not fetch audio stream: {e}",
                )

            logging.info(
                "[%s] Got stream URL. Duration: %s",
                job_id,
                f"{int(song_duration)}s" if song_duration else "unknown",
            )

            if song_duration and song_duration > MAX_DURATION_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Video is too long ({int(song_duration)}s). Maximum is {MAX_DURATION_SECONDS // 60} minutes.",
                )

            # Step 2: Download audio via ffmpeg (handles all formats, converts to WAV)
            logging.info("[%s] Downloading audio...", job_id)
            dl_result = await asyncio.to_thread(
                subprocess.run,
                [
                    "ffmpeg", "-y",
                    "-i", stream_url,
                    "-vn",                  # drop video
                    "-acodec", "pcm_s16le", # convert to WAV
                    "-ar", "44100",         # standard sample rate
                    str(input_file),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if dl_result.returncode != 0:
                err = (dl_result.stderr or "")[:200]
                logging.error("[%s] ffmpeg download failed: %s", job_id, err)
                raise HTTPException(status_code=400, detail="Failed to download audio.")

            if input_file.stat().st_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="Audio file too large (max 50MB).")

            logging.info(
                "[%s] Download complete (%.1fMB)",
                job_id,
                input_file.stat().st_size / 1e6,
            )

            # Step 3: Run Demucs vocal separation
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
                    str(input_file),
                ],
                capture_output=True,
                text=True,
                timeout=demucs_timeout,
            )
            elapsed = time.time() - t0

            if demucs_result.returncode != 0:
                combined = ((demucs_result.stderr or "") + (demucs_result.stdout or ""))[:400]
                combined = combined.replace(str(WORK_DIR), "[workdir]")
                logging.error("[%s] Demucs failed after %.1fs: %s", job_id, elapsed, combined)
                raise HTTPException(status_code=500, detail=f"Vocal separation failed: {combined}")

            logging.info("[%s] Vocal separation complete in %.1fs", job_id, elapsed)

            # Step 4: Find the instrumental (no_vocals) track
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
