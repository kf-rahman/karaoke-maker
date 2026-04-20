import os
import re
import uuid
import shutil
import subprocess
import asyncio
import logging
import urllib.request
import urllib.parse
import json
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
# YouTube cookies
# ---------------------------------------------------------------------------
COOKIES_FILE = Path("/tmp/yt_cookies.txt")
_cookies_content = os.environ.get("YOUTUBE_COOKIES", "").strip()
if _cookies_content:
    COOKIES_FILE.write_text(_cookies_content)
    logging.info("YouTube cookies loaded from environment.")
else:
    logging.warning("YOUTUBE_COOKIES not set — downloads may fail on server IPs.")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 3

# ---------------------------------------------------------------------------
# Concurrency guard
# ---------------------------------------------------------------------------
MAX_CONCURRENT_JOBS = 1
_job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# ---------------------------------------------------------------------------
# Job state store  {job_id -> {status, audio_url, error, started_at}}
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}

app.mount("/static", StaticFiles(directory="static"), name="static")

WORK_DIR = Path("/tmp/karaoke_work")
WORK_DIR.mkdir(exist_ok=True)

MAX_DURATION_SECONDS = 600

_YT_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=[\w\-]{11}|youtu\.be/[\w\-]{11}|youtube\.com/shorts/[\w\-]{11})"
)


def _yt_dlp_base_args() -> list[str]:
    args = [
        "yt-dlp", "--no-playlist",
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
        raise HTTPException(status_code=429, detail="Too many requests. Please wait before trying again.")
    _rate_limit_store[client_ip].append(now)


async def _run_job(job_id: str, url: str):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    input_file = job_dir / "input.wav"

    logging.info("[%s] New job: %s", job_id, url)

    try:
        logging.info("[%s] Waiting for processing slot...", job_id)
        try:
            await asyncio.wait_for(_job_semaphore.acquire(), timeout=1800)
        except asyncio.TimeoutError:
            _jobs[job_id] = {"status": "error", "error": "Server is busy. Please try again in a few minutes."}
            shutil.rmtree(job_dir, ignore_errors=True)
            return

        try:
            # Step 0: Probe duration + title
            logging.info("[%s] Probing video metadata...", job_id)
            probe = await asyncio.to_thread(
                subprocess.run,
                _yt_dlp_base_args() + ["--dump-json", "--skip-download", url],
                capture_output=True, text=True, timeout=30,
            )
            song_duration: float | None = None
            song_title: str | None = None
            if probe.returncode == 0:
                try:
                    meta = json.loads(probe.stdout.strip())
                    song_duration = float(meta.get("duration") or 0) or None
                    song_title = meta.get("title") or None
                    if song_duration:
                        logging.info("[%s] Duration: %ds, title: %s", job_id, int(song_duration), song_title)
                    if song_duration and song_duration > MAX_DURATION_SECONDS:
                        _jobs[job_id] = {"status": "error", "error": f"Video is too long ({int(song_duration)}s). Maximum is {MAX_DURATION_SECONDS // 60} minutes."}
                        return
                except (ValueError, json.JSONDecodeError):
                    pass
            if song_title:
                _jobs[job_id]["title"] = song_title

            # Step 1: Download audio
            logging.info("[%s] Downloading audio...", job_id)
            _jobs[job_id]["step"] = "downloading"
            dl = await asyncio.to_thread(
                subprocess.run,
                _yt_dlp_base_args() + [
                    "-x", "--audio-format", "wav",
                    "--max-filesize", "50M",
                    "--no-exec", "--no-batch-file", "--no-config",
                    "-o", str(input_file),
                    "--", url,
                ],
                capture_output=True, text=True, timeout=300,
            )

            if dl.returncode != 0:
                err = (dl.stderr or "")[:300].replace(str(WORK_DIR), "[workdir]")
                logging.error("[%s] Download failed: %s", job_id, err)
                _jobs[job_id] = {"status": "error", "error": f"Failed to download audio: {err}"}
                return

            actual_files = list(job_dir.glob("input*"))
            if not actual_files:
                _jobs[job_id] = {"status": "error", "error": "Download succeeded but file not found."}
                return
            actual_input = actual_files[0]
            logging.info("[%s] Download complete (%.1fMB)", job_id, actual_input.stat().st_size / 1e6)

            # Step 2: Vocal separation with Spleeter
            spleeter_out = job_dir / "spleeter_out"
            spleeter_out.mkdir(exist_ok=True)
            spleeter_timeout = int(max(120, min(song_duration * 3, 600))) if song_duration else 600
            logging.info("[%s] Starting vocal separation (timeout: %ds)...", job_id, spleeter_timeout)
            _jobs[job_id]["step"] = "separating"
            t0 = time.time()
            sep = await asyncio.to_thread(
                subprocess.run,
                [
                    "spleeter", "separate",
                    "-p", "spleeter:2stems",
                    "-c", "mp3",
                    "-o", str(spleeter_out),
                    str(actual_input),
                ],
                capture_output=True, text=True, timeout=spleeter_timeout,
            )
            elapsed = time.time() - t0

            if sep.returncode != 0:
                combined = ((sep.stderr or "") + (sep.stdout or ""))
                combined = combined.replace(str(WORK_DIR), "[workdir]")
                logging.error("[%s] Spleeter failed after %.1fs: %s", job_id, elapsed, combined)
                _jobs[job_id] = {"status": "error", "error": f"Vocal separation failed: {combined[-500:]}"}
                return

            logging.info("[%s] Vocal separation complete in %.1fs", job_id, elapsed)

            # Step 3: Find output
            accompaniment = spleeter_out / actual_input.stem / "accompaniment.mp3"
            if not accompaniment.exists():
                _jobs[job_id] = {"status": "error", "error": "Output file not found."}
                return

            serve_name = f"{job_id}_karaoke.mp3"
            shutil.copy2(accompaniment, WORK_DIR / serve_name)
            logging.info("[%s] Job complete. Serving: %s", job_id, serve_name)
            _jobs[job_id] = {"status": "complete", "audio_url": f"/api/audio/{serve_name}"}

        finally:
            _job_semaphore.release()
            shutil.rmtree(job_dir, ignore_errors=True)

    except subprocess.TimeoutExpired:
        logging.error("[%s] Job timed out", job_id)
        _jobs[job_id] = {"status": "error", "error": "Processing timed out. Try a shorter song."}
        shutil.rmtree(job_dir, ignore_errors=True)
    except Exception:
        logging.exception("[%s] Unexpected error", job_id)
        _jobs[job_id] = {"status": "error", "error": "An internal error occurred."}
        shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/api/process")
async def process_video(req: ProcessRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "processing", "step": "queued", "started_at": time.time()}
    asyncio.create_task(_run_job(job_id, req.url))
    return JSONResponse({"job_id": job_id})


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    if not re.fullmatch(r"[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)


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
        file_path, media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _clean_title_for_search(title: str) -> str:
    """Strip common YouTube noise before sending to lyrics search."""
    # Remove parenthetical/bracketed suffixes like (Official Video), [4K], (Remastered 2011)
    title = re.sub(r"[\(\[][^\)\]]{0,40}[\)\]]", "", title)
    # Remove common filler words that hurt search
    title = re.sub(r"\b(official|video|audio|lyrics|lyric|hd|hq|mv|ft\.?|feat\.?)\b", "", title, flags=re.IGNORECASE)
    return title.strip(" -–|")


@app.get("/api/lyrics/{job_id}")
async def get_lyrics(job_id: str):
    if not re.fullmatch(r"[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    title = job.get("title")
    if not title:
        return JSONResponse({"lyrics": None, "title": None})

    query = _clean_title_for_search(title)
    logging.info("[%s] Fetching lyrics for: %s", job_id, query)

    try:
        encoded = urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(
            f"https://lrclib.net/api/search?{encoded}",
            headers={"User-Agent": "karaoke-maker/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            results = json.loads(resp.read().decode())

        if not results:
            return JSONResponse({"lyrics": None, "title": title})

        # Prefer results that have plain lyrics; take the first match
        best = next((r for r in results if r.get("plainLyrics")), None)
        if not best:
            return JSONResponse({"lyrics": None, "title": title})

        return JSONResponse({
            "lyrics": best["plainLyrics"],
            "title": title,
            "matched": best.get("trackName") or best.get("title"),
            "artist": best.get("artistName"),
        })

    except Exception as e:
        logging.warning("[%s] Lyrics fetch failed: %s", job_id, e)
        return JSONResponse({"lyrics": None, "title": title})


@app.get("/api/health")
async def health():
    return {"status": "ok"}


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
                # Clean up old job records
                for jid in list(_jobs.keys()):
                    if time.time() - _jobs[jid].get("started_at", time.time()) > 7200:
                        del _jobs[jid]
            except Exception:
                pass
    asyncio.create_task(_cleanup_loop())
