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
# Job state store  {job_id -> {status, audio_url, error, started_at, ...}}
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}

app.mount("/static", StaticFiles(directory="static"), name="static")

WORK_DIR = Path("/tmp/karaoke_work")
WORK_DIR.mkdir(exist_ok=True)

MAX_DURATION_SECONDS = 600

_YT_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=[\w\-]{11}|youtu\.be/[\w\-]{11}|youtube\.com/shorts/[\w\-]{11})"
)

# ---------------------------------------------------------------------------
# Whisper model — loaded once at startup, reused across jobs
# ---------------------------------------------------------------------------
_whisper_model = None


def _load_whisper_model():
    global _whisper_model
    try:
        import whisper
        logging.info("Loading Whisper medium model...")
        _whisper_model = whisper.load_model("small")
        logging.info("Whisper model ready.")
    except Exception:
        logging.exception("Failed to load Whisper model — transcription fallback disabled.")


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


def _clean_title_for_search(title: str) -> str:
    title = re.sub(r"[\(\[][^\)\]]{0,40}[\)\]]", "", title)
    title = re.sub(r"\b(official|video|audio|lyrics|lyric|hd|hq|mv|ft\.?|feat\.?)\b", "", title, flags=re.IGNORECASE)
    return title.strip(" -–|")


def _segments_to_lrc(segments: list) -> str:
    """Convert Whisper segments to LRC timestamp format."""
    lines = []
    for seg in segments:
        t = seg["start"]
        minutes = int(t // 60)
        seconds = t % 60
        lines.append(f"[{minutes:02d}:{seconds:05.2f}] {seg['text'].strip()}")
    return "\n".join(lines)


async def _fetch_lrclib(title: str) -> dict | None:
    """Search lrclib. Prefers synced lyrics. Returns dict or None."""
    query = _clean_title_for_search(title)
    logging.info("lrclib search: %r", query)

    def _do_request():
        encoded = urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(
            f"https://lrclib.net/api/search?{encoded}",
            headers={"User-Agent": "karaoke-maker/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())

    try:
        results = await asyncio.to_thread(_do_request)
        if not results:
            return None
        best = (
            next((r for r in results if r.get("syncedLyrics")), None) or
            next((r for r in results if r.get("plainLyrics")), None)
        )
        if not best:
            return None
        synced = bool(best.get("syncedLyrics"))
        return {
            "lyrics": best["syncedLyrics"] if synced else best["plainLyrics"],
            "lyrics_synced": synced,
            "lyrics_source": "lrclib",
            "lyrics_artist": best.get("artistName"),
            "lyrics_matched": best.get("trackName"),
        }
    except Exception as e:
        logging.warning("lrclib fetch failed: %s", e)
        return None


def _run_whisper(audio_path: Path) -> dict | None:
    """Blocking Whisper transcription — run via asyncio.to_thread."""
    if _whisper_model is None:
        return None
    try:
        logging.info("Whisper transcribing: %s", audio_path.name)
        t0 = time.time()
        result = _whisper_model.transcribe(str(audio_path), task="transcribe")
        elapsed = time.time() - t0
        logging.info("Whisper done in %.1fs, detected language: %s", elapsed, result.get("language"))
        lrc = _segments_to_lrc(result["segments"])
        return {
            "lyrics": lrc,
            "lyrics_synced": True,
            "lyrics_source": "whisper",
            "lyrics_artist": None,
            "lyrics_matched": None,
        }
    except Exception:
        logging.exception("Whisper transcription failed")
        return None


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
            # ------------------------------------------------------------------
            # Step 0: Probe — get title + duration
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # Step 1: lrclib search + download in parallel
            #   lrclib resolves in ~1s; download takes ~10-20s
            #   By the time download finishes we know if Whisper is needed
            # ------------------------------------------------------------------
            lrclib_task = asyncio.create_task(_fetch_lrclib(song_title)) if song_title else None

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
                if lrclib_task:
                    lrclib_task.cancel()
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

            # Collect lrclib result (almost certainly done by now)
            lrclib_result: dict | None = None
            if lrclib_task:
                lrclib_result = await lrclib_task
                if lrclib_result:
                    logging.info("[%s] lrclib hit: %s — %s", job_id, lrclib_result.get("lyrics_artist"), lrclib_result.get("lyrics_matched"))
                    _jobs[job_id].update(lrclib_result)
                else:
                    logging.info("[%s] lrclib miss — Whisper will transcribe", job_id)

            # ------------------------------------------------------------------
            # Step 2: Spleeter + Whisper in parallel
            #   Both read actual_input independently
            #   Whisper only runs when lrclib found nothing
            # ------------------------------------------------------------------
            spleeter_out = job_dir / "spleeter_out"
            spleeter_out.mkdir(exist_ok=True)
            spleeter_timeout = int(max(120, min(song_duration * 3, 600))) if song_duration else 600

            logging.info("[%s] Starting separation + transcription (timeout: %ds)...", job_id, spleeter_timeout)
            _jobs[job_id]["step"] = "separating"

            spleeter_state: dict = {}
            whisper_state: dict = {}

            async def run_spleeter():
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
                    combined = ((sep.stderr or "") + (sep.stdout or "")).replace(str(WORK_DIR), "[workdir]")
                    logging.error("[%s] Spleeter failed after %.1fs (returncode=%s): %r", job_id, elapsed, sep.returncode, combined[-500:])
                    spleeter_state["error"] = f"Vocal separation failed: {combined[-500:]}"
                    # Surface the error immediately — don't wait for Whisper to
                    # finish since without audio there's nothing to show
                    _jobs[job_id].update({"status": "error", "error": spleeter_state["error"]})
                else:
                    logging.info("[%s] Spleeter done in %.1fs", job_id, elapsed)
                    spleeter_state["ok"] = True

            async def run_whisper():
                if lrclib_result or _whisper_model is None:
                    return
                result = await asyncio.to_thread(_run_whisper, actual_input)
                if result:
                    whisper_state.update(result)

            await asyncio.gather(run_spleeter(), run_whisper())

            # ------------------------------------------------------------------
            # Step 3: Finish up
            # ------------------------------------------------------------------
            if spleeter_state.get("error"):
                _jobs[job_id].update({"status": "error", "error": spleeter_state["error"]})
                return

            accompaniment = spleeter_out / actual_input.stem / "accompaniment.mp3"
            if not accompaniment.exists():
                _jobs[job_id] = {"status": "error", "error": "Output file not found."}
                return

            if whisper_state.get("lyrics"):
                logging.info("[%s] Whisper lyrics stored", job_id)
                _jobs[job_id].update(whisper_state)

            serve_name = f"{job_id}_karaoke.mp3"
            shutil.copy2(accompaniment, WORK_DIR / serve_name)
            logging.info("[%s] Job complete. Serving: %s", job_id, serve_name)
            _jobs[job_id].update({"status": "complete", "audio_url": f"/api/audio/{serve_name}"})

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


@app.get("/api/lyrics/{job_id}")
async def get_lyrics(job_id: str):
    if not re.fullmatch(r"[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse({
        "lyrics": job.get("lyrics"),
        "lyrics_synced": job.get("lyrics_synced", False),
        "lyrics_source": job.get("lyrics_source"),
        "title": job.get("title"),
        "artist": job.get("lyrics_artist"),
        "matched": job.get("lyrics_matched"),
    })


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def _on_startup():
    # Load Whisper in a background thread so it doesn't block server startup
    asyncio.create_task(asyncio.to_thread(_load_whisper_model))

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
                for jid in list(_jobs.keys()):
                    if time.time() - _jobs[jid].get("started_at", time.time()) > 7200:
                        del _jobs[jid]
            except Exception:
                pass

    asyncio.create_task(_cleanup_loop())
