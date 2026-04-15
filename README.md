---
title: Karaoke Maker
emoji: 🎤
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
---

# Karaoke Maker

Paste a YouTube link, get back an instrumental (vocals-removed) version you can sing over.

Uses [Demucs](https://github.com/facebookresearch/demucs) (Meta's source separation model) to strip vocals. Works well on most songs; results vary by recording.

---

## How it works

1. User pastes a YouTube URL
2. Server downloads the audio via `yt-dlp`
3. Demucs separates the vocals from the instrumental
4. The instrumental track is served back for playback and download

---

## Running locally

Requires Docker.

```bash
# Build (ARM64 Mac — uses Dockerfile.local)
docker build -f Dockerfile.local -t karaoke-local .

# Run
docker run -p 8000:8000 karaoke-local
```

Then open [http://localhost:8000](http://localhost:8000).

> The first build takes ~10–15 minutes — it downloads PyTorch and the Demucs model (~800MB total) and bakes them into the image. Subsequent builds are fast due to layer caching.

---

## Limitations

- YouTube only (no other sources)
- Max song length: 10 minutes
- Rate limit: 3 requests per IP per minute
- Max 1 song processing at a time (CPU constraint)
- Vocal separation quality varies — works best on songs with clear vocal/instrumental separation

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + uvicorn |
| Audio download | yt-dlp |
| Vocal separation | Demucs `mdx_extra_q` model |
| ML framework | PyTorch (CPU-only) |
| Frontend | Vanilla HTML/CSS/JS |
| Hosting | Hugging Face Spaces |
