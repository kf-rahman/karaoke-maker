FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home appuser

WORKDIR /app

# Install Demucs (pulls in torch + torchaudio), Whisper, and web stack
RUN pip install --no-cache-dir \
    demucs \
    openai-whisper \
    soundfile \
    unidecode \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    pydantic==2.9.2 \
    yt-dlp

# Copy application
COPY . .

# Create work directory with correct permissions
RUN mkdir -p /tmp/karaoke_work && chown appuser:appuser /tmp/karaoke_work

USER appuser

# Pre-download HTDemucs and Whisper models so they're baked into the image
RUN python3 -c "from demucs.pretrained import get_model; get_model('htdemucs')" && \
    python3 -c "import whisper; whisper.load_model('small')"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300"]
