FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home appuser

WORKDIR /app

# Install Spleeter (pulls in TensorFlow automatically)
RUN pip install --no-cache-dir spleeter

# Install web framework, YouTube downloader, and Whisper
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    pydantic==2.9.2 \
    yt-dlp \
    openai-whisper

# Copy application
COPY . .

# Create work directory with correct permissions
RUN mkdir -p /tmp/karaoke_work && chown appuser:appuser /tmp/karaoke_work

USER appuser

# Tell Spleeter to store models in the user's home dir (writable by appuser)
ENV MODEL_PATH=/home/appuser/pretrained_models

# Pre-download Spleeter and Whisper models so they're baked into the image
RUN ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t 1 /tmp/test.wav && \
    spleeter separate -p spleeter:2stems -o /tmp/sp_test /tmp/test.wav && \
    python3 -c "import whisper; whisper.load_model('small')" && \
    rm -rf /tmp/test.wav /tmp/sp_test

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300"]
