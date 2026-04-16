FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home appuser

WORKDIR /app

# Install CPU-only PyTorch (small footprint, no CUDA)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchaudio==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install demucs and its deps separately (pin numpy<2 for demucs 4.0.1 compatibility)
RUN pip install --no-cache-dir "numpy<2" demucs==4.0.1 diffq

# Install web framework
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    pydantic==2.9.2 \
    yt-dlp

# Copy application
COPY . .

# Create work directory with correct permissions
RUN mkdir -p /tmp/karaoke_work && chown appuser:appuser /tmp/karaoke_work

USER appuser

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "620"]
