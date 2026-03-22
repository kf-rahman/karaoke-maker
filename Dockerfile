FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home appuser

WORKDIR /app

# Install CPU-only PyTorch (small footprint, no CUDA)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchaudio==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install demucs and its deps separately
RUN pip install --no-cache-dir demucs==4.0.1

# Install web framework
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    pydantic==2.9.2 \
    yt-dlp==2024.12.23

# Copy application
COPY . .

# Create work directory with correct permissions
RUN mkdir -p /tmp/karaoke_work && chown appuser:appuser /tmp/karaoke_work

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "620"]
