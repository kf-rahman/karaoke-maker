FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create work directory with correct permissions
RUN mkdir -p /tmp/karaoke_work && chown appuser:appuser /tmp/karaoke_work

# Switch to non-root user
USER appuser

EXPOSE 8000

# Use multiple workers only if the instance has enough RAM for Demucs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "620"]
