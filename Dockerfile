# Dockerfile for Flask ML Backend
# Optimized for Cloud Run, Fly.io, Render, etc.

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Remove proxy environment variables that cause issues with httpx/supabase
# (Railway and some platforms set these)
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV http_proxy=""
ENV https_proxy=""
ENV ALL_PROXY=""
ENV all_proxy=""

# Expose port (Cloud Run uses PORT env var, others can override)
EXPOSE 8080

# Use gunicorn with single worker (ML models are memory-intensive)
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --threads 4 app:app

