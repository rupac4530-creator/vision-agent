FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libgl1-mesa-glx libglib2.0-0 curl gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Pre-download YOLOv8n model (cached in Docker layer)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLO model will download at first run"

# Copy application code
COPY backend /app/backend

WORKDIR /app/backend

# Create required directories
RUN mkdir -p uploads frames analysis static logs stream_uploads

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use PORT env var for Railway/Render compatibility
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1

