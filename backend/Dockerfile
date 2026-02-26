FROM python:3.11-slim

WORKDIR /app

# System deps: ffmpeg, libGL for OpenCV, gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLOv8n model at build time (cached layer)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLO model will be downloaded at first run"

# Copy application
COPY . .

# Create required directories
RUN mkdir -p uploads frames analysis static

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
