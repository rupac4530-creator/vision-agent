# backend/vision_worker.py
"""
Singleton YOLO vision worker with per-frame inference, latency tracking,
object stability scoring, and configurable model upgrade.

Usage:
    from vision_worker import vision_worker
    detections = vision_worker.infer_frame("path/to/frame.jpg")
    batch = vision_worker.infer_frames_batch(["frame1.jpg", "frame2.jpg"])
    metrics = vision_worker.get_metrics()
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Optional
from collections import defaultdict

logger = logging.getLogger("vision_worker")

# Use yolov8s (small) for better accuracy — 2× more accurate than nano
# Falls back to yolov8n if not available
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8s.pt")

# Detection parameters — tuned for balanced accuracy
CONF_THRESHOLD = float(os.getenv("YOLO_CONF", "0.20"))  # lower = catch more objects
NMS_IOU = float(os.getenv("YOLO_IOU", "0.50"))          # NMS threshold
MAX_DET = int(os.getenv("YOLO_MAX_DET", "50"))           # max detections per frame


class VisionWorker:
    """Thread-safe singleton YOLO inference worker with latency tracking and
    object stability scoring."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._model_name_loaded = None
        self._metrics = {
            "total_inferences": 0,
            "total_frames": 0,
            "latencies_ms": [],  # keep last 200
            "last_detections": [],
        }
        # Object stability tracking (rolling memory across frames)
        self._object_memory: dict[str, list[float]] = defaultdict(list)  # label → [confidence history]
        self._stable_objects: set[str] = set()  # labels seen in ≥3 consecutive frames

    def _ensure_model(self):
        """Lazy-load YOLO model on first use. Try yolov8s first, fallback to yolov8n."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from ultralytics import YOLO
                    import torch

                    model_to_load = MODEL_NAME
                    logger.info("Loading YOLO model: %s", model_to_load)
                    t0 = time.time()

                    try:
                        self._model = YOLO(model_to_load)
                        self._model_name_loaded = model_to_load
                    except Exception as e:
                        # Fallback to nano if requested model fails
                        logger.warning("Failed to load %s: %s  — falling back to yolov8n.pt",
                                       model_to_load, e)
                        self._model = YOLO("yolov8n.pt")
                        self._model_name_loaded = "yolov8n.pt"

                    # Use half precision on GPU for speed
                    if torch.cuda.is_available():
                        self._model.to("cuda")
                        logger.info("YOLO running on CUDA GPU")
                    else:
                        # Limit CPU threads to avoid system overload
                        torch.set_num_threads(max(2, os.cpu_count() // 2))

                    logger.info("YOLO loaded in %.2fs (model=%s)",
                                time.time() - t0, self._model_name_loaded)

    def infer_frame(self, frame_path: str, conf: float = None) -> list[dict]:
        """
        Run inference on a single frame.
        Returns: [{label, confidence, bbox: [x1, y1, x2, y2], stable: bool}]
        """
        self._ensure_model()
        if conf is None:
            conf = CONF_THRESHOLD
        t0 = time.time()

        try:
            results = self._model.predict(
                source=frame_path,
                conf=conf,
                iou=NMS_IOU,
                max_det=MAX_DET,
                verbose=False,
            )
            detections = []
            r = results[0]

            # Track which labels appear in this frame
            frame_labels = set()

            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                label = self._model.names.get(cls_id, str(cls_id))
                # xyxy format: [x1, y1, x2, y2]
                coords = box.xyxy[0].tolist()

                frame_labels.add(label)

                # Confidence-based quality tag
                if conf_val >= 0.70:
                    quality = "high"
                elif conf_val >= 0.40:
                    quality = "medium"
                else:
                    quality = "low"

                detections.append({
                    "label": label,
                    "confidence": round(conf_val, 3),
                    "bbox": [round(c, 1) for c in coords],
                    "quality": quality,
                    "stable": label in self._stable_objects,
                })

            # Update object stability memory
            self._update_object_memory(frame_labels)

            latency_ms = round((time.time() - t0) * 1000, 1)
            self._update_metrics(latency_ms, detections)
            return detections

        except Exception as e:
            logger.error("Inference error on %s: %s", frame_path, e)
            return []

    def _update_object_memory(self, current_labels: set[str]):
        """Track object presence across frames for stability scoring."""
        # Increment count for current labels, reset for absent ones
        with self._lock:
            all_labels = set(self._object_memory.keys()) | current_labels
            for label in all_labels:
                if label in current_labels:
                    self._object_memory[label].append(1.0)
                    if len(self._object_memory[label]) > 10:
                        self._object_memory[label] = self._object_memory[label][-10:]
                    # Stable if seen in ≥3 of last 5 frames
                    recent = self._object_memory[label][-5:]
                    if sum(recent) >= 3:
                        self._stable_objects.add(label)
                    else:
                        self._stable_objects.discard(label)
                else:
                    self._object_memory[label].append(0.0)
                    if len(self._object_memory[label]) > 10:
                        self._object_memory[label] = self._object_memory[label][-10:]
                    recent = self._object_memory[label][-5:]
                    if sum(recent) < 2:
                        self._stable_objects.discard(label)

    def infer_frames_batch(self, frame_paths: list[str], conf: float = None) -> list[list[dict]]:
        """
        Run inference on multiple frames.
        Returns: [[detections_frame_1], [detections_frame_2], ...]
        """
        self._ensure_model()
        all_detections = []
        t0 = time.time()

        for fp in frame_paths:
            dets = self.infer_frame(fp, conf)
            all_detections.append(dets)

        total_ms = round((time.time() - t0) * 1000, 1)
        logger.info("Batch inference: %d frames in %.1fms (%.1fms/frame)",
                     len(frame_paths), total_ms,
                     total_ms / max(len(frame_paths), 1))
        return all_detections

    def _update_metrics(self, latency_ms: float, detections: list):
        """Thread-safe metrics update."""
        with self._lock:
            self._metrics["total_inferences"] += 1
            self._metrics["total_frames"] += 1
            self._metrics["latencies_ms"].append(latency_ms)
            # Keep only last 200
            if len(self._metrics["latencies_ms"]) > 200:
                self._metrics["latencies_ms"] = self._metrics["latencies_ms"][-200:]
            self._metrics["last_detections"] = detections

    def get_metrics(self) -> dict:
        """Return current performance metrics."""
        with self._lock:
            lats = self._metrics["latencies_ms"]
            if lats:
                sorted_lats = sorted(lats)
                n = len(sorted_lats)
                avg = round(sum(sorted_lats) / n, 1)
                median = round(sorted_lats[n // 2], 1)
                p90 = round(sorted_lats[int(n * 0.9)], 1) if n >= 10 else round(sorted_lats[-1], 1)
                fps = round(1000.0 / avg, 1) if avg > 0 else 0
            else:
                avg = median = p90 = fps = 0

            # Average confidence of recent detections
            recent_dets = self._metrics["last_detections"]
            avg_conf = 0
            if recent_dets:
                avg_conf = round(sum(d["confidence"] for d in recent_dets) / len(recent_dets), 3)

            return {
                "total_inferences": self._metrics["total_inferences"],
                "total_frames": self._metrics["total_frames"],
                "avg_latency_ms": avg,
                "median_latency_ms": median,
                "p90_latency_ms": p90,
                "model_fps": fps,
                "last_detections_count": len(recent_dets),
                "avg_confidence": avg_conf,
                "stable_objects": list(self._stable_objects),
                "model": self._model_name_loaded or MODEL_NAME,
            }


# ── Singleton instance ───────────────────────────────────────────────
vision_worker = VisionWorker()
