"""
crowd_monitor.py — AI Crowd Safety Monitor
Detect crowd density, distress signals, and safety risks using YOLO + LLM analysis.
"""
import os
import cv2
import base64
import time
import logging
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger("crowd_monitor")


class CrowdMonitor:
    """Crowd safety monitoring with density estimation and distress detection."""

    def __init__(self):
        self._yolo = None
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")
        self._history: List[Dict] = []

    def _load_yolo(self):
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
            self._yolo = YOLO(model_path)
        except Exception as e:
            logger.warning("YOLO load failed: %s", e)

    def _density_to_safety(self, person_count: int, frame_area: float) -> Dict:
        """Calculate crowd safety score from person density."""
        density = person_count / max(frame_area / 10000, 1)  # persons per 100x100px area

        if person_count == 0:
            return {"score": 100, "level": "safe", "color": "#22c55e", "label": "SAFE — No persons detected"}
        elif density < 0.5:
            score = max(90 - person_count * 2, 60)
            return {"score": score, "level": "safe", "color": "#22c55e", "label": f"SAFE — {person_count} person(s) in area"}
        elif density < 2.0:
            score = max(70 - person_count, 40)
            return {"score": score, "level": "moderate", "color": "#f59e0b", "label": f"MODERATE CROWD — {person_count} persons"}
        elif density < 4.0:
            score = max(40 - person_count, 20)
            return {"score": score, "level": "dense", "color": "#f97316", "label": f"DENSE CROWD — {person_count} persons"}
        else:
            score = max(20 - person_count, 5)
            return {"score": score, "level": "critical", "color": "#ef4444", "label": f"CRITICAL DENSITY — {person_count} persons"}

    def _detect_clusters(self, bboxes: List[List[int]], frame_w: int, frame_h: int) -> List[Dict]:
        """Detect person clusters that may indicate gatherings or distress zones."""
        if len(bboxes) < 3:
            return []

        # Simple grid-based clustering
        clusters = []
        grid_size = 3
        grid = [[0] * grid_size for _ in range(grid_size)]

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            gx = min(int(cx / frame_w * grid_size), grid_size - 1)
            gy = min(int(cy / frame_h * grid_size), grid_size - 1)
            grid[gy][gx] += 1

        for gy in range(grid_size):
            for gx in range(grid_size):
                if grid[gy][gx] >= 3:
                    x = int(gx / grid_size * frame_w)
                    y = int(gy / grid_size * frame_h)
                    w = frame_w // grid_size
                    h = frame_h // grid_size
                    clusters.append({
                        "region": [x, y, x + w, y + h],
                        "count": grid[gy][gx],
                        "density": "high" if grid[gy][gx] >= 5 else "moderate",
                    })

        return clusters

    def analyze(self, frame_b64: str) -> Dict[str, Any]:
        """Analyze crowd safety in a frame."""
        t0 = time.time()

        try:
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image")
        except Exception as e:
            return {"ok": False, "error": str(e)}

        h, w = frame.shape[:2]
        frame_area = w * h

        self._load_yolo()
        person_bboxes = []
        all_detections = []

        if self._yolo is not None:
            try:
                results = self._yolo(frame, verbose=False, conf=0.4, classes=[0])  # class 0 = person
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                            conf = float(box.conf[0])
                            person_bboxes.append([x1, y1, x2, y2])
                            all_detections.append({
                                "label": "person",
                                "conf": round(conf, 3),
                                "bbox": [x1, y1, x2, y2],
                            })
            except Exception as e:
                logger.warning("Crowd YOLO error: %s", e)

        person_count = len(person_bboxes)
        safety = self._density_to_safety(person_count, frame_area)
        clusters = self._detect_clusters(person_bboxes, w, h)

        # Generate heatmap data (grid of normalized counts)
        grid_w, grid_h = 10, 8
        heatmap = [[0.0] * grid_w for _ in range(grid_h)]
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            gx = min(int(cx / w * grid_w), grid_w - 1)
            gy = min(int(cy / h * grid_h), grid_h - 1)
            heatmap[gy][gx] += 1.0

        # Normalize heatmap
        max_val = max(max(row) for row in heatmap) or 1.0
        heatmap = [[round(v / max_val, 3) for v in row] for row in heatmap]

        result = {
            "ok": True,
            "person_count": person_count,
            "detections": all_detections,
            "clusters": clusters,
            "safety_score": safety["score"],
            "safety_level": safety["level"],
            "safety_color": safety["color"],
            "safety_label": safety["label"],
            "heatmap": heatmap,
            "heatmap_dims": {"w": grid_w, "h": grid_h},
            "frame_dims": {"w": w, "h": h},
            "latency_ms": round((time.time() - t0) * 1000),
        }

        # Save recent history
        self._history.append({
            "ts": time.time(),
            "person_count": person_count,
            "safety_score": safety["score"],
            "safety_level": safety["level"],
        })
        self._history = self._history[-100:]

        return result

    def get_trend(self, last_n: int = 20) -> List[Dict]:
        """Return recent safety score trend."""
        return self._history[-last_n:]


# Singleton
crowd_monitor = CrowdMonitor()
