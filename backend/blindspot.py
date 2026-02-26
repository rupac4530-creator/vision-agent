"""
blindspot.py — BlindSpot Guardian
Dashcam AI agent that identifies road hazards and alerts drivers in real-time.
Detects: vehicles, pedestrians, cyclists, obstacles, traffic lights, stop signs.
Classifies proximity and urgency, generates voice-ready alert messages.
"""
import os
import cv2
import base64
import time
import logging
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger("blindspot")

# Hazard classification
HAZARD_MAP = {
    # High priority: immediate danger
    "person": {"priority": 3, "alert": "⚠️ Pedestrian detected!", "color": "#ef4444", "level": "danger"},
    "bicycle": {"priority": 3, "alert": "⚠️ Cyclist ahead!", "color": "#ef4444", "level": "danger"},
    "motorcycle": {"priority": 3, "alert": "⚠️ Motorcycle nearby!", "color": "#ef4444", "level": "danger"},
    "dog": {"priority": 2, "alert": "🐕 Animal on road!", "color": "#f97316", "level": "warning"},
    "cat": {"priority": 2, "alert": "🐈 Animal on road!", "color": "#f97316", "level": "warning"},
    # Medium priority: traffic hazards
    "car": {"priority": 1, "alert": "🚗 Vehicle nearby", "color": "#f59e0b", "level": "caution"},
    "truck": {"priority": 2, "alert": "🚛 Large vehicle ahead", "color": "#f97316", "level": "warning"},
    "bus": {"priority": 2, "alert": "🚌 Bus ahead", "color": "#f97316", "level": "warning"},
    "traffic light": {"priority": 1, "alert": "🚦 Traffic light detected", "color": "#3b82f6", "level": "info"},
    "stop sign": {"priority": 2, "alert": "🛑 Stop sign!", "color": "#ef4444", "level": "warning"},
    # Low priority
    "bench": {"priority": 0, "alert": "🪑 Obstacle detected", "color": "#6366f1", "level": "info"},
    "fire hydrant": {"priority": 1, "alert": "🚒 Obstacle ahead", "color": "#f59e0b", "level": "caution"},
    "potted plant": {"priority": 0, "alert": "🌿 Obstacle ahead", "color": "#6366f1", "level": "info"},
}

SAFE_STATUS = {"level": "clear", "color": "#22c55e", "label": "✅ ROAD CLEAR"}


def _bbox_proximity_score(bbox: List[int], frame_w: int, frame_h: int) -> float:
    """
    Estimate proximity of detected object based on bounding box size + position.
    Returns 0-1 where 1 = very close.
    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    size_score = min(area / (frame_area * 0.01), 1.0)  # normalize: >1% of frame = close
    # Objects in lower half of frame are closer (forward camera)
    center_y = (y1 + y2) / 2
    vertical_score = center_y / frame_h
    return round(min((size_score * 0.6 + vertical_score * 0.4), 1.0), 3)


class BlindSpotGuardian:
    """AI dashcam guardian for real-time road hazard detection."""

    def __init__(self):
        self._yolo = None
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")
        self._alert_history: List[Dict] = []
        self._frame_count = 0
        self._total_hazards = 0
        self._session_start = time.time()

    def _load_yolo(self):
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
            self._yolo = YOLO(model_path)
            logger.info("BlindSpot YOLO loaded")
        except Exception as e:
            logger.warning("BlindSpot YOLO load failed: %s", e)

    def analyze(self, frame_b64: str) -> Dict[str, Any]:
        """Analyze dashcam frame for hazards."""
        t0 = time.time()
        self._frame_count += 1

        try:
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode frame")
        except Exception as e:
            return {"ok": False, "error": str(e)}

        h, w = frame.shape[:2]
        self._load_yolo()

        detections = []
        hazards = []
        max_priority = -1
        top_alert = None
        top_color = SAFE_STATUS["color"]
        top_level = "clear"

        if self._yolo is not None:
            try:
                results = self._yolo(frame, verbose=False, conf=0.4)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = r.names[cls_id].lower()
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        proximity = _bbox_proximity_score([x1, y1, x2, y2], w, h)

                        det = {
                            "label": label,
                            "conf": round(conf, 3),
                            "bbox": [x1, y1, x2, y2],
                            "proximity": proximity,
                        }
                        detections.append(det)

                        if label in HAZARD_MAP:
                            haz_info = HAZARD_MAP[label]
                            priority = haz_info["priority"]
                            # Boost priority if object is close
                            effective_priority = priority + (1 if proximity > 0.5 else 0)

                            hazards.append({
                                **det,
                                "alert_message": haz_info["alert"],
                                "color": haz_info["color"],
                                "level": haz_info["level"],
                                "effective_priority": effective_priority,
                            })
                            if effective_priority > max_priority:
                                max_priority = effective_priority
                                top_alert = haz_info["alert"]
                                top_color = haz_info["color"]
                                top_level = haz_info["level"]
            except Exception as e:
                logger.warning("BlindSpot YOLO error: %s", e)

        # Sort hazards by priority (highest first)
        hazards.sort(key=lambda x: -x["effective_priority"])

        if top_alert and top_level in ("danger", "warning"):
            self._total_hazards += 1
            self._alert_history.append({
                "timestamp": time.time(),
                "level": top_level,
                "message": top_alert,
                "detections": [h["label"] for h in hazards[:3]],
                "frame_id": self._frame_count,
            })
            self._alert_history = self._alert_history[-50:]

        driving_tips = self._get_driving_tip(hazards)

        return {
            "ok": True,
            "frame_id": self._frame_count,
            "detections": detections,
            "hazards": hazards,
            "hazard_count": len(hazards),
            "top_alert": top_alert or SAFE_STATUS["label"],
            "top_level": top_level,
            "top_color": top_color,
            "driving_tip": driving_tips,
            "total_hazards_session": self._total_hazards,
            "session_duration_s": round(time.time() - self._session_start),
            "latency_ms": round((time.time() - t0) * 1000),
        }

    def _get_driving_tip(self, hazards: List[Dict]) -> str:
        """Generate a quick driving tip based on detected hazards."""
        if not hazards:
            return "Road clear — maintain safe following distance."
        top = hazards[0]
        label = top["label"]
        prox = top["proximity"]

        tips = {
            "person": "Slow down, prepare to stop for pedestrian crossing.",
            "bicycle": "Give cyclists 1.5m clearance, reduce speed.",
            "motorcycle": "Check mirrors, motorcycles can be in blind spots.",
            "car": "Maintain safe following distance, check mirrors.",
            "truck": "Stay out of truck blind spots, don't cut in front.",
            "bus": "Watch for passengers alighting, prepare to stop.",
            "stop sign": "Come to a full stop, check all directions.",
            "traffic light": "Prepare for light change, reduce speed.",
            "dog": "Slow down, animals may dart into road unexpectedly.",
        }
        tip = tips.get(label, "Hazard detected, reduce speed and stay alert.")
        if prox > 0.7:
            tip = "⚠️ IMMINENT HAZARD — BRAKE NOW! " + tip
        return tip

    def get_alert_history(self, limit: int = 20) -> List[Dict]:
        return self._alert_history[-limit:]

    def get_stats(self) -> Dict:
        return {
            "frames_analyzed": self._frame_count,
            "total_hazards": self._total_hazards,
            "session_duration_s": round(time.time() - self._session_start),
            "alerts_logged": len(self._alert_history),
        }

    def reset_session(self):
        self._frame_count = 0
        self._total_hazards = 0
        self._alert_history = []
        self._session_start = time.time()


# Singleton
blindspot_guardian = BlindSpotGuardian()
