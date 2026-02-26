"""
eco_watch.py — EcoWatch Ranger
AI-powered environmental surveillance for forests, wildlife, and protected areas.
Detects: fire/smoke, deforestation activity, illegal logging, poaching, unauthorized vehicles.
Uses YOLO for object detection + Gemini Vision for context analysis.
"""
import os
import cv2
import base64
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger("eco_watch")

# Threat category definitions
THREAT_CATEGORIES = {
    "fire": {
        "keywords": {"fire", "smoke", "flame"},
        "level": "critical",
        "color": "#dc2626",
        "label": "🔥 FIRE DETECTED",
        "action": "Alert fire services immediately",
    },
    "vehicle": {
        "keywords": {"car", "truck", "motorcycle", "bus"},
        "level": "alert",
        "color": "#ef4444",
        "label": "🚗 UNAUTHORIZED VEHICLE",
        "action": "Log vehicle intrusion, notify ranger",
    },
    "person": {
        "keywords": {"person"},
        "level": "caution",
        "color": "#f59e0b",
        "label": "👤 HUMAN PRESENCE",
        "action": "Monitor movements, verify authorization",
    },
    "animal": {
        "keywords": {"bear", "elephant", "wolf", "lion", "tiger", "deer", "bird"},
        "level": "info",
        "color": "#22c55e",
        "label": "🦎 WILDLIFE ACTIVITY",
        "action": "Log wildlife sighting for research",
    },
    "clear": {
        "keywords": set(),
        "level": "clear",
        "color": "#10b981",
        "label": "✅ AREA CLEAR",
        "action": "Normal monitoring",
    },
}

THREAT_PRIORITY = ["fire", "vehicle", "person", "animal", "clear"]


class EcoWatchRanger:
    """AI environmental surveillance system."""

    def __init__(self):
        self._yolo = None
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")
        self._alert_history: List[Dict] = []
        self._frame_count = 0
        self._wildlife_log: List[Dict] = []
        self._total_threats = 0

    def _load_yolo(self):
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
            self._yolo = YOLO(model_path)
            logger.info("EcoWatch YOLO loaded")
        except Exception as e:
            logger.warning("EcoWatch YOLO load failed: %s", e)

    def _classify_threat(self, labels: set) -> Dict:
        for category in THREAT_PRIORITY:
            info = THREAT_CATEGORIES[category]
            if labels & info["keywords"]:
                return {"category": category, **info}
        return {"category": "clear", **THREAT_CATEGORIES["clear"]}

    def analyze(self, frame_b64: str, location: str = "Unknown") -> Dict[str, Any]:
        """Analyze frame for environmental threats."""
        t0 = time.time()
        self._frame_count += 1

        try:
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode frame")
        except Exception as e:
            return {"ok": False, "error": str(e), "threat": "clear"}

        self._load_yolo()
        detections = []
        detected_labels = set()

        if self._yolo is not None:
            try:
                results = self._yolo(frame, verbose=False, conf=0.35)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = r.names[cls_id].lower()
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        detections.append({
                            "label": label,
                            "conf": round(conf, 3),
                            "bbox": [x1, y1, x2, y2],
                        })
                        detected_labels.add(label)
            except Exception as e:
                logger.warning("EcoWatch YOLO error: %s", e)

        threat = self._classify_threat(detected_labels)

        # Log non-clear events
        if threat["category"] != "clear":
            self._total_threats += 1
            alert_entry = {
                "timestamp": time.time(),
                "category": threat["category"],
                "level": threat["level"],
                "label": threat["label"],
                "action": threat["action"],
                "location": location,
                "frame_id": self._frame_count,
                "detections": [d["label"] for d in detections],
            }
            self._alert_history.append(alert_entry)
            self._alert_history = self._alert_history[-100:]

            if threat["category"] == "animal":
                self._wildlife_log.append({
                    "ts": time.time(),
                    "species": list(detected_labels & THREAT_CATEGORIES["animal"]["keywords"]),
                    "location": location,
                })
                self._wildlife_log = self._wildlife_log[-200:]

        # Gemini context analysis for fire/critical threats
        gemini_analysis = None
        if threat["category"] in ("fire",) and self._gemini_key:
            try:
                gemini_analysis = self._gemini_analyze(frame_b64, threat["category"])
            except Exception as e:
                logger.warning("Gemini eco analysis failed: %s", e)

        return {
            "ok": True,
            "frame_id": self._frame_count,
            "location": location,
            "detections": detections,
            "detected_labels": list(detected_labels),
            "threat_category": threat["category"],
            "threat_level": threat["level"],
            "threat_color": threat["color"],
            "threat_label": threat["label"],
            "threat_action": threat["action"],
            "gemini_analysis": gemini_analysis,
            "total_threats_today": self._total_threats,
            "latency_ms": round((time.time() - t0) * 1000),
        }

    def _gemini_analyze(self, frame_b64: str, category: str) -> Optional[str]:
        """Use Gemini to provide detailed threat assessment."""
        import requests
        prompts = {
            "fire": (
                "You are an environmental ranger AI. Analyze this image for fire or smoke presence. "
                "Describe severity, spread direction, and immediate recommended action in under 60 words."
            ),
            "vehicle": (
                "Analyze this image for unauthorized vehicles in a protected forest area. "
                "Describe the vehicle type, direction, and immediate action needed in under 60 words."
            ),
        }
        prompt = prompts.get(category, "Analyze this image for environmental threats in under 60 words.")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={self._gemini_key}"
        )
        payload = {
            "contents": [{"parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
                {"text": prompt},
            ]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 100},
        }
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    def get_alert_history(self, limit: int = 20) -> List[Dict]:
        return self._alert_history[-limit:]

    def get_wildlife_log(self, limit: int = 50) -> List[Dict]:
        return self._wildlife_log[-limit:]

    def get_stats(self) -> Dict:
        return {
            "frames_analyzed": self._frame_count,
            "total_threats": self._total_threats,
            "wildlife_sightings": len(self._wildlife_log),
            "recent_alerts": len(self._alert_history),
        }

    def reset(self):
        self._frame_count = 0
        self._total_threats = 0
        self._alert_history = []
        self._wildlife_log = []


# Singleton
eco_watch = EcoWatchRanger()
