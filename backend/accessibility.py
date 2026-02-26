"""
accessibility.py — Accessibility Agent
Real-time vision assistance for visually impaired users.
Features:
- Live scene description (every N seconds)
- Object and text detection
- Reading assistance (point camera at text → AI reads it)
- Navigation hints (detect doors, stairs, obstacles)
- Currency/product recognition
Uses Gemini Vision for rich descriptions + YOLO for spatial object awareness.
"""
import os
import cv2
import base64
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger("accessibility")

# Objects relevant for navigation/accessibility
NAVIGATION_OBJECTS = {
    "door", "stairs", "elevator", "escalator", "chair", "table", "bed",
    "couch", "sofa", "toilet", "sink", "bottle", "cup", "keyboard",
    "laptop", "cell phone", "book", "remote", "clock", "person",
    "dog", "cat", "car", "bicycle", "traffic light", "stop sign",
}

DESCRIPTION_MODES = {
    "scene": "Describe what you see in this scene in 2-3 short sentences for a blind person.",
    "read": "Read all visible text in this image. List each text element on a new line.",
    "navigate": (
        "Describe what is directly ahead for safe navigation. "
        "Mention any obstacles, steps, doors, or hazards. Be concise and directional."
    ),
    "identify": (
        "Identify the main object or item in the center of this image. "
        "Describe it clearly including color, size, and any text/labels visible."
    ),
    "currency": (
        "Identify any currency (coins or notes) visible in this image. "
        "State denomination and currency type. If not currency, say what you see."
    ),
}


class AccessibilityAgent:
    """Real-time accessibility vision assistant."""

    def __init__(self):
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")
        self._yolo = None
        self._caption_history: List[Dict] = []
        self._last_description = ""
        self._last_description_ts = 0
        self._frame_count = 0
        self._description_interval = 3.0  # seconds between auto-descriptions

    def _load_yolo(self):
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
            self._yolo = YOLO(model_path)
        except Exception as e:
            logger.warning("Accessibility YOLO load failed: %s", e)

    def describe_scene(
        self,
        frame_b64: str,
        mode: str = "scene",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Describe a frame using Gemini Vision.
        mode: 'scene' | 'read' | 'navigate' | 'identify' | 'currency'
        """
        t0 = time.time()
        self._frame_count += 1

        # Rate-limit auto-refresh unless forced
        now = time.time()
        if (
            not force_refresh
            and (now - self._last_description_ts) < self._description_interval
            and mode == "scene"
            and self._last_description
        ):
            return {
                "ok": True,
                "description": self._last_description,
                "mode": mode,
                "cached": True,
                "latency_ms": 0,
            }

        # Get YOLO spatial context
        spatial_context = self._get_spatial_context(frame_b64)

        # Gemini Vision description
        description = None
        provider = "yolo-only"

        if self._gemini_key:
            try:
                description, provider = self._gemini_describe(frame_b64, mode, spatial_context)
            except Exception as e:
                logger.warning("Gemini accessibility describe failed: %s", e)

        if not description:
            # Fallback: use YOLO detections to build a simple description
            description = self._yolo_description(spatial_context)
            provider = "yolo-extractive"

        # Store in history
        self._last_description = description
        self._last_description_ts = now

        entry = {
            "ts": now,
            "mode": mode,
            "description": description,
            "provider": provider,
            "objects_detected": spatial_context.get("objects", []),
        }
        self._caption_history.append(entry)
        self._caption_history = self._caption_history[-100:]

        return {
            "ok": True,
            "description": description,
            "mode": mode,
            "provider": provider,
            "spatial_context": spatial_context,
            "cached": False,
            "latency_ms": round((time.time() - t0) * 1000),
        }

    def _get_spatial_context(self, frame_b64: str) -> Dict:
        """Run YOLO to get object positions for spatial context."""
        try:
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"objects": [], "detections": []}
        except Exception:
            return {"objects": [], "detections": []}

        h, w = frame.shape[:2]
        self._load_yolo()

        detections = []
        if self._yolo:
            try:
                results = self._yolo(frame, verbose=False, conf=0.4)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = r.names[cls_id].lower()
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        # Compute position description
                        pos_h = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
                        pos_v = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"

                        detections.append({
                            "label": label,
                            "conf": round(conf, 2),
                            "position": f"{pos_v}-{pos_h}",
                            "bbox": [x1, y1, x2, y2],
                        })
            except Exception as e:
                logger.warning("Accessibility YOLO inference error: %s", e)

        # Filter for navigation-relevant objects
        nav_objects = [d for d in detections if d["label"] in NAVIGATION_OBJECTS]

        return {
            "objects": [f"{d['label']} ({d['position']})" for d in nav_objects[:6]],
            "detections": detections,
            "total_objects": len(detections),
        }

    def _gemini_describe(self, frame_b64: str, mode: str, spatial: Dict) -> tuple:
        import requests
        prompt = DESCRIPTION_MODES.get(mode, DESCRIPTION_MODES["scene"])
        if spatial.get("objects"):
            objects_hint = ", ".join(spatial["objects"][:5])
            prompt += f"\n\nYOLO detected nearby: {objects_hint}"

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={self._gemini_key}"
        )
        payload = {
            "contents": [{"parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
                {"text": prompt},
            ]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 150},
        }
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text, "gemini-2.0-flash"

    def _yolo_description(self, spatial: Dict) -> str:
        objects = spatial.get("objects", [])
        detections = spatial.get("detections", [])
        if not detections:
            return "Scene is clear. No objects detected by local sensor."
        labels = [d["label"] for d in detections[:6]]
        unique = list(dict.fromkeys(labels))
        return f"Detected: {', '.join(unique)}. " + (
            f"Nearby navigation objects: {', '.join(objects[:4])}." if objects else ""
        )

    def get_live_caption(self) -> Dict:
        """Return the most recent description as a live caption."""
        return {
            "caption": self._last_description or "Waiting for scene description...",
            "ts": self._last_description_ts,
            "age_seconds": round(time.time() - self._last_description_ts, 1),
        }

    def get_history(self, limit: int = 20) -> List[Dict]:
        return self._caption_history[-limit:]

    def set_description_interval(self, seconds: float):
        self._description_interval = max(1.0, min(30.0, seconds))

    def get_stats(self) -> Dict:
        return {
            "frames_processed": self._frame_count,
            "captions_generated": len(self._caption_history),
            "last_caption_age_s": round(time.time() - self._last_description_ts, 1),
            "description_interval_s": self._description_interval,
        }


# Singleton
accessibility_agent = AccessibilityAgent()
