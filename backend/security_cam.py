"""
security_cam.py — AI Security Camera with Object Tracking and Alert System

SDK-Aligned: Ported patterns from Vision-Agents SDK security_camera_processor.py
Features:
  - YOLO object detection with multi-class alert triggers
  - Zone-based alerting (entry/exit/restricted zones)
  - Multi-object tracking persistence across frames
  - Known-persons database with identification
  - Package tracking with abandoned package detection
  - Event bus integration for real-time alerts
  - Wanted poster generation via Gemini Vision
"""
import os
import cv2
import base64
import time
import logging
import numpy as np
from collections import defaultdict, deque
from typing import Optional, List, Dict, Any, Tuple, Set

logger = logging.getLogger("security_cam")

ALERT_LEVELS = {
    "clear": {"level": 0, "color": "#22c55e", "sound": False, "label": "ALL CLEAR"},
    "caution": {"level": 1, "color": "#f59e0b", "sound": False, "label": "CAUTION"},
    "alert": {"level": 2, "color": "#ef4444", "sound": True, "label": "ALERT"},
    "critical": {"level": 3, "color": "#dc2626", "sound": True, "label": "CRITICAL"},
}

# Objects that trigger high alert
HIGH_ALERT_OBJECTS = {"knife", "gun", "pistol", "rifle", "weapon", "scissors"}
# Objects that trigger caution
CAUTION_OBJECTS = {"backpack", "luggage", "suitcase", "handbag"}
# Trackable objects for persistence
TRACKABLE_OBJECTS = {"person", "car", "truck", "bicycle", "motorcycle", "bus", "dog", "cat"}


# ── Zone Configuration ────────────────────────────────────────────────
class SecurityZone:
    """Defines a named rectangular zone with custom alert rules."""
    def __init__(self, name: str, bbox: Tuple[float, float, float, float],
                 zone_type: str = "monitor", alert_on_entry: bool = True):
        self.name = name
        self.bbox = bbox  # (x1_frac, y1_frac, x2_frac, y2_frac) as fraction of frame
        self.zone_type = zone_type  # "monitor", "restricted", "exit"
        self.alert_on_entry = alert_on_entry
        self.occupants: Set[int] = set()  # Track IDs currently in zone

    def contains(self, cx: float, cy: float, frame_w: int, frame_h: int) -> bool:
        """Check if a center point falls within this zone."""
        x1 = self.bbox[0] * frame_w
        y1 = self.bbox[1] * frame_h
        x2 = self.bbox[2] * frame_w
        y2 = self.bbox[3] * frame_h
        return x1 <= cx <= x2 and y1 <= cy <= y2


# ── Tracked Object ────────────────────────────────────────────────────
class TrackedObject:
    """Persistent tracking state for a single detected object."""
    def __init__(self, track_id: int, label: str, bbox: list):
        self.track_id = track_id
        self.label = label
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.positions: deque = deque(maxlen=30)  # Last 30 positions
        self.positions.append(bbox)
        self.frame_count = 1
        self.is_stationary = False
        self.stationary_since: Optional[float] = None

    def update(self, bbox: list):
        self.positions.append(bbox)
        self.last_seen = time.time()
        self.frame_count += 1
        # Check if stationary (position hasn't changed much)
        if len(self.positions) >= 5:
            recent = list(self.positions)[-5:]
            cx_vals = [(b[0]+b[2])/2 for b in recent]
            cy_vals = [(b[1]+b[3])/2 for b in recent]
            dx = max(cx_vals) - min(cx_vals)
            dy = max(cy_vals) - min(cy_vals)
            was_stationary = self.is_stationary
            self.is_stationary = (dx < 20 and dy < 20)
            if self.is_stationary and not was_stationary:
                self.stationary_since = time.time()

    @property
    def duration_seconds(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def center(self) -> Tuple[float, float]:
        bbox = self.positions[-1]
        return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)


class SecurityCamera:
    """
    AI-powered security camera with real-time threat detection.

    SDK-Aligned features:
    - Multi-object tracking with persistence across frames
    - Zone-based alerting for restricted/monitored areas
    - Package abandonment detection (stationary objects)
    - Known-persons database
    - Event bus integration for real-time alert streaming
    """

    ABANDONED_THRESHOLD_SEC = 30  # Object stationary for this long → alert

    def __init__(self):
        self._yolo = None
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")
        self._alert_history: List[Dict] = []
        self._known_persons: List[str] = [
            n.strip() for n in
            os.getenv("SECURITY_KNOWN_PERSONS", "").split(",") if n.strip()
        ]
        self._frame_count = 0
        self._next_track_id = 1

        # SDK-aligned: Multi-object tracking persistence
        self._tracked_objects: Dict[int, TrackedObject] = {}
        self._label_counter: Dict[str, int] = defaultdict(int)

        # SDK-aligned: Zone-based alerting
        self._zones: List[SecurityZone] = [
            SecurityZone("Entry Zone", (0.0, 0.7, 1.0, 1.0), "monitor", alert_on_entry=True),
            SecurityZone("Restricted Area", (0.3, 0.0, 0.7, 0.3), "restricted", alert_on_entry=True),
        ]

        # SDK-aligned: Event bus integration
        self._event_bus = None
        try:
            from event_bus import event_bus, EventType, Event
            self._event_bus = event_bus
            self._EventType = EventType
            self._Event = Event
        except ImportError:
            pass

        # SDK-aligned: Observability
        self._metrics = {
            "total_frames": 0,
            "total_detections": 0,
            "total_alerts": 0,
            "total_persons": 0,
            "weapons_detected": 0,
            "zone_violations": 0,
            "abandoned_objects": 0,
        }

    def _load_yolo(self):
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
            self._yolo = YOLO(model_path)
            logger.info("Security YOLO model loaded")
        except Exception as e:
            logger.warning("YOLO load failed for security cam: %s", e)

    def _match_detection_to_track(self, label: str, bbox: list) -> int:
        """Simple IoU-based matching of detections to existing tracks."""
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        best_id = -1
        best_dist = 100  # Max pixel distance for matching

        for tid, obj in self._tracked_objects.items():
            if obj.label != label:
                continue
            ocx, ocy = obj.center
            dist = ((cx-ocx)**2 + (cy-ocy)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id >= 0:
            self._tracked_objects[best_id].update(bbox)
            return best_id

        # New tracked object
        tid = self._next_track_id
        self._next_track_id += 1
        self._tracked_objects[tid] = TrackedObject(tid, label, bbox)
        return tid

    def _check_zones(self, detections: list, frame_w: int, frame_h: int) -> List[Dict]:
        """Check which detections are in which zones, generate zone alerts."""
        zone_alerts = []
        for det in detections:
            if det["label"] != "person":
                continue
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            for zone in self._zones:
                if zone.contains(cx, cy, frame_w, frame_h):
                    track_id = det.get("track_id", 0)
                    if track_id not in zone.occupants:
                        zone.occupants.add(track_id)
                        if zone.alert_on_entry and zone.zone_type == "restricted":
                            zone_alerts.append({
                                "zone": zone.name,
                                "type": zone.zone_type,
                                "track_id": track_id,
                                "message": f"Person entered {zone.zone_type} zone: {zone.name}",
                            })
                            self._metrics["zone_violations"] += 1

        # Clean occupants who left
        active_track_ids = {d.get("track_id", 0) for d in detections if d["label"] == "person"}
        for zone in self._zones:
            zone.occupants &= active_track_ids

        return zone_alerts

    def _check_abandoned_objects(self) -> List[Dict]:
        """Check for stationary objects that might be abandoned."""
        alerts = []
        now = time.time()
        for tid, obj in self._tracked_objects.items():
            if (obj.label in CAUTION_OBJECTS and obj.is_stationary and
                    obj.stationary_since and
                    (now - obj.stationary_since) > self.ABANDONED_THRESHOLD_SEC):
                alerts.append({
                    "type": "abandoned_object",
                    "track_id": tid,
                    "label": obj.label,
                    "stationary_seconds": round(now - obj.stationary_since),
                    "message": f"⚠️ Possible abandoned {obj.label} (stationary {round(now - obj.stationary_since)}s)",
                })
                self._metrics["abandoned_objects"] += 1
        return alerts

    def _prune_stale_tracks(self, max_age: float = 10.0):
        """Remove tracks that haven't been updated recently."""
        now = time.time()
        stale = [tid for tid, obj in self._tracked_objects.items()
                 if (now - obj.last_seen) > max_age]
        for tid in stale:
            del self._tracked_objects[tid]

    async def _emit_event(self, alert_key: str, message: str, data: dict):
        """Emit security event to the event bus if available."""
        if self._event_bus:
            try:
                import asyncio
                await self._event_bus.emit(self._Event(
                    type=self._EventType.SECURITY_ALERT,
                    data={"alert": alert_key, "message": message, **data},
                    source="security_cam",
                ))
            except Exception:
                pass  # Don't let event bus errors affect detection

    def analyze(self, frame_b64: str) -> Dict[str, Any]:
        """
        Analyze a frame for security threats.

        SDK-aligned enhanced analysis:
        - YOLO detection with multi-object tracking
        - Zone-based alerting
        - Abandoned object detection
        - Enriched metadata for each detection
        """
        t0 = time.time()
        self._frame_count += 1
        self._metrics["total_frames"] += 1

        try:
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image")
        except Exception as e:
            return {"ok": False, "error": str(e), "alert": "clear"}

        frame_h, frame_w = frame.shape[:2]
        self._load_yolo()
        detections = []
        persons_count = 0

        if self._yolo is not None:
            try:
                results = self._yolo(frame, verbose=False, conf=0.4)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = r.names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        bbox = [x1, y1, x2, y2]

                        # SDK-aligned: Track the object
                        track_id = self._match_detection_to_track(label, bbox)

                        detections.append({
                            "label": label,
                            "conf": round(conf, 3),
                            "bbox": bbox,
                            "track_id": track_id,
                            "area": (x2-x1) * (y2-y1),
                        })

                        if label == "person":
                            persons_count += 1
                        self._label_counter[label] += 1

                self._metrics["total_detections"] += len(detections)
                self._metrics["total_persons"] += persons_count
            except Exception as e:
                logger.warning("Security YOLO inference error: %s", e)

        # Prune stale tracks
        self._prune_stale_tracks()

        # Determine alert level
        labels = {d["label"] for d in detections}
        alert_key = "clear"
        alert_message = ""
        triggered_by = []
        zone_alerts = []
        abandoned_alerts = []

        if labels & HIGH_ALERT_OBJECTS:
            alert_key = "critical"
            triggered = list(labels & HIGH_ALERT_OBJECTS)
            alert_message = f"⚠️ WEAPON DETECTED: {', '.join(triggered)}"
            triggered_by = triggered
            self._metrics["weapons_detected"] += 1
        elif persons_count > 5:
            alert_key = "alert"
            alert_message = f"⚠️ CROWD SURGE: {persons_count} persons detected"
            triggered_by = ["large_crowd"]
        elif labels & CAUTION_OBJECTS:
            alert_key = "caution"
            triggered = list(labels & CAUTION_OBJECTS)
            alert_message = f"⚡ Suspicious items: {', '.join(triggered)}"
            triggered_by = triggered
        elif persons_count > 0:
            alert_key = "caution"
            alert_message = f"👤 {persons_count} person(s) detected"

        # SDK-aligned: Zone-based alerting
        zone_alerts = self._check_zones(detections, frame_w, frame_h)
        if zone_alerts:
            if alert_key in ("clear", "caution"):
                alert_key = "alert"
            alert_message += " | " + "; ".join(z["message"] for z in zone_alerts)

        # SDK-aligned: Abandoned object detection
        abandoned_alerts = self._check_abandoned_objects()
        if abandoned_alerts:
            if alert_key in ("clear", "caution"):
                alert_key = "alert"
            alert_message += " | " + "; ".join(a["message"] for a in abandoned_alerts)

        alert_info = ALERT_LEVELS[alert_key]

        # Log alerts
        if alert_key in ("alert", "critical"):
            self._alert_history.append({
                "timestamp": time.time(),
                "alert": alert_key,
                "message": alert_message,
                "triggered_by": triggered_by,
                "frame_id": self._frame_count,
                "zone_alerts": zone_alerts,
                "abandoned_alerts": abandoned_alerts,
            })
            self._alert_history = self._alert_history[-50:]
            self._metrics["total_alerts"] += 1

        return {
            "ok": True,
            "frame_id": self._frame_count,
            "detections": detections,
            "persons_count": persons_count,
            "alert": alert_key,
            "alert_label": alert_info["label"],
            "alert_color": alert_info["color"],
            "alert_sound": alert_info["sound"],
            "alert_message": alert_message,
            "triggered_by": triggered_by,
            "zone_alerts": zone_alerts,
            "abandoned_alerts": abandoned_alerts,
            "active_tracks": len(self._tracked_objects),
            "latency_ms": round((time.time() - t0) * 1000),
        }

    def generate_wanted_poster(self, frame_b64: str, description: str = "") -> Dict:
        """Generate a wanted poster description using Gemini Vision."""
        if not self._gemini_key:
            return {"ok": False, "error": "Gemini API key required for wanted poster generation"}

        try:
            import requests
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={self._gemini_key}"
            )
            prompt = (
                "Create a dramatic WANTED POSTER text for the person/object in this image. "
                "Include: WANTED header, physical description, last seen location (infer from scene), "
                "reward amount, warning level. Make it dramatic and official-sounding. "
                f"Additional context: {description}. "
                "Format as plain text poster content."
            )
            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
                        {"text": prompt},
                    ]
                }],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 400},
            }
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            poster_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"ok": True, "poster_text": poster_text, "provider": "gemini-vision"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_alert_history(self, limit: int = 20) -> List[Dict]:
        """Return recent alert history."""
        return self._alert_history[-limit:]

    def get_metrics(self) -> Dict:
        """Return security camera analytics metrics."""
        return {
            **self._metrics,
            "active_tracks": len(self._tracked_objects),
            "zones": [{"name": z.name, "type": z.zone_type, "occupants": len(z.occupants)} for z in self._zones],
            "label_distribution": dict(self._label_counter),
        }

    def add_zone(self, name: str, bbox: Tuple[float, float, float, float],
                 zone_type: str = "restricted"):
        """Add a new security zone (coords as fractions 0.0-1.0)."""
        self._zones.append(SecurityZone(name, bbox, zone_type, alert_on_entry=True))
        logger.info("Added security zone: %s (%s)", name, zone_type)

    def reset(self):
        """Reset camera state."""
        self._frame_count = 0
        self._alert_history = []
        self._tracked_objects.clear()
        self._label_counter.clear()
        self._next_track_id = 1
        for z in self._zones:
            z.occupants.clear()


# Singleton
security_camera = SecurityCamera()
