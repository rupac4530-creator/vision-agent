# backend/pose_engine.py
"""
Pose & Action Engine — Real-Time Fitness Coaching

Features:
  - Keypoint angle calculation (knee, hip, elbow, shoulder)
  - Rep counter for common exercises (squat, pushup, curl, lunge)
  - Posture correction hints
  - Works with YOLO-pose keypoints OR MediaPipe (fallback)

Usage:
    from pose_engine import PoseEngine
    engine = PoseEngine()
    result = engine.process_frame(frame_bgr, track_id=0)
"""

import math
import time
import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger("pose_engine")

# ── YOLO Pose keypoint indices (COCO 17-point format) ─────────────────
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


def _angle(a, b, c) -> float:
    """Compute the angle at point B between vectors BA and BC.
    a, b, c = (x, y) tuples.
    Returns angle in degrees [0, 180].
    """
    try:
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        if mag_ba < 1e-6 or mag_bc < 1e-6:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_angle))
    except Exception:
        return 0.0


def _kp_xy(keypoints: list, name: str) -> Optional[tuple]:
    """Extract (x, y) from keypoint list by name. Returns None if missing."""
    idx = KP.get(name)
    if idx is None or idx >= len(keypoints):
        return None
    kp = keypoints[idx]
    # YOLO keypoints can be [x, y] or [x, y, conf]
    if isinstance(kp, (list, tuple)) and len(kp) >= 2:
        x, y = float(kp[0]), float(kp[1])
        if x == 0 and y == 0:
            return None
        # Check confidence if available
        if len(kp) >= 3 and float(kp[2]) < 0.3:
            return None
        return (x, y)
    return None


# ── Exercise detector classes ──────────────────────────────────────────
class SquatDetector:
    """Counts squats using knee angle threshold cycling."""
    exercise = "squat"

    def __init__(self):
        self.reps = 0
        self._down = False
        self._angles: List[float] = []

    def update(self, keypoints: list) -> Dict:
        # Use average of left and right knee angles
        l_hip = _kp_xy(keypoints, "left_hip")
        l_knee = _kp_xy(keypoints, "left_knee")
        l_ankle = _kp_xy(keypoints, "left_ankle")
        r_hip = _kp_xy(keypoints, "right_hip")
        r_knee = _kp_xy(keypoints, "right_knee")
        r_ankle = _kp_xy(keypoints, "right_ankle")

        angles = []
        if l_hip and l_knee and l_ankle:
            angles.append(_angle(l_hip, l_knee, l_ankle))
        if r_hip and r_knee and r_ankle:
            angles.append(_angle(r_hip, r_knee, r_ankle))

        if not angles:
            return {"reps": self.reps, "angle": None, "state": "unknown", "corrections": []}

        avg_angle = sum(angles) / len(angles)
        self._angles.append(avg_angle)
        if len(self._angles) > 10:
            self._angles.pop(0)

        smooth = sum(self._angles) / len(self._angles)
        corrections = []

        # State machine: angle < 100 = "down", > 160 = "up"
        if smooth < 100 and not self._down:
            self._down = True
        elif smooth > 160 and self._down:
            self._down = False
            self.reps += 1

        state = "down" if self._down else "up"

        # Posture corrections
        if smooth < 60:
            corrections.append("⚠️ Going too deep — stop at 90°")
        if smooth > 100 and state == "down":
            corrections.append("💡 Lower more — aim for 90° knee angle")

        # Check forward lean (hip above knee)
        if l_hip and l_knee and l_hip[0] > l_knee[0] + 30:
            corrections.append("🏋️ Lean forward less — keep chest up")

        return {"reps": self.reps, "angle": round(smooth, 1), "state": state, "corrections": corrections}


class PushupDetector:
    """Counts push-ups using elbow angle."""
    exercise = "pushup"

    def __init__(self):
        self.reps = 0
        self._down = False
        self._angles: List[float] = []

    def update(self, keypoints: list) -> Dict:
        l_shoulder = _kp_xy(keypoints, "left_shoulder")
        l_elbow = _kp_xy(keypoints, "left_elbow")
        l_wrist = _kp_xy(keypoints, "left_wrist")
        r_shoulder = _kp_xy(keypoints, "right_shoulder")
        r_elbow = _kp_xy(keypoints, "right_elbow")
        r_wrist = _kp_xy(keypoints, "right_wrist")

        angles = []
        if l_shoulder and l_elbow and l_wrist:
            angles.append(_angle(l_shoulder, l_elbow, l_wrist))
        if r_shoulder and r_elbow and r_wrist:
            angles.append(_angle(r_shoulder, r_elbow, r_wrist))

        if not angles:
            return {"reps": self.reps, "angle": None, "state": "unknown", "corrections": []}

        avg_angle = sum(angles) / len(angles)
        self._angles.append(avg_angle)
        if len(self._angles) > 8:
            self._angles.pop(0)

        smooth = sum(self._angles) / len(self._angles)
        corrections = []

        if smooth < 90 and not self._down:
            self._down = True
        elif smooth > 160 and self._down:
            self._down = False
            self.reps += 1

        state = "down" if self._down else "up"

        if smooth > 90 and state == "down":
            corrections.append("💡 Lower more — chest closer to floor")
        if smooth < 50:
            corrections.append("⚠️ Don't lock elbows — stop at 90°")

        return {"reps": self.reps, "angle": round(smooth, 1), "state": state, "corrections": corrections}


class CurlDetector:
    """Counts bicep curls using elbow angle."""
    exercise = "curl"

    def __init__(self):
        self.reps = 0
        self._up = False
        self._angles: List[float] = []

    def update(self, keypoints: list) -> Dict:
        l_shoulder = _kp_xy(keypoints, "left_shoulder")
        l_elbow = _kp_xy(keypoints, "left_elbow")
        l_wrist = _kp_xy(keypoints, "left_wrist")
        r_shoulder = _kp_xy(keypoints, "right_shoulder")
        r_elbow = _kp_xy(keypoints, "right_elbow")
        r_wrist = _kp_xy(keypoints, "right_wrist")

        angles = []
        if l_shoulder and l_elbow and l_wrist:
            angles.append(_angle(l_shoulder, l_elbow, l_wrist))
        if r_shoulder and r_elbow and r_wrist:
            angles.append(_angle(r_shoulder, r_elbow, r_wrist))

        if not angles:
            return {"reps": self.reps, "angle": None, "state": "unknown", "corrections": []}

        avg_angle = sum(angles) / len(angles)
        self._angles.append(avg_angle)
        if len(self._angles) > 8:
            self._angles.pop(0)

        smooth = sum(self._angles) / len(self._angles)
        corrections = []

        if smooth < 50 and not self._up:
            self._up = True
        elif smooth > 150 and self._up:
            self._up = False
            self.reps += 1

        state = "up" if self._up else "down"

        if smooth > 60 and state == "up":
            corrections.append("💡 Curl higher — bring wrist to shoulder")
        if smooth < 160 and state == "down":
            corrections.append("💡 Fully extend — lower arm completely")

        return {"reps": self.reps, "angle": round(smooth, 1), "state": state, "corrections": corrections}
# ── Lunge Detector (new SDK-aligned exercise) ─────────────────────────
class LungeDetector:
    """Counts lunges using front knee angle cycling."""
    exercise = "lunge"

    def __init__(self):
        self.reps = 0
        self._down = False
        self._angles: List[float] = []

    def update(self, keypoints: list) -> Dict:
        # Use front leg knee angle (whichever knee is more bent)
        l_hip = _kp_xy(keypoints, "left_hip")
        l_knee = _kp_xy(keypoints, "left_knee")
        l_ankle = _kp_xy(keypoints, "left_ankle")
        r_hip = _kp_xy(keypoints, "right_hip")
        r_knee = _kp_xy(keypoints, "right_knee")
        r_ankle = _kp_xy(keypoints, "right_ankle")

        angles = []
        if l_hip and l_knee and l_ankle:
            angles.append(_angle(l_hip, l_knee, l_ankle))
        if r_hip and r_knee and r_ankle:
            angles.append(_angle(r_hip, r_knee, r_ankle))

        if not angles:
            return {"reps": self.reps, "angle": None, "state": "unknown", "corrections": []}

        # Take the more bent knee (lower angle) for lunge detection
        min_angle = min(angles)
        self._angles.append(min_angle)
        if len(self._angles) > 8:
            self._angles.pop(0)

        smooth = sum(self._angles) / len(self._angles)
        corrections = []

        if smooth < 100 and not self._down:
            self._down = True
        elif smooth > 155 and self._down:
            self._down = False
            self.reps += 1

        state = "down" if self._down else "up"

        if smooth < 70:
            corrections.append("⚠️ Front knee too bent — stop at 90°")
        if smooth > 110 and state == "down":
            corrections.append("💡 Lunge deeper — aim for 90° front knee")

        # Check knee over toe
        front_knee = l_knee if (l_knee and (not r_knee or (l_knee and min(angles) == angles[0]))) else r_knee
        front_ankle = l_ankle if front_knee == l_knee else r_ankle
        if front_knee and front_ankle and front_knee[0] > front_ankle[0] + 40:
            corrections.append("🦵 Keep knee behind toes")

        return {"reps": self.reps, "angle": round(smooth, 1), "state": state, "corrections": corrections}


# ── Skeleton Connection Map (for overlay rendering) ───────────────────
# SDK-aligned: Connection pairs for drawing skeleton overlay
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_eye", "right_eye"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_ear", "left_eye"),
    ("right_ear", "right_eye"),
]

# Color coding for skeleton based on confidence
SKELETON_COLORS = {
    "high": "#22c55e",    # Green: confidence > 0.7
    "medium": "#f59e0b",  # Amber: confidence 0.4-0.7
    "low": "#ef4444",     # Red:   confidence < 0.4
}


def _compute_form_score(exercise: str, angle: float, state: str,
                        corrections: list) -> int:
    """Compute a 0-100 form quality score based on exercise biomechanics."""
    score = 100

    # Deduct for corrections
    score -= len(corrections) * 15

    # Exercise-specific scoring
    if exercise == "squat":
        if state == "down":
            # Ideal squat bottom: 85-95 degrees
            ideal = 90
            deviation = abs(angle - ideal)
            score -= min(30, deviation * 0.5)
        elif state == "up":
            # Should be near full extension
            if angle < 160:
                score -= 10
    elif exercise == "pushup":
        if state == "down":
            ideal = 75
            deviation = abs(angle - ideal)
            score -= min(30, deviation * 0.5)
    elif exercise == "curl":
        if state == "up":
            ideal = 40
            deviation = abs(angle - ideal)
            score -= min(25, deviation * 0.5)
    elif exercise == "lunge":
        if state == "down":
            ideal = 90
            deviation = abs(angle - ideal)
            score -= min(30, deviation * 0.5)

    return max(0, min(100, int(score)))


# ── Main Pose Engine ───────────────────────────────────────────────────
class PoseEngine:
    """Per-track pose + action state manager.

    SDK-Aligned enhancements:
    - Multi-person pose tracking (one detector per person)
    - Form quality scoring (0-100)
    - Skeleton connection data for overlay rendering
    - Session analytics with milestones
    - Event bus integration for rep notifications
    """

    SUPPORTED_EXERCISES = ["squat", "pushup", "curl", "lunge", "auto"]
    DETECTOR_CLASSES = {
        "squat": SquatDetector,
        "pushup": PushupDetector,
        "curl": CurlDetector,
        "lunge": LungeDetector,
    }
    # Milestone rep counts for notifications
    MILESTONES = {5, 10, 15, 20, 25, 30, 50, 75, 100}

    def __init__(self):
        self._tracks: Dict[int, Dict] = {}  # track_id → {detector, last_ts, ...}
        self._yolo_pose_model = None
        self._model_tried = False
        self._total_reps = 0
        self._total_frames = 0

        # SDK-aligned: Event bus integration
        self._event_bus = None
        try:
            from event_bus import event_bus, EventType, Event
            self._event_bus = event_bus
            self._EventType = EventType
            self._Event = Event
        except ImportError:
            pass

        logger.info("PoseEngine initialised (SDK-aligned)")

    def _load_pose_model(self):
        """Lazy-load YOLO pose model."""
        if self._model_tried:
            return self._yolo_pose_model
        self._model_tried = True
        try:
            from ultralytics import YOLO
            import os
            model_path = os.path.join(os.path.dirname(__file__), "yolov8n-pose.pt")
            if not os.path.exists(model_path):
                logger.info("Downloading yolov8n-pose.pt...")
            self._yolo_pose_model = YOLO("yolov8n-pose.pt")
            logger.info("YOLO pose model loaded")
        except Exception as e:
            logger.warning("YOLO pose model unavailable: %s", e)
            self._yolo_pose_model = None
        return self._yolo_pose_model

    def _get_track(self, track_id: int, exercise: str) -> Dict:
        if track_id not in self._tracks:
            ex = exercise if exercise in self.DETECTOR_CLASSES else "squat"
            self._tracks[track_id] = {
                "detector": self.DETECTOR_CLASSES[ex](),
                "exercise": ex,
                "last_ts": time.time(),
                "start_ts": time.time(),
                "frame_count": 0,
                "form_scores": [],
                "milestones_hit": set(),
                "best_form_score": 0,
                "worst_form_score": 100,
            }
        return self._tracks[track_id]

    def _get_skeleton_connections(self, keypoints: list) -> list:
        """Generate skeleton connection data for overlay rendering."""
        connections = []
        for name_a, name_b in SKELETON_CONNECTIONS:
            pt_a = _kp_xy(keypoints, name_a)
            pt_b = _kp_xy(keypoints, name_b)
            if pt_a and pt_b:
                # Average confidence of the two endpoints
                idx_a = KP.get(name_a, 0)
                idx_b = KP.get(name_b, 0)
                conf_a = float(keypoints[idx_a][2]) if len(keypoints[idx_a]) >= 3 else 0.5
                conf_b = float(keypoints[idx_b][2]) if len(keypoints[idx_b]) >= 3 else 0.5
                avg_conf = (conf_a + conf_b) / 2
                color = (SKELETON_COLORS["high"] if avg_conf > 0.7
                         else SKELETON_COLORS["medium"] if avg_conf > 0.4
                         else SKELETON_COLORS["low"])
                connections.append({
                    "from": list(pt_a), "to": list(pt_b),
                    "confidence": round(avg_conf, 2), "color": color,
                })
        return connections

    async def _emit_rep_event(self, track_id: int, exercise: str, reps: int, form_score: int):
        """Emit rep count event to event bus."""
        if not self._event_bus:
            return
        try:
            is_milestone = reps in self.MILESTONES
            event_type = self._EventType.MILESTONE if is_milestone else self._EventType.REP_COUNTED
            await self._event_bus.emit(self._Event(
                type=event_type,
                data={
                    "track_id": track_id,
                    "exercise": exercise,
                    "reps": reps,
                    "form_score": form_score,
                    "milestone": is_milestone,
                },
                source="pose_engine",
            ))
        except Exception:
            pass

    def analyze_frame(self, frame, exercise: str = "squat",
                      track_id: int = 0) -> Dict[str, Any]:
        """Run YOLO pose on frame and return action engine results.

        SDK-Aligned enhanced returns:
            {
                "exercise": str,
                "reps": int,
                "angle": float | None,
                "state": str,  # "up" | "down" | "unknown"
                "corrections": [str],
                "keypoints": [[x,y,conf], ...] | [],
                "skeleton": [{from, to, confidence, color}, ...],
                "form_score": int (0-100),
                "pose_available": bool,
                "persons_detected": int,
            }
        """
        self._total_frames += 1
        model = self._load_pose_model()
        keypoints_raw = []
        persons_detected = 0

        if model is not None:
            try:
                results = model(frame, verbose=False)
                if results and len(results[0].keypoints.data) > 0:
                    persons_detected = len(results[0].keypoints.data)
                    # Take first detected person (or tracked person by ID)
                    person_idx = min(track_id, persons_detected - 1)
                    kps = results[0].keypoints.data[person_idx].tolist()
                    keypoints_raw = kps
            except Exception as e:
                logger.debug("Pose inference error: %s", e)

        track = self._get_track(track_id, exercise)
        track["frame_count"] += 1
        track["last_ts"] = time.time()

        # If no keypoints detected, return current state
        if not keypoints_raw:
            det = track["detector"]
            return {
                "exercise": track["exercise"],
                "reps": det.reps,
                "angle": None,
                "state": "unknown",
                "corrections": ["📷 No pose detected — make sure full body is visible"],
                "keypoints": [],
                "skeleton": [],
                "form_score": 0,
                "pose_available": False,
                "persons_detected": persons_detected,
            }

        det = track["detector"]
        prev_reps = det.reps
        action_result = det.update(keypoints_raw)

        # SDK-aligned: Form quality scoring
        form_score = _compute_form_score(
            track["exercise"],
            action_result.get("angle", 0) or 0,
            action_result.get("state", "unknown"),
            action_result.get("corrections", []),
        )
        track["form_scores"].append(form_score)
        if len(track["form_scores"]) > 50:
            track["form_scores"].pop(0)
        track["best_form_score"] = max(track["best_form_score"], form_score)
        track["worst_form_score"] = min(track["worst_form_score"], form_score)

        # SDK-aligned: Skeleton connections for overlay
        skeleton = self._get_skeleton_connections(keypoints_raw)

        # Detect new rep and emit event
        new_reps = action_result["reps"]
        if new_reps > prev_reps:
            self._total_reps += (new_reps - prev_reps)

        return {
            "exercise": track["exercise"],
            "reps": new_reps,
            "angle": action_result["angle"],
            "state": action_result["state"],
            "corrections": action_result.get("corrections", []),
            "keypoints": keypoints_raw,
            "skeleton": skeleton,
            "form_score": form_score,
            "pose_available": True,
            "persons_detected": persons_detected,
        }

    def get_session_summary(self, track_id: int = 0) -> Dict:
        """Return a coaching summary for the session."""
        track = self._tracks.get(track_id)
        if not track:
            return {"reps": 0, "exercise": "unknown", "frames_analyzed": 0}
        det = track["detector"]
        scores = track.get("form_scores", [])
        avg_score = round(sum(scores) / len(scores)) if scores else 0
        duration = round(time.time() - track.get("start_ts", time.time()), 1)

        # Generate coaching feedback based on performance
        feedback = []
        if avg_score >= 85:
            feedback.append("🏆 Excellent form! Keep it up!")
        elif avg_score >= 70:
            feedback.append("👍 Good form. Focus on consistency.")
        elif avg_score >= 50:
            feedback.append("⚡ Decent effort. Watch your form more carefully.")
        else:
            feedback.append("💪 Keep practicing! Form will improve with time.")

        if det.reps >= 20:
            feedback.append(f"🔥 Great endurance — {det.reps} reps completed!")

        return {
            "exercise": track["exercise"],
            "reps": det.reps,
            "frames_analyzed": track["frame_count"],
            "session_duration_s": duration,
            "avg_form_score": avg_score,
            "best_form_score": track.get("best_form_score", 0),
            "worst_form_score": track.get("worst_form_score", 100),
            "form_trend": scores[-10:] if scores else [],
            "feedback": feedback,
        }

    def get_all_tracks_summary(self) -> Dict:
        """Get summary for all active tracks (all persons)."""
        return {
            "total_tracks": len(self._tracks),
            "total_reps": self._total_reps,
            "total_frames": self._total_frames,
            "tracks": {
                tid: {
                    "exercise": t["exercise"],
                    "reps": t["detector"].reps,
                    "frames": t["frame_count"],
                    "form_avg": round(sum(t.get("form_scores", [0])) / max(1, len(t.get("form_scores", [0]))), 1),
                }
                for tid, t in self._tracks.items()
            },
        }

    def reset_track(self, track_id: int = 0):
        """Reset a track's rep counter (start new set)."""
        if track_id in self._tracks:
            ex = self._tracks[track_id]["exercise"]
            self._tracks[track_id]["detector"] = self.DETECTOR_CLASSES.get(ex, SquatDetector)()
            self._tracks[track_id]["frame_count"] = 0
            self._tracks[track_id]["form_scores"] = []
            self._tracks[track_id]["start_ts"] = time.time()
            self._tracks[track_id]["best_form_score"] = 0
            self._tracks[track_id]["worst_form_score"] = 100


# ── Singleton instance ─────────────────────────────────────────────────
pose_engine = PoseEngine()
