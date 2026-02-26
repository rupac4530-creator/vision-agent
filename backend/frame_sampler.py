"""
frame_sampler.py — Intelligent Frame Sampling Pipeline
Throttle FPS, detect motion, skip static frames, pre-annotate with YOLO.
Prevents VLM bottleneck by only sending meaningful frames.
"""
import cv2
import time
import logging
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("frame_sampler")


class FrameSampler:
    """
    Intelligent frame sampler with:
    - Configurable FPS throttle
    - Motion-based frame selection (skip static frames)
    - Frame quality scoring
    - Optional YOLO pre-annotation before sending to VLM
    """

    def __init__(self, target_fps: float = 8.0, motion_threshold: float = 0.02):
        self.target_fps = target_fps
        self.motion_threshold = motion_threshold  # fraction of pixels that changed
        self._interval = 1.0 / target_fps
        self._last_send_time = 0.0
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count = 0
        self._sent_count = 0
        self._motion_buffer = deque(maxlen=10)
        self._stats: Dict[str, Any] = {
            "total_received": 0,
            "total_sent": 0,
            "motion_skip": 0,
            "fps_skip": 0,
            "avg_motion": 0.0,
        }

    def should_process(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """
        Decide whether this frame should be processed/sent to the VLM.
        Returns (should_process, info_dict).
        """
        self._frame_count += 1
        self._stats["total_received"] = self._frame_count

        now = time.time()
        info = {"frame_id": self._frame_count, "sent": False, "reason": ""}

        # FPS throttle
        elapsed = now - self._last_send_time
        if elapsed < self._interval:
            self._stats["fps_skip"] += 1
            info["reason"] = f"fps_throttle ({elapsed*1000:.0f}ms < {self._interval*1000:.0f}ms)"
            return False, info

        # Motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_score = 0.0
        if self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_score = np.sum(thresh > 0) / thresh.size
            self._motion_buffer.append(motion_score)
            avg_motion = np.mean(self._motion_buffer)
            self._stats["avg_motion"] = round(float(avg_motion), 4)

            if motion_score < self.motion_threshold:
                self._stats["motion_skip"] += 1
                self._prev_gray = gray
                info["reason"] = f"no_motion (score={motion_score:.4f} < {self.motion_threshold})"
                return False, info

        self._prev_gray = gray

        # Frame passes — send it
        self._last_send_time = now
        self._sent_count += 1
        self._stats["total_sent"] = self._sent_count
        info["sent"] = True
        info["motion_score"] = round(motion_score, 4)
        info["reason"] = "selected"

        return True, info

    def quality_score(self, frame: np.ndarray) -> float:
        """Return frame quality 0-1 (based on sharpness / Laplacian variance)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize: 0 = blurry, 1 = sharp (cap at 1000)
        return round(min(lap_var / 1000.0, 1.0), 3)

    @property
    def stats(self) -> Dict:
        efficiency = 0
        if self._frame_count > 0:
            efficiency = round(self._sent_count / self._frame_count * 100, 1)
        return {**self._stats, "efficiency_pct": efficiency}

    def reset(self):
        self._prev_gray = None
        self._last_send_time = 0.0
        self._frame_count = 0
        self._sent_count = 0
        self._motion_buffer.clear()
        self._stats = {"total_received": 0, "total_sent": 0, "motion_skip": 0, "fps_skip": 0, "avg_motion": 0.0}


# Default singleton sampler (8 FPS, 2% motion threshold)
frame_sampler = FrameSampler(target_fps=8.0, motion_threshold=0.02)
