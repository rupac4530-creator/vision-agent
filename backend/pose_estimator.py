"""
Lightweight pose estimation wrapper used by the Live Stream fitness coach.

Uses MediaPipe Pose if installed. This is intentionally optional:
- If MediaPipe isn't available, the rest of the platform still works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Compute the angle ABC (in degrees) for 2D points a,b,c where b is the vertex.
    """
    import math

    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    dot = bax * bcx + bay * bcy
    mag1 = math.hypot(bax, bay)
    mag2 = math.hypot(bcx, bcy)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosang))


@dataclass
class PoseFrame:
    ok: bool
    angles: Dict[str, float]
    keypoints: Dict[str, Tuple[float, float]]
    meta: Dict[str, Any]


class PoseEstimator:
    """
    MediaPipe pose estimator that returns a small set of keypoints + angles useful for coaching.
    """

    def __init__(self):
        self._mp_pose = None
        self._pose = None

        try:
            import mediapipe as mp  # type: ignore

            self._mp_pose = mp.solutions.pose
            # model_complexity=0 is fastest; smooth_landmarks helps jitter
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                enable_segmentation=False,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            self._mp_pose = None
            self._pose = None

    @property
    def available(self) -> bool:
        return self._pose is not None

    def infer_bgr(self, image_bgr) -> PoseFrame:
        """
        Infer pose from an OpenCV BGR frame.

        Returns a PoseFrame with:
        - keypoints: a small set of named joints (pixel coords)
        - angles: knee/elbow angles (degrees)
        """
        if self._pose is None:
            return PoseFrame(ok=False, angles={}, keypoints={}, meta={"reason": "mediapipe_not_installed"})

        import cv2

        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self._pose.process(img_rgb)
        if not res.pose_landmarks:
            return PoseFrame(ok=False, angles={}, keypoints={}, meta={"reason": "no_pose"})

        lm = res.pose_landmarks.landmark

        def px(idx: int) -> Tuple[float, float]:
            return (lm[idx].x * w, lm[idx].y * h)

        # MediaPipe Pose landmark indices
        KP = {
            "left_shoulder": px(11),
            "right_shoulder": px(12),
            "left_elbow": px(13),
            "right_elbow": px(14),
            "left_wrist": px(15),
            "right_wrist": px(16),
            "left_hip": px(23),
            "right_hip": px(24),
            "left_knee": px(25),
            "right_knee": px(26),
            "left_ankle": px(27),
            "right_ankle": px(28),
        }

        # Compute angles (use left side primarily; right side as fallback)
        left_knee = _angle_deg(KP["left_hip"], KP["left_knee"], KP["left_ankle"])
        right_knee = _angle_deg(KP["right_hip"], KP["right_knee"], KP["right_ankle"])
        left_elbow = _angle_deg(KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"])
        right_elbow = _angle_deg(KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"])

        angles = {
            "left_knee": round(left_knee, 1),
            "right_knee": round(right_knee, 1),
            "left_elbow": round(left_elbow, 1),
            "right_elbow": round(right_elbow, 1),
        }

        # Quick quality signal: visibility of key landmarks
        vis = {
            "left_knee": float(lm[25].visibility),
            "right_knee": float(lm[26].visibility),
            "left_hip": float(lm[23].visibility),
            "right_hip": float(lm[24].visibility),
        }

        return PoseFrame(ok=True, angles=angles, keypoints=KP, meta={"w": w, "h": h, "visibility": vis})

