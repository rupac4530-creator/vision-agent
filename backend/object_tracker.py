"""
object_tracker.py — Persistent Object Tracker
Simple IoU-based multi-object tracker (SORT-inspired) for maintaining object IDs across frames.
Tracks objects' positions, velocities, labels, and last_seen timestamps.
"""
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("object_tracker")


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection-over-Union between two bboxes [x1,y1,x2,y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class TrackedObject:
    """A single tracked object with ID, state, and velocity."""

    _next_id = 1

    def __init__(self, bbox: List[float], label: str, conf: float):
        self.id = TrackedObject._next_id
        TrackedObject._next_id += 1
        self.bbox = list(bbox)
        self.label = label
        self.conf = conf
        self.last_seen = time.time()
        self.age = 0  # frames alive
        self.missed = 0  # consecutive missed frames
        self.velocity = [0.0, 0.0]  # [vx, vy] in pixels/frame
        self.prev_center = self._center()

    def _center(self) -> List[float]:
        return [(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2]

    def update(self, bbox: List[float], conf: float):
        """Update position and compute velocity."""
        new_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.velocity = [
            new_center[0] - self.prev_center[0],
            new_center[1] - self.prev_center[1],
        ]
        self.prev_center = new_center
        self.bbox = list(bbox)
        self.conf = conf
        self.last_seen = time.time()
        self.age += 1
        self.missed = 0

    def predict(self) -> List[float]:
        """Predict next position using velocity."""
        return [
            self.bbox[0] + self.velocity[0],
            self.bbox[1] + self.velocity[1],
            self.bbox[2] + self.velocity[0],
            self.bbox[3] + self.velocity[1],
        ]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "conf": round(self.conf, 3),
            "bbox": [round(v, 1) for v in self.bbox],
            "velocity": [round(v, 2) for v in self.velocity],
            "age_frames": self.age,
            "last_seen": round(time.time() - self.last_seen, 2),
        }


class ObjectTracker:
    """
    Multi-object tracker using IoU matching (SORT-inspired algorithm).
    Maintains persistent object IDs across frames even with brief occlusions.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5, max_age_seconds: float = 3.0):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_age_seconds = max_age_seconds
        self._tracks: Dict[int, TrackedObject] = {}
        self._frame_count = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        detections: [{"label": str, "conf": float, "bbox": [x1,y1,x2,y2]}, ...]
        Returns: list of tracked objects with persistent IDs
        """
        self._frame_count += 1
        now = time.time()

        # Remove stale tracks
        stale_ids = [
            tid for tid, t in self._tracks.items()
            if t.missed > self.max_missed or (now - t.last_seen) > self.max_age_seconds
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        if not detections:
            for t in self._tracks.values():
                t.missed += 1
            return [t.to_dict() for t in self._tracks.values()]

        # Predict next positions for all tracks
        track_ids = list(self._tracks.keys())
        predicted = {tid: self._tracks[tid].predict() for tid in track_ids}

        # Build IoU cost matrix
        det_bboxes = [d["bbox"] for d in detections]
        matched_dets = set()
        matched_tracks = set()

        # Greedy matching
        iou_pairs = []
        for ti, tid in enumerate(track_ids):
            for di, det_bbox in enumerate(det_bboxes):
                score = iou(predicted[tid], det_bbox)
                if score >= self.iou_threshold:
                    iou_pairs.append((score, tid, di))

        # Sort by IoU descending
        iou_pairs.sort(key=lambda x: -x[0])

        for score, tid, di in iou_pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            det = detections[di]
            self._tracks[tid].update(det["bbox"], det["conf"])
            self._tracks[tid].label = det["label"]
            matched_tracks.add(tid)
            matched_dets.add(di)

        # Mark unmatched tracks as missed
        for tid in track_ids:
            if tid not in matched_tracks:
                self._tracks[tid].missed += 1

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_dets:
                new_track = TrackedObject(det["bbox"], det["label"], det["conf"])
                self._tracks[new_track.id] = new_track

        return [t.to_dict() for t in self._tracks.values() if t.missed == 0]

    def get_all_tracks(self) -> List[Dict]:
        """Return all currently tracked objects."""
        return [t.to_dict() for t in self._tracks.values()]

    def reset(self):
        """Clear all tracks."""
        self._tracks.clear()
        self._frame_count = 0
        TrackedObject._next_id = 1

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks.values() if t.missed == 0)


# Singleton
object_tracker = ObjectTracker()
