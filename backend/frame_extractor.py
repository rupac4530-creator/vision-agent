# backend/frame_extractor.py
"""
Extract frames from a video file at a configurable sample rate (frames per second).
Uses OpenCV for fast, reliable frame extraction.
"""

import cv2
import os
from pathlib import Path


def extract_frames(video_path: str, out_dir: str, fps_sample: int = 1) -> int:
    """
    Extract frames from *video_path* at *fps_sample* frames per second.

    Parameters
    ----------
    video_path : str
        Absolute path to the input video file.
    out_dir : str
        Directory where extracted JPEG frames will be saved.
    fps_sample : int
        How many frames to keep per second of video (default 1).

    Returns
    -------
    int
        Number of frames saved to *out_dir*.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Determine native FPS; fall back to 25.0 if metadata is missing
    video_fps = vidcap.get(cv2.CAP_PROP_FPS) or 25.0
    if video_fps <= 0:
        video_fps = 25.0

    # Interval in raw frames between each saved frame
    frame_interval = max(1, int(round(video_fps / float(fps_sample))))

    saved = 0
    frame_idx = 0

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(out_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    vidcap.release()
    return saved
