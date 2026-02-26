# backend/detect.py
"""
Per-frame object detection using YOLOv8 (ultralytics).
Uses yolov8n (nano) by default for speed; swap to yolov8s/m for accuracy.

This module provides both batch detection (detect_frames) and
single-frame detection (detect_single_frame) with full bbox coordinates.
"""

from ultralytics import YOLO
from pathlib import Path
import json
import time
import os

MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8s.pt")


def detect_single_frame(frame_path: str, conf_threshold: float = 0.20) -> list[dict]:
    """
    Run YOLOv8 on a single frame. Returns detection list with bboxes.

    Returns
    -------
    list[dict]
        Each dict: {label, confidence, bbox: [x1, y1, x2, y2]}
    """
    model = YOLO(MODEL_NAME)
    res = model.predict(source=frame_path, conf=conf_threshold, verbose=False)
    detections = []
    r = res[0]
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names.get(cls, str(cls))
        coords = box.xyxy[0].tolist()
        quality = "high" if conf >= 0.70 else ("medium" if conf >= 0.40 else "low")
        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [round(c, 1) for c in coords],
            "quality": quality,
        })
    return detections


def detect_frames(frames_dir: str, out_dir: str, conf_threshold: float = 0.20) -> dict:
    """
    Run YOLOv8 on every frame in *frames_dir* and save detection results.

    Parameters
    ----------
    frames_dir : str
        Folder containing extracted JPEG frames.
    out_dir : str
        Where to write detections.json.
    conf_threshold : float
        Minimum confidence for a detection to be kept.

    Returns
    -------
    dict
        Keys: summary (frames, time_seconds), results (per-frame detections)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)
    frames = sorted(
        p for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    results_list = []
    t0 = time.time()

    for f in frames:
        res = model.predict(source=str(f), conf=conf_threshold, verbose=False)
        detections = []
        r = res[0]
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names.get(cls, str(cls))
            coords = box.xyxy[0].tolist()
            quality = "high" if conf >= 0.70 else ("medium" if conf >= 0.40 else "low")
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [round(c, 1) for c in coords],
                "quality": quality,
            })
        results_list.append({
            "frame": f.name,
            "path": str(f),
            "detections": detections,
        })

    total_time = time.time() - t0

    # Aggregate label counts for summary
    label_counts = {}
    for r in results_list:
        for d in r.get("detections", []):
            lbl = d["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    summary = {
        "frames": len(frames),
        "time_seconds": round(total_time, 3),
        "label_counts": label_counts,
        "total_detections": sum(label_counts.values()),
    }

    # Persist detections
    det_path = os.path.join(out_dir, "detections.json")
    with open(det_path, "w") as fh:
        json.dump({"summary": summary, "results": results_list}, fh, indent=2)

    return {"summary": summary, "results": results_list}
