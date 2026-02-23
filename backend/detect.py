# backend/detect.py
"""
Per-frame object detection using YOLOv8 (ultralytics).
Uses yolov8n (nano) by default for speed; swap to yolov8s/m for accuracy.
"""

from ultralytics import YOLO
from pathlib import Path
import json
import time
import os

MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")


def detect_frames(frames_dir: str, out_dir: str, conf_threshold: float = 0.3) -> dict:
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
            detections.append({"label": label, "confidence": round(conf, 3)})
        results_list.append({
            "frame": f.name,
            "path": str(f),
            "detections": detections,
        })

    total_time = time.time() - t0
    summary = {"frames": len(frames), "time_seconds": round(total_time, 3)}

    # Persist detections
    det_path = os.path.join(out_dir, "detections.json")
    with open(det_path, "w") as fh:
        json.dump({"summary": summary, "results": results_list}, fh, indent=2)

    return {"summary": summary, "results": results_list}
