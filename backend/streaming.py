# backend/streaming.py
"""
Real-time streaming endpoints for Vision Agent.

• POST /stream_chunk   — accept a short video chunk (2-5 s), process immediately
                         (frames + transcription + detection), return instant JSON.
• POST /stream_finalize — stitch all uploaded chunks via ffmpeg, run full analysis.
"""

import os
import shutil
import time
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from frame_extractor import extract_frames
from transcribe import transcribe_audio_whisper
from detect import detect_frames

try:
    import ffmpeg as ffmpeg_lib
except ImportError:
    ffmpeg_lib = None  # gracefully degrade if ffmpeg-python not installed

router = APIRouter(tags=["streaming"])

STREAM_ROOT = Path(__file__).resolve().parent / "stream_uploads"
STREAM_ROOT.mkdir(exist_ok=True)


def _extract_audio_ffmpeg(video_path: str, out_audio: str) -> str:
    """Extract mono 16 kHz WAV from *video_path* using ffmpeg-python."""
    if ffmpeg_lib is None:
        raise RuntimeError("ffmpeg-python is not installed")
    (
        ffmpeg_lib.input(video_path)
        .output(out_audio, ac=1, ar="16000", vn=None)
        .overwrite_output()
        .run(quiet=True)
    )
    return out_audio


# ── /stream_chunk ──────────────────────────────────────────────────────
@router.post("/stream_chunk")
async def stream_chunk(
    video_stem: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(0),
    file: UploadFile = File(...),
):
    """
    Accept a small chunk, process it synchronously, and return a quick
    summary (transcript snippet + top detected labels + timing).
    """
    vs_folder = STREAM_ROOT / video_stem
    chunks_folder = vs_folder / "chunks"
    chunks_folder.mkdir(parents=True, exist_ok=True)

    # Save chunk
    ext = Path(file.filename or "chunk.mp4").suffix or ".mp4"
    chunk_path = chunks_folder / f"chunk_{chunk_index:04d}{ext}"
    with chunk_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    t0 = time.time()

    # 1) Extract frames (1 fps)
    frames_out = vs_folder / f"frames_chunk_{chunk_index:04d}"
    frames_out.mkdir(parents=True, exist_ok=True)
    try:
        frames_count = extract_frames(str(chunk_path), str(frames_out), fps_sample=1)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"frame extract: {e}"}, status_code=500)

    # 2) Extract audio
    audio_path = vs_folder / f"chunk_{chunk_index:04d}.wav"
    try:
        _extract_audio_ffmpeg(str(chunk_path), str(audio_path))
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"audio extract: {e}"}, status_code=500)

    # 3) Transcribe
    t_tr0 = time.time()
    try:
        transcript = transcribe_audio_whisper(str(audio_path), str(vs_folder))
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"transcription: {e}"}, status_code=500)
    t_tr = time.time() - t_tr0

    # 4) Detect objects
    t_det0 = time.time()
    try:
        detections = detect_frames(str(frames_out), str(vs_folder), conf_threshold=0.3)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"detection: {e}"}, status_code=500)
    t_det = time.time() - t_det0

    # Aggregate top labels
    label_counts: dict[str, int] = {}
    for r in detections.get("results", []):
        for d in r.get("detections", []):
            lbl = d["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:6]

    elapsed = time.time() - t0

    return {
        "ok": True,
        "video_stem": video_stem,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "frames_count": frames_count,
        "transcript_snippet": (transcript.get("text") or "")[:400],
        "transcription_time": round(t_tr, 3),
        "detection_time": round(t_det, 3),
        "top_labels": [{"label": k, "count": v} for k, v in top_labels],
        "elapsed_seconds": round(elapsed, 3),
        "sample_frames": [str(p) for p in sorted(frames_out.glob("*.jpg"))[:3]],
    }


# ── /stream_finalize ──────────────────────────────────────────────────
@router.post("/stream_finalize")
async def stream_finalize(video_stem: str = Form(...)):
    """
    Stitch all uploaded chunks into one video via ffmpeg concat, then
    run the full analysis pipeline (frames + transcription + detection).
    """
    vs_folder = STREAM_ROOT / video_stem
    chunks_folder = vs_folder / "chunks"
    if not chunks_folder.exists():
        raise HTTPException(status_code=404, detail="No chunks found")

    chunk_files = sorted(chunks_folder.iterdir())
    if not chunk_files:
        raise HTTPException(status_code=404, detail="Chunks folder is empty")

    # Build ffmpeg concat list
    filelist = vs_folder / "ffmpeg_list.txt"
    with filelist.open("w") as fh:
        for f in chunk_files:
            fh.write(f"file '{f}'\n")

    stitched = vs_folder / f"{video_stem}_stitched.mp4"

    if ffmpeg_lib is None:
        raise HTTPException(status_code=500, detail="ffmpeg-python not installed")

    # Attempt concat copy, fall back to re-encode
    try:
        (
            ffmpeg_lib.input(str(filelist), format="concat", safe=0)
            .output(str(stitched), c="copy")
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception:
        try:
            (
                ffmpeg_lib.input(str(filelist), format="concat", safe=0)
                .output(str(stitched), vcodec="libx264", acodec="aac")
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Stitch failed: {e2}")

    # Run full analysis on the stitched video via internal HTTP call
    import requests

    try:
        with open(stitched, "rb") as fh:
            resp = requests.post(
                "http://127.0.0.1:8000/analyze",
                files={"file": ("stitched.mp4", fh, "video/mp4")},
                timeout=300,
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Analyze failed: {resp.text}")
        return JSONResponse(resp.json())
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Analyze call failed: {e}")
