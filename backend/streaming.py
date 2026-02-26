# backend/streaming.py
"""
Real-time streaming endpoints for Vision Agent.

• POST /stream_chunk     — accept a short video chunk, process immediately
• POST /stream_finalize  — stitch all chunks, run full analysis

STRATEGY FOR WEBM CHUNKS:
MediaRecorder produces fragmented WebM where only chunk 1 has the EBML
initialization segment. Chunks 2+ are bare clusters that may lack keyframes.

Fix: maintain a GROWING WebM file (init segment from chunk 1 + all received
clusters). For each new chunk, append its data, transcode the GROWING file
to MP4, and extract only the NEW frames (skip previously extracted ones).

This guarantees ffmpeg always has the full reference chain.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import threading

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from frame_extractor import extract_frames
from transcribe import transcribe_audio_whisper
from detect import detect_frames
from vision_worker import vision_worker

router = APIRouter(tags=["streaming"])

STREAM_ROOT = Path(__file__).resolve().parent / "stream_uploads"
STREAM_ROOT.mkdir(exist_ok=True)

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── Per-stream metrics store (thread-safe) ────────────────────────────
_stream_metrics_lock = threading.Lock()
_stream_metrics: dict[str, dict] = {}  # video_stem -> metrics


def _record_chunk_metrics(video_stem: str, frames: int, detections: list, latency_ms: float):
    """Thread-safe update of per-stream metrics."""
    with _stream_metrics_lock:
        if video_stem not in _stream_metrics:
            _stream_metrics[video_stem] = {
                "chunks_processed": 0,
                "frames_total": 0,
                "latencies_ms": [],
                "last_detections": [],
                "total_detections": 0,
            }
        m = _stream_metrics[video_stem]
        m["chunks_processed"] += 1
        m["frames_total"] += frames
        m["latencies_ms"].append(latency_ms)
        if len(m["latencies_ms"]) > 200:
            m["latencies_ms"] = m["latencies_ms"][-200:]
        m["last_detections"] = detections
        m["total_detections"] += len(detections)


# ── Logging setup ─────────────────────────────────────────────────────
def _get_stream_logger(video_stem: str) -> logging.Logger:
    """Get or create a per-stream logger that writes to logs/stream_<stem>.log."""
    name = f"stream.{video_stem}"
    lgr = logging.getLogger(name)
    if not lgr.handlers:
        lgr.setLevel(logging.DEBUG)
        fh = logging.FileHandler(LOGS_DIR / f"stream_{video_stem}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lgr.addHandler(fh)
    return lgr


# ── Helpers ────────────────────────────────────────────────────────────

def _check_ffmpeg() -> tuple[bool, str]:
    """Check if ffmpeg is available and return version string."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=5
        )
        version_line = r.stdout.decode(errors="ignore").split("\n")[0]
        return True, version_line
    except Exception as e:
        return False, str(e)


def _extract_webm_init_segment(data: bytes) -> bytes:
    """
    Extract the WebM init segment (EBML + Segment + Tracks) from a complete
    WebM file. Cuts before the first Cluster element (ID 0x1F43B675).
    """
    cluster_id = b'\x1f\x43\xb6\x75'
    pos = data.find(cluster_id)
    if pos > 0:
        return data[:pos]
    return data[:4096]


def _transcode_to_mp4(src: str, dst: str, crf: int = 23,
                       codec: str = "libx264", lgr=None) -> tuple[bool, str]:
    """Transcode src to mp4. Returns (success, error_tail)."""
    cmd = [
        "ffmpeg", "-y",
        "-err_detect", "ignore_err",    # Tolerate minor stream errors
        "-i", str(src),
        "-c:v", codec,
        "-preset", "veryfast",
        "-crf", str(crf),
        "-an",
        str(dst),
    ]
    if lgr:
        lgr.debug("Transcode cmd: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=60,
        )
        if lgr:
            lgr.debug("Transcode OK: %s -> %s", src, dst)
        return True, ""
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore")
        tail = stderr[-800:] if len(stderr) > 800 else stderr
        if lgr:
            lgr.error("Transcode FAILED: %s", tail)
        return False, tail
    except subprocess.TimeoutExpired:
        if lgr:
            lgr.error("Transcode TIMEOUT: %s", src)
        return False, "ffmpeg transcode timed out (>60s)"
    except Exception as e:
        if lgr:
            lgr.error("Transcode ERROR: %s", e)
        return False, str(e)


def _extract_frames_ffmpeg(src: str, out_dir: str, fps: int = 1,
                            start_number: int = 1) -> int:
    """Extract frames via ffmpeg directly. Returns count of frames extracted."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-i", str(src),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-start_number", str(start_number),
        str(Path(out_dir) / "frame_%04d.jpg"),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       check=True, timeout=60)
        return len(list(Path(out_dir).glob("*.jpg")))
    except Exception:
        return 0


def _extract_audio_chunk(video_path: str, out_audio: str) -> bool:
    """Extract audio from a video chunk. Returns True if successful."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_audio),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       check=True, timeout=30)
        return True
    except Exception:
        return False


# ── /stream_chunk ──────────────────────────────────────────────────────
@router.post("/stream_chunk")
async def stream_chunk(
    video_stem: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(0),
    file: UploadFile = File(...),
):
    """
    Accept a small chunk, process it, return instant JSON summary.
    Uses growing-file strategy for robust WebM decoding.
    """
    lgr = _get_stream_logger(video_stem)
    lgr.info("=== Chunk %d received (total=%d) ===", chunk_index, total_chunks)

    # Check ffmpeg
    ff_ok, ff_ver = _check_ffmpeg()
    if not ff_ok:
        lgr.error("ffmpeg not found: %s", ff_ver)
        return JSONResponse(
            {"ok": False, "error": f"ffmpeg not found: {ff_ver}"},
            status_code=503,
        )
    lgr.debug("ffmpeg: %s", ff_ver)

    vs_folder = STREAM_ROOT / video_stem
    chunks_folder = vs_folder / "chunks"
    chunks_folder.mkdir(parents=True, exist_ok=True)

    # Save incoming chunk
    ext = Path(file.filename or "chunk.webm").suffix or ".webm"
    chunk_path = chunks_folder / f"chunk_{chunk_index:04d}{ext}"
    raw_data = await file.read()
    chunk_path.write_bytes(raw_data)
    lgr.info("Chunk %d saved: %d bytes, ext=%s", chunk_index, len(raw_data), ext)

    t0 = time.time()
    needs_transcode = ext.lower() in (".webm", ".mkv", ".ogg", ".ogv")

    # ─── Growing-file strategy for WebM ───────────────────────────────
    if needs_transcode:
        growing_webm = vs_folder / "growing_stream.webm"

        if chunk_index <= 1:
            # Chunk 1: save as-is (contains init segment + first cluster)
            growing_webm.write_bytes(raw_data)

            # Also save init segment separately for reference
            init_seg = _extract_webm_init_segment(raw_data)
            (vs_folder / "init_segment.bin").write_bytes(init_seg)
            lgr.info("Init segment saved: %d bytes (from %d byte chunk 1)",
                      len(init_seg), len(raw_data))
        else:
            # Chunk N>1: append to growing file
            # If growing file doesn't exist (e.g., server restart), reconstruct
            if not growing_webm.exists():
                lgr.warning("Growing file missing — reconstructing from all chunks")
                init_path = vs_folder / "init_segment.bin"
                if init_path.exists():
                    growing_webm.write_bytes(init_path.read_bytes())
                else:
                    # Try to find chunk 1
                    for c_ext in (".webm", ".mkv"):
                        c1 = chunks_folder / f"chunk_0001{c_ext}"
                        if c1.exists():
                            c1_data = c1.read_bytes()
                            init_seg = _extract_webm_init_segment(c1_data)
                            growing_webm.write_bytes(c1_data)
                            break
                    else:
                        lgr.error("Cannot reconstruct — no chunk 1 or init segment")
                        return JSONResponse(
                            {"ok": False, "error": "Missing chunk 1 init segment — cannot decode subsequent chunks"},
                            status_code=500,
                        )

                # Also append any chunks between 1 and current
                for i in range(2, chunk_index):
                    for c_ext in (".webm", ".mkv"):
                        ci = chunks_folder / f"chunk_{i:04d}{c_ext}"
                        if ci.exists():
                            with open(growing_webm, "ab") as gf:
                                gf.write(ci.read_bytes())
                            break

            # Append current chunk data
            with open(growing_webm, "ab") as gf:
                gf.write(raw_data)
                gf.flush()
                os.fsync(gf.fileno())

            lgr.info("Growing file appended: now %d bytes total",
                      growing_webm.stat().st_size)

        # Count frames already extracted (to know which are new)
        frames_out = vs_folder / f"frames_chunk_{chunk_index:04d}"
        frames_out.mkdir(parents=True, exist_ok=True)
        # previous_frames = sum of all frames from previous chunks
        prev_frame_count = 0
        for i in range(1, chunk_index):
            prev_dir = vs_folder / f"frames_chunk_{i:04d}"
            if prev_dir.exists():
                prev_frame_count += len(list(prev_dir.glob("*.jpg")))

        # Transcode growing file to MP4
        mp4_path = vs_folder / f"growing_chunk_{chunk_index:04d}.mp4"

        ok, err = _transcode_to_mp4(str(growing_webm), str(mp4_path),
                                     crf=23, codec="libx264", lgr=lgr)
        if not ok:
            # Try mpeg4 fallback
            ok, err = _transcode_to_mp4(str(growing_webm), str(mp4_path),
                                         crf=8, codec="mpeg4", lgr=lgr)

        if not ok:
            # Save error log
            err_file = vs_folder / f"ffmpeg_stderr_chunk_{chunk_index:04d}.txt"
            err_file.write_text(err, encoding="utf-8")
            lgr.error("All transcode attempts failed for chunk %d", chunk_index)
            return JSONResponse(
                {"ok": False, "error": f"Transcode failed. Error: {err[-200:]}",
                 "error_file": str(err_file)},
                status_code=500,
            )

        # Extract ALL frames from the transcoded growing file
        all_frames_dir = vs_folder / "all_frames_temp"
        all_frames_dir.mkdir(parents=True, exist_ok=True)
        # Clear temp frames
        for f in all_frames_dir.glob("*.jpg"):
            f.unlink()

        try:
            total_frames = extract_frames(str(mp4_path), str(all_frames_dir), fps_sample=1)
        except Exception:
            total_frames = _extract_frames_ffmpeg(str(mp4_path), str(all_frames_dir))

        lgr.info("Total frames from growing file: %d (prev=%d)", total_frames, prev_frame_count)

        # Copy only the NEW frames to this chunk's frame directory
        all_frame_files = sorted(all_frames_dir.glob("*.jpg"))
        new_frames = all_frame_files[prev_frame_count:]
        frames_count = len(new_frames)
        for i, src_frame in enumerate(new_frames):
            dst_frame = frames_out / f"frame_{i+1:04d}.jpg"
            shutil.copy2(src_frame, dst_frame)

        lgr.info("New frames for chunk %d: %d", chunk_index, frames_count)

        # Clean up temp mp4 (keep growing webm)
        if mp4_path.exists():
            mp4_path.unlink()

        # Audio: extract from the growing file's last few seconds
        process_path = str(growing_webm)

    else:
        # Non-WebM: process directly
        process_path = str(chunk_path)
        frames_out = vs_folder / f"frames_chunk_{chunk_index:04d}"
        frames_out.mkdir(parents=True, exist_ok=True)
        try:
            frames_count = extract_frames(process_path, str(frames_out), fps_sample=1)
        except Exception:
            frames_count = _extract_frames_ffmpeg(process_path, str(frames_out))
            if frames_count == 0:
                return JSONResponse(
                    {"ok": False, "error": "Frame extraction failed"},
                    status_code=500,
                )

    # ── Audio + Transcription ─────────────────────────────────────────
    audio_path = vs_folder / f"chunk_{chunk_index:04d}.wav"
    t_tr0 = time.time()
    audio_ok = _extract_audio_chunk(process_path, str(audio_path))

    if audio_ok:
        try:
            transcript = transcribe_audio_whisper(str(audio_path), str(vs_folder))
        except Exception as e:
            lgr.warning("Transcription failed: %s", e)
            transcript = {"text": "(transcription failed)", "segments": [],
                          "model": "none", "time_seconds": 0}
    else:
        transcript = {"text": "(audio extraction skipped)", "segments": [],
                      "model": "none", "time_seconds": 0}
    t_tr = time.time() - t_tr0

    # ── Object Detection (using vision_worker for bbox + latency) ─────
    t_det0 = time.time()
    frame_files = sorted(str(f) for f in frames_out.glob("*.jpg"))
    all_frame_detections = []
    try:
        all_frame_detections = vision_worker.infer_frames_batch(frame_files, conf=0.20)
    except Exception:
        # Fallback to detect_frames
        try:
            det_result = detect_frames(str(frames_out), str(vs_folder), conf_threshold=0.20)
            all_frame_detections = [r.get("detections", []) for r in det_result.get("results", [])]
        except Exception:
            all_frame_detections = []
    t_det = time.time() - t_det0

    # Flatten detections for this chunk
    chunk_detections = []
    for frame_dets in all_frame_detections:
        chunk_detections.extend(frame_dets)

    # Aggregate top labels
    label_counts: dict[str, int] = {}
    for d in chunk_detections:
        lbl = d.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:6]

    # Per-frame detections with frame names (for bounding box overlay)
    per_frame_detections = []
    for i, fp in enumerate(frame_files):
        dets = all_frame_detections[i] if i < len(all_frame_detections) else []
        per_frame_detections.append({
            "frame": Path(fp).name,
            "detections": dets,
        })

    elapsed = time.time() - t0
    elapsed_ms = round(elapsed * 1000, 1)
    lgr.info("Chunk %d complete: %d frames, %.1fs elapsed", chunk_index, frames_count, elapsed)

    # Record metrics for /stream_status
    _record_chunk_metrics(video_stem, frames_count, chunk_detections, elapsed_ms)

    return {
        "ok": True,
        "video_stem": video_stem,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "frames_count": frames_count,
        "transcript_snippet": (transcript.get("text") or "")[:400],
        "transcription_time": round(t_tr, 3),
        "detection_time": round(t_det, 3),
        "detections_count": len(chunk_detections),
        "top_labels": [{"label": k, "count": v} for k, v in top_labels],
        "per_frame_detections": per_frame_detections,
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_ms": elapsed_ms,
        "sample_frames": frame_files[:3],
        "transcoded": needs_transcode,
    }


# ── /stream_status ────────────────────────────────────────────────────
@router.get("/stream_status")
async def stream_status(video_stem: str = Query(...)):
    """
    Return real-time metrics for a stream: total frames, chunks processed,
    average chunk latency, detections summary, and model performance.
    """
    with _stream_metrics_lock:
        m = _stream_metrics.get(video_stem)

    if not m:
        return {
            "ok": True,
            "video_stem": video_stem,
            "chunks_processed": 0,
            "frames_total": 0,
            "avg_chunk_latency_ms": 0,
            "median_chunk_latency_ms": 0,
            "p90_chunk_latency_ms": 0,
            "detections_last_chunk": 0,
            "total_detections": 0,
            "model_metrics": vision_worker.get_metrics(),
        }

    lats = m["latencies_ms"]
    sorted_lats = sorted(lats) if lats else [0]
    n = len(sorted_lats)

    return {
        "ok": True,
        "video_stem": video_stem,
        "chunks_processed": m["chunks_processed"],
        "frames_total": m["frames_total"],
        "avg_chunk_latency_ms": round(sum(sorted_lats) / n, 1),
        "median_chunk_latency_ms": round(sorted_lats[n // 2], 1),
        "p90_chunk_latency_ms": round(sorted_lats[int(n * 0.9)], 1) if n >= 10 else round(sorted_lats[-1], 1),
        "detections_last_chunk": len(m["last_detections"]),
        "total_detections": m["total_detections"],
        "model_metrics": vision_worker.get_metrics(),
    }


# ── /stream_finalize ──────────────────────────────────────────────────
@router.post("/stream_finalize")
async def stream_finalize(video_stem: str = Form(...)):
    """
    Stitch all uploaded chunks into one video via ffmpeg concat, then
    run the full analysis pipeline.
    """
    ff_ok, ff_ver = _check_ffmpeg()
    if not ff_ok:
        raise HTTPException(status_code=503, detail=f"ffmpeg not found: {ff_ver}")

    lgr = _get_stream_logger(video_stem)
    lgr.info("=== FINALIZE START ===")

    vs_folder = STREAM_ROOT / video_stem
    chunks_folder = vs_folder / "chunks"
    if not chunks_folder.exists():
        raise HTTPException(status_code=404, detail="No chunks found")

    # Best option: use the growing webm file directly
    growing_webm = vs_folder / "growing_stream.webm"
    if growing_webm.exists():
        stitched = vs_folder / f"{video_stem}_stitched.mp4"
        ok, err = _transcode_to_mp4(str(growing_webm), str(stitched),
                                     crf=23, codec="libx264", lgr=lgr)
        if not ok:
            ok, err = _transcode_to_mp4(str(growing_webm), str(stitched),
                                         crf=8, codec="mpeg4", lgr=lgr)
        if not ok:
            raise HTTPException(status_code=500, detail=f"Stitch transcode failed: {err[-300:]}")
    else:
        # Fallback: concat mp4 chunks
        mp4_files = sorted(chunks_folder.glob("chunk_????.mp4"))
        if not mp4_files:
            mp4_files = sorted(f for f in chunks_folder.iterdir()
                               if f.suffix.lower() in ('.webm', '.mp4', '.mkv'))
        if not mp4_files:
            raise HTTPException(status_code=404, detail="No chunks to stitch")

        filelist = vs_folder / "ffmpeg_list.txt"
        with filelist.open("w") as fh:
            for f in mp4_files:
                fh.write(f"file '{str(f).replace(chr(92), '/')}'\n")

        stitched = vs_folder / f"{video_stem}_stitched.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(filelist), "-c", "copy", str(stitched),
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           check=True, timeout=60)
        except Exception:
            cmd_re = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(filelist),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", str(stitched),
            ]
            try:
                subprocess.run(cmd_re, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=True, timeout=120)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Stitch failed: {e2}")

    lgr.info("Stitched video: %s (%d bytes)", stitched, stitched.stat().st_size)

    # Run full analysis
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
        lgr.info("=== FINALIZE COMPLETE ===")
        return JSONResponse(resp.json())
    except requests.RequestException as e:
        lgr.error("Analyze call failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Analyze call failed: {e}")
