# backend/main.py
"""
Vision Agent — FastAPI application.

Endpoints
---------
GET  /              → Premium upload & dashboard UI
GET  /demo          → Demo UI with timeline, QA chat, quiz, MathJax
POST /upload        → Save video + extract frames (Day 1)
POST /analyze       → Full pipeline: frames + audio + transcription + detection (Day 2)
POST /generate_notes → LLM-powered notes — returns job_id for polling (Day 3)
POST /stream_chunk  → Real-time per-chunk processing (Day 4, via streaming router)
POST /stream_finalize → Stitch chunks + full analysis (Day 4, via streaming router)
POST /ask           → Contextual QA over notes + transcript (Day 5)
POST /generate_quiz → MCQ + short-answer quiz from notes (Day 5)
"""

from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import shutil
import subprocess
import time
import threading
import traceback
import logging

import uvicorn

from frame_extractor import extract_frames
from transcribe import transcribe_audio_whisper
from detect import detect_frames
from generate_notes import generate_notes_from_analysis
from llm_provider import provider as llm_provider_instance
from llm_provider import _safe_parse_json as safe_parse_json
from llm_provider import LLMQuotaError, LLMTimeoutError
from streaming import router as streaming_router
from url_ingest import router as url_ingest_router
import jobs

logger = logging.getLogger("main")

# ── App setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Vision Agent",
    description="Real-time multimodal AI video agent — WeMakeDevs Hackathon",
    version="2.0.0",
)

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
FRAMES_DIR = BASE / "frames"
ANALYSIS_DIR = BASE / "analysis"
UPLOAD_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

# ── LLM response cache (in-memory, process-local) ─────────────────────
LLM_CACHE: dict = {}


def cached_llm(key: str, call_fn, ttl: int = 3600):
    """Return cached result if present, else call and cache."""
    now = time.time()
    if key in LLM_CACHE and now - LLM_CACHE[key]["ts"] < ttl:
        return LLM_CACHE[key]["val"]
    val = call_fn()
    LLM_CACHE[key] = {"ts": now, "val": val}
    return val


# ── Mount static directories ──────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
app.mount("/analysis", StaticFiles(directory=str(ANALYSIS_DIR)), name="analysis")
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Include routers
app.include_router(streaming_router)
app.include_router(url_ingest_router)
app.include_router(jobs.router)  # Shared /jobs/{id} endpoint

_start_time = time.time()


@app.get("/health")
def health_check():
    """System health / status endpoint."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "ffmpeg_available": _is_ffmpeg_available(),
        "uploads_count": len(list(UPLOAD_DIR.glob("*"))) if UPLOAD_DIR.exists() else 0,
        "frames_count": len(list(FRAMES_DIR.glob("*"))) if FRAMES_DIR.exists() else 0,
        "analyses_count": len(list(ANALYSIS_DIR.glob("*.json"))) if ANALYSIS_DIR.exists() else 0,
        "llm_provider": getattr(__import__("llm_provider", fromlist=["ACTIVE_PROVIDER"]), "ACTIVE_PROVIDER", "unknown"),
    }


# ── Helpers ────────────────────────────────────────────────────────────
def _is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_audio(video_path: str, out_audio: str) -> str:
    video_p = Path(video_path)
    if not video_p.exists():
        raise RuntimeError(f"Video file not found: {video_path}")
    if not _is_ffmpeg_available():
        raise RuntimeError("ffmpeg not found in PATH.")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_p),
        "-ac", "1", "-ar", "16000", "-vn", str(out_audio),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore")[:2000]
        raise RuntimeError(f"ffmpeg error:\n{err}")
    return out_audio


# ── GET / ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = BASE / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── GET /demo ──────────────────────────────────────────────────────────
@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    html_path = BASE / "static" / "demo.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── POST /upload (Day 1) ──────────────────────────────────────────────
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    fname = file.filename or "uploaded_video.mp4"
    ext = Path(fname).suffix or ".mp4"
    saved_path = UPLOAD_DIR / f"video{ext}"

    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    out_folder = FRAMES_DIR / saved_path.stem
    if out_folder.exists():
        for p in out_folder.iterdir():
            if p.is_file():
                p.unlink()
    out_folder.mkdir(parents=True, exist_ok=True)

    try:
        count = extract_frames(str(saved_path), str(out_folder), fps_sample=1)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    return {
        "ok": True,
        "video": str(saved_path),
        "frames_dir": str(out_folder),
        "frames_count": count,
    }


# ── POST /analyze (Day 2) ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """Full pipeline: upload → frames → audio → transcript → detection."""
    fname = file.filename or "uploaded_video.mp4"
    ext = Path(fname).suffix or ".mp4"
    saved_path = UPLOAD_DIR / f"video{ext}"

    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    video_stem = saved_path.stem
    frames_folder = FRAMES_DIR / video_stem
    frames_folder.mkdir(parents=True, exist_ok=True)

    for p in frames_folder.iterdir():
        if p.is_file():
            p.unlink()

    t_start = time.time()
    warnings = []

    frames_count = extract_frames(str(saved_path), str(frames_folder), fps_sample=1)
    t_frames = time.time() - t_start

    analysis_dir = ANALYSIS_DIR / video_stem
    analysis_dir.mkdir(parents=True, exist_ok=True)
    audio_path = analysis_dir / "audio.wav"
    audio_ok = False

    try:
        _extract_audio(str(saved_path), str(audio_path))
        audio_ok = True
    except Exception as e:
        warnings.append(f"Audio extraction failed: {e}")

    t0 = time.time()
    if audio_ok:
        transcript = transcribe_audio_whisper(str(audio_path), str(analysis_dir))
    else:
        transcript = {
            "text": "(Audio extraction failed — transcription skipped.)",
            "segments": [], "model": "none", "time_seconds": 0,
        }
        (analysis_dir / "transcript.json").write_text(
            json.dumps(transcript, indent=2), encoding="utf-8")
        warnings.append("Transcription skipped due to audio extraction failure.")
    t_transcribe = time.time() - t0

    t0 = time.time()
    detections = detect_frames(str(frames_folder), str(analysis_dir))
    t_detect = time.time() - t0

    total_time = time.time() - t_start

    analysis = {
        "ok": True, "video": str(saved_path), "video_stem": video_stem,
        "frames_count": frames_count,
        "frames_time_seconds": round(t_frames, 3),
        "transcription_time_seconds": round(t_transcribe, 3),
        "detection_time_seconds": round(t_detect, 3),
        "total_time_seconds": round(total_time, 3),
        "transcript": transcript,
        "detections_summary": detections["summary"],
    }
    if warnings:
        analysis["warnings"] = warnings

    with open(analysis_dir / "analysis.json", "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)

    return analysis


# ── POST /generate_notes (Day 3) — ASYNC JOB FLOW ─────────────────────
def _run_notes_job(job_id: str, video_stem: str, analysis_file: str):
    """Background worker for notes generation."""
    try:
        jobs.set_progress(job_id, 10, "Loading analysis data")
        logger.info("[%s] Notes job started for stem=%s", job_id, video_stem)

        jobs.set_progress(job_id, 30, "Calling LLM provider")
        notes = generate_notes_from_analysis(analysis_file)

        jobs.set_progress(job_id, 80, "Saving notes")
        out_path = Path(analysis_file).parent / "notes.json"
        out_path.write_text(json.dumps(notes, indent=2), encoding="utf-8")

        provider_name = notes.get("_llm_meta", {}).get("provider", "unknown")
        is_fallback = notes.get("_fallback", False)

        jobs.set_done(job_id, {
            "video_stem": video_stem,
            "notes_path": str(out_path),
            "llm_provider": provider_name,
            "is_fallback": is_fallback,
        })
        logger.info("[%s] Notes job completed (provider=%s)", job_id, provider_name)

    except LLMQuotaError as e:
        logger.warning("[%s] LLM quota error: %s", job_id, e)
        # Fall back to generating context-aware placeholder notes
        try:
            from llm_provider import FallbackProvider
            fallback = FallbackProvider()
            analysis_data = json.loads(Path(analysis_file).read_text(encoding="utf-8"))
            notes = fallback.generate_notes(analysis_data)
            out_path = Path(analysis_file).parent / "notes.json"
            out_path.write_text(json.dumps(notes, indent=2), encoding="utf-8")
            jobs.set_done(job_id, {
                "video_stem": video_stem,
                "notes_path": str(out_path),
                "llm_provider": "fallback",
                "is_fallback": True,
                "warning": f"LLM quota exceeded — auto-summary generated instead. ({e})",
            })
        except Exception as e2:
            jobs.set_failed(job_id, f"LLM quota error and fallback also failed: {e2}")

    except LLMTimeoutError as e:
        logger.warning("[%s] LLM timeout: %s", job_id, e)
        jobs.set_failed(job_id, f"LLM timed out: {e}. Please try again.")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("[%s] Notes job error: %s\n%s", job_id, e, tb)
        jobs.set_failed(job_id, str(e))


@app.post("/generate_notes")
async def generate_notes_endpoint(video_stem: str = "video"):
    """
    Start a background notes generation job. Returns job_id for polling.
    Poll GET /jobs/{job_id} for progress and results.
    """
    analysis_file = ANALYSIS_DIR / video_stem / "analysis.json"
    if not analysis_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"analysis.json not found for '{video_stem}'. Run /analyze first.",
        )

    job_id = jobs.create_job("notes", {"video_stem": video_stem})

    thread = threading.Thread(
        target=_run_notes_job,
        args=(job_id, video_stem, str(analysis_file)),
        daemon=True,
    )
    thread.start()

    return JSONResponse({
        "ok": True,
        "job_id": job_id,
        "status_url": f"/jobs/{job_id}",
    })


# ── POST /ask (2-Tier Agent) ───────────────────────────────────────────
@app.post("/ask")
async def ask_question(
    video_stem: str = Body(...),
    question: str = Body(...),
    llm_provider_name: str = Body(default="auto"),
    llm_model: str = Body(default=""),
):
    """
    Two-tier agent response:
    - Tier A (FastReply): instant deterministic reply (<500ms) from YOLO labels + transcript
    - Tier B (PolishReply): background LLM job, poll via /jobs/{job_id}

    Returns: {fast_reply: {reply, provenance, source, latency_ms}, job_id, status}
    """
    from agent_core import ask_agent

    t_start = time.time()
    result = ask_agent(video_stem, question, llm_provider_name, llm_model)
    result["elapsed_seconds"] = round(time.time() - t_start, 3)
    return result


# ── POST /generate_quiz (Day 5) ───────────────────────────────────────
@app.post("/generate_quiz")
async def generate_quiz(video_stem: str = Body(...), count: int = Body(5)):
    """Generate MCQs and short-answer questions from notes."""
    t_start = time.time()

    notes_file = ANALYSIS_DIR / video_stem / "notes.json"
    if not notes_file.exists():
        raise HTTPException(status_code=404, detail="notes.json not found.")

    notes = json.loads(notes_file.read_text(encoding="utf-8"))

    schema_example = json.dumps({
        "mcq": [{"question": "...", "options": ["A", "B", "C", "D"], "answer": "A", "explanation": "..."}],
        "short": [{"question": "...", "answer": "..."}],
    }, indent=2)

    messages = [
        {"role": "system", "content": "You generate study questions from lecture notes. Return valid JSON ONLY."},
        {"role": "user", "content": (
            "NOTES:\n" + json.dumps(notes, indent=2)[:2000]
            + f"\n\nGenerate exactly {count} MCQs and {count} short-answer "
            "questions.\n\n" f"OUTPUT SCHEMA:\n{schema_example}"
        )},
    ]

    def _call():
        try:
            text, _ = llm_provider_instance.chat(messages, max_tokens=1200, temperature=0.0)
            return safe_parse_json(text)
        except (LLMQuotaError, LLMTimeoutError) as e:
            return {"mcq": [], "short": [], "error": f"LLM unavailable: {e}"}
        except Exception as e:
            return {"mcq": [], "short": [], "error": f"Error: {e}"}

    cache_key = f"quiz::{video_stem}::{count}"
    quiz = cached_llm(cache_key, _call, ttl=3600)
    quiz["elapsed_seconds"] = round(time.time() - t_start, 3)

    quiz_path = ANALYSIS_DIR / video_stem / "quiz.json"
    quiz_path.write_text(json.dumps(quiz, indent=2), encoding="utf-8")

    return quiz


# ── Run server ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
