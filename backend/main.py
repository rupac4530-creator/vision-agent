# backend/main.py
"""
Vision Agent — FastAPI application.

Endpoints
---------
GET  /              → Premium upload & dashboard UI
GET  /demo          → Demo UI
POST /upload        → Save video + extract frames
POST /analyze       → Full pipeline: frames + audio + transcription + detection
POST /generate_notes → LLM-powered notes
POST /ask           → Contextual QA over notes + transcript
POST /generate_quiz → MCQ + short-answer quiz from notes
POST /pose_analyze  → YOLO pose analysis + rep counting
POST /stt           → Speech-to-text transcription
POST /character_chat → Character/persona AI chat
POST /character_reset → Reset character chat session
GET  /personas      → List available character personas
POST /security_analyze → Security camera threat detection
POST /wanted_poster → Generate wanted poster via Gemini Vision
GET  /security_alerts → Recent security alert history
POST /crowd_analyze → Crowd density & safety analysis
GET  /crowd_trend   → Crowd safety trend history
POST /gaming_analyze → Gaming screenshot strategy analysis
GET  /gaming_history → Past gaming analysis history
GET  /metrics       → Real-time performance metrics
POST /track_objects → Persistent multi-object tracking
POST /eco_analyze   → EcoWatch Ranger forest surveillance
GET  /eco_alerts    → EcoWatch alert history
GET  /eco_wildlife  → Wildlife sighting log
POST /blindspot_analyze → BlindSpot dashcam hazard detection
GET  /blindspot_alerts  → BlindSpot alert history
POST /meeting_frame → Meeting assistant video frame analysis
POST /meeting_transcript_add → Add transcript segment to meeting
GET  /meeting_transcript → Get meeting transcript
POST /meeting_summarize → Summarize meeting with LLM
GET  /meeting_info  → Get meeting session info
POST /accessibility_describe → Accessibility scene description
GET  /accessibility_caption → Get latest live caption
"""

from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import base64
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
from llm_provider import provider as llm_provider_instance, CascadeProvider, FallbackProvider
from llm_provider import _safe_parse_json as safe_parse_json
from llm_provider import LLMQuotaError, LLMTimeoutError
from streaming import router as streaming_router

# ── New feature modules (lazy-imported at runtime to avoid startup failures) ─
# Each module uses a singleton pattern: from X import X_singleton
from stt_engine import stt_engine
from character_agent import character_agent
from security_cam import security_camera
from crowd_monitor import crowd_monitor
from gaming_companion import gaming_companion
from frame_sampler import frame_sampler
from object_tracker import object_tracker
from eco_watch import eco_watch
from blindspot import blindspot_guardian
from meeting_assistant import meeting_assistant
from accessibility import accessibility_agent
from url_ingest import router as url_ingest_router
from pose_engine import pose_engine
import jobs

# ── SDK-Aligned modules (Phase 1 integration) ─────────────────────────
from function_registry import tool_registry
from event_bus import event_bus, EventType, Event
from observability import health_tracker, platform_metrics
from video_processor import create_default_pipeline, ProcessorPipeline
from conversation import conversation_manager

# ── SDK-Aligned modules (Phase 4 — full SDK extraction) ───────────────
from llm_types import ToolSchema, NormalizedResponse, NormalizedStatus, Role, Message
from agent_core import default_agent, AgentCore, AgentConfig
from rag_engine import rag_engine, Document
from instructions import Instructions, InstructionPresets
from warmup_cache import warmup_cache
from mcp_tools import mcp_manager
from profiling import profiler
from turn_detection import SilenceTurnDetector
from transcript_buffer import TranscriptBuffer

# ── SDK-Aligned modules (Deep extraction — every remaining module) ─────
from llm_base import LLMBase, LLMResponseEvent, ToolCallEvent
from stt_base import STTBase, TranscriptResponse, STTEvent
from tts_base import TTSBase, TTSAudioChunk, TTSEvent
from vad import EnergyVAD, VADResult
from edge_types import Participant, PcmData, VideoFrame, StreamMetadata, AudioFormat
from http_transport import http_transport, HTTPTransport
from config import config, Config
from event_manager import EventManager, BaseEvent, ToolStartEvent, ToolEndEvent

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
        "llm_provider": getattr(llm_provider_instance, "name", "unknown"),
        "llm_display": getattr(llm_provider_instance, "display_name", "Unknown"),
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


# ── GET /models — list all configured LLM providers ─────────────────
@app.get("/models")
def list_models():
    """Return all available LLM providers in cascade order."""
    if isinstance(llm_provider_instance, CascadeProvider):
        providers = llm_provider_instance.all_providers_info()
    else:
        providers = [llm_provider_instance.info()]
    active = getattr(llm_provider_instance, "name", "unknown")
    return {
        "active": active,
        "active_display": getattr(llm_provider_instance, "display_name", active),
        "providers": providers,
        "cascade_chain": " → ".join(p["name"] for p in providers),
    }


# ── POST /pose_analyze — analyze frame via YOLO-pose ──────────────────
@app.post("/pose_analyze")
async def pose_analyze(
    exercise: str = Body("squat"),
    track_id: int = Body(0),
    frame_b64: str = Body(...),
):
    """Analyze a base64-encoded JPEG frame with YOLO pose model.
    Returns rep count, angles, state, and posture corrections.
    """
    import numpy as np
    import cv2
    try:
        img_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = pose_engine.analyze_frame(frame, exercise=exercise, track_id=track_id)
    return {"ok": True, **result}


# ── POST /pose_reset — reset rep counter for a session ───────────────
@app.post("/pose_reset")
async def pose_reset(track_id: int = Body(0)):
    """Reset rep counter for a pose session (start new set)."""
    pose_engine.reset_track(track_id)
    return {"ok": True, "message": f"Track {track_id} reset"}


# ── GET /pose_summary — get session coaching summary ─────────────────
@app.get("/pose_summary")
def pose_summary(track_id: int = 0):
    """Get coaching summary for a pose session."""
    return pose_engine.get_session_summary(track_id)


# ── POST /tts — generate TTS script for browser speech synthesis ──────
@app.post("/tts")
async def tts_endpoint(
    text: str = Body(...),
    voice_name: str = Body(default=""),
    rate: float = Body(default=1.0),
):
    """Return a TTS script payload for use with browser Web Speech API.
    The client should call speechSynthesis.speak() with these parameters.
    """
    # Clean text for TTS (remove markdown)
    import re
    clean = re.sub(r'[\*_`#>\[\]\(\)]', '', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return {
        "ok": True,
        "text": clean,
        "voice_name": voice_name,
        "rate": rate,
        "lang": "en-US",
    }


# ── POST /stt — speech-to-text via Whisper or Gemini STT ─────────────────
@app.post("/stt")
async def stt_endpoint(
    audio_b64: str = Body(...),
    fmt: str = Body(default="wav"),
):
    """Transcribe base64-encoded audio bytes. Supports WAV, MP3, WebM."""
    result = stt_engine.transcribe_base64(audio_b64, fmt=fmt)
    return {"ok": True, **result}


# ── POST /character_chat — persona AI conversation ─────────────────────
@app.post("/character_chat")
async def character_chat_endpoint(
    persona: str = Body("companion"),
    message: str = Body(...),
    session_id: str = Body(default="default"),
    frame_description: str = Body(default=""),
):
    """Chat with a character persona (Aldrich, Coach, Teacher, Guardian, etc.)."""
    result = character_agent.chat(
        persona_key=persona,
        user_message=message,
        session_id=session_id,
        frame_description=frame_description if frame_description else None,
    )
    return {"ok": True, **result}


# ── POST /character_reset — clear session history ──────────────────────
@app.post("/character_reset")
async def character_reset(
    session_id: str = Body(default="default"),
):
    character_agent.reset_session(session_id)
    return {"ok": True, "message": f"Session '{session_id}' cleared"}


# ── GET /personas — list all available personas ─────────────────────────
@app.get("/personas")
def list_personas():
    """Return all available AI character personas."""
    return {"personas": character_agent.list_personas()}


# ── POST /security_analyze — security camera threat detection ────────────
@app.post("/security_analyze")
async def security_analyze_endpoint(
    frame_b64: str = Body(...),
):
    """Analyze a video frame for security threats. Returns detections + alert level."""
    result = security_camera.analyze(frame_b64)
    return result


# ── POST /wanted_poster — generate wanted poster via Gemini Vision ───────
@app.post("/wanted_poster")
async def wanted_poster_endpoint(
    frame_b64: str = Body(...),
    description: str = Body(default=""),
):
    """Generate a wanted poster description for persons/objects in the frame."""
    result = security_camera.generate_wanted_poster(frame_b64, description)
    return result


# ── GET /security_alerts — recent alert history ──────────────────────
@app.get("/security_alerts")
def get_security_alerts(limit: int = 20):
    """Return recent security alert history."""
    return {
        "alerts": security_camera.get_alert_history(limit),
        "total": len(security_camera._alert_history),
    }


# ── POST /crowd_analyze — crowd density + safety score ────────────────
@app.post("/crowd_analyze")
async def crowd_analyze_endpoint(
    frame_b64: str = Body(...),
):
    """Analyze crowd density and safety in a video frame."""
    result = crowd_monitor.analyze(frame_b64)
    return result


# ── GET /crowd_trend — recent safety score history ────────────────────
@app.get("/crowd_trend")
def crowd_trend(last_n: int = 20):
    """Return recent crowd safety score trend."""
    return {"trend": crowd_monitor.get_trend(last_n)}


# ── POST /gaming_analyze — game screenshot strategy advice ────────────
@app.post("/gaming_analyze")
async def gaming_analyze_endpoint(
    frame_b64: str = Body(...),
    game_type: str = Body(default="general"),
    context: str = Body(default=""),
):
    """Analyze a game screenshot and return strategic advice."""
    result = gaming_companion.analyze(frame_b64, game_type=game_type, extra_context=context)
    return result


# ── GET /gaming_history — past gaming advice ────────────────────────
@app.get("/gaming_history")
def gaming_history():
    return {"history": gaming_companion.get_history()}


# ── GET /metrics — real-time performance metrics ─────────────────────
_metrics_store: dict = {
    "total_requests": 0,
    "pose_frames": 0,
    "security_frames": 0,
    "crowd_frames": 0,
    "gaming_frames": 0,
    "stt_calls": 0,
    "character_chats": 0,
    "start_time": time.time(),
}

@app.get("/metrics")
def get_metrics():
    """Return real-time performance metrics for the dashboard."""
    uptime = round(time.time() - _metrics_store["start_time"])
    sampler_stats = frame_sampler.stats
    tracker_count = object_tracker.active_count

    active_provider = getattr(llm_provider_instance, "name", "unknown")
    cascade_info = []
    if isinstance(llm_provider_instance, CascadeProvider):
        cascade_info = llm_provider_instance.all_providers_info()

    return {
        "uptime_seconds": uptime,
        "total_requests": _metrics_store["total_requests"],
        "pose_frames_processed": _metrics_store["pose_frames"],
        "security_frames_processed": _metrics_store["security_frames"],
        "crowd_frames_processed": _metrics_store["crowd_frames"],
        "gaming_analyses": _metrics_store["gaming_frames"],
        "stt_calls": _metrics_store["stt_calls"],
        "character_chats": _metrics_store["character_chats"],
        "frame_sampler": sampler_stats,
        "active_tracker_objects": tracker_count,
        "active_provider": active_provider,
        "llm_cascade": cascade_info,
        "server": "vision-agent/2.0",
    }


# ── POST /track_objects — persistent object tracking ─────────────────
@app.post("/track_objects")
async def track_objects_endpoint(
    detections: list = Body(default=[]),
):
    """Update object tracker with new detections, get persistent tracked objects."""
    tracked = object_tracker.update(detections)
    return {"ok": True, "tracked": tracked, "active_count": object_tracker.active_count}


# ── POST /character_reset — reset character session ───────────────────
@app.post("/character_reset")
async def character_reset_endpoint(session_id: str = Body(default="default")):
    """Reset a character chat session."""
    character_agent.reset_session(session_id)
    return {"ok": True, "session_id": session_id, "message": "Session reset."}


# ══════════════════════════════════════════════════════════════════════
# ECOWATCH RANGER ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

# ── POST /eco_analyze — forest/wildlife frame analysis ────────────────
@app.post("/eco_analyze")
async def eco_analyze_endpoint(
    frame_b64: str = Body(...),
    location: str = Body(default="Unknown"),
):
    """Analyze a frame for environmental threats (forest, wildlife, fire)."""
    _metrics_store["total_requests"] += 1
    result = eco_watch.analyze(frame_b64, location=location)
    return result


# ── GET /eco_alerts — EcoWatch alert history ──────────────────────────
@app.get("/eco_alerts")
def eco_alerts(limit: int = 20):
    return {"alerts": eco_watch.get_alert_history(limit), "stats": eco_watch.get_stats()}


# ── GET /eco_wildlife — wildlife sighting log ─────────────────────────
@app.get("/eco_wildlife")
def eco_wildlife(limit: int = 50):
    return {"sightings": eco_watch.get_wildlife_log(limit)}


# ── POST /eco_reset — reset EcoWatch session ──────────────────────────
@app.post("/eco_reset")
def eco_reset():
    eco_watch.reset()
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════
# BLINDSPOT GUARDIAN ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

# ── POST /blindspot_analyze — dashcam hazard detection ────────────────
@app.post("/blindspot_analyze")
async def blindspot_analyze_endpoint(frame_b64: str = Body(...)):
    """Analyze dashcam frame for road hazards."""
    _metrics_store["total_requests"] += 1
    result = blindspot_guardian.analyze(frame_b64)
    return result


# ── GET /blindspot_alerts — recent hazard alert history ───────────────
@app.get("/blindspot_alerts")
def blindspot_alerts(limit: int = 20):
    return {
        "alerts": blindspot_guardian.get_alert_history(limit),
        "stats": blindspot_guardian.get_stats(),
    }


# ── POST /blindspot_reset — reset session ─────────────────────────────
@app.post("/blindspot_reset")
def blindspot_reset():
    blindspot_guardian.reset_session()
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════
# MEETING ASSISTANT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

# ── POST /meeting_frame — analyze meeting video frame ─────────────────
@app.post("/meeting_frame")
async def meeting_frame_endpoint(
    frame_b64: str = Body(...),
    session_id: str = Body(default="default"),
):
    """Analyze video frame for participant count in a meeting."""
    _metrics_store["total_requests"] += 1
    return meeting_assistant.process_frame(session_id, frame_b64)


# ── POST /meeting_transcript_add — add transcript segment ─────────────
@app.post("/meeting_transcript_add")
async def meeting_transcript_add_endpoint(
    text: str = Body(...),
    session_id: str = Body(default="default"),
    speaker: str = Body(default="Speaker"),
):
    """Add a transcript segment and detect action items/decisions."""
    return meeting_assistant.add_transcript_segment(session_id, text, speaker)


# ── GET /meeting_transcript — get current transcript ──────────────────
@app.get("/meeting_transcript")
def meeting_transcript_endpoint(session_id: str = "default", last_n: int = 30):
    return {
        "transcript": meeting_assistant.get_transcript(session_id, last_n),
        "session": meeting_assistant.get_session_info(session_id),
    }


# ── POST /meeting_summarize — LLM meeting summary ─────────────────────
@app.post("/meeting_summarize")
async def meeting_summarize_endpoint(session_id: str = Body(default="default")):
    """Generate a meeting summary from transcript using LLM."""
    return meeting_assistant.summarize(session_id)


# ── GET /meeting_info — session info ─────────────────────────────────
@app.get("/meeting_info")
def meeting_info_endpoint(session_id: str = "default"):
    info = meeting_assistant.get_session_info(session_id)
    if not info:
        return {"ok": False, "error": "Session not found"}
    return {"ok": True, **info}


# ── POST /meeting_clear — clear a session ─────────────────────────────
@app.post("/meeting_clear")
def meeting_clear_endpoint(session_id: str = Body(default="default")):
    meeting_assistant.clear_session(session_id)
    return {"ok": True, "session_id": session_id}


# ══════════════════════════════════════════════════════════════════════
# ACCESSIBILITY AGENT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

# ── POST /accessibility_describe — Gemini scene description ───────────
@app.post("/accessibility_describe")
async def accessibility_describe_endpoint(
    frame_b64: str = Body(...),
    mode: str = Body(default="scene"),
    force_refresh: bool = Body(default=False),
):
    """Describe a scene for accessibility (mode: scene|read|navigate|identify|currency)."""
    _metrics_store["total_requests"] += 1
    return accessibility_agent.describe_scene(frame_b64, mode=mode, force_refresh=force_refresh)


# ── GET /accessibility_caption — latest live caption ─────────────────
@app.get("/accessibility_caption")
def accessibility_caption_endpoint():
    return accessibility_agent.get_live_caption()


# ── GET /accessibility_history — caption history ──────────────────────
@app.get("/accessibility_history")
def accessibility_history_endpoint(limit: int = 20):
    return {"history": accessibility_agent.get_history(limit)}


# ── POST /accessibility_interval — set description interval ──────────
@app.post("/accessibility_interval")
async def accessibility_interval_endpoint(seconds: float = Body(...)):
    accessibility_agent.set_description_interval(seconds)
    return {"ok": True, "interval_seconds": seconds}


# ══════════════════════════════════════════════════════════════════════
# SDK-ALIGNED ENDPOINTS (Phase 1)
# ══════════════════════════════════════════════════════════════════════

# ── GET /tools — list all registered LLM-callable tools ──────────────
@app.get("/tools")
def list_tools():
    """Return all registered function-calling tools with schemas."""
    return {"tools": tool_registry.list_functions(), "count": tool_registry.count}


# ── GET /tools/schema — full OpenAI-compatible tool schemas ──────────
@app.get("/tools/schema")
def tools_schema():
    """Return OpenAI-compatible tool schemas for all registered functions."""
    return {"tools": tool_registry.get_tools_schema()}


# ── POST /tools/call — execute a registered tool ─────────────────────
@app.post("/tools/call")
async def call_tool(
    name: str = Body(...),
    arguments: dict = Body(default={}),
):
    """Execute a registered tool by name with given arguments."""
    result = await tool_registry.execute(name, arguments)
    platform_metrics.tool_calls += 1
    if result.get("ok"):
        await event_bus.emit(Event(
            type=EventType.TOOL_RESULT,
            data={"function": name, "result": result.get("result"), "latency_ms": result.get("latency_ms")},
            source="api",
        ))
    return result


# ── POST /agent/chat — unified agent chat with function calling ──────
@app.post("/agent/chat")
async def agent_chat(
    message: str = Body(...),
    session_id: str = Body(default="default"),
    use_tools: bool = Body(default=True),
):
    """
    Unified agent chat endpoint with conversation memory and function calling.
    Maintains per-session conversation history and optionally calls tools.
    """
    session = conversation_manager.get_or_create(session_id)
    session.add_message("user", message)
    platform_metrics.questions_answered += 1

    await event_bus.emit(Event(
        type=EventType.LLM_REQUEST,
        data={"session": session_id, "message": message[:100]},
        source="agent_chat",
    ))

    # Build messages with tools
    messages = session.get_context(max_messages=15)
    tools = tool_registry.get_tools_schema() if use_tools else None

    start = time.time()
    try:
        response = llm_provider_instance.chat(
            messages=messages,
            tools=tools,
        ) if tools else llm_provider_instance.chat(messages=messages)

        latency_ms = (time.time() - start) * 1000
        health_tracker.record_success(llm_provider_instance.name, latency_ms)

        # Handle tool calls in response
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tc in response.tool_calls:
                tr = await tool_registry.execute_tool_call(tc)
                tool_results.append(tr)
            reply = json.dumps(tool_results, default=str)
        elif hasattr(response, 'choices'):
            reply = response.choices[0].message.content or ""
        elif isinstance(response, str):
            reply = response
        else:
            reply = str(response)

        session.add_message("assistant", reply)
        session.save()

        await event_bus.emit_llm_response(
            provider=llm_provider_instance.name,
            model=getattr(llm_provider_instance, 'model_id', ''),
            latency_ms=latency_ms,
        )

        return {
            "reply": reply,
            "session_id": session_id,
            "provider": llm_provider_instance.name,
            "latency_ms": round(latency_ms, 1),
            "message_count": session.message_count,
        }

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        health_tracker.record_error(llm_provider_instance.name, str(e), latency_ms)
        session.add_message("assistant", f"[Error: {str(e)[:100]}]")

        return {
            "reply": f"I encountered an error: {str(e)[:200]}",
            "session_id": session_id,
            "error": True,
            "latency_ms": round(latency_ms, 1),
        }


# ── GET /agent/conversation/{session_id} — conversation history ──────
@app.get("/agent/conversation/{session_id}")
def get_conversation(session_id: str):
    """Get conversation history for a session."""
    session = conversation_manager.get(session_id)
    if not session:
        return {"session_id": session_id, "messages": [], "exists": False}
    return session.to_dict()


# ── GET /agent/sessions — list all conversation sessions ─────────────
@app.get("/agent/sessions")
def list_sessions():
    """List all conversation sessions."""
    return {"sessions": conversation_manager.list_sessions(), "count": conversation_manager.count}


# ── DELETE /agent/conversation/{session_id} — delete a session ────────
@app.delete("/agent/conversation/{session_id}")
def delete_conversation(session_id: str):
    ok = conversation_manager.delete(session_id)
    return {"ok": ok, "session_id": session_id}


# ── GET /events — SSE stream of real-time agent events ───────────────
from starlette.responses import StreamingResponse

@app.get("/events")
async def events_stream():
    """Server-Sent Events stream for real-time agent events."""
    async def generate():
        yield "data: {\"type\": \"connected\", \"message\": \"Event stream connected\"}\n\n"
        async for sse in event_bus.sse_stream():
            yield sse

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


# ── GET /events/history — recent event history ────────────────────────
@app.get("/events/history")
def events_history(limit: int = 50, event_type: str = ""):
    """Get recent event history."""
    et = None
    if event_type:
        try:
            et = EventType(event_type)
        except ValueError:
            pass
    return {"events": event_bus.get_history(limit=limit, event_type=et),
            "stats": event_bus.get_stats()}


# ── GET /provider_health — LLM provider health dashboard ─────────────
@app.get("/provider_health")
def provider_health():
    """Get health status and metrics for all LLM providers."""
    return health_tracker.get_all_stats()


# ── GET /metrics/prometheus — Prometheus-compatible metrics ───────────
@app.get("/metrics/prometheus")
def prometheus_metrics():
    """Prometheus-compatible text metrics for monitoring."""
    from starlette.responses import Response
    text = health_tracker.to_prometheus() + "\n" + platform_metrics.to_prometheus()
    return Response(content=text, media_type="text/plain")


# ── GET /metrics/platform — platform usage metrics ───────────────────
@app.get("/metrics/platform")
def platform_metrics_endpoint():
    return platform_metrics.to_dict()


# ── GET /pipeline/stats — video processor pipeline stats ─────────────
_default_pipeline = create_default_pipeline()

@app.get("/pipeline/stats")
def pipeline_stats():
    return _default_pipeline.get_stats()


# ══════════════════════════════════════════════════════════════════════
# SDK-ALIGNED ENDPOINTS (Phase 4 — Full SDK Extraction)
# ══════════════════════════════════════════════════════════════════════

# ── Global instances for Phase 4 endpoints ────────────────────────────
_turn_detector = SilenceTurnDetector()
_transcript_buf = TranscriptBuffer()


# ── POST /rag/add — add documents to RAG index ───────────────────────
@app.post("/rag/add")
async def rag_add(text: str = Body(...), source: str = Body(default="api")):
    """Add a document to the RAG knowledge base."""
    doc = Document(text=text, source=source)
    chunks = await rag_engine.add_documents([doc])
    return {"ok": True, "chunks_indexed": chunks, "source": source}


# ── POST /rag/search — search the RAG knowledge base ─────────────────
@app.post("/rag/search")
async def rag_search(query: str = Body(...), top_k: int = Body(default=3)):
    """Search the RAG knowledge base."""
    results = await rag_engine.search(query, top_k=top_k)
    return {"query": query, "results": results, "stats": rag_engine.get_stats()}


# ── GET /rag/stats — RAG index statistics ─────────────────────────────
@app.get("/rag/stats")
def rag_stats():
    return rag_engine.get_stats()


# ── POST /instructions/parse — parse instruction text with @-mentions ─
@app.post("/instructions/parse")
async def instructions_parse(
    text: str = Body(...),
    base_dir: str = Body(default=""),
):
    """Parse instruction text, inlining any @mentioned markdown files."""
    inst = Instructions(input_text=text, base_dir=base_dir)
    return {
        "full_reference": inst.full_reference,
        "referenced_files": inst.referenced_files,
        "info": inst.to_dict(),
    }


# ── GET /instructions/presets — list available instruction presets ────
@app.get("/instructions/presets")
def instructions_presets():
    return {
        "presets": {
            "security_camera": InstructionPresets.security_camera()[:100] + "...",
            "fitness_coach": InstructionPresets.fitness_coach()[:100] + "...",
            "meeting_assistant": InstructionPresets.meeting_assistant()[:100] + "...",
            "accessibility_helper": InstructionPresets.accessibility_helper()[:100] + "...",
        }
    }


# ── GET /warmup/stats — model warmup cache statistics ─────────────────
@app.get("/warmup/stats")
def warmup_stats():
    return warmup_cache.get_stats()


# ── GET /profiling/stats — performance profiling statistics ───────────
@app.get("/profiling/stats")
def profiling_stats(name: str = ""):
    """Get profiling statistics (all or for a specific operation)."""
    return profiler.get_stats(name=name if name else None)


# ── POST /agent/run — run the default agent with tool-calling loop ────
@app.post("/agent/run")
async def agent_run(
    input: str = Body(...),
    context: dict = Body(default={}),
    session_id: str = Body(default=""),
):
    """Run the default agent with optional context and tool calling."""
    response = await default_agent.run(input, context=context or None, session_id=session_id or None)
    return response.to_dict()


# ── GET /agent/stats — agent statistics ───────────────────────────────
@app.get("/agent/stats")
def agent_stats():
    return default_agent.get_stats()


# ── POST /mcp/add_server — add an MCP tool server ────────────────────
@app.post("/mcp/add_server")
async def mcp_add_server(
    name: str = Body(...),
    url: str = Body(default=""),
    command: list = Body(default=[]),
):
    """Connect to an MCP tool server (local or remote)."""
    server = await mcp_manager.add_server(
        name=name,
        url=url if url else None,
        command=command if command else None,
    )
    return {"ok": True, "server": server.get_stats()}


# ── GET /mcp/stats — MCP manager statistics ───────────────────────────
@app.get("/mcp/stats")
def mcp_stats():
    return mcp_manager.get_stats()


# ── POST /transcript/push — push STT result to transcript buffer ─────
@app.post("/transcript/push")
async def transcript_push(
    text: str = Body(...),
    is_final: bool = Body(default=True),
    speaker_id: str = Body(default=""),
):
    """Push a speech-to-text result to the transcript buffer."""
    if is_final:
        _transcript_buf.push_final(text, speaker_id=speaker_id or None)
    else:
        _transcript_buf.push_interim(text, speaker_id=speaker_id or None)
    return {
        "current_text": _transcript_buf.get_current_text(),
        "complete_sentences": _transcript_buf.get_complete_sentences(),
        "stats": _transcript_buf.get_stats(),
    }


# ── GET /transcript/stats — transcript buffer statistics ─────────────
@app.get("/transcript/stats")
def transcript_stats():
    return _transcript_buf.get_stats()


# ── POST /turn/audio_level — feed audio level to turn detector ────────
@app.post("/turn/audio_level")
async def turn_audio_level(level: float = Body(...)):
    """Feed an audio level sample to the turn detector."""
    event = _turn_detector.on_audio_level(level)
    return {
        "is_speaking": _turn_detector.is_speaking,
        "should_respond": _turn_detector.should_respond(),
        "event": {"type": event.type, "duration_ms": event.duration_ms} if event else None,
        "stats": _turn_detector.get_stats(),
    }


# ── GET /turn/stats — turn detector statistics ────────────────────────
@app.get("/turn/stats")
def turn_stats():
    return _turn_detector.get_stats()


# ── Global VAD instance ──────────────────────────────────────────────
_vad = EnergyVAD()

# ── GET /config — platform configuration (safe, no secrets) ──────────
@app.get("/config")
def config_endpoint():
    return config.to_dict()


# ── POST /vad/process — process audio for voice activity ─────────────
@app.post("/vad/process")
async def vad_process(audio_level: float = Body(...)):
    """Feed audio level to VAD (energy-based voice activity detection)."""
    # Simulate PCM processing with energy level
    result = _vad.process(int(audio_level * 32768).to_bytes(2, 'little', signed=True) * 160)
    return {
        "is_speech": result.is_speech,
        "energy": round(result.energy, 4),
        "threshold": round(result.threshold, 4),
        "stats": _vad.get_stats(),
    }


# ── GET /vad/stats — VAD statistics ──────────────────────────────────
@app.get("/vad/stats")
def vad_stats():
    return _vad.get_stats()


# ── GET /transport/stats — HTTP transport adapter stats ──────────────
@app.get("/transport/stats")
def transport_stats():
    return http_transport.get_stats()


# ── GET /modules — list all loaded SDK modules ───────────────────────
@app.get("/modules")
def list_modules():
    """List all loaded Vision-Agents SDK modules with status."""
    modules = {
        # Phase 1-3 core
        "function_registry": {"status": "loaded", "phase": 1},
        "event_bus": {"status": "loaded", "phase": 1},
        "observability": {"status": "loaded", "phase": 1},
        "video_processor": {"status": "loaded", "phase": 1},
        "conversation": {"status": "loaded", "phase": 1},
        # Phase 4 SDK extraction
        "llm_types": {"status": "loaded", "phase": 4},
        "agent_core": {"status": "loaded", "phase": 4},
        "rag_engine": {"status": "loaded", "phase": 4},
        "instructions": {"status": "loaded", "phase": 4},
        "warmup_cache": {"status": "loaded", "phase": 4},
        "mcp_tools": {"status": "loaded", "phase": 4},
        "profiling": {"status": "loaded", "phase": 4},
        "turn_detection": {"status": "loaded", "phase": 4},
        "transcript_buffer": {"status": "loaded", "phase": 4},
        # Deep extraction (Phase 5)
        "llm_base": {"status": "loaded", "phase": 5},
        "stt_base": {"status": "loaded", "phase": 5},
        "tts_base": {"status": "loaded", "phase": 5},
        "vad": {"status": "loaded", "phase": 5},
        "edge_types": {"status": "loaded", "phase": 5},
        "http_transport": {"status": "loaded", "phase": 5},
        "config": {"status": "loaded", "phase": 5},
        "event_manager": {"status": "loaded", "phase": 5},
    }
    return {
        "modules": modules,
        "total": len(modules),
        "by_phase": {
            "phase_1_3": sum(1 for m in modules.values() if m["phase"] <= 3),
            "phase_4": sum(1 for m in modules.values() if m["phase"] == 4),
            "phase_5": sum(1 for m in modules.values() if m["phase"] == 5),
        }
    }


# ── Startup event: emit agent started ─────────────────────────────────
@app.on_event("startup")
async def on_startup():
    await event_bus.emit(Event(
        type=EventType.AGENT_STARTED,
        data={
            "version": "3.0.0-full-sdk",
            "tools_registered": tool_registry.count,
            "llm_provider": llm_provider_instance.name,
            "modules_loaded": [
                "llm_types", "agent_core", "rag_engine", "instructions",
                "warmup_cache", "mcp_tools", "profiling", "turn_detection",
                "transcript_buffer", "function_registry", "event_bus",
                "observability", "video_processor", "conversation",
            ],
            "rag_stats": rag_engine.get_stats(),
            "warmup_stats": warmup_cache.get_stats(),
        },
        source="main",
    ))
    logger.info("Vision Agent v3.0 started — %d tools, %d modules",
                tool_registry.count, 14)


# ── Shutdown event: save conversations ────────────────────────────────
@app.on_event("shutdown")
async def on_shutdown():
    conversation_manager.save_all()
    await event_bus.emit(Event(type=EventType.AGENT_STOPPED, source="main"))
    logger.info("Vision Agent shutting down, conversations saved")


# ── Run server ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
