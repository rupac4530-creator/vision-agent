# backend/main.py
"""
Vision Agent — FastAPI application.

Endpoints
---------
GET  /              → Premium upload & dashboard UI
GET  /demo          → Demo UI with timeline, QA chat, quiz, MathJax
POST /upload        → Save video + extract frames (Day 1)
POST /analyze       → Full pipeline: frames + audio + transcription + detection (Day 2)
POST /generate_notes → LLM-powered notes from analysis.json (Day 3)
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

import uvicorn

from frame_extractor import extract_frames
from transcribe import transcribe_audio_whisper
from detect import detect_frames
from generate_notes import generate_notes_from_analysis
from llm_helpers import call_llm, safe_parse_json
from streaming import router as streaming_router

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
# Main UI assets
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
# Serve analysis outputs (notes.json, detections.json, etc.)
app.mount("/analysis", StaticFiles(directory=str(ANALYSIS_DIR)), name="analysis")
# Serve extracted frames (for timeline thumbnails)
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")
# Serve uploaded videos (for video player)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Include real-time streaming router (Day 4)
app.include_router(streaming_router)


# ── Helpers ────────────────────────────────────────────────────────────
def _is_ffmpeg_available() -> bool:
    """Return True if the ffmpeg binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def _extract_audio(video_path: str, out_audio: str) -> str:
    """
    Extract mono 16 kHz WAV from video using ffmpeg subprocess.
    Raises RuntimeError with a clear message if ffmpeg is missing or fails.
    """
    video_p = Path(video_path)
    if not video_p.exists():
        raise RuntimeError(f"Video file not found: {video_path}")

    if not _is_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg binary not found in PATH. "
            "Install ffmpeg (choco install ffmpeg) or run the Docker image."
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_p),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(out_audio),
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
    """Full pipeline: upload → frames → audio → transcript → detection.
    
    RESILIENT: If audio extraction fails (e.g. ffmpeg missing), the pipeline
    continues with frames + detection and returns partial results with warnings.
    """
    fname = file.filename or "uploaded_video.mp4"
    ext = Path(fname).suffix or ".mp4"
    saved_path = UPLOAD_DIR / f"video{ext}"

    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    video_stem = saved_path.stem
    frames_folder = FRAMES_DIR / video_stem
    frames_folder.mkdir(parents=True, exist_ok=True)

    # Clean old frames
    for p in frames_folder.iterdir():
        if p.is_file():
            p.unlink()

    t_start = time.time()
    warnings = []

    # Step 0: Extract frames
    frames_count = extract_frames(str(saved_path), str(frames_folder), fps_sample=1)
    t_frames = time.time() - t_start

    # Step 1: Extract audio (resilient — continues if fails)
    analysis_dir = ANALYSIS_DIR / video_stem
    analysis_dir.mkdir(parents=True, exist_ok=True)
    audio_path = analysis_dir / "audio.wav"
    audio_ok = False

    try:
        _extract_audio(str(saved_path), str(audio_path))
        audio_ok = True
    except Exception as e:
        warnings.append(f"Audio extraction failed: {e}")

    # Step 2: Transcribe (skip if audio extraction failed)
    t0 = time.time()
    if audio_ok:
        transcript = transcribe_audio_whisper(str(audio_path), str(analysis_dir))
    else:
        transcript = {
            "text": "(Audio extraction failed — transcription skipped. See warnings.)",
            "segments": [],
            "model": "none",
            "time_seconds": 0,
        }
        # Save placeholder transcript
        (analysis_dir / "transcript.json").write_text(
            json.dumps(transcript, indent=2), encoding="utf-8"
        )
        warnings.append("Transcription skipped due to audio extraction failure.")
    t_transcribe = time.time() - t0

    # Step 3: Object detection (always runs)
    t0 = time.time()
    detections = detect_frames(str(frames_folder), str(analysis_dir))
    t_detect = time.time() - t0

    total_time = time.time() - t_start

    analysis = {
        "ok": True,
        "video": str(saved_path),
        "video_stem": video_stem,
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

    # Persist analysis.json
    with open(analysis_dir / "analysis.json", "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)

    return analysis


# ── POST /generate_notes (Day 3) ──────────────────────────────────────
@app.post("/generate_notes")
async def generate_notes_endpoint(video_stem: str = "video"):
    """
    Generate structured study notes from analysis.json using an LLM.
    Requires OPENAI_API_KEY environment variable.
    """
    analysis_file = ANALYSIS_DIR / video_stem / "analysis.json"
    if not analysis_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"analysis.json not found for '{video_stem}'. Run /analyze first.",
        )

    try:
        notes = generate_notes_from_analysis(str(analysis_file))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Persist notes.json
    out_path = analysis_file.parent / "notes.json"
    out_path.write_text(json.dumps(notes, indent=2), encoding="utf-8")

    return notes


# ── POST /ask (Day 5) — Contextual QA ─────────────────────────────────
@app.post("/ask")
async def ask_question(video_stem: str = Body(...), question: str = Body(...)):
    """
    Answer a question using notes.json + transcript + detections.
    Returns: {answer, provenance, confidence, elapsed_seconds}
    """
    t_start = time.time()

    analysis_dir = ANALYSIS_DIR / video_stem
    notes_file = analysis_dir / "notes.json"
    analysis_file = analysis_dir / "analysis.json"

    if not notes_file.exists() or not analysis_file.exists():
        raise HTTPException(
            status_code=404,
            detail="notes.json / analysis.json not found. Run /analyze and /generate_notes first.",
        )

    notes = json.loads(notes_file.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_file.read_text(encoding="utf-8"))

    # Build context: summary + highlights + transcript snippet
    ctx = []
    ctx.append("SUMMARY:\n" + (notes.get("summary", ""))[:1200])

    for h in notes.get("highlights", [])[:3]:
        ctx.append(f"HIGHLIGHT ({h.get('timestamp')}s): {h.get('text')}")

    # Key concepts + formulas
    concepts = notes.get("key_concepts", [])
    if concepts:
        ctx.append("KEY CONCEPTS: " + ", ".join(concepts[:10]))
    formulas = notes.get("formulas", [])
    if formulas:
        ctx.append("FORMULAS: " + "; ".join(f.get("latex", "") for f in formulas[:5]))

    ctx.append(
        "TRANSCRIPT_SNIPPET:\n"
        + analysis.get("transcript", {}).get("text", "")[:1200]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise assistant that answers questions about a recorded "
                "lecture. Use ONLY the provided context. If the answer is not in the "
                "context, say 'I don't know — not enough info.' Cite transcript "
                "excerpts when possible. Return valid JSON: "
                '{"answer":"...","provenance":"excerpt","confidence":0.0-1.0}'
            ),
        },
        {
            "role": "user",
            "content": "CONTEXT:\n" + "\n\n".join(ctx) + f"\n\nQUESTION: {question}",
        },
    ]

    def _call():
        text, _ = call_llm(messages, max_tokens=400, temperature=0.0)
        try:
            return safe_parse_json(text)
        except Exception:
            return {"answer": text.strip(), "provenance": None, "confidence": 0.0}

    cache_key = f"ask::{video_stem}::{question}"
    result = cached_llm(cache_key, _call, ttl=3600)
    result["elapsed_seconds"] = round(time.time() - t_start, 3)
    return result


# ── POST /generate_quiz (Day 5) — MCQ + short answer ──────────────────
@app.post("/generate_quiz")
async def generate_quiz(video_stem: str = Body(...), count: int = Body(5)):
    """
    Generate *count* MCQs and *count* short-answer questions from notes.json.
    Returns: {mcq: [...], short: [...], elapsed_seconds}
    """
    t_start = time.time()

    notes_file = ANALYSIS_DIR / video_stem / "notes.json"
    if not notes_file.exists():
        raise HTTPException(status_code=404, detail="notes.json not found. Run /generate_notes first.")

    notes = json.loads(notes_file.read_text(encoding="utf-8"))

    schema_example = json.dumps(
        {
            "mcq": [
                {
                    "question": "...",
                    "options": ["A", "B", "C", "D"],
                    "answer": "A",
                    "explanation": "...",
                }
            ],
            "short": [{"question": "...", "answer": "..."}],
        },
        indent=2,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You generate well-written study questions from lecture notes. "
                "Return valid JSON ONLY. Follow the schema exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                "NOTES:\n" + json.dumps(notes, indent=2)[:2000]
                + f"\n\nGenerate exactly {count} MCQs and {count} short-answer "
                "questions. MCQ options must be plausible distractors. Provide "
                "the correct answer and a 1-line explanation.\n\n"
                f"OUTPUT SCHEMA:\n{schema_example}"
            ),
        },
    ]

    def _call():
        text, _ = call_llm(messages, max_tokens=1200, temperature=0.0)
        return safe_parse_json(text)

    cache_key = f"quiz::{video_stem}::{count}"
    quiz = cached_llm(cache_key, _call, ttl=3600)
    quiz["elapsed_seconds"] = round(time.time() - t_start, 3)

    # Persist quiz.json
    quiz_path = ANALYSIS_DIR / video_stem / "quiz.json"
    quiz_path.write_text(json.dumps(quiz, indent=2), encoding="utf-8")

    return quiz


# ── Run server ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
