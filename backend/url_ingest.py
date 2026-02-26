# backend/url_ingest.py
"""
URL-based video ingestion for Vision Agent — ASYNC JOB FLOW.

POST /fetch_and_analyze → returns {ok, job_id, status_url} immediately
Background thread runs: download → transcode → extract → detect → transcribe → notes
Uses shared job store from jobs.py for status polling via GET /jobs/{id}
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import threading
import traceback
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from frame_extractor import extract_frames
from transcribe import transcribe_audio_whisper
from detect import detect_frames
from generate_notes import generate_notes_from_analysis
import jobs

router = APIRouter(tags=["url_ingest"])

BASE = Path(__file__).resolve().parent
FETCH_DIR = BASE / "remote_fetch"
ANALYSIS_DIR = BASE / "analysis"
FRAMES_DIR = BASE / "frames"
LOGS_DIR = BASE / "logs"
FETCH_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

CACHE_FILE = FETCH_DIR / "index.json"

ALLOWED_DOMAINS = {
    "youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com",
    "vimeo.com", "www.vimeo.com",
    "facebook.com", "www.facebook.com", "fb.watch",
    "twitter.com", "x.com", "www.twitter.com",
    "twitch.tv", "www.twitch.tv", "clips.twitch.tv",
    "dailymotion.com", "www.dailymotion.com",
    "instagram.com", "www.instagram.com",
}

_rate_limit: dict[str, list[float]] = {}
MAX_REQUESTS_PER_MINUTE = 5

logger = logging.getLogger("url_ingest")


# ── URL validation ────────────────────────────────────────────────────

def _validate_url(url: str) -> tuple[bool, str]:
    if not url or len(url) > 2048:
        return False, "URL is empty or too long"
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    if not url_pattern.match(url):
        return False, "Invalid URL format"
    from urllib.parse import urlparse
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    blocked = ("localhost", "127.0.0.1", "0.0.0.0", "10.", "192.168.", "172.")
    if any(hostname.startswith(b) for b in blocked):
        return False, "Internal/private URLs are not allowed"
    domain_parts = hostname.split(".")
    for i in range(len(domain_parts)):
        candidate = ".".join(domain_parts[i:])
        if candidate in ALLOWED_DOMAINS:
            return True, ""
    return False, f"Domain '{hostname}' not in allowed list"


def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    if client_ip not in _rate_limit:
        _rate_limit[client_ip] = []
    _rate_limit[client_ip] = [t for t in _rate_limit[client_ip] if now - t < 60]
    if len(_rate_limit[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        return False
    _rate_limit[client_ip].append(now)
    return True


# ── Cache ─────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


# ── yt-dlp download ──────────────────────────────────────────────────

def _download_video(url: str, workdir: Path, job_id: str) -> tuple[bool, str, str]:
    """Download video via yt_dlp Python API."""
    try:
        import yt_dlp
    except ImportError:
        return False, "", "yt_dlp not installed. pip install yt-dlp"

    out_template = str(workdir / "video.%(ext)s")
    opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "max_filesize": 500 * 1024 * 1024,
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        return False, "", f"Download failed: {e}"

    for f in workdir.iterdir():
        if f.name.startswith("video.") and f.suffix.lower() in (".mp4", ".mkv", ".webm"):
            return True, str(f), ""
    return False, "", "Download completed but no video file found"


def _transcode_to_mp4(src: str, dst: str) -> tuple[bool, str]:
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(dst),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       check=True, timeout=300)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode(errors="ignore")[-500:]
    except Exception as e:
        return False, str(e)


def _fetch_captions(url: str, workdir: Path) -> tuple[bool, str, str]:
    try:
        import yt_dlp
    except ImportError:
        return False, "", "yt_dlp not installed"

    opts = {
        "writesubtitles": True, "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt", "skip_download": True,
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
        for ext in ("vtt", "srt"):
            for sub_file in workdir.glob(f"*.{ext}"):
                raw = sub_file.read_text(encoding="utf-8", errors="ignore")
                lines = [l.strip() for l in raw.split("\n")
                         if l.strip() and not re.match(r"^\d{2}:\d{2}", l.strip())
                         and l.strip() != "WEBVTT" and "-->" not in l
                         and not l.strip().startswith("Kind:") and not l.strip().startswith("Language:")]
                if lines:
                    return True, " ".join(lines)[:8000], "auto"
    except Exception as e:
        logger.warning("Caption fetch failed: %s", e)
    return False, "", ""


# ── Background job worker ─────────────────────────────────────────────

def _run_fetch_job(job_id: str, url: str, use_captions: bool):
    """Background worker — runs the entire URL ingest pipeline."""
    log_file = LOGS_DIR / f"job_{job_id}.log"

    try:
        # Set up per-job file logging
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        jlog = logging.getLogger(f"job.{job_id}")
        jlog.addHandler(fh)
        jlog.setLevel(logging.DEBUG)

        jobs.set_progress(job_id, 5, "Validating URL")
        jlog.info("Job started for URL: %s", url)

        uhash = _url_hash(url)
        workdir = FETCH_DIR / uhash
        workdir.mkdir(parents=True, exist_ok=True)

        # Check cache
        cache = _load_cache()
        if uhash in cache:
            cached = cache[uhash]
            stem = cached.get("analysis_stem", "")
            analysis_json = ANALYSIS_DIR / stem / "analysis.json"
            if stem and analysis_json.exists():
                jlog.info("Cache hit for %s", url)
                jobs.set_done(job_id, {
                    "analysis_stem": stem, "cached": True,
                    "source_url": url,
                    "transcript_source": cached.get("transcript_source", "unknown"),
                })
                return

        # Check dependencies
        jobs.set_progress(job_id, 8, "Checking dependencies")
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            jobs.set_failed(job_id, "yt_dlp not installed. Run: pip install yt-dlp")
            return
        if not shutil.which("ffmpeg"):
            jobs.set_failed(job_id, "ffmpeg not found in PATH")
            return

        transcript_text = ""
        transcript_source = "none"
        warnings_list = []

        # Captions
        if use_captions:
            jobs.set_progress(job_id, 12, "Fetching captions")
            cap_ok, cap_text, cap_src = _fetch_captions(url, workdir)
            if cap_ok and cap_text:
                transcript_text = cap_text
                transcript_source = f"captions ({cap_src})"
                jlog.info("Captions found: %d chars", len(cap_text))

        # Download
        jobs.set_progress(job_id, 20, "Downloading video")
        jlog.info("Downloading video...")
        dl_ok, dl_path, dl_err = _download_video(url, workdir, job_id)
        if not dl_ok:
            jobs.set_failed(job_id, f"Download failed: {dl_err}")
            jlog.error("Download failed: %s", dl_err)
            return
        jlog.info("Downloaded: %s", dl_path)

        # Transcode if needed
        jobs.set_progress(job_id, 35, "Transcoding video")
        mp4_path = dl_path
        if not dl_path.endswith(".mp4"):
            mp4_path = str(workdir / "video_transcoded.mp4")
            tc_ok, tc_err = _transcode_to_mp4(dl_path, mp4_path)
            if not tc_ok:
                jobs.set_failed(job_id, f"Transcode failed: {tc_err}")
                jlog.error("Transcode failed: %s", tc_err)
                return
            jlog.info("Transcoded to MP4")

        # Extract frames
        jobs.set_progress(job_id, 50, "Extracting frames")
        video_stem = f"url_{uhash}_{int(time.time())}"
        frames_dir = FRAMES_DIR / video_stem
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            frames_count = extract_frames(mp4_path, str(frames_dir), fps_sample=1)
        except Exception as e:
            warnings_list.append(f"Frame extraction warning: {e}")
            frames_count = 0

        if frames_count == 0:
            cmd = ["ffmpeg", "-y", "-i", mp4_path, "-vf", "fps=1", "-q:v", "2",
                   str(frames_dir / "frame_%04d.jpg")]
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=True, timeout=120)
                frames_count = len(list(frames_dir.glob("*.jpg")))
            except Exception:
                pass
        jlog.info("Frames extracted: %d", frames_count)

        # Object detection
        jobs.set_progress(job_id, 65, "Running object detection")
        analysis_out = ANALYSIS_DIR / video_stem
        analysis_out.mkdir(parents=True, exist_ok=True)

        try:
            detections = detect_frames(str(frames_dir), str(analysis_out), conf_threshold=0.20)
        except Exception as e:
            detections = {"results": [], "summary": {"frames": 0}}
            warnings_list.append(f"Detection warning: {e}")

        # Transcription (if no captions)
        jobs.set_progress(job_id, 75, "Transcribing audio")
        if not transcript_text:
            try:
                tr = transcribe_audio_whisper(mp4_path, str(analysis_out))
                transcript_text = tr.get("text", "")
                tr_model = tr.get("model", "")
                if transcript_text.strip() and tr_model and tr_model != "none":
                    transcript_source = f"whisper ({tr_model})"
                elif transcript_text.strip():
                    transcript_source = "whisper (auto)"
                else:
                    transcript_text = ""
                    transcript_source = "no speech detected"
            except Exception as e:
                warnings_list.append(f"Transcription warning: {e}")
                transcript_source = "transcription failed"

        # Save analysis.json
        analysis_data = {
            "video_stem": video_stem, "source_url": url, "source_type": "url_ingest",
            "transcript": {"text": transcript_text, "source": transcript_source},
            "detections_summary": detections.get("summary", {}),
            "frames_count": frames_count,
            "timestamp": datetime.now().isoformat(),
            "warnings": warnings_list,
        }
        (analysis_out / "analysis.json").write_text(
            json.dumps(analysis_data, indent=2), encoding="utf-8")
        if detections.get("results"):
            (analysis_out / "detections.json").write_text(
                json.dumps(detections, indent=2), encoding="utf-8")

        # Generate notes
        jobs.set_progress(job_id, 85, "Generating AI notes")
        llm_provider_name = "unknown"
        is_fallback = False
        try:
            notes = generate_notes_from_analysis(str(analysis_out / "analysis.json"))
            (analysis_out / "notes.json").write_text(
                json.dumps(notes, indent=2), encoding="utf-8")
            llm_provider_name = notes.get("_llm_meta", {}).get("provider", "unknown")
            is_fallback = notes.get("_fallback", False)
        except Exception as e:
            jlog.warning("Notes generation failed: %s", e)
            warnings_list.append(f"Notes warning: {e}")
            is_fallback = True

        # Update cache
        cache[uhash] = {
            "url": url, "analysis_stem": video_stem,
            "transcript_source": transcript_source,
            "timestamp": datetime.now().isoformat(),
        }
        _save_cache(cache)

        # Done!
        jobs.set_done(job_id, {
            "analysis_stem": video_stem, "source_url": url,
            "transcript_source": transcript_source, "frames_count": frames_count,
            "warnings": warnings_list, "cached": False,
            "llm_provider": llm_provider_name, "is_fallback": is_fallback,
        })
        jlog.info("Job completed: stem=%s", video_stem)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("[%s] Unhandled error: %s\n%s", job_id, e, tb)
        try:
            log_file.write_text(tb, encoding="utf-8")
        except Exception:
            pass
        jobs.set_failed(job_id, str(e))


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/fetch_and_analyze")
async def fetch_and_analyze(request: Request, body: dict = Body(...)):
    """Start an async ingestion job. Returns job_id immediately."""
    url = (body.get("url") or "").strip()
    consent = body.get("consent", False)
    use_captions = body.get("use_captions_if_available", True)

    consent_header = request.headers.get("X-User-Consent", "").lower()
    if not consent and consent_header != "true":
        return JSONResponse(
            {"ok": False, "error": "User consent required."},
            status_code=403)

    valid, reason = _validate_url(url)
    if not valid:
        return JSONResponse({"ok": False, "error": reason}, status_code=400)

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return JSONResponse(
            {"ok": False, "error": "Rate limit exceeded. Try again in a minute."},
            status_code=429)

    # Create job in shared store
    job_id = jobs.create_job("fetch", {"url": url})

    # Start background thread
    thread = threading.Thread(
        target=_run_fetch_job,
        args=(job_id, url, use_captions),
        daemon=True,
    )
    thread.start()

    return JSONResponse({
        "ok": True,
        "job_id": job_id,
        "status_url": f"/jobs/{job_id}",
    })
