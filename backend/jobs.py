# backend/jobs.py
"""
Shared in-memory job store for background tasks.

Used by /generate_notes, /fetch_and_analyze, agent_core, and other async endpoints.
Provides create/update/get helpers and a unified GET /jobs/{id} endpoint.
Includes a watchdog thread that auto-fails stuck jobs after 120 seconds.
"""

import threading
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(tags=["jobs"])
logger = logging.getLogger("jobs")

BASE = Path(__file__).resolve().parent
LOGS_DIR = BASE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── In-memory store ───────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_lock = threading.Lock()

JOB_TIMEOUT_SECONDS = 120  # Auto-fail jobs stuck for >120s


def create_job(job_type: str, metadata: Optional[dict] = None) -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())[:12]
    job = {
        "job_id": job_id,
        "type": job_type,
        "status": "pending",
        "progress": 0,
        "step": "Queued",
        "created_at": datetime.now().isoformat(),
        "started_at": time.time(),
        "result": None,
        "error": None,
        "warnings": [],
    }
    if metadata:
        job.update(metadata)
    with _lock:
        _jobs[job_id] = job
    logger.info("[%s] Job created (type=%s)", job_id, job_type)
    return job_id


def update_job(job_id: str, **kwargs):
    """Thread-safe job update."""
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def set_progress(job_id: str, progress: int, step: str):
    """Update job progress and step description."""
    update_job(job_id, status="running", progress=min(progress, 99), step=step)
    logger.info("[%s] %d%% — %s", job_id, progress, step)


def set_done(job_id: str, result: Optional[dict] = None):
    """Mark job as completed."""
    update_job(job_id, status="done", progress=100, step="Complete", result=result or {})
    logger.info("[%s] Job completed", job_id)


def set_failed(job_id: str, error: str):
    """Mark job as failed with error message."""
    update_job(job_id, status="failed", progress=100, error=error, step="Failed")
    logger.error("[%s] Job failed: %s", job_id, error)


def get_job(job_id: str) -> Optional[dict]:
    """Get a copy of the job state."""
    with _lock:
        if job_id in _jobs:
            return _jobs[job_id].copy()
    return None


# ── Watchdog: auto-fail stuck jobs ────────────────────────────────────
def _watchdog():
    """Background thread that checks for stuck jobs every 15 seconds."""
    while True:
        time.sleep(15)
        now = time.time()
        with _lock:
            for jid, job in list(_jobs.items()):
                if job["status"] in ("done", "failed"):
                    continue
                started = job.get("started_at", now)
                if now - started > JOB_TIMEOUT_SECONDS:
                    job["status"] = "failed"
                    job["progress"] = 100
                    job["error"] = f"Job timed out after {JOB_TIMEOUT_SECONDS}s"
                    job["step"] = "Timed out"
                    logger.warning("[%s] Watchdog: auto-failed stuck job", jid)

_watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
_watchdog_thread.start()


# ── Endpoint ──────────────────────────────────────────────────────────
@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll job status. Returns current progress and result when done."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JSONResponse(job)
