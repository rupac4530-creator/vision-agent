# backend/agent_core.py
"""
Two-tier agent reasoning engine for the Vision Agent.

Tier A — FastReply (deterministic, <500ms):
  Template-based responses from YOLO labels + transcript snippet.
  Cached: same question → same fast reply.

Tier B — PolishReply (LLM, ~3-8s, async background job):
  Cloud LLM call with full context (labels + transcript + notes).
  Returns structured response with provenance links.
  Auto-fallback if LLM quota hit.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import jobs
from llm_provider import provider as llm_provider_instance
from llm_provider import LLMQuotaError, LLMTimeoutError

logger = logging.getLogger("agent_core")

ANALYSIS_ROOT = Path(__file__).resolve().parent / "analysis"


# ── FastReply Cache ───────────────────────────────────────────────────
_fast_cache: dict[str, dict] = {}  # key = f"{video_stem}:{question_hash}" → reply


def _question_key(video_stem: str, question: str) -> str:
    """Simple cache key for fast replies."""
    import hashlib
    q_hash = hashlib.md5(question.strip().lower().encode()).hexdigest()[:12]
    return f"{video_stem}:{q_hash}"


# ── Context Loader ────────────────────────────────────────────────────
def _load_context(video_stem: str) -> dict:
    """
    Load all available context for a video: detections, transcript, notes.
    Returns a dict with keys: detections, transcript_text, notes_summary, top_labels.
    """
    ctx = {
        "detections": [],
        "transcript_text": "",
        "notes_summary": "",
        "top_labels": [],
    }

    analysis_dir = ANALYSIS_ROOT / video_stem
    if not analysis_dir.exists():
        # Also check stream_uploads
        analysis_dir = Path(__file__).resolve().parent / "stream_uploads" / video_stem
        if not analysis_dir.exists():
            return ctx

    # Detections
    det_path = analysis_dir / "detections.json"
    if det_path.exists():
        try:
            with open(det_path) as f:
                det_data = json.load(f)
            results = det_data.get("results", [])
            ctx["detections"] = results

            # Aggregate top labels
            label_counts: dict[str, int] = {}
            for r in results:
                for d in r.get("detections", []):
                    lbl = d.get("label", "unknown")
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            ctx["top_labels"] = sorted(label_counts.items(), key=lambda x: -x[1])[:10]
        except Exception as e:
            logger.warning("Failed to load detections: %s", e)

    # Transcript
    tx_path = analysis_dir / "transcript.json"
    if tx_path.exists():
        try:
            with open(tx_path) as f:
                tx_data = json.load(f)
            ctx["transcript_text"] = tx_data.get("text", "")
        except Exception as e:
            logger.warning("Failed to load transcript: %s", e)

    # Notes
    notes_path = analysis_dir / "notes.json"
    if notes_path.exists():
        try:
            with open(notes_path) as f:
                notes_data = json.load(f)
            ctx["notes_summary"] = notes_data.get("summary", "")
        except Exception as e:
            logger.warning("Failed to load notes: %s", e)

    return ctx


# ── Tier A: FastReply (Upgraded — scene + activity intelligence) ──────
def fast_reply(video_stem: str, question: str) -> dict:
    """
    Generate an instant intelligent reply (<500ms) based on available context.
    Includes scene classification, activity inference, and dominant object analysis.

    Returns
    -------
    dict
        {reply, provenance: [{type, detail}], source: 'fast', cached: bool}
    """
    t0 = time.time()
    cache_key = _question_key(video_stem, question)

    # Check cache
    if cache_key in _fast_cache:
        cached = _fast_cache[cache_key]
        cached["cached"] = True
        cached["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return cached

    ctx = _load_context(video_stem)
    q_lower = question.strip().lower()

    provenance = []
    reply_parts = []

    # ── Scene & Activity Inference ─────────────────────────────────
    top_labels = ctx["top_labels"]
    label_set = {lbl.lower() for lbl, _ in top_labels}

    # Scene classification heuristic
    outdoor_cues = {"car", "truck", "bus", "bicycle", "motorcycle", "traffic light",
                    "stop sign", "bird", "tree", "dog", "horse", "sheep", "cow"}
    indoor_cues = {"chair", "couch", "bed", "dining table", "tv", "laptop", "keyboard",
                   "mouse", "remote", "microwave", "oven", "refrigerator", "sink", "toilet"}
    urban_cues = {"car", "truck", "bus", "traffic light", "stop sign", "person"}

    outdoor_score = len(label_set & outdoor_cues)
    indoor_score = len(label_set & indoor_cues)
    urban_score = len(label_set & urban_cues)

    if outdoor_score > indoor_score and outdoor_score > 0:
        scene = "outdoor"
    elif indoor_score > outdoor_score and indoor_score > 0:
        scene = "indoor"
    elif urban_score >= 2:
        scene = "urban/street"
    else:
        scene = "general"

    # Activity inference from object combinations
    activity = ""
    if "person" in label_set:
        if label_set & {"laptop", "keyboard", "mouse"}:
            activity = "working at a computer"
        elif label_set & {"sports ball", "tennis racket", "baseball bat", "skateboard", "surfboard"}:
            activity = "engaged in sports/physical activity"
        elif label_set & {"book", "backpack"}:
            activity = "studying or reading"
        elif label_set & {"cell phone"}:
            activity = "using a phone"
        elif label_set & {"cup", "bowl", "fork", "knife", "spoon", "dining table"}:
            activity = "dining"
        elif label_set & {"car", "truck", "bus", "bicycle", "motorcycle"}:
            activity = "in a transportation context"
        elif "person" in label_set and sum(c for l, c in top_labels if l == "person") > 2:
            activity = "in a group/social setting"
        else:
            activity = "present in the scene"

    # ── Label-based questions ─────────────────────────────────────
    if top_labels:
        label_str = ", ".join(f"{lbl} (×{cnt})" for lbl, cnt in top_labels[:6])
        total_dets = sum(cnt for _, cnt in top_labels)
        dominant = top_labels[0] if top_labels else None

        # "What do you see?" / "What's in the video?"
        if any(kw in q_lower for kw in ["see", "detect", "find", "what", "objects", "show", "identify", "describe"]):
            reply_parts.append(f"I detected {total_dets} objects across the video frames.")
            reply_parts.append(f"**Scene**: {scene.title()}")
            if dominant:
                reply_parts.append(f"**Dominant element**: {dominant[0]} ({dominant[1]} instances)")
            reply_parts.append(f"**All detections**: {label_str}.")
            if activity:
                reply_parts.append(f"**Activity**: The subject appears to be {activity}.")
            provenance.append({"type": "detection", "detail": f"{total_dets} total detections, scene={scene}"})

        # "How many?" / "count"
        elif any(kw in q_lower for kw in ["how many", "count", "number"]):
            reply_parts.append(f"Total: {total_dets} object detections across all frames.")
            for lbl, cnt in top_labels[:8]:
                reply_parts.append(f"  • {lbl}: {cnt}")
            provenance.append({"type": "detection", "detail": f"{total_dets} objects counted"})

        # Specific object query: "Is there a person?"
        for lbl, cnt in top_labels:
            if lbl.lower() in q_lower:
                reply_parts.append(f"Yes, I found **{lbl}** {cnt} time(s) in the video frames.")
                if cnt > 3:
                    reply_parts.append(f"It appears consistently, suggesting it's a central element.")
                provenance.append({"type": "detection", "detail": f"{lbl}: {cnt} occurrences"})
                break

    # ── Transcript-based questions ────────────────────────────────
    transcript = ctx["transcript_text"]
    if transcript and len(transcript) > 10:
        if any(kw in q_lower for kw in ["said", "say", "transcript", "speech", "audio", "hear", "talk", "speak"]):
            snippet = transcript[:400]
            reply_parts.append(f"From the audio: \"{snippet}{'…' if len(transcript) > 400 else ''}\"")
            provenance.append({"type": "transcript", "detail": f"{len(transcript)} characters"})

    # ── Notes-based questions ─────────────────────────────────────
    notes = ctx["notes_summary"]
    if notes and len(notes) > 10:
        if any(kw in q_lower for kw in ["summary", "notes", "main point", "about", "explain"]):
            reply_parts.append(f"Summary: {notes[:400]}{'…' if len(notes) > 400 else ''}")
            provenance.append({"type": "notes", "detail": "AI-generated summary"})

    # ── Default fallback (context-rich) ───────────────────────────
    if not reply_parts:
        if top_labels:
            label_str = ", ".join(f"{lbl} (×{cnt})" for lbl, cnt in top_labels[:6])
            reply_parts.append(f"🔍 **Scene Analysis**: {scene.title()} environment")
            reply_parts.append(f"**Detected objects**: {label_str}")
            if activity:
                reply_parts.append(f"**Inferred activity**: {activity}")
            if transcript and len(transcript) > 10:
                reply_parts.append(f"**Audio context**: \"{transcript[:150]}…\"")
            reply_parts.append("\n_For a deeper analysis, wait for the full AI response below._")
            provenance.append({"type": "detection", "detail": f"scene={scene}, activity={activity}"})
        else:
            reply_parts.append("I don't have enough context yet. Upload or stream a video first, then ask me about what's in it.")

    result = {
        "reply": "\n".join(reply_parts),
        "provenance": provenance,
        "source": "fast",
        "cached": False,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "scene": scene,
        "activity": activity,
    }

    # Cache it
    _fast_cache[cache_key] = result
    return result


# ── Tier B: PolishReply (LLM) ─────────────────────────────────────────
def polish_reply_worker(job_id: str, video_stem: str, question: str,
                        provider_name: str = "auto", model: str = ""):
    """
    Background worker for LLM-powered polished reply.
    Runs in a thread via jobs.py — updates job store with progress + result.
    """
    try:
        jobs.set_progress(job_id, 10, "Loading context...")
        ctx = _load_context(video_stem)

        jobs.set_progress(job_id, 20, "Building prompt...")

        # Build system prompt with context
        system_parts = [
            "You are a Vision Agent AI assistant analyzing video content.",
            "You have access to the following data from the video:",
        ]

        if ctx["top_labels"]:
            label_str = ", ".join(f"{lbl} (×{cnt})" for lbl, cnt in ctx["top_labels"][:10])
            system_parts.append(f"\nDetected objects: {label_str}")

        if ctx["transcript_text"]:
            system_parts.append(f"\nAudio transcript: {ctx['transcript_text'][:1500]}")

        if ctx["notes_summary"]:
            system_parts.append(f"\nSummary notes: {ctx['notes_summary'][:800]}")

        system_parts.append("\nProvide a detailed, helpful answer. Include specific references to what you observed.")
        system_parts.append("At the end, include a JSON provenance block: {\"sources\": [{\"type\": \"detection|transcript|notes\", \"detail\": \"...\"}]}")

        system_prompt = "\n".join(system_parts)
        user_prompt = f"User question: {question}"

        jobs.set_progress(job_id, 40, "Calling LLM...")

        # Use the singleton LLM provider
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        llm_response, _meta = llm_provider_instance.chat(messages, max_tokens=800, temperature=0.3)

        jobs.set_progress(job_id, 80, "Parsing response...")

        # Try to extract provenance JSON from response
        provenance = []
        reply_text = llm_response
        try:
            # Look for JSON block at end
            if "{" in llm_response and "sources" in llm_response:
                json_start = llm_response.rfind("{")
                json_end = llm_response.rfind("}") + 1
                if json_start > 0 and json_end > json_start:
                    prov_json = json.loads(llm_response[json_start:json_end])
                    provenance = prov_json.get("sources", [])
                    reply_text = llm_response[:json_start].strip()
        except (json.JSONDecodeError, Exception):
            pass

        if not provenance:
            # Build provenance from context
            if ctx["top_labels"]:
                provenance.append({"type": "detection", "detail": f"{sum(c for _, c in ctx['top_labels'])} total detections"})
            if ctx["transcript_text"]:
                provenance.append({"type": "transcript", "detail": f"{len(ctx['transcript_text'])} chars"})
            if ctx["notes_summary"]:
                provenance.append({"type": "notes", "detail": "AI-generated summary"})

        result = {
            "reply": reply_text,
            "provenance": provenance,
            "source": "llm",
            "provider": llm_provider_instance.name,
            "model": getattr(llm_provider_instance, '_model', 'unknown'),
        }

        jobs.set_done(job_id, result)
        logger.info("PolishReply done for job %s", job_id)

    except (LLMQuotaError, LLMTimeoutError) as e:
        logger.warning("LLM error for polish reply %s: %s", job_id, e)
        # Fallback: return enhanced fast reply
        fast = fast_reply(video_stem, question)
        result = {
            "reply": fast["reply"] + "\n\n⚠️ Full AI analysis unavailable (quota/timeout). This is a fast analysis based on detected objects and transcript.",
            "provenance": fast["provenance"],
            "source": "fallback",
            "error": str(e),
        }
        jobs.set_done(job_id, result)

    except Exception as e:
        logger.error("PolishReply failed for %s: %s", job_id, e, exc_info=True)
        jobs.set_failed(job_id, f"Agent error: {str(e)}")

    finally:
        # Safety net: if worker exits without setting terminal state, mark failed
        job = jobs.get_job(job_id)
        if job and job.get("status") not in ("done", "failed"):
            jobs.set_failed(job_id, "Worker exited unexpectedly")


# ── Convenience: Full 2-tier response ─────────────────────────────────
def ask_agent(video_stem: str, question: str,
              provider: str = "auto", model: str = "") -> dict:
    """
    Two-tier ask: returns fast reply immediately + starts background LLM job.

    Returns
    -------
    dict
        {fast_reply: {...}, job_id: str, status: 'processing'}
    """
    # Tier A: instant
    fast = fast_reply(video_stem, question)

    # Tier B: background LLM job
    job_id = jobs.create_job("agent_polish", {"video_stem": video_stem})

    import threading
    t = threading.Thread(
        target=polish_reply_worker,
        args=(job_id, video_stem, question, provider, model),
        daemon=True,
    )
    t.start()

    return {
        "fast_reply": fast,
        "job_id": job_id,
        "status": "processing",
    }
