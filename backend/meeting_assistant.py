"""
meeting_assistant.py — AI Meeting Assistant
Real-time meeting analysis: participant tracking, live transcription,
action item extraction, and meeting summarization via LLM.
Supports speaker diarization (count by face detection) and keyword flagging.
"""
import os
import cv2
import base64
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger("meeting_assistant")

# Keywords that flag action items or important moments
ACTION_KEYWORDS = [
    "will", "should", "must", "need to", "action item", "todo", "follow up",
    "by monday", "by friday", "deadline", "assign", "responsible", "owner",
    "next step", "let's", "we should", "i'll", "you'll", "they should",
]

DECISION_KEYWORDS = [
    "decided", "agreed", "confirmed", "approved", "rejected", "concluded",
    "we will", "we won't", "final decision", "resolution", "verdict",
]


class MeetingAssistant:
    """AI-powered meeting assistant with real-time analysis."""

    def __init__(self):
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")
        self._yolo = None
        self._sessions: Dict[str, Dict] = {}

    def _load_yolo(self):
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
            self._yolo = YOLO(model_path)
        except Exception as e:
            logger.warning("Meeting YOLO load failed: %s", e)

    def get_or_create_session(self, session_id: str) -> Dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "id": session_id,
                "started_at": time.time(),
                "transcript_segments": [],
                "action_items": [],
                "decisions": [],
                "participants_seen": 0,
                "max_participants": 0,
                "frame_count": 0,
                "last_summary": None,
                "last_summary_ts": 0,
            }
        return self._sessions[session_id]

    def process_frame(self, session_id: str, frame_b64: str) -> Dict[str, Any]:
        """Analyze video frame for participant count."""
        session = self.get_or_create_session(session_id)
        session["frame_count"] += 1
        t0 = time.time()

        person_count = 0
        try:
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._load_yolo()
                if self._yolo:
                    results = self._yolo(frame, verbose=False, conf=0.45, classes=[0])
                    for r in results:
                        person_count = len(r.boxes)
                session["participants_seen"] = person_count
                if person_count > session["max_participants"]:
                    session["max_participants"] = person_count
        except Exception as e:
            logger.warning("Meeting frame analysis error: %s", e)

        return {
            "ok": True,
            "session_id": session_id,
            "participants_visible": person_count,
            "max_participants_seen": session["max_participants"],
            "frame_count": session["frame_count"],
            "latency_ms": round((time.time() - t0) * 1000),
        }

    def add_transcript_segment(self, session_id: str, text: str, speaker: str = "Speaker") -> Dict:
        """Add a transcript segment and extract action items/decisions."""
        session = self.get_or_create_session(session_id)
        ts = time.time()

        segment = {
            "ts": ts,
            "speaker": speaker,
            "text": text,
            "is_action": False,
            "is_decision": False,
        }

        text_lower = text.lower()
        action_flags = [kw for kw in ACTION_KEYWORDS if kw in text_lower]
        decision_flags = [kw for kw in DECISION_KEYWORDS if kw in text_lower]

        if action_flags:
            segment["is_action"] = True
            session["action_items"].append({
                "ts": ts,
                "text": text,
                "speaker": speaker,
                "keywords": action_flags,
            })
            session["action_items"] = session["action_items"][-50:]

        if decision_flags:
            segment["is_decision"] = True
            session["decisions"].append({
                "ts": ts,
                "text": text,
                "speaker": speaker,
                "keywords": decision_flags,
            })
            session["decisions"] = session["decisions"][-50:]

        session["transcript_segments"].append(segment)
        session["transcript_segments"] = session["transcript_segments"][-500:]

        return {
            "ok": True,
            "is_action": segment["is_action"],
            "is_decision": segment["is_decision"],
            "action_items_count": len(session["action_items"]),
            "decisions_count": len(session["decisions"]),
        }

    def get_transcript(self, session_id: str, last_n: int = 30) -> List[Dict]:
        session = self._sessions.get(session_id, {})
        return session.get("transcript_segments", [])[-last_n:]

    def summarize(self, session_id: str) -> Dict[str, Any]:
        """Generate meeting summary using LLM."""
        session = self._sessions.get(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}

        # Build transcript text
        segments = session.get("transcript_segments", [])
        transcript_text = "\n".join(
            f"{s['speaker']}: {s['text']}" for s in segments[-100:]
        )

        action_items = session.get("action_items", [])
        decisions = session.get("decisions", [])

        # Try LLM summarization
        summary_text = None
        summary_provider = "extractive"

        if self._gemini_key and transcript_text.strip():
            try:
                summary_text, summary_provider = self._gemini_summarize(
                    transcript_text, action_items, decisions
                )
            except Exception as e:
                logger.warning("Gemini meeting summary failed: %s", e)

        if not summary_text:
            # Extractive fallback
            summary_text = self._extractive_summary(segments, action_items, decisions)

        session["last_summary"] = summary_text
        session["last_summary_ts"] = time.time()

        duration_min = round((time.time() - session["started_at"]) / 60, 1)

        return {
            "ok": True,
            "session_id": session_id,
            "summary": summary_text,
            "provider": summary_provider,
            "action_items": action_items,
            "decisions": decisions,
            "participants_max": session["max_participants"],
            "duration_minutes": duration_min,
            "transcript_segments": len(segments),
        }

    def _gemini_summarize(self, transcript: str, actions: List, decisions: List) -> tuple:
        import requests
        prompt = (
            "You are a professional meeting secretary AI. "
            "Analyze this meeting transcript and provide:\n"
            "1. Brief summary (3-4 sentences)\n"
            "2. Key decisions made\n"
            "3. Action items with owners (if mentioned)\n"
            "4. Next steps\n\n"
            f"TRANSCRIPT:\n{transcript[:3000]}\n\n"
            f"Detected action items: {len(actions)}\n"
            f"Detected decisions: {len(decisions)}\n\n"
            "Respond in plain text, concise and professional."
        )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={self._gemini_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500},
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text, "gemini-2.0-flash"

    def _extractive_summary(self, segments: List, actions: List, decisions: List) -> str:
        n = len(segments)
        a = len(actions)
        d = len(decisions)
        lines = [f"Meeting had {n} transcript segments."]
        if a:
            lines.append(f"Action items detected: {a}")
            for item in actions[:3]:
                lines.append(f"  • {item['text'][:80]}")
        if d:
            lines.append(f"Decisions made: {d}")
            for dec in decisions[:3]:
                lines.append(f"  • {dec['text'][:80]}")
        if not a and not d:
            lines.append("No action items or decisions flagged automatically.")
        return "\n".join(lines)

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        return {
            "id": session_id,
            "started_at": session["started_at"],
            "duration_s": round(time.time() - session["started_at"]),
            "participants_max": session["max_participants"],
            "participants_visible": session["participants_seen"],
            "transcript_count": len(session["transcript_segments"]),
            "action_items_count": len(session["action_items"]),
            "decisions_count": len(session["decisions"]),
            "last_summary": session.get("last_summary"),
        }

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def clear_session(self, session_id: str):
        self._sessions.pop(session_id, None)


# Singleton
meeting_assistant = MeetingAssistant()
