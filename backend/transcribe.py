# backend/transcribe.py
"""
Audio transcription using the OpenAI Whisper API (cloud).
Falls back to a stub if no API key is set, so the server can still start.

Set OPENAI_API_KEY in your environment for real transcription.
"""

import os
import json
import time
from pathlib import Path

from openai import OpenAI


def transcribe_audio_whisper(audio_path: str, out_dir: str) -> dict:
    """
    Transcribe *audio_path* using the OpenAI Whisper API and save
    transcript.json to *out_dir*.

    Returns
    -------
    dict
        Keys: text, segments, model, time_seconds
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # No API key — return empty transcript so server doesn't crash
        out = {
            "text": "(No OPENAI_API_KEY set — transcription skipped)",
            "segments": [],
            "model": "none",
            "time_seconds": 0,
        }
        _save(out, out_dir)
        return out

    client = OpenAI(api_key=api_key)

    t0 = time.time()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )
    duration = time.time() - t0

    # response is a Transcription object; convert to dict
    resp_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)

    out = {
        "text": resp_dict.get("text", "").strip(),
        "segments": resp_dict.get("segments", []),
        "model": "whisper-1",
        "time_seconds": round(duration, 3),
    }

    _save(out, out_dir)
    return out


def _save(out: dict, out_dir: str):
    transcript_path = os.path.join(out_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
