# backend/generate_notes.py
"""
Generate structured study notes from video analysis using the LLM provider.

Uses llm_provider for Gemini/OpenAI/Fallback abstraction.
"""

import json
from pathlib import Path
from typing import Dict


def generate_notes_from_analysis(analysis_path: str) -> Dict:
    """
    Read *analysis_path* (analysis.json), call the LLM provider, and return
    structured notes as a Python dict.

    Provider selection (via env vars):
      - GEMINI_API_KEY  → Google Gemini
      - OPENAI_API_KEY  → OpenAI
      - Neither         → Context-aware fallback placeholder
    """
    from llm_provider import provider

    p = Path(analysis_path)
    analysis = json.loads(p.read_text(encoding="utf-8"))

    # Enrich with detection results if present
    det_path = p.parent / "detections.json"
    if det_path.exists():
        det_data = json.loads(det_path.read_text(encoding="utf-8"))
        analysis["detections_results"] = det_data.get("results", [])

    notes = provider.generate_notes(analysis)
    return notes
