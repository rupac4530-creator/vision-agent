# backend/generate_notes.py
"""
Prompt-engineer an LLM to convert multimodal video analysis
(transcript + per-frame object labels) into structured study notes.

Output schema: summary, key_concepts, formulas (LaTeX), viva_questions
(easy / medium / hard), timestamped highlights, and provenance.
"""

import json
from pathlib import Path
from typing import Dict

from llm_helpers import call_llm, safe_parse_json

# ── System prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a precise academic assistant. You receive multimodal video analysis
(transcript text + per-frame object-detection labels) and produce a compact,
structured study package.

RULES:
1. Output **valid JSON only** — no commentary outside the JSON object.
2. Follow the exact schema shown in the user message.
3. Be concise and factual.
4. Include timestamps (seconds) when possible.
5. Use LaTeX notation for formulas (e.g. E = mc^2).
6. Viva questions must cover easy, medium, and hard difficulty.
"""

# Desired output schema (included literally in prompt so LLM knows types)
SCHEMA_EXAMPLE = {
    "summary": "1-3 sentence summary of the video content.",
    "key_concepts": ["concept 1", "concept 2", "..."],
    "formulas": [
        {
            "latex": "E = mc^2",
            "explanation": "mass-energy equivalence",
            "timestamp": 12.3,
        }
    ],
    "viva_questions": {
        "easy": ["question 1", "question 2"],
        "medium": ["question 3", "question 4"],
        "hard": ["question 5"],
    },
    "highlights": [
        {
            "timestamp": 4.2,
            "text": "Important moment description",
            "frame": "frame_0003.jpg",
        }
    ],
    "provenance": {
        "transcript_excerpt": "short excerpt supporting a key point",
        "detection_examples": [
            {"frame": "frame_0003.jpg", "labels": ["person", "whiteboard"]}
        ],
    },
}


def _build_messages(analysis: Dict) -> list:
    """Build the chat messages list from an analysis dict."""
    transcript_text = analysis.get("transcript", {}).get("text", "")
    # Keep a reasonable snippet to stay within token budget
    transcript_snip = transcript_text[:2000]

    # Compile frame-level label summary (top 10 frames)
    frames_info_lines: list[str] = []
    detections_results = analysis.get("detections_results") or []
    for item in detections_results[:10]:
        labels = ", ".join(d["label"] for d in item.get("detections", [])[:6])
        frames_info_lines.append(f"{item.get('frame', '?')}: {labels}")

    user_content = (
        "Transcript (snippet):\n"
        f'"""{transcript_snip}"""\n\n'
        "Top frames (frame: labels):\n"
        f'"""\n{chr(10).join(frames_info_lines)}\n"""\n\n'
        "OUTPUT SCHEMA (return valid JSON matching this schema exactly):\n"
        f"{json.dumps(SCHEMA_EXAMPLE, indent=2)}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_notes_from_analysis(analysis_path: str) -> Dict:
    """
    Read *analysis_path* (analysis.json), call the LLM, and return
    structured notes as a Python dict.

    If OPENAI_API_KEY is not set, returns the sample notes.json as fallback.
    """
    import os

    p = Path(analysis_path)
    analysis = json.loads(p.read_text(encoding="utf-8"))

    # ── Graceful fallback when no API key ──────────────────────────
    if not os.getenv("OPENAI_API_KEY"):
        sample_path = Path(__file__).resolve().parent / "analysis" / "sample" / "notes.json"
        if sample_path.exists():
            notes = json.loads(sample_path.read_text(encoding="utf-8"))
            notes["_fallback"] = True
            notes["_fallback_reason"] = "OPENAI_API_KEY not set — returning sample notes"
            return notes
        return {
            "summary": "[Notes unavailable: set OPENAI_API_KEY to generate real notes]",
            "key_concepts": [],
            "formulas": [],
            "viva_questions": {"easy": [], "medium": [], "hard": []},
            "highlights": [],
            "_fallback": True,
            "_fallback_reason": "OPENAI_API_KEY not set and no sample notes found",
        }

    # Embed detection results if detections.json exists alongside analysis.json
    det_path = p.parent / "detections.json"
    if det_path.exists():
        det_data = json.loads(det_path.read_text(encoding="utf-8"))
        analysis["detections_results"] = det_data.get("results", [])

    messages = _build_messages(analysis)
    text, raw_resp = call_llm(messages, max_tokens=1200, temperature=0.0)

    notes = safe_parse_json(text)
    # Attach provenance metadata about the LLM call
    notes["_llm_meta"] = {
        "model": raw_resp.get("model", "unknown"),
        "raw_preview": text[:300],
    }
    return notes

