# backend/llm_helpers.py
"""
LLM call wrapper with retry logic and robust JSON extraction.
Supports OpenAI Chat API (>= v1.0).
Set OPENAI_API_KEY environment variable before use.
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple

from openai import OpenAI

# Initialise client — reads OPENAI_API_KEY from env automatically
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def call_llm(
    messages: List[dict],
    max_tokens: int = 1200,
    temperature: float = 0.0,
    retries: int = 2,
) -> Tuple[str, dict]:
    """
    Send *messages* to the LLM and return (text_content, raw_response_dict).
    Retries on transient errors with exponential back-off.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
            )
            content = resp.choices[0].message.content or ""
            return content, resp.model_dump()
        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt * 2)
    raise last_err  # type: ignore[misc]


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from *text*.
    Handles cases where the LLM wraps JSON in markdown code fences.
    """
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_nl = cleaned.index("\n")
        cleaned = cleaned[first_nl + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -3]

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(cleaned[start : end + 1])
