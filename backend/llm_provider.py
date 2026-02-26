# backend/llm_provider.py
"""
LLM Provider Abstraction Layer.

Supports:
  - Google Gemini  (GEMINI_API_KEY)  — preferred / default
  - OpenAI         (OPENAI_API_KEY)  — alternative
  - Auto-summary   (no key)          — context-aware placeholder

Usage:
    from llm_provider import provider
    result = provider.chat(messages, max_tokens=1200)
    notes  = provider.generate_notes(analysis_dict)
"""

import json
import os
import time
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional

# ── Load .env file ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

logger = logging.getLogger("llm_provider")

# ── Configuration from env ────────────────────────────────────────────
LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER", "auto").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_MODEL = os.getenv("CLOUDFLARE_MODEL", "@cf/qwen/qwen1.5-14b-chat-awq")


# ── Notes schema ──────────────────────────────────────────────────────
NOTES_SCHEMA = {
    "summary": "1-3 sentence summary of the video content.",
    "key_concepts": ["concept 1", "concept 2"],
    "formulas": [{"latex": "E = mc^2", "explanation": "mass-energy equivalence", "timestamp": 12.3}],
    "viva_questions": {
        "easy": ["question 1", "question 2"],
        "medium": ["question 3"],
        "hard": ["question 4"],
    },
    "highlights": [{"timestamp": 4.2, "text": "Important moment", "frame": "frame_0003.jpg"}],
    "provenance": {
        "transcript_excerpt": "short excerpt supporting a key point",
        "detection_examples": [{"frame": "frame_0003.jpg", "labels": ["person", "whiteboard"]}],
    },
}

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


def _build_notes_prompt(analysis: Dict) -> str:
    """Build the user prompt for notes generation from analysis dict."""
    transcript_text = analysis.get("transcript", {}).get("text", "")
    transcript_snip = transcript_text[:2000]

    frames_lines: list[str] = []
    for item in (analysis.get("detections_results") or [])[:10]:
        labels = ", ".join(d["label"] for d in item.get("detections", [])[:6])
        frames_lines.append(f"{item.get('frame', '?')}: {labels}")

    return (
        "Transcript (snippet):\n"
        f'"""{transcript_snip}"""\n\n'
        "Top frames (frame: labels):\n"
        f'"""\n{chr(10).join(frames_lines)}\n"""\n\n'
        "OUTPUT SCHEMA (return valid JSON matching this schema exactly):\n"
        f"{json.dumps(NOTES_SCHEMA, indent=2)}"
    )


def _safe_parse_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from LLM text, handling code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.index("\n")
        cleaned = cleaned[first_nl + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(cleaned[start: end + 1])


# ── Custom error for LLM quota/auth issues ────────────────────────────
class LLMQuotaError(Exception):
    """Raised when the LLM provider returns a quota or auth error."""
    pass


class LLMTimeoutError(Exception):
    """Raised when the LLM provider times out."""
    pass


# ═══════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════
class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def chat(self, messages: List[dict], max_tokens: int = 1200,
             temperature: float = 0.0) -> Tuple[str, dict]:
        """Send chat messages, return (text_content, raw_meta_dict)."""
        ...

    def generate_notes(self, analysis: Dict) -> Dict:
        """Generate structured study notes from an analysis dict."""
        prompt = _build_notes_prompt(analysis)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text, meta = self.chat(messages, max_tokens=1200, temperature=0.0)
        notes = _safe_parse_json(text)
        notes["_llm_meta"] = {"provider": self.name, "model": meta.get("model", "unknown")}
        return notes


# ═══════════════════════════════════════════════════════════════════════
# OpenAI Provider
# ═══════════════════════════════════════════════════════════════════════
class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("OpenAI provider initialised (model=%s)", self._model)

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        last_err = None
        for attempt in range(2):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    timeout=45,
                )
                content = resp.choices[0].message.content or ""
                return content, {"model": resp.model, "provider": "openai"}
            except Exception as e:
                err_str = str(e).lower()
                # Don't retry on quota/auth — fail fast
                if any(k in err_str for k in ("quota", "rate_limit", "429", "401", "403", "insufficient")):
                    raise LLMQuotaError(f"OpenAI quota/auth error: {e}")
                last_err = e
                time.sleep(2)
        raise last_err


# ═══════════════════════════════════════════════════════════════════════
# Google Gemini Provider  (REST API only — reliable)
# ═══════════════════════════════════════════════════════════════════════
class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        logger.info("Gemini provider initialised (model=%s, REST API)", self._model_name)

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        """Call Gemini via REST API with strict timeouts."""
        import requests

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model_name}:generateContent?key={self._api_key}"
        )

        prompt_text = "\n\n".join(msg["content"] for msg in messages)
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        last_err = None
        for attempt in range(2):  # max 2 attempts (1 retry)
            try:
                resp = requests.post(url, json=payload, timeout=(10, 45))

                # Handle HTTP errors specifically
                if resp.status_code == 429:
                    body = resp.json().get("error", {}).get("message", "Rate limited")
                    raise LLMQuotaError(f"Gemini rate limit (429): {body}")

                if resp.status_code == 403:
                    body = resp.json().get("error", {}).get("message", "Forbidden")
                    raise LLMQuotaError(f"Gemini quota/auth (403): {body}")

                if resp.status_code == 400:
                    body = resp.json().get("error", {}).get("message", "Bad request")
                    # Check if it's a quota error disguised as 400
                    if "quota" in body.lower() or "limit" in body.lower() or "exhausted" in body.lower():
                        raise LLMQuotaError(f"Gemini quota exhausted: {body}")
                    raise ValueError(f"Gemini bad request: {body}")

                resp.raise_for_status()
                data = resp.json()

                candidates = data.get("candidates", [])
                if not candidates:
                    block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                    raise ValueError(f"Gemini returned no candidates (blockReason={block_reason})")

                text = candidates[0]["content"]["parts"][0]["text"]
                return text, {"model": self._model_name, "provider": "gemini"}

            except (LLMQuotaError, LLMTimeoutError):
                raise  # Don't retry quota/timeout errors — fail fast

            except requests.exceptions.Timeout:
                raise LLMTimeoutError(f"Gemini API timed out after 45s (attempt {attempt + 1})")

            except requests.exceptions.HTTPError as e:
                last_err = e
                try:
                    err_body = e.response.json().get("error", {}).get("message", str(e))
                except Exception:
                    err_body = str(e)
                # Check for quota-related errors in any HTTP error
                if any(k in err_body.lower() for k in ("quota", "rate", "limit", "exhausted", "free_tier")):
                    raise LLMQuotaError(f"Gemini quota error: {err_body}")
                logger.warning("Gemini HTTP error (attempt %d): %s", attempt + 1, err_body)
                time.sleep(3)

            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if any(k in err_str for k in ("quota", "rate", "limit", "free_tier")):
                    raise LLMQuotaError(f"Gemini quota error: {e}")
                logger.warning("Gemini error (attempt %d): %s", attempt + 1, e)
                time.sleep(3)

        raise last_err


# ═══════════════════════════════════════════════════════════════════════
# Cloudflare Workers AI Provider (Open / low-cost)
# ═══════════════════════════════════════════════════════════════════════
class CloudflareProvider(LLMProvider):
    name = "cloudflare"

    def __init__(self, account_id: str, api_token: str, model: str):
        self._account_id = account_id
        self._api_token = api_token
        self._model = model or "@cf/qwen/qwen1.5-14b-chat-awq"
        logger.info("Cloudflare provider initialised (model=%s)", self._model)

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        import requests

        if not self._account_id or not self._api_token:
            raise LLMQuotaError("Cloudflare credentials missing (CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN)")

        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/ai/run/{self._model}"
        )
        headers = {"Authorization": f"Bearer {self._api_token}"}
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(10, 45))
        except requests.exceptions.Timeout:
            raise LLMTimeoutError("Cloudflare Workers AI timed out after 45s")

        if resp.status_code in (401, 403):
            raise LLMQuotaError(f"Cloudflare auth/quota error ({resp.status_code})")
        if resp.status_code == 429:
            raise LLMQuotaError("Cloudflare rate limit (429)")

        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Cloudflare Workers AI HTTP error: {e}")

        data = resp.json()
        # Workers AI responses vary by model; try common shapes.
        text = ""
        if isinstance(data, dict):
            # Most common: {"result": {"response": "..."}}
            if isinstance(data.get("result"), dict) and "response" in data["result"]:
                text = data["result"]["response"] or ""
            # Some models: {"result": {"output_text": "..."}}
            elif isinstance(data.get("result"), dict) and "output_text" in data["result"]:
                text = data["result"]["output_text"] or ""
            # Fallback: {"response": "..."}
            elif "response" in data:
                text = data.get("response") or ""

        return text, {"model": self._model, "provider": "cloudflare"}


# ═══════════════════════════════════════════════════════════════════════
# Fallback Provider (no API key — intelligent local summary mode)
# ═══════════════════════════════════════════════════════════════════════
class FallbackProvider(LLMProvider):
    name = "local-summary"

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        # Extract the user content for a basic response
        user_text = ""
        for m in messages:
            if m.get("role") == "user":
                user_text = m["content"]
        return json.dumps({"note": "Generated by local-summary engine (no external LLM).", "input_length": len(user_text)}), {"model": "local-summary", "provider": "local-summary"}

    def generate_notes(self, analysis: Dict) -> Dict:
        """Generate intelligent notes using NLP extraction from transcript and detections."""
        transcript_text = analysis.get("transcript", {}).get("text", "")
        transcript_source = analysis.get("transcript", {}).get("source", "unknown")

        # ── Smart sentence extraction ──────────────────────────────────
        import re
        # Split into sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+', transcript_text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]

        # ── Summary: first 3 meaningful sentences ─────────────────────
        summary_candidates = [s for s in sentences if len(s) > 30]
        if summary_candidates:
            summary = " ".join(summary_candidates[:3])[:600]
        elif transcript_text.strip():
            summary = transcript_text[:500]
        else:
            summary = "Video content analyzed. No speech transcript available — notes generated from visual detection data only."

        # ── Key concepts: extract unique noun phrases ─────────────────
        key_concepts = []
        seen_lower = set()
        # Use detection labels as concepts
        det_results = analysis.get("detections_results", [])
        det_summary = analysis.get("detections_summary", {})
        label_counts = det_summary.get("label_counts", {})

        if label_counts:
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:6]:
                concept = f"{label.title()} detected ({count} instances)"
                key_concepts.append(concept)
                seen_lower.add(label.lower())

        # Also extract from transcript sentences
        for s in sentences[:12]:
            short = s.rstrip(".!?").strip()
            if 20 < len(short) < 120 and short.lower() not in seen_lower:
                key_concepts.append(short)
                seen_lower.add(short.lower())
            if len(key_concepts) >= 8:
                break

        # ── Viva questions ────────────────────────────────────────────
        easy_q = ["What are the main objects or subjects visible in this video?"]
        medium_q = ["How do the visual elements relate to the spoken content?"]
        hard_q = ["What conclusions can be drawn from the combination of visual and audio analysis?"]

        if key_concepts:
            easy_q.append(f"Describe the role of '{key_concepts[0].split('(')[0].strip()}' in this video.")
        if sentences:
            medium_q.append(f"Explain the significance of: '{sentences[0][:80]}...'")
        if len(label_counts) > 2:
            labels_str = ", ".join(list(label_counts.keys())[:4])
            hard_q.append(f"Analyze the relationship between these detected elements: {labels_str}")

        # ── Highlights from detection data ────────────────────────────
        highlights = []
        for item in det_results[:5]:
            frame = item.get("frame", "")
            dets = item.get("detections", [])
            if dets:
                labels = ", ".join(set(d["label"] for d in dets[:4]))
                highlights.append({
                    "timestamp": 0,
                    "text": f"Detected: {labels}",
                    "frame": frame,
                })

        # ── Provenance ────────────────────────────────────────────────
        provenance = {"transcript_excerpt": "", "detection_examples": []}
        if sentences:
            provenance["transcript_excerpt"] = sentences[0][:200]
        for item in det_results[:3]:
            frame = item.get("frame", "")
            labels = [d["label"] for d in item.get("detections", [])[:5]]
            if labels:
                provenance["detection_examples"].append({"frame": frame, "labels": labels})

        return {
            "summary": summary,
            "key_concepts": key_concepts if key_concepts else ["Visual content analysis", "Object detection results"],
            "formulas": [],
            "viva_questions": {
                "easy": easy_q,
                "medium": medium_q,
                "hard": hard_q,
            },
            "highlights": highlights,
            "provenance": provenance,
            "_fallback": True,
            "_fallback_reason": "Generated by local-summary engine (no external LLM quota available).",
            "_llm_meta": {"provider": "local-summary", "model": "extractive-v1"},
        }


# ═══════════════════════════════════════════════════════════════════════
# Cascade Provider — tries providers in order, auto-fallback on quota/error
# ═══════════════════════════════════════════════════════════════════════
class CascadeProvider(LLMProvider):
    """Wraps multiple providers and tries them in order.
    If the primary fails with quota/timeout/error, silently falls to next."""

    def __init__(self, providers: list[LLMProvider]):
        self._providers = providers
        self.name = providers[0].name if providers else "none"

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        last_err = None
        for p in self._providers:
            try:
                result = p.chat(messages, max_tokens, temperature)
                self.name = p.name  # update name to reflect which actually worked
                return result
            except (LLMQuotaError, LLMTimeoutError) as e:
                logger.warning("Cascade: %s failed (%s), trying next provider", p.name, e)
                last_err = e
                continue
            except Exception as e:
                logger.warning("Cascade: %s error (%s), trying next provider", p.name, e)
                last_err = e
                continue
        # All failed — use fallback chat
        self.name = "local-summary"
        return FallbackProvider().chat(messages, max_tokens, temperature)

    def generate_notes(self, analysis: Dict) -> Dict:
        last_err = None
        for p in self._providers:
            try:
                notes = p.generate_notes(analysis)
                self.name = p.name
                return notes
            except (LLMQuotaError, LLMTimeoutError) as e:
                logger.warning("Cascade notes: %s failed (%s), trying next", p.name, e)
                last_err = e
                continue
            except Exception as e:
                logger.warning("Cascade notes: %s error (%s), trying next", p.name, e)
                last_err = e
                continue
        # All failed — use intelligent fallback
        self.name = "local-summary"
        fb = FallbackProvider()
        notes = fb.generate_notes(analysis)
        if last_err:
            notes["_fallback_reason"] = f"All LLM providers exhausted ({last_err}). Using local-summary."
        return notes


# ═══════════════════════════════════════════════════════════════════════
# Provider factory — builds cascade chain
# ═══════════════════════════════════════════════════════════════════════
def _create_provider() -> LLMProvider:
    """Create a cascade provider chain based on environment config."""
    chain: list[LLMProvider] = []

    if LLM_PROVIDER_NAME == "cloudflare" and CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN:
        chain.append(CloudflareProvider(
            account_id=CLOUDFLARE_ACCOUNT_ID,
            api_token=CLOUDFLARE_API_TOKEN,
            model=CLOUDFLARE_MODEL,
        ))
    elif LLM_PROVIDER_NAME == "gemini" and GEMINI_API_KEY:
        chain.append(GeminiProvider(api_key=GEMINI_API_KEY))
    elif LLM_PROVIDER_NAME == "openai" and OPENAI_API_KEY:
        chain.append(OpenAIProvider(api_key=OPENAI_API_KEY))
    elif LLM_PROVIDER_NAME == "auto":
        # Auto mode: Cloudflare (if configured) → Gemini → OpenAI
        if CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN:
            chain.append(CloudflareProvider(
                account_id=CLOUDFLARE_ACCOUNT_ID,
                api_token=CLOUDFLARE_API_TOKEN,
                model=CLOUDFLARE_MODEL,
            ))
        if GEMINI_API_KEY:
            chain.append(GeminiProvider(api_key=GEMINI_API_KEY))
        if OPENAI_API_KEY:
            chain.append(OpenAIProvider(api_key=OPENAI_API_KEY))
    else:
        if CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN:
            chain.append(CloudflareProvider(
                account_id=CLOUDFLARE_ACCOUNT_ID,
                api_token=CLOUDFLARE_API_TOKEN,
                model=CLOUDFLARE_MODEL,
            ))
        if GEMINI_API_KEY:
            chain.append(GeminiProvider(api_key=GEMINI_API_KEY))
        if OPENAI_API_KEY:
            chain.append(OpenAIProvider(api_key=OPENAI_API_KEY))

    # Always add fallback at the end
    chain.append(FallbackProvider())

    if len(chain) == 1:
        logger.warning("No LLM API keys configured — using local-summary only")
        return chain[0]

    logger.info("LLM cascade chain: %s", " → ".join(p.name for p in chain))
    return CascadeProvider(chain)


# ── Singleton provider instance ───────────────────────────────────────
provider = _create_provider()
logger.info("LLM provider active: %s", provider.name)
