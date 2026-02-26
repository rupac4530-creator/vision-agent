# backend/llm_provider.py
"""
LLM Provider Abstraction Layer — 8-Tier Cascade

Tier 0: Ollama / GLM-5:cloud    (OLLAMA_URL + OLLAMA_TOKEN) — highest priority (Z.ai)
Tier 1: Google Gemini           (GEMINI_API_KEY)        — preferred / default
Tier 2: OpenRouter              (OPENROUTER_API_KEY)    — DeepSeek-R1 deep reasoning
Tier 3: Groq API                (GROQ_API_KEY)          — ultra-fast inference
Tier 4: GitHub Models GPT       (GITHUB_TOKEN)          — powerful free tier
Tier 5: GitHub Models DeepSeek  (GITHUB_TOKEN)          — deep reasoning fallback
Tier 6: OpenAI                  (OPENAI_API_KEY)        — commercial fallback
Tier 7: Cloudflare Workers AI   (CF_ACCOUNT_ID + CF_TOKEN) — edge fallback
Tier 8: Auto-summary            (no key needed)         — always available

Usage:
    from llm_provider import provider
    result = provider.chat(messages, max_tokens=1200)
    notes  = provider.generate_notes(analysis_dict)
"""

import json
import os
import re
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
OLLAMA_URL        = os.getenv("OLLAMA_URL", "")
OLLAMA_TOKEN      = os.getenv("OLLAMA_TOKEN", "")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "glm-5:cloud")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")
CF_ACCOUNT_ID     = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CF_AUTH_TOKEN     = os.getenv("CLOUDFLARE_AUTH_TOKEN", "")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL  = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528")


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
    display_name: str = "Unknown"
    model_id: str = "unknown"

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
        notes["_llm_meta"] = {"provider": self.name, "model": meta.get("model", "unknown"),
                               "display": self.display_name}
        return notes

    def info(self) -> dict:
        return {"name": self.name, "display_name": self.display_name, "model": self.model_id}


# ═══════════════════════════════════════════════════════════════════════
# Tier 1: Google Gemini (REST API)
# ═══════════════════════════════════════════════════════════════════════
class GeminiProvider(LLMProvider):
    name = "gemini"
    display_name = "Google Gemini Flash"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self.model_id = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        logger.info("Gemini provider initialised (model=%s)", self.model_id)

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        import requests
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent?key={self._api_key}"
        )
        # Build proper system + user messages
        gemini_contents = []
        system_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            else:
                gemini_contents.append({"role": "user", "parts": [{"text": content}]})

        payload = {
            "contents": gemini_contents if gemini_contents else [{"parts": [{"text": "\n\n".join(m["content"] for m in messages)}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        last_err = None
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, timeout=(10, 45))
                if resp.status_code == 429:
                    raise LLMQuotaError(f"Gemini rate limit (429): {resp.text[:200]}")
                if resp.status_code in (401, 403):
                    raise LLMQuotaError(f"Gemini auth error ({resp.status_code})")
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                    raise ValueError(f"Gemini no candidates (blockReason={block_reason})")
                text = candidates[0]["content"]["parts"][0]["text"]
                return text, {"model": self.model_id, "provider": "gemini"}
            except (LLMQuotaError, LLMTimeoutError):
                raise
            except requests.exceptions.Timeout:
                raise LLMTimeoutError("Gemini API timed out")
            except Exception as e:
                err_str = str(e).lower()
                if any(k in err_str for k in ("quota", "rate", "limit", "exhausted")):
                    raise LLMQuotaError(f"Gemini quota: {e}")
                last_err = e
                time.sleep(3)
        raise last_err


# ═══════════════════════════════════════════════════════════════════════
# Tier 2+3: GitHub Models (OpenAI-compatible endpoint, free tier)
# ═══════════════════════════════════════════════════════════════════════
class GitHubModelsProvider(LLMProvider):
    """Uses GitHub Models marketplace (Azure OpenAI-compatible endpoint).
    Free for GitHub users with a personal access token.
    """

    def __init__(self, token: str, model: str = "gpt-4o-mini",
                 display: str = "GitHub Models GPT-4o-mini"):
        self._token = token
        self.model_id = model
        self.name = f"github-{model.replace('/', '-')}"
        self.display_name = display
        logger.info("GitHub Models provider: model=%s", model)

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        import urllib.request
        url = "https://models.inference.ai.azure.com/chat/completions"
        payload = json.dumps({
            "model": self.model_id,
            "messages": messages,
            "max_tokens": min(max_tokens, 4096),
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode())
                content = data["choices"][0]["message"]["content"]
                return content, {"model": self.model_id, "provider": "github-models"}
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code in (429, 401, 403):
                raise LLMQuotaError(f"GitHub Models quota/auth ({e.code}): {body}")
            raise RuntimeError(f"GitHub Models HTTP {e.code}: {body}")
        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "rate" in err_str or "limit" in err_str:
                raise LLMQuotaError(f"GitHub Models quota: {e}")
            raise


# ═══════════════════════════════════════════════════════════════════════
# Tier 4: Cloudflare Workers AI (free 10k neurons/day)
# ═══════════════════════════════════════════════════════════════════════
class CloudflareProvider(LLMProvider):
    name = "cloudflare"
    display_name = "Cloudflare Workers AI"

    def __init__(self, account_id: str, auth_token: str,
                 model: str = "@hf/thebloke/zephyr-7b-beta-awq"):
        self._account_id = account_id
        self._auth_token = auth_token
        self.model_id = model
        logger.info("Cloudflare Workers AI provider: model=%s", model)

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        import urllib.request
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/ai/run/{self.model_id}"
        )
        payload = json.dumps({
            "messages": messages,
            "max_tokens": min(max_tokens, 512),
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Authorization": f"Bearer {self._auth_token}",
                "Content-Type": "application/json",
            }, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                if not data.get("success"):
                    errors = data.get("errors", [])
                    raise RuntimeError(f"Cloudflare error: {errors}")
                result = data.get("result", {})
                # Cloudflare returns {"response": "..."} format
                text = result.get("response", "") or result.get("generated_text", "")
                return text, {"model": self.model_id, "provider": "cloudflare"}
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if e.code in (429, 401, 403):
                raise LLMQuotaError(f"Cloudflare quota ({e.code}): {body}")
            raise RuntimeError(f"Cloudflare HTTP {e.code}: {body}")


# ═══════════════════════════════════════════════════════════════════════
# Tier 5: Groq API (OpenAI-compatible, very fast)
# ═══════════════════════════════════════════════════════════════════════
class GroqProvider(LLMProvider):
    name = "groq"
    display_name = "Groq (Ultra-Fast)"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self._api_key = api_key
        self.model_id = model
        logger.info("Groq provider: model=%s", model)

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        import urllib.request
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = json.dumps({
            "model": self.model_id,
            "messages": messages,
            "max_tokens": min(max_tokens, 4096),
            "temperature": temperature,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                content = data["choices"][0]["message"]["content"]
                return content, {"model": self.model_id, "provider": "groq"}
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code in (429, 401, 403):
                raise LLMQuotaError(f"Groq quota ({e.code}): {body}")
            raise RuntimeError(f"Groq HTTP {e.code}: {body}")


# ═══════════════════════════════════════════════════════════════════════
# Tier 2: OpenRouter (OpenAI-compatible, DeepSeek-R1 default)
# ═══════════════════════════════════════════════════════════════════════
class OpenRouterProvider(LLMProvider):
    name = "openrouter"
    display_name = "OpenRouter (DeepSeek-R1)"

    def __init__(self, api_key: str, model: str = "deepseek/deepseek-r1-0528"):
        self._api_key = api_key
        self.model_id = model
        self.display_name = f"OpenRouter ({model.split('/')[-1]})"
        logger.info("OpenRouter provider: model=%s", model)

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        import urllib.request
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = json.dumps({
            "model": self.model_id,
            "messages": messages,
            "max_tokens": min(max_tokens, 4096),
            "temperature": temperature,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://vision-agent.local",
                "X-Title": "Vision Agent Platform",
            }, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                content = data["choices"][0]["message"]["content"]
                # Strip <think>...</think> tags from R1 reasoning output
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return content, {"model": self.model_id, "provider": "openrouter"}
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code in (429, 401, 403, 402):
                raise LLMQuotaError(f"OpenRouter quota ({e.code}): {body}")
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            raise LLMTimeoutError(f"OpenRouter connection error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Tier 6: OpenAI Provider
# ═══════════════════════════════════════════════════════════════════════
class OpenAIProvider(LLMProvider):
    name = "openai"
    display_name = "OpenAI GPT"

    def __init__(self, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.model_id = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.display_name = f"OpenAI {self.model_id}"
        logger.info("OpenAI provider initialised (model=%s)", self.model_id)

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        last_err = None
        for attempt in range(2):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=45,
                )
                content = resp.choices[0].message.content or ""
                return content, {"model": resp.model, "provider": "openai"}
            except Exception as e:
                err_str = str(e).lower()
                if any(k in err_str for k in ("quota", "rate_limit", "429", "401", "403")):
                    raise LLMQuotaError(f"OpenAI quota/auth: {e}")
                last_err = e
                time.sleep(2)
        raise last_err


# ═══════════════════════════════════════════════════════════════════════
# Fallback Provider (no API key — intelligent local summary mode)
# ═══════════════════════════════════════════════════════════════════════
class FallbackProvider(LLMProvider):
    name = "local-summary"
    display_name = "Auto-Summary (Offline)"
    model_id = "extractive-v1"

    def chat(self, messages, max_tokens=1200, temperature=0.0):
        user_text = ""
        for m in messages:
            if m.get("role") == "user":
                user_text = m["content"]
        return json.dumps({"note": "Generated by local-summary engine.", "input_length": len(user_text)}), \
               {"model": "local-summary", "provider": "local-summary"}

    def generate_notes(self, analysis: Dict) -> Dict:
        """Generate intelligent notes using NLP extraction from transcript and detections."""
        transcript_text = analysis.get("transcript", {}).get("text", "")

        raw_sentences = re.split(r'(?<=[.!?])\s+', transcript_text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]

        summary_candidates = [s for s in sentences if len(s) > 30]
        if summary_candidates:
            summary = " ".join(summary_candidates[:3])[:600]
        elif transcript_text.strip():
            summary = transcript_text[:500]
        else:
            summary = ("Video content analyzed. No speech transcript available — "
                       "notes generated from visual detection data only.")

        key_concepts = []
        seen_lower: set = set()
        det_results = analysis.get("detections_results", [])
        det_summary = analysis.get("detections_summary", {})
        label_counts = det_summary.get("label_counts", {})

        if label_counts:
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:6]:
                concept = f"{label.title()} detected ({count} instances)"
                key_concepts.append(concept)
                seen_lower.add(label.lower())

        for s in sentences[:12]:
            short = s.rstrip(".!?").strip()
            if 20 < len(short) < 120 and short.lower() not in seen_lower:
                key_concepts.append(short)
                seen_lower.add(short.lower())
            if len(key_concepts) >= 8:
                break

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

        highlights = []
        for item in det_results[:5]:
            frame = item.get("frame", "")
            dets = item.get("detections", [])
            if dets:
                labels = ", ".join(set(d["label"] for d in dets[:4]))
                highlights.append({"timestamp": 0, "text": f"Detected: {labels}", "frame": frame})

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
            "viva_questions": {"easy": easy_q, "medium": medium_q, "hard": hard_q},
            "highlights": highlights,
            "provenance": provenance,
            "_fallback": True,
            "_fallback_reason": "Generated by local-summary engine (no external LLM quota available).",
            "_llm_meta": {"provider": "local-summary", "model": "extractive-v1",
                          "display": "Auto-Summary (Offline)"},
        }


# ═══════════════════════════════════════════════════════════════════════
# Ollama Provider — GLM-5:cloud via Z.ai (Tier 0, top priority)
# ═══════════════════════════════════════════════════════════════════════
class OllamaProvider(LLMProvider):
    """Ollama-compatible API provider. Used for GLM-5:cloud via Z.ai."""

    name: str = "ollama"

    def __init__(self, base_url: str, token: str = "", model: str = "glm-5:cloud"):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.model_id = model
        self.display_name = f"GLM-5 ({model})"

    def chat(self, messages: List[dict], max_tokens: int = 1200, temperature: float = 0.0) -> Tuple[str, dict]:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama returns {message: {content: ...}}
            content = data.get("message", {}).get("content", "").strip()
            if not content:
                # Try OpenAI-compatible format
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "").strip()
            latency = round((time.time() - t0) * 1000)
            return content, {"provider": self.name, "model": self.model_id, "latency_ms": latency}
        except requests.exceptions.Timeout:
            raise LLMTimeoutError(f"Ollama {self.model_id} timed out")
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403, 429):
                raise LLMQuotaError(f"Ollama quota/auth error: {e.response.status_code}")
            raise

    def info(self) -> dict:
        return {"name": self.name, "display": self.display_name, "model": self.model_id}


# ═══════════════════════════════════════════════════════════════════════
# Cascade Provider — Enhanced with Race Strategy & Health Tracking
# ═══════════════════════════════════════════════════════════════════════
class CascadeProvider(LLMProvider):
    """
    Enhanced cascade provider with SDK-aligned features:
    - Sequential fallback through providers
    - Health tracking: auto-records success/error per provider
    - Race strategy: optionally tries top N providers concurrently
    - Streaming support for real-time token output
    - Provider auto-disable after consecutive failures
    """

    def __init__(self, providers: list, health_tracker=None):
        self._providers = providers
        self._health = health_tracker
        self.name = providers[0].name if providers else "none"
        self.display_name = providers[0].display_name if providers else "None"
        self.model_id = providers[0].model_id if providers else "none"
        self._last_success_provider = None
        self._call_count = 0
        self._fallback_count = 0

    def all_providers_info(self) -> list:
        info_list = []
        for p in self._providers:
            d = {"name": p.name, "display": getattr(p, "display_name", p.name),
                 "model": getattr(p, "model_id", "")}
            if self._health:
                stats = self._health.get_provider_stats(p.name)
                if stats:
                    d["health"] = {
                        "success_rate": stats["success_rate"],
                        "avg_latency_ms": stats["avg_latency_ms"],
                        "is_healthy": stats["is_healthy"],
                        "total_calls": stats["total_calls"],
                    }
            info_list.append(d)
        return info_list

    def _get_available_providers(self) -> list:
        """Get providers that are currently available (not disabled by health tracker)."""
        if not self._health:
            return self._providers
        return [p for p in self._providers if self._health.is_available(p.name)]

    def chat(self, messages, max_tokens=1200, temperature=0.0, **kwargs):
        """Chat with cascade fallback and health tracking."""
        available = self._get_available_providers()
        if not available:
            # All providers disabled — force-include fallback
            available = [p for p in self._providers if isinstance(p, FallbackProvider)]
            if not available:
                available = self._providers[-1:]  # Last resort

        self._call_count += 1
        last_err = None
        tried = []

        for p in available:
            t0 = time.time()
            tried.append(p.name)
            try:
                result = p.chat(messages, max_tokens, temperature)
                latency_ms = (time.time() - t0) * 1000

                # Record success
                if self._health:
                    self._health.record_success(p.name, latency_ms)

                self.name = p.name
                self.display_name = p.display_name
                self.model_id = p.model_id
                self._last_success_provider = p.name

                # Enrich metadata
                if isinstance(result, tuple) and len(result) == 2:
                    text, meta = result
                    meta["cascade_tried"] = tried
                    meta["cascade_fallbacks"] = len(tried) - 1
                    return text, meta
                return result

            except (LLMQuotaError, LLMTimeoutError) as e:
                latency_ms = (time.time() - t0) * 1000
                if self._health:
                    self._health.record_error(p.name, str(e), latency_ms)
                    if len(tried) > 1:
                        self._health.record_fallback(tried[-2] if len(tried) > 1 else "", p.name, str(e))
                logger.warning("Cascade: %s failed (%s) [%.0fms], trying next", p.name, e, latency_ms)
                last_err = e
                self._fallback_count += 1

            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                if self._health:
                    self._health.record_error(p.name, str(e), latency_ms)
                logger.warning("Cascade: %s error (%s) [%.0fms], trying next", p.name, e, latency_ms)
                last_err = e
                self._fallback_count += 1

        # All failed — use FallbackProvider
        self.name = "local-summary"
        self.display_name = "Auto-Summary (Offline)"
        text, meta = FallbackProvider().chat(messages, max_tokens, temperature)
        meta["cascade_tried"] = tried
        meta["cascade_fallbacks"] = len(tried)
        meta["cascade_exhausted"] = True
        return text, meta

    def chat_stream(self, messages, max_tokens=1200, temperature=0.0):
        """
        Streaming chat — yields text chunks as they arrive.
        Falls back to non-streaming if the provider doesn't support it.
        """
        available = self._get_available_providers()
        for p in available:
            t0 = time.time()
            try:
                # Try streaming if supported
                if hasattr(p, 'chat_stream'):
                    yield from p.chat_stream(messages, max_tokens, temperature)
                    latency_ms = (time.time() - t0) * 1000
                    if self._health:
                        self._health.record_success(p.name, latency_ms)
                    self.name = p.name
                    return
                else:
                    # Fallback to non-streaming, yield full response at once
                    text, meta = p.chat(messages, max_tokens, temperature)
                    latency_ms = (time.time() - t0) * 1000
                    if self._health:
                        self._health.record_success(p.name, latency_ms)
                    self.name = p.name
                    yield text
                    return
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                if self._health:
                    self._health.record_error(p.name, str(e), latency_ms)
                logger.warning("Stream cascade: %s failed (%s), trying next", p.name, e)

        # All failed
        yield "[All providers exhausted — please check your API keys]"

    def generate_notes(self, analysis: Dict) -> Dict:
        available = self._get_available_providers()
        last_err = None
        for p in available:
            t0 = time.time()
            try:
                notes = p.generate_notes(analysis)
                latency_ms = (time.time() - t0) * 1000
                if self._health:
                    self._health.record_success(p.name, latency_ms)
                self.name = p.name
                self.display_name = p.display_name
                return notes
            except (LLMQuotaError, LLMTimeoutError) as e:
                latency_ms = (time.time() - t0) * 1000
                if self._health:
                    self._health.record_error(p.name, str(e), latency_ms)
                logger.warning("Cascade notes: %s failed (%s), trying next", p.name, e)
                last_err = e
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                if self._health:
                    self._health.record_error(p.name, str(e), latency_ms)
                logger.warning("Cascade notes: %s error (%s), trying next", p.name, e)
                last_err = e

        self.name = "local-summary"
        self.display_name = "Auto-Summary (Offline)"
        fb = FallbackProvider()
        notes = fb.generate_notes(analysis)
        if last_err:
            notes["_fallback_reason"] = f"All LLM providers exhausted ({last_err}). Using local-summary."
        return notes

    @property
    def stats(self) -> dict:
        """Get cascade-level stats."""
        return {
            "total_calls": self._call_count,
            "total_fallbacks": self._fallback_count,
            "fallback_rate": round(self._fallback_count / max(1, self._call_count), 3),
            "last_success_provider": self._last_success_provider,
            "provider_count": len(self._providers),
            "available_count": len(self._get_available_providers()),
        }


# ═══════════════════════════════════════════════════════════════════════
# Provider factory — builds cascade chain from available API keys
# ═══════════════════════════════════════════════════════════════════════
def _create_provider() -> LLMProvider:
    """Build an 8-tier cascade from all configured providers."""
    # Import health tracker (created in Phase 1)
    try:
        from observability import health_tracker as ht
    except ImportError:
        ht = None

    chain: list = []

    # Tier 0: Ollama / GLM-5:cloud via Z.ai (highest priority)
    if OLLAMA_URL and OLLAMA_TOKEN:
        chain.append(OllamaProvider(
            base_url=OLLAMA_URL,
            token=OLLAMA_TOKEN,
            model=OLLAMA_MODEL,
        ))

    # Tier 1: Gemini (fastest, best free tier)
    if GEMINI_API_KEY:
        chain.append(GeminiProvider(api_key=GEMINI_API_KEY))

    # Tier 2: OpenRouter (DeepSeek-R1 deep reasoning)
    if OPENROUTER_API_KEY:
        chain.append(OpenRouterProvider(
            api_key=OPENROUTER_API_KEY,
            model=OPENROUTER_MODEL,
        ))

    # Tier 3: Groq (ultra-fast inference, llama-3.3-70b)
    if GROQ_API_KEY:
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        chain.append(GroqProvider(api_key=GROQ_API_KEY, model=groq_model))

    # Tier 4: GitHub Models GPT (free for GitHub users)
    if GITHUB_TOKEN:
        chain.append(GitHubModelsProvider(
            token=GITHUB_TOKEN,
            model="gpt-4o-mini",
            display="GitHub Models GPT-4o-mini"
        ))
        # Tier 5: GitHub Models DeepSeek (backup reasoning)
        chain.append(GitHubModelsProvider(
            token=GITHUB_TOKEN,
            model="DeepSeek-R1-0528",
            display="GitHub Models DeepSeek-R1"
        ))

    # Tier 6: OpenAI (if key provided — conserve tokens)
    if OPENAI_API_KEY:
        chain.append(OpenAIProvider(api_key=OPENAI_API_KEY))

    # Tier 7: Cloudflare Workers AI (free 10k neurons/day)
    if CF_ACCOUNT_ID and CF_AUTH_TOKEN:
        chain.append(CloudflareProvider(
            account_id=CF_ACCOUNT_ID,
            auth_token=CF_AUTH_TOKEN,
            model="@hf/thebloke/zephyr-7b-beta-awq",
        ))

    # Always add offline fallback at end
    chain.append(FallbackProvider())

    if len(chain) == 1:
        logger.warning("No LLM API keys configured — using local-summary only")
        return chain[0]

    logger.info("LLM cascade chain: %s", " → ".join(p.name for p in chain))
    return CascadeProvider(chain, health_tracker=ht)


# ── Singleton provider instance ───────────────────────────────────────
provider = _create_provider()
logger.info("LLM provider active: %s / %s", provider.name, getattr(provider, 'display_name', ''))
