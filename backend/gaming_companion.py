"""
gaming_companion.py — AI Gaming Companion
Analyzes game screenshots and provides real-time strategy advice + commentary.
"""
import os
import base64
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("gaming_companion")

GAME_CONTEXTS = {
    "chess": "You are a chess grandmaster. Analyze this board position and suggest the best move with strategic reasoning.",
    "fps": "You are an FPS game coach. Analyze this screenshot for positioning, cover usage, threat awareness, and tactical advice.",
    "moba": "You are a MOBA expert (LoL/Dota). Analyze this game state for map control, item builds, teamfight positioning.",
    "rts": "You are an RTS expert (SC2/AoE). Analyze this screenshot for economy, unit control, and strategic map control.",
    "puzzle": "You are a puzzle solver. Analyze this game state and suggest the next logical steps to solve the puzzle.",
    "racing": "You are a racing coach. Analyze this moment for racing line, braking points, and cornering technique.",
    "sports": "You are a sports game analyst. Analyze this play for tactical opportunities and player positioning.",
    "general": "You are a professional gaming coach. Analyze this screenshot and provide helpful strategic advice.",
}


class GamingCompanion:
    """AI gaming assistant that coaches players via visual analysis."""

    def __init__(self):
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")
        self._ollama_url = os.getenv("OLLAMA_URL", "")
        self._ollama_token = os.getenv("OLLAMA_TOKEN", "")
        self._ollama_model = os.getenv("OLLAMA_MODEL", "glm-5:cloud")
        self._advice_history = []

    def analyze(self, frame_b64: str, game_type: str = "general", extra_context: str = "") -> Dict[str, Any]:
        """Analyze a game screenshot and return strategic advice."""
        t0 = time.time()

        context_prompt = GAME_CONTEXTS.get(game_type.lower(), GAME_CONTEXTS["general"])
        if extra_context:
            context_prompt += f"\n\nAdditional context: {extra_context}"
        context_prompt += "\n\nGive concise, actionable advice in under 80 words. Be specific and direct."

        # Try Gemini Vision (best for image analysis)
        if self._gemini_key:
            try:
                import requests
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-2.0-flash:generateContent?key={self._gemini_key}"
                )
                payload = {
                    "contents": [{
                        "parts": [
                            {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
                            {"text": context_prompt},
                        ]
                    }],
                    "generationConfig": {"temperature": 0.4, "maxOutputTokens": 200},
                }
                resp = requests.post(url, json=payload, timeout=25)
                resp.raise_for_status()
                data = resp.json()
                advice = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                result = {
                    "ok": True,
                    "game_type": game_type,
                    "advice": advice,
                    "provider": "gemini-vision",
                    "latency_ms": round((time.time() - t0) * 1000),
                }
                self._save_history(game_type, advice)
                return result
            except Exception as e:
                logger.warning("Gemini gaming analysis error: %s", e)

        # Fallback: text-only analysis via LLM cascade
        try:
            from llm_provider import provider
            messages = [
                {"role": "system", "content": context_prompt},
                {"role": "user", "content": "Please analyze the current game situation based on the context provided and give your best strategic advice."},
            ]
            text, meta = provider.chat(messages, max_tokens=200, temperature=0.4)
            result = {
                "ok": True,
                "game_type": game_type,
                "advice": text.strip(),
                "provider": meta.get("provider", "cascade") if meta else "cascade",
                "latency_ms": round((time.time() - t0) * 1000),
                "note": "Vision analysis unavailable — using text reasoning only",
            }
            self._save_history(game_type, text.strip())
            return result
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "advice": "Unable to analyze game state. Please check API configuration.",
                "latency_ms": round((time.time() - t0) * 1000),
            }

    def commentary(self, action: str, game_type: str = "general") -> str:
        """Generate exciting commentary for a game action."""
        try:
            from llm_provider import provider
            messages = [
                {"role": "system", "content": "You are an exciting sports/game commentator. Generate 1-2 sentences of energetic commentary. No markdown."},
                {"role": "user", "content": f"Game: {game_type}. Action: {action}. Give energetic commentary!"},
            ]
            text, _ = provider.chat(messages, max_tokens=80, temperature=0.8)
            return text.strip()
        except Exception as e:
            return f"Incredible play! {action}"

    def _save_history(self, game_type: str, advice: str):
        self._advice_history.append({
            "ts": time.time(),
            "game_type": game_type,
            "advice": advice,
        })
        self._advice_history = self._advice_history[-20:]

    def get_history(self) -> list:
        return self._advice_history


# Singleton
gaming_companion = GamingCompanion()
