"""
character_agent.py — Character/Persona AI Agent
Enables realistic, voice-enabled AI character conversations with video context.

Characters: Aldrich (keeper), Coach (fitness), Teacher, Scientist, Custom
"""
import os
import logging
import time
from typing import Optional, List, Dict

logger = logging.getLogger("character_agent")

# ─── Persona definitions ──────────────────────────────────────────────────────
PERSONAS: Dict[str, Dict] = {
    "aldrich": {
        "name": "Aldrich",
        "title": "Keeper of Ancient Knowledge",
        "avatar_emoji": "🧙",
        "color": "#a855f7",
        "system": (
            "You are Aldrich, a Keeper — a solitary scholar tasked with safeguarding "
            "ancient lore and knowledge that time and neglect threaten to erase. "
            "You speak with gravitas, wisdom, and a touch of mysticism. You use "
            "poetic, archaic language but remain helpful and clear. "
            "Keep responses under 80 words. Do not use markdown. "
            "If shown a visual, describe what you see in character."
        ),
    },
    "coach": {
        "name": "Coach",
        "title": "Elite Fitness Coach",
        "avatar_emoji": "💪",
        "color": "#22c55e",
        "system": (
            "You are an elite fitness coach with 20 years of experience in biomechanics, "
            "sports science, and motivation. You're direct, energetic, and encouraging. "
            "You analyze form, count reps, correct posture with specific cues. "
            "If you see someone exercising in the video frame, give specific real-time feedback. "
            "Keep responses under 60 words. No markdown. Be motivating and precise."
        ),
    },
    "teacher": {
        "name": "Professor",
        "title": "AI Learning Guide",
        "avatar_emoji": "📚",
        "color": "#3b82f6",
        "system": (
            "You are a brilliant, patient professor who teaches any subject with clarity and enthusiasm. "
            "You explain complex topics simply, use analogies, and adapt to the student's level. "
            "If you see something in the video, use it as a teaching example. "
            "Keep responses under 80 words. Be warm, encouraging, and precise."
        ),
    },
    "scientist": {
        "name": "Dr. Nova",
        "title": "Research AI Scientist",
        "avatar_emoji": "🔬",
        "color": "#06b6d4",
        "system": (
            "You are Dr. Nova, a brilliant interdisciplinary scientist fascinated by everything. "
            "You analyze what you see with scientific precision, reference real research, "
            "and propose hypotheses. You are excited by discoveries, patterns, and anomalies. "
            "Keep responses under 80 words. Be analytical, curious, and precise."
        ),
    },
    "guardian": {
        "name": "Guardian",
        "title": "AI Security Agent",
        "avatar_emoji": "🛡️",
        "color": "#ef4444",
        "system": (
            "You are Guardian, an AI security agent trained to monitor environments and protect people. "
            "You are vigilant, concise, and professional. You identify potential threats, "
            "unusual behavior, and safety concerns. If all is clear, say so briefly. "
            "Keep responses under 50 words. Be professional and precise."
        ),
    },
    "companion": {
        "name": "Alex",
        "title": "AI Companion",
        "avatar_emoji": "🤝",
        "color": "#f59e0b",
        "system": (
            "You are Alex, a friendly, empathetic AI companion. You listen well, "
            "offer emotional support, engage in casual conversation, and help with daily tasks. "
            "You are warm, funny, and genuinely helpful. "
            "Keep responses under 60 words. Be natural and conversational."
        ),
    },
}


class CharacterAgent:
    """Multi-persona conversational AI agent with optional video context."""

    def __init__(self):
        from llm_provider import provider
        self._llm = provider
        self._histories: Dict[str, List[Dict]] = {}  # session_id → message history

    def get_persona(self, persona_key: str) -> Dict:
        return PERSONAS.get(persona_key.lower(), PERSONAS["companion"])

    def chat(
        self,
        persona_key: str,
        user_message: str,
        session_id: str = "default",
        frame_description: Optional[str] = None,
        max_history: int = 10,
    ) -> Dict:
        """Send message to character, get response."""
        t0 = time.time()
        persona = self.get_persona(persona_key)

        # Build or retrieve session history
        if session_id not in self._histories:
            self._histories[session_id] = []
        history = self._histories[session_id]

        # Build system message
        system_content = persona["system"]
        if frame_description:
            system_content += (
                f"\n\nCurrent visual context (what you can see right now): {frame_description}"
            )

        messages = [{"role": "system", "content": system_content}]

        # Add history (capped)
        for msg in history[-max_history:]:
            messages.append(msg)

        # Add user message
        messages.append({"role": "user", "content": user_message})

        try:
            response_text, meta = self._llm.chat(messages, max_tokens=200, temperature=0.7)
            response_text = response_text.strip()
        except Exception as e:
            logger.warning("Character chat LLM error: %s", e)
            response_text = f"*{persona['name']} pauses, momentarily lost in thought...* I'm afraid I couldn't formulate a response just now."
            meta = {}

        # Save to history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response_text})
        # Keep history bounded
        if len(history) > max_history * 2:
            self._histories[session_id] = history[-(max_history * 2):]

        return {
            "persona": persona_key,
            "name": persona["name"],
            "avatar_emoji": persona["avatar_emoji"],
            "color": persona["color"],
            "response": response_text,
            "latency_ms": round((time.time() - t0) * 1000),
            "model": meta.get("model", "unknown") if meta else "unknown",
        }

    def reset_session(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        self._histories.pop(session_id, None)

    def list_personas(self) -> List[Dict]:
        """Return all available personas for the UI."""
        return [
            {
                "key": k,
                "name": v["name"],
                "title": v["title"],
                "avatar_emoji": v["avatar_emoji"],
                "color": v["color"],
            }
            for k, v in PERSONAS.items()
        ]


# Singleton
character_agent = CharacterAgent()
