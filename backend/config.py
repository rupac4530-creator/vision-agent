# backend/config.py
# adapted from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned Configuration & Feature Flags.

Central configuration for the Vision Agent platform.
All external-provider integrations are conditional via env vars.
"""

import os
import logging
from typing import Any, Dict

logger = logging.getLogger("config")


class Config:
    """Platform configuration loaded from environment variables.

    Feature flags allow enabling/disabling each SDK plugin at runtime.
    """

    # ── LLM Provider flags ────────────────────────────────────────
    ENABLE_GEMINI: bool = os.getenv("ENABLE_GEMINI", "true").lower() == "true"
    ENABLE_OPENAI: bool = os.getenv("ENABLE_OPENAI", "false").lower() == "true"
    ENABLE_ANTHROPIC: bool = os.getenv("ENABLE_ANTHROPIC", "false").lower() == "true"
    ENABLE_GROQ: bool = os.getenv("ENABLE_GROQ", "false").lower() == "true"
    ENABLE_CLOUDFLARE: bool = os.getenv("ENABLE_CLOUDFLARE", "false").lower() == "true"
    ENABLE_OLLAMA: bool = os.getenv("ENABLE_OLLAMA", "false").lower() == "true"
    ENABLE_HUGGINGFACE: bool = os.getenv("ENABLE_HUGGINGFACE", "false").lower() == "true"
    ENABLE_MISTRAL: bool = os.getenv("ENABLE_MISTRAL", "false").lower() == "true"
    ENABLE_AWS: bool = os.getenv("ENABLE_AWS", "false").lower() == "true"

    # ── STT/TTS Provider flags ────────────────────────────────────
    ENABLE_DEEPGRAM: bool = os.getenv("ENABLE_DEEPGRAM", "false").lower() == "true"
    ENABLE_ELEVENLABS: bool = os.getenv("ENABLE_ELEVENLABS", "false").lower() == "true"
    ENABLE_CARTESIA: bool = os.getenv("ENABLE_CARTESIA", "false").lower() == "true"

    # ── Feature flags ─────────────────────────────────────────────
    ENABLE_RAG: bool = os.getenv("ENABLE_RAG", "true").lower() == "true"
    ENABLE_MCP: bool = os.getenv("ENABLE_MCP", "true").lower() == "true"
    ENABLE_PROFILING: bool = os.getenv("ENABLE_PROFILING", "true").lower() == "true"
    ENABLE_VAD: bool = os.getenv("ENABLE_VAD", "true").lower() == "true"
    ENABLE_TURN_DETECTION: bool = os.getenv("ENABLE_TURN_DETECTION", "true").lower() == "true"
    ENABLE_WARMUP_CACHE: bool = os.getenv("ENABLE_WARMUP_CACHE", "true").lower() == "true"

    # ── API Keys (loaded from .env, never committed) ──────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    CLOUDFLARE_API_TOKEN: str = os.getenv("CLOUDFLARE_API_TOKEN", "")
    CLOUDFLARE_ACCOUNT_ID: str = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

    # ── Server settings ──────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "500"))

    # ── Model settings ───────────────────────────────────────────
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.0-flash")
    DEFAULT_STT_MODEL: str = os.getenv("DEFAULT_STT_MODEL", "whisper-small")
    DEFAULT_TTS_VOICE: str = os.getenv("DEFAULT_TTS_VOICE", "alloy")
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    POSE_MODEL: str = os.getenv("POSE_MODEL", "yolov8n-pose.pt")

    @classmethod
    def get_enabled_providers(cls) -> Dict[str, bool]:
        """Get all provider feature flags."""
        return {
            "gemini": cls.ENABLE_GEMINI,
            "openai": cls.ENABLE_OPENAI,
            "anthropic": cls.ENABLE_ANTHROPIC,
            "groq": cls.ENABLE_GROQ,
            "cloudflare": cls.ENABLE_CLOUDFLARE,
            "ollama": cls.ENABLE_OLLAMA,
            "huggingface": cls.ENABLE_HUGGINGFACE,
            "mistral": cls.ENABLE_MISTRAL,
            "aws": cls.ENABLE_AWS,
            "deepgram": cls.ENABLE_DEEPGRAM,
            "elevenlabs": cls.ENABLE_ELEVENLABS,
            "cartesia": cls.ENABLE_CARTESIA,
        }

    @classmethod
    def get_enabled_features(cls) -> Dict[str, bool]:
        """Get all feature flags."""
        return {
            "rag": cls.ENABLE_RAG,
            "mcp": cls.ENABLE_MCP,
            "profiling": cls.ENABLE_PROFILING,
            "vad": cls.ENABLE_VAD,
            "turn_detection": cls.ENABLE_TURN_DETECTION,
            "warmup_cache": cls.ENABLE_WARMUP_CACHE,
        }

    @classmethod
    def to_dict(cls) -> Dict:
        """Get safe (non-secret) configuration summary."""
        return {
            "providers": cls.get_enabled_providers(),
            "features": cls.get_enabled_features(),
            "server": {
                "host": cls.HOST,
                "port": cls.PORT,
                "debug": cls.DEBUG,
                "max_upload_mb": cls.MAX_UPLOAD_MB,
            },
            "models": {
                "llm": cls.DEFAULT_LLM_MODEL,
                "stt": cls.DEFAULT_STT_MODEL,
                "tts": cls.DEFAULT_TTS_VOICE,
                "yolo": cls.YOLO_MODEL,
                "pose": cls.POSE_MODEL,
            },
            "has_keys": {
                "gemini": bool(cls.GEMINI_API_KEY),
                "openai": bool(cls.OPENAI_API_KEY),
                "groq": bool(cls.GROQ_API_KEY),
            },
        }


# ── Singleton ─────────────────────────────────────────────────────────
config = Config()
logger.info("Config loaded: %d providers enabled, %d features enabled",
            sum(config.get_enabled_providers().values()),
            sum(config.get_enabled_features().values()))
