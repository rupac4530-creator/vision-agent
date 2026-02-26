# backend/conversation.py
"""
Conversation Memory — SDK-Aligned Session Management

Ported from Vision-Agents SDK (agents-core/vision_agents/core/agents/conversation.py)
Manages multi-turn conversation history per session with context windowing,
summary generation, and persistent storage.
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("conversation")

CONVERSATIONS_DIR = os.path.join(os.path.dirname(__file__), "conversations")
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_llm_message(self) -> dict:
        """Format for LLM API (OpenAI-compatible)."""
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationSession:
    """
    A conversation session with message history and metadata.
    SDK-aligned: supports context windowing and auto-summarization.
    """
    session_id: str
    messages: List[Message] = field(default_factory=list)
    system_prompt: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_messages: int = 100
    max_context_messages: int = 20

    def add_message(self, role: str, content: str, **metadata):
        """Add a message to the conversation."""
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self.last_active = time.time()

        # Trim old messages if needed
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self, max_messages: Optional[int] = None,
                    include_system: bool = True) -> List[dict]:
        """
        Get conversation context formatted for LLM API calls.
        Returns the most recent N messages in OpenAI format.
        """
        limit = max_messages or self.max_context_messages
        context = []

        if include_system and self.system_prompt:
            context.append({"role": "system", "content": self.system_prompt})

        recent = self.messages[-limit:]
        for msg in recent:
            context.append(msg.to_llm_message())

        return context

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "message_count": self.message_count,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "messages": [m.to_dict() for m in self.messages[-50:]],  # Last 50
            "metadata": self.metadata,
        }

    def save(self):
        """Persist conversation to disk."""
        path = os.path.join(CONVERSATIONS_DIR, f"{self.session_id}.json")
        data = {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, session_id: str) -> Optional["ConversationSession"]:
        """Load a conversation from disk."""
        path = os.path.join(CONVERSATIONS_DIR, f"{session_id}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            session = cls(
                session_id=data["session_id"],
                system_prompt=data.get("system_prompt", ""),
                created_at=data.get("created_at", time.time()),
                last_active=data.get("last_active", time.time()),
                metadata=data.get("metadata", {}),
            )
            for m in data.get("messages", []):
                session.messages.append(Message(
                    role=m["role"],
                    content=m["content"],
                    metadata=m.get("metadata", {}),
                ))
            return session
        except Exception as e:
            logger.error("Failed to load conversation %s: %s", session_id, e)
            return None


class ConversationManager:
    """
    Manages multiple conversation sessions.
    Provides session creation, retrieval, and persistence.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are Vision Agent, an AI assistant with advanced visual understanding "
        "capabilities. You can analyze images, video streams, detect objects and poses, "
        "and provide intelligent insights. You also have access to specialized tools."
    )

    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}
        self._load_existing()

    def _load_existing(self):
        """Load existing conversations from disk."""
        if not os.path.isdir(CONVERSATIONS_DIR):
            return
        count = 0
        for fname in os.listdir(CONVERSATIONS_DIR):
            if not fname.endswith(".json"):
                continue
            sid = fname.replace(".json", "")
            session = ConversationSession.load(sid)
            if session:
                self._sessions[sid] = session
                count += 1
        if count:
            logger.info("Loaded %d existing conversations", count)

    def get_or_create(self, session_id: str, system_prompt: str = "") -> ConversationSession:
        """Get existing session or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(
                session_id=session_id,
                system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            )
            logger.info("Created new conversation: %s", session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[ConversationSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[dict]:
        """List all sessions with summary info."""
        sessions = []
        for sid, s in sorted(self._sessions.items(), key=lambda x: x[1].last_active, reverse=True):
            sessions.append({
                "session_id": sid,
                "message_count": s.message_count,
                "created_at": s.created_at,
                "last_active": s.last_active,
                "last_message": s.last_message.content[:80] if s.last_message else "",
            })
        return sessions

    def save_all(self):
        """Persist all sessions to disk."""
        for s in self._sessions.values():
            s.save()

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            path = os.path.join(CONVERSATIONS_DIR, f"{session_id}.json")
            if os.path.isfile(path):
                os.remove(path)
            return True
        return False

    @property
    def count(self) -> int:
        return len(self._sessions)


# ══════════════════════════════════════════════════════════════════════
# Global singleton
# ══════════════════════════════════════════════════════════════════════

conversation_manager = ConversationManager()
logger.info("ConversationManager ready (%d sessions)", conversation_manager.count)
