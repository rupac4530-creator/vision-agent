# backend/llm_types.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned LLM type system.

Provides normalized contracts for multi-provider LLM communication:
- ContentPart types (text, image, audio, JSON)
- Message and Role definitions
- ToolSchema for function calling
- NormalizedResponse for unified provider output
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field


# ── Content Parts ──────────────────────────────────────────────────────

class TextPart:
    """Text content part."""
    type: str = "text"
    def __init__(self, text: str):
        self.text = text
    def to_dict(self):
        return {"type": "text", "text": self.text}


class ImageBytesPart:
    """Image content from raw bytes."""
    type: str = "image"
    def __init__(self, data: bytes, mime_type: str = "image/jpeg"):
        self.data = data
        self.mime_type = mime_type
    def to_dict(self):
        return {"type": "image", "mime_type": self.mime_type, "data_len": len(self.data)}


class ImageURLPart:
    """Image content from URL."""
    type: str = "image"
    def __init__(self, url: str, mime_type: str = "image/jpeg"):
        self.url = url
        self.mime_type = mime_type
    def to_dict(self):
        return {"type": "image", "url": self.url, "mime_type": self.mime_type}


class AudioPart:
    """Audio content part."""
    type: str = "audio"
    def __init__(self, data: bytes, mime_type: str = "audio/wav",
                 sample_rate: int = 16000, channels: int = 1):
        self.data = data
        self.mime_type = mime_type
        self.sample_rate = sample_rate
        self.channels = channels
    def to_dict(self):
        return {"type": "audio", "mime_type": self.mime_type,
                "sample_rate": self.sample_rate, "channels": self.channels}


class JsonPart:
    """Structured JSON content part."""
    type: str = "json"
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    def to_dict(self):
        return {"type": "json", "data": self.data}


# Union type for all content parts
ContentPart = Union[TextPart, ImageBytesPart, ImageURLPart, AudioPart, JsonPart]


# ── Roles & Messages ──────────────────────────────────────────────────

class Role(str, Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""
    role: Role
    content: List[ContentPart] = field(default_factory=list)
    name: Optional[str] = None  # For tool messages

    def text_content(self) -> str:
        """Extract plain text from all content parts."""
        texts = []
        for part in self.content:
            if isinstance(part, TextPart):
                texts.append(part.text)
        return " ".join(texts)

    def to_dict(self) -> Dict:
        return {
            "role": self.role.value,
            "content": [p.to_dict() for p in self.content],
            **({"name": self.name} if self.name else {}),
        }


# ── Tool Schema ────────────────────────────────────────────────────────

@dataclass
class ToolSchema:
    """Schema for a function/tool that can be called by an LLM."""
    name: str
    description: str = ""
    parameters_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


@dataclass
class ResponseFormat:
    """Format specification for structured LLM output."""
    json_schema: Optional[Dict[str, Any]] = None
    strict: bool = False


# ── Normalized Response ────────────────────────────────────────────────

class NormalizedStatus(str, Enum):
    """Status of an LLM response."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass
class NormalizedUsage:
    """Token usage statistics from an LLM response."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    raw_usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class NormalizedTextItem:
    """Text output item from LLM."""
    type: str = "text"
    text: str = ""
    index: int = 0


@dataclass
class NormalizedToolCallItem:
    """Tool call output item from LLM."""
    type: str = "tool_call"
    name: str = ""
    arguments_json: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "type": "tool_call",
            "name": self.name,
            "arguments": self.arguments_json,
            **({"id": self.id} if self.id else {}),
        }


@dataclass
class NormalizedToolResultItem:
    """Tool result item to feed back to LLM."""
    type: str = "tool_result"
    name: str = ""
    result_json: Dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


NormalizedOutputItem = Union[NormalizedTextItem, NormalizedToolCallItem, NormalizedToolResultItem]


@dataclass
class NormalizedResponse:
    """Unified response from any LLM provider."""
    id: str = ""
    model: str = ""
    status: NormalizedStatus = NormalizedStatus.COMPLETED
    output: List[NormalizedOutputItem] = field(default_factory=list)
    usage: Optional[NormalizedUsage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    raw: Optional[Any] = None

    @property
    def output_text(self) -> str:
        """Extract concatenated text from all text output items."""
        return " ".join(
            item.text for item in self.output
            if isinstance(item, NormalizedTextItem)
        )

    @property
    def tool_calls(self) -> List[NormalizedToolCallItem]:
        """Extract all tool call items."""
        return [item for item in self.output if isinstance(item, NormalizedToolCallItem)]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "model": self.model,
            "status": self.status.value,
            "output_text": self.output_text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "usage": self.usage.to_dict() if self.usage else None,
            "metadata": self.metadata,
        }
