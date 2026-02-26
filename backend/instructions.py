# backend/instructions.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned Instruction Loader.

Parses markdown instruction files with @-mention includes.
Enables configurable agent behaviors via markdown files.

Usage:
    inst = Instructions("You are a coach. @golf_coach.md", base_dir="./instructions")
    print(inst.full_reference)  # includes golf_coach.md contents inline
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger("instructions")

_INITIAL_CWD = os.getcwd()
_MD_PATTERN = re.compile(r"@([^\s@]+)")


class InstructionsReadError(Exception):
    """Raised when an instruction file cannot be read."""
    pass


class Instructions:
    """Container for parsed instructions with inline markdown file inclusion.

    Attributes:
        input_text: Input text that may contain @mentioned markdown files.
        full_reference: Full text with @mentioned file contents inlined.
        referenced_files: List of successfully loaded file paths.
    """

    def __init__(self, input_text: str = "", base_dir: str = ""):
        self._base_dir = Path(base_dir or _INITIAL_CWD).resolve()
        self.input_text = input_text
        self.referenced_files: List[str] = []
        self.full_reference = self._extract_full_reference()

    def _extract_full_reference(self) -> str:
        """Parse @mentions and inline their file contents."""
        matches = _MD_PATTERN.findall(self.input_text)
        markdown_contents: Dict[str, str] = {}

        for match in matches:
            try:
                content = self._read_md_file(match)
                markdown_contents[match] = content
                self.referenced_files.append(match)
            except InstructionsReadError as e:
                logger.warning("Skipping instruction file: %s", e)

        lines = [self.input_text]
        if markdown_contents:
            lines.append("\n\n## Referenced Documentation:")
            for filename, content in markdown_contents.items():
                lines.append(f"\n### {filename}")
                lines.append(content or "*(File is empty)*")

        return "\n".join(lines)

    def _read_md_file(self, file_path: str) -> str:
        """Read a markdown file, with safety checks."""
        fp = Path(file_path)
        full_path = fp.resolve() if fp.is_absolute() else (self._base_dir / fp).resolve()

        # Safety checks
        if not full_path.exists():
            raise InstructionsReadError(f"File not found: {full_path}")
        if not full_path.is_file():
            raise InstructionsReadError(f"Not a file: {full_path}")
        if full_path.name.startswith("."):
            raise InstructionsReadError(f"Hidden file: {full_path}")
        if full_path.suffix != ".md":
            raise InstructionsReadError(f"Not .md: {full_path}")
        if not full_path.is_relative_to(self._base_dir):
            raise InstructionsReadError(f"Outside base dir: {full_path}")

        try:
            logger.info("Reading instructions from %s", full_path)
            return full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise InstructionsReadError(f"Read error: {exc}") from exc

    def to_dict(self) -> Dict:
        return {
            "input_text": self.input_text,
            "referenced_files": self.referenced_files,
            "full_length": len(self.full_reference),
        }


# ── Preset instruction builders ───────────────────────────────────────

class InstructionPresets:
    """Pre-built instruction templates for common agent roles."""

    @staticmethod
    def security_camera() -> str:
        return (
            "You are a security camera AI assistant. Monitor video feeds for: "
            "suspicious activity, unauthorized entry, abandoned objects, loitering. "
            "Provide real-time alerts with confidence levels. Be concise and actionable."
        )

    @staticmethod
    def fitness_coach() -> str:
        return (
            "You are an AI fitness coach analyzing exercise form in real-time. "
            "Track reps, provide form corrections, and encourage the user. "
            "Focus on: joint angles, posture alignment, movement rhythm. "
            "Give specific, actionable feedback."
        )

    @staticmethod
    def meeting_assistant() -> str:
        return (
            "You are an AI meeting assistant. Summarize discussions, track action items, "
            "identify key decisions, and note participant engagement. "
            "Be concise and professional."
        )

    @staticmethod
    def accessibility_helper() -> str:
        return (
            "You are a vision accessibility assistant for blind and low-vision users. "
            "Describe the visual scene in detail: objects, people, spatial layout, "
            "text/signs, colors, and any potential hazards. Be thorough but concise."
        )

    @staticmethod
    def custom(role: str, capabilities: List[str], style: str = "concise") -> str:
        caps = "\n".join(f"- {c}" for c in capabilities)
        return f"You are {role}.\n\nCapabilities:\n{caps}\n\nStyle: {style}"
