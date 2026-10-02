"""Prompt Manager for Dynamic Prompt Composition.

Composes prompt modules based on AgentState (language, role, attempt, etc.).
Implements Prompt Composition over Monolithic Prompt architecture.
"""

import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

from src.state import AgentState


class PromptManager:
    """Manages and composes modular prompt templates."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or Path(__file__).resolve().parent.parent.parent / "prompts"
        self._cache: Dict[str, str] = {}

    # === Core Composition Methods ===

    def compose_system_prompt(self, state: AgentState) -> str:
        """Build system prompt from role + style + format + safety."""
        parts = []

        # 1. Role (default: local_guide)
        role_name = state.metadata.get("role", "local_guide")
        parts.append(self._load_role(role_name))

        # 2. Style (language-specific)
        parts.append(self._load_style(state.language))

        # 3. Format (default: markdown_output)
        format_name = state.metadata.get("output_format", "markdown_output")
        parts.append(self._load_format(format_name))

        # 4. Safety (always included)
        parts.append(self._load_safety())

        # 5. Tool schemas (injected at runtime by AgentCore)
        # Not included here - AgentCore adds tool definitions separately

        composed = self._combine(parts)

        # Record prompt version for traceability
        self._record_prompt_version(state, composed)

        return composed

    def compose_retry_prompt(self, state: AgentState, errors: List[Dict]) -> str:
        """Build retry hint from feedback + error_explanation."""
        parts = []

        # Base retry hint template
        parts.append(self._load_feedback("retry_hint"))

        # Error-specific explanations
        for error in errors:
            error_type = error.get("error_type", "unknown")
            parts.append(self._load_feedback("error_explanation", error_type=error_type))

        # Recent feedback summary
        if state.feedback:
            parts.append(f"\n## 歷史回饋摘要\n{state.feedback[-1]}")

        return self._combine(parts)

    def compose_judge_prompt(self, eval_type: str, **kwargs) -> str:
        """Build judge prompt for groundedness/quality evaluation."""
        if eval_type == "groundedness":
            return self._load_judge("groundedness")
        elif eval_type == "quality":
            return self._load_judge("quality")
        else:
            raise ValueError(f"Unknown eval_type: {eval_type}")

    # === Module Loaders ===

    def _load_role(self, role_name: str) -> str:
        """Load role prompt (e.g., local_guide, professional_cs, cultural_expert)."""
        return self._load_module("roles", f"{role_name}.md")

    def _load_style(self, language: str) -> str:
        """Load style prompt for language (zh-TW, en, ja)."""
        # Normalize language code
        lang_map = {"zh": "zh-TW", "zh-CN": "zh-TW", "zh-TW": "zh-TW", "en": "en", "ja": "ja"}
        lang = lang_map.get(language, "zh-TW")
        return self._load_module("styles", f"{lang}.md")

    def _load_format(self, format_name: str) -> str:
        """Load format guardrail (json_output, markdown_output, structured_text)."""
        return self._load_module("formats", f"{format_name}.md")

    def _load_safety(self) -> str:
        """Load safety rules (always included)."""
        return self._load_module("system", "safety.md")

    def _load_judge(self, judge_name: str) -> str:
        """Load judge prompt (groundedness, quality)."""
        return self._load_module("judge", f"{judge_name}.md")

    def _load_feedback(self, feedback_name: str, error_type: Optional[str] = None) -> str:
        """Load feedback template with optional error-specific section."""
        content = self._load_module("feedback", f"{feedback_name}.md")

        # For error_explanation, extract relevant error_type section
        if feedback_name == "error_explanation" and error_type:
            content = self._extract_error_section(content, error_type)

        return content

    def _extract_error_section(self, content: str, error_type: str) -> str:
        """Extract error-type specific section from error_explanation.md."""
        # Simple extraction: find section for this error_type
        lines = content.split("\n")
        in_section = False
        result = []

        for line in lines:
            if line.strip().startswith(f"### {error_type}") or line.strip() == f"## {error_type}":
                in_section = True
                continue
            elif in_section and line.strip().startswith("### ") and error_type not in line:
                break
            if in_section:
                result.append(line)

        if result:
            return "\n".join(result).strip()
        return content  # Fallback to full content

    def _load_module(self, category: str, filename: str) -> str:
        """Load a prompt module from disk with caching."""
        cache_key = f"{category}/{filename}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.prompts_dir / category / filename
        if not path.exists():
            # Try without extension
            path = self.prompts_dir / category / f"{filename}.md"

        if path.exists():
            content = path.read_text(encoding="utf-8")
        else:
            content = f"[WARNING: Prompt module not found: {category}/{filename}]"

        self._cache[cache_key] = content
        return content

    def _combine(self, parts: List[str]) -> str:
        """Combine prompt parts with clear separators."""
        separator = "\n\n" + "=" * 60 + "\n\n"
        return separator.join(part.strip() for part in parts if part.strip())

    # === Version Tracking ===

    def _record_prompt_version(self, state: AgentState, composed_prompt: str):
        """Record prompt version hashes in state metadata for traceability."""
        if "prompt_version" not in state.metadata:
            state.metadata["prompt_version"] = {}

        # Compute hash of each module used
        role_name = state.metadata.get("role", "local_guide")
        style_name = state.language
        format_name = state.metadata.get("output_format", "markdown_output")

        state.metadata["prompt_version"] = {
            "role": role_name,
            "role_hash": self._file_hash("roles", f"{role_name}.md"),
            "style": style_name,
            "style_hash": self._file_hash("styles", f"{style_name}.md"),
            "format": format_name,
            "format_hash": self._file_hash("formats", f"{format_name}.md"),
            "safety_hash": self._file_hash("system", "safety.md"),
            "composed_hash": hashlib.sha256(composed_prompt.encode()).hexdigest()[:16],
        }

    def _file_hash(self, category: str, filename: str) -> str:
        """Compute SHA256 hash of a prompt file."""
        path = self.prompts_dir / category / filename
        if not path.exists():
            path = self.prompts_dir / category / f"{filename}.md"
        if path.exists():
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        return "missing"

    # === Utility Methods ===

    def clear_cache(self):
        """Clear the prompt cache (useful for development/hot-reload)."""
        self._cache.clear()

    def list_available(self, category: str) -> List[str]:
        """List available prompt modules in a category."""
        cat_dir = self.prompts_dir / category
        if not cat_dir.exists():
            return []
        return [f.stem for f in cat_dir.glob("*.md")]

    def get_prompt_preview(self, state: AgentState, max_chars: int = 500) -> str:
        """Get preview of composed system prompt for debugging."""
        prompt = self.compose_system_prompt(state)
        return prompt[:max_chars] + ("..." if len(prompt) > max_chars else "")


# Factory function
def create_prompt_manager(prompts_dir: Optional[Path] = None) -> PromptManager:
    """Factory function to create PromptManager."""
    return PromptManager(prompts_dir)