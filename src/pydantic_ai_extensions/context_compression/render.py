"""Message renderers for context compression.

Two independent renderers:
- `render_messages_to_text`: flat prose rendering, used only for token estimation
  (§5.4). Includes `ModelRequest.instructions`.
- `render_structured`: XML-ish structured rendering, used as the summarizer input
  (§5.8, the only input mode).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast
from xml.sax.saxutils import escape, quoteattr

from pydantic_ai.messages import (
    CompactionPart,
    FilePart,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

__all__ = ["render_messages_to_text", "render_structured", "stringify"]


def stringify(x: Any) -> str:
    """Best-effort text rendering of a part payload (args/content)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if hasattr(x, "data") and hasattr(x, "media_type"):
        return "<binary>"  # BinaryContent (duck-typed)
    if isinstance(x, (dict, list)):
        y = cast(Any, x)
        try:
            return json.dumps(y, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(y)
    # pydantic models / dataclasses / other objects
    if hasattr(x, "model_dump_json"):
        return x.model_dump_json()
    return str(x)


def _user_prompt_text(content: str | Sequence[Any]) -> str:
    """Extract text from `UserPromptPart.content` (str or sequence of UserContent)."""
    if isinstance(content, str):
        return content
    if hasattr(content, "data") and hasattr(content, "media_type"):
        return "<media>"  # BinaryContent (duck-typed)
    # Sequence of UserContent items (str, TextContent, ImageUrl, BinaryContent, ...)
    if isinstance(content, (list, tuple)):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif hasattr(item, "data") and hasattr(item, "media_type"):
                # BinaryContent (duck-typed to avoid isinstance-on-Protocol issues in some IDEs)
                chunks.append("<media>")
            elif hasattr(item, "content") and isinstance(item.content, str):
                # TextContent (duck-typed)
                chunks.append(item.content)  # type: ignore[union-attr]
            else:
                chunks.append("<media>")
        return "\n".join(chunks)
    return stringify(content)


def render_messages_to_text(messages: Sequence[ModelMessage], *, include_thinking: bool = True) -> str:
    """Flat prose rendering of messages, for token estimation only.

    Includes each `ModelRequest.instructions` (the agent system prompt field) before
    its parts. Tool calls/returns are rendered as readable prose. Multi-modal items
    become `<media>` (lossy).
    """
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest) and msg.instructions:
            lines.append(msg.instructions)
        for part in msg.parts:
            text = _render_part_flat(part, include_thinking=include_thinking)
            if text is not None:
                lines.append(text)
    return "\n".join(lines)


def _render_part_flat(part: Any, *, include_thinking: bool) -> str | None:
    """Render a single part to flat prose. Returns None to skip (e.g. thinking when disabled)."""
    if isinstance(part, SystemPromptPart):
        return part.content
    if isinstance(part, UserPromptPart):
        return _user_prompt_text(part.content)
    if isinstance(part, TextPart):
        return part.content
    if isinstance(part, ThinkingPart):
        return part.content if include_thinking else None
    if isinstance(part, (ToolCallPart, NativeToolCallPart)):
        return f"assistant called tool {part.tool_name} with {stringify(part.args)}"
    if isinstance(part, (ToolReturnPart, NativeToolReturnPart)):
        return f"tool {part.tool_name} returned {stringify(part.content)}"
    if isinstance(part, RetryPromptPart):
        return f"tool {part.tool_name} retry: {stringify(part.content)}"
    if isinstance(part, InstructionPart):
        return part.content
    if isinstance(part, CompactionPart):
        return part.content or "<encrypted compaction>"
    if isinstance(part, FilePart):
        return "<media>"
    return stringify(part)  # future part types


def render_structured(messages: Sequence[ModelMessage]) -> str:
    """XML-ish structured rendering of messages, used as the summarizer input.

    Each part becomes a `<message>` tagged with its role; tool calls/returns carry
    structured attributes (`tool_call`, `tool_name`, `tool_call_id`, `arguments`).
    Wrapped in `<conversation-history>`. Multi-modal items become `<media>`.
    """
    inner: list[str] = ["<conversation-history>"]
    for msg in messages:
        if isinstance(msg, ModelRequest) and msg.instructions:
            inner.append(f'  <message role="system">{escape(msg.instructions)}</message>')
        for part in msg.parts:
            inner.append(f"  {_render_part_structured(part)}")
    inner.append("</conversation-history>")
    return "\n".join(inner)


def _render_part_structured(part: Any) -> str:
    if isinstance(part, SystemPromptPart):
        return f'<message role="system">{escape(part.content)}</message>'
    if isinstance(part, UserPromptPart):
        return f'<message role="user">{escape(_user_prompt_text(part.content))}</message>'
    if isinstance(part, TextPart):
        return f'<message role="assistant">{escape(part.content)}</message>'
    if isinstance(part, ThinkingPart):
        return f'<message role="assistant" thinking>{escape(part.content)}</message>'
    if isinstance(part, ToolCallPart | NativeToolCallPart):
        args = stringify(part.args)
        # quoteattr picks single quotes when the value contains double quotes (e.g. JSON),
        # so `arguments='{"city": "SF"}'` is valid.
        return f'<message role="assistant" tool_call={quoteattr(part.tool_name)} arguments={quoteattr(args)} />'
    if isinstance(part, ToolReturnPart | NativeToolReturnPart):
        return (
            f'<message role="tool" tool_name={quoteattr(part.tool_name)}'
            f" tool_call_id={quoteattr(part.tool_call_id)}>"
            f"{escape(stringify(part.content))}</message>"
        )
    if isinstance(part, RetryPromptPart):
        return (
            f'<message role="tool" tool_name={quoteattr(part.tool_name or "")} retry>'
            f"{escape(stringify(part.content))}</message>"
        )
    if isinstance(part, InstructionPart):
        return f'<message role="system">{escape(part.content)}</message>'
    if isinstance(part, CompactionPart):
        if part.content is None:
            return '<message role="system" compaction />'
        return f'<message role="system" compaction>{escape(part.content)}</message>'
    if isinstance(part, FilePart):
        return '<message role="user"><media /></message>'
    return f'<message role="unknown">{escape(stringify(part))}</message>'  # pragma: no cover
