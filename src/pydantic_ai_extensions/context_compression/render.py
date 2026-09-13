"""Message renderers for context compression.

Two independent renderers:
- `render_messages_to_text`: flat prose rendering, used only for token estimation
  (§5.4). Includes the agent system prompt (`ModelRequest.instructions`) *once* --
  the framework stamps instructions onto every request, but the wire format sends
  them as a single system prompt per API call.
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

__all__ = ["render_message_texts", "render_messages_to_text", "render_structured", "stringify"]


def _is_media_like(x: Any) -> bool:
    """Duck-type BinaryContent (any object with `data` + `media_type`)."""
    return hasattr(x, "data") and hasattr(x, "media_type")


def _json_default(o: Any) -> str:
    if _is_media_like(o) or isinstance(o, (bytes, bytearray, memoryview)):
        return "<binary>"
    return str(o)


def stringify(x: Any) -> str:
    """Best-effort text rendering of a part payload (args/content).

    Binary media (`BinaryContent` or anything duck-typed by `data`+`media_type`) --
    including media nested inside dicts/lists/pydantic models -- renders as
    ``<binary>`` so raw bytes never inflate the text (which would skew token
    estimates and leak into truncation output / the summarizer prompt).
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray, memoryview)) or _is_media_like(x):
        return "<binary>"
    if isinstance(x, (dict, list, tuple)):
        y = cast(Any, x)
        try:
            return json.dumps(y, ensure_ascii=False, default=_json_default)
        except (TypeError, ValueError):
            return str(y)
    # pydantic models / dataclasses / other objects
    if hasattr(x, "model_dump_json"):
        try:
            return json.dumps(x.model_dump(), ensure_ascii=False, default=_json_default)
        except Exception:
            return x.model_dump_json()  # type: ignore[no-any-return]
    return str(x)


def _user_prompt_text(content: str | Sequence[Any]) -> str:
    """Extract text from `UserPromptPart.content` (str or sequence of UserContent)."""
    if isinstance(content, str):
        return content
    if _is_media_like(content):
        return "<media>"  # BinaryContent (duck-typed)
    # Sequence of UserContent items (str, TextContent, ImageUrl, BinaryContent, ...)
    if isinstance(content, (list, tuple)):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif _is_media_like(item):
                chunks.append("<media>")
            elif hasattr(item, "content") and isinstance(item.content, str):
                # TextContent (duck-typed)
                chunks.append(item.content)  # type: ignore[union-attr]
            else:
                chunks.append("<media>")
        return "\n".join(chunks)
    return stringify(content)


def render_message_texts(messages: Sequence[ModelMessage], *, include_thinking: bool = True) -> list[str]:
    """Render each message to a flat prose string (one entry per message).

    The agent system prompt (`ModelRequest.instructions`) is included exactly once --
    attached to the *last* request that carries it. pydantic-ai stamps instructions
    onto every request it appends to the history, but providers receive them as a
    single system prompt per API call, so counting each stamped copy would inflate
    the estimate in proportion to the number of turns. Attributing the prompt to the
    last carrier keeps every suffix estimate (``messages[k:]``) correct too: a suffix
    contains a carrier iff it contains the global last carrier.
    """
    last_instructions_idx = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest) and msg.instructions:
            last_instructions_idx = i

    texts: list[str] = []
    for i, msg in enumerate(messages):
        lines: list[str] = []
        if i == last_instructions_idx and isinstance(msg, ModelRequest) and msg.instructions:
            lines.append(msg.instructions)
        for part in msg.parts:
            text = _render_part_flat(part, include_thinking=include_thinking)
            if text is not None:
                lines.append(text)
        texts.append("\n".join(lines))
    return texts


def render_messages_to_text(messages: Sequence[ModelMessage], *, include_thinking: bool = True) -> str:
    """Flat prose rendering of messages, for token estimation only.

    Includes the agent system prompt (`ModelRequest.instructions`) once (see
    `render_message_texts`). Tool calls/returns are rendered as readable prose.
    Multi-modal items become `<media>` (lossy).
    """
    return "\n".join(render_message_texts(messages, include_thinking=include_thinking))


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
    Wrapped in `<conversation-history>`. Multi-modal items become `<media>`. The
    agent system prompt (`ModelRequest.instructions`) is included once (last carrier),
    mirroring `render_messages_to_text` -- the framework stamps it onto every request.
    """
    inner: list[str] = ["<conversation-history>"]
    last_instructions_idx = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest) and msg.instructions:
            last_instructions_idx = i
    for i, msg in enumerate(messages):
        if i == last_instructions_idx and isinstance(msg, ModelRequest) and msg.instructions:
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
