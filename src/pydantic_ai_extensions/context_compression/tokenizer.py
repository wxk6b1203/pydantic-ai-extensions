"""Token estimation and trigger logic (§5.4).

`ContextSize` is the shared spec for both the compression trigger
(``compress_threshold``) and the recent-window keep (``keep``):
``('messages', N)`` / ``('tokens', N)`` / ``('fraction', F)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic_ai.messages import ModelMessage

from .render import render_messages_to_text

__all__ = [
    "DEFAULT_ENCODING",
    "DEFAULT_MAX_TOKENS",
    "ContextSize",
    "estimate_text_tokens",
    "estimate_tokens",
    "should_trigger",
    "truncate_text_to_tokens",
]

ContextSize = tuple[Literal["messages"], int] | tuple[Literal["tokens"], int] | tuple[Literal["fraction"], float]
"""Trigger threshold / keep-window spec (borrowed from summarization-pydantic-ai)."""

DEFAULT_MAX_TOKENS = 128_000
"""Fallback for `fraction` mode when `max_tokens` is unknown.

pydantic-ai's model profiles do not expose the input context window, so true
auto-detection needs an external source (e.g. genai-prices). Until then, callers
should pass `max_tokens` explicitly; otherwise this conservative default is used.
"""

DEFAULT_ENCODING = "o200k_base"
"""tiktoken encoding for OpenAI/DeepSeek-family models (approximate for others)."""


def estimate_text_tokens(
    text: str,
    *,
    encoding: str | None = None,
    char_per_token: int = 4,
) -> int:
    """Estimate the token count of a plain string (for tool-output truncation)."""
    enc_name = encoding or DEFAULT_ENCODING
    try:
        import tiktoken

        enc = tiktoken.get_encoding(enc_name)
        return len(enc.encode(text, disallowed_special=()))
    except (ImportError, ValueError):
        return max(1, len(text) // char_per_token)


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    *,
    encoding: str | None = None,
    char_per_token: int = 4,
) -> str:
    """Middle-cut `text` so it fits within `max_tokens` (head + marker + tail).

    Token-precise when tiktoken is available; falls back to a character budget
    (``max_tokens * char_per_token``) otherwise. Used as the last-resort truncation
    for tool outputs and for capping summary length.
    """
    enc_name = encoding or DEFAULT_ENCODING
    try:
        import tiktoken

        enc = tiktoken.get_encoding(enc_name)
        tokens = enc.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        half = max(max_tokens // 2, 1)
        head, tail = enc.decode(tokens[:half]), enc.decode(tokens[len(tokens) - half :])
        return f"{head}\n...[truncated ~{len(tokens) - 2 * half} tokens]...\n{tail}"
    except (ImportError, ValueError):
        budget = max_tokens * char_per_token
        if len(text) <= budget:
            return text
        half = max(budget // 2, 1)
        return f"{text[:half]}\n...[truncated {len(text) - 2 * half} chars]...\n{text[len(text) - half :]}"


def estimate_tokens(
    messages: Sequence[ModelMessage],
    *,
    encoding: str | None = None,
    char_per_token: int = 4,
    include_thinking: bool = True,
) -> int:
    """Estimate the token count of `messages`.

    Uses tiktoken with the given encoding (default ``o200k_base``); falls back to
    ``len(text) // char_per_token`` if tiktoken is unavailable or the encoding is
    unknown. Includes `ModelRequest.instructions`.
    """
    text = render_messages_to_text(messages, include_thinking=include_thinking)
    enc_name = encoding or DEFAULT_ENCODING
    try:
        import tiktoken

        enc = tiktoken.get_encoding(enc_name)
        return len(enc.encode(text, disallowed_special=()))
    except (ImportError, ValueError):
        return max(1, len(text) // char_per_token)


def should_trigger(
    messages: Sequence[ModelMessage],
    compress_threshold: ContextSize,
    max_tokens: int | None,
    *,
    encoding: str | None = None,
    char_per_token: int = 4,
    include_thinking: bool = True,
) -> bool:
    """Whether compression should fire, per the `ContextSize` spec."""
    match compress_threshold:
        case ("messages", count):
            return len(messages) >= count
        case ("tokens", budget):
            return (
                estimate_tokens(
                    messages,
                    encoding=encoding,
                    char_per_token=char_per_token,
                    include_thinking=include_thinking,
                )
                >= budget
            )
        case ("fraction", frac):
            budget = int((max_tokens or DEFAULT_MAX_TOKENS) * frac)
            return (
                estimate_tokens(
                    messages,
                    encoding=encoding,
                    char_per_token=char_per_token,
                    include_thinking=include_thinking,
                )
                >= budget
            )
        case _:
            raise ValueError(f"Invalid compress_threshold: {compress_threshold!r}")
