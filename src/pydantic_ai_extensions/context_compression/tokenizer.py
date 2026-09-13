"""Token estimation and trigger logic (§5.4).

`ContextSize` is the shared spec for both the compression trigger
(``compress_threshold``) and the recent-window keep (``keep``):
``('messages', N)`` / ``('tokens', N)`` / ``('fraction', F)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
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


@lru_cache(maxsize=8)
def _get_encoder(enc_name: str):
    """Cached tiktoken encoder lookup (unknown names raise ValueError, uncached)."""
    import tiktoken

    return tiktoken.get_encoding(enc_name)


def _encode(text: str, enc_name: str) -> list[int]:
    return _get_encoder(enc_name).encode(text, disallowed_special=())


def estimate_text_tokens(
    text: str,
    *,
    encoding: str | None = None,
    char_per_token: int = 4,
) -> int:
    """Estimate the token count of a plain string (for tool-output truncation)."""
    enc_name = encoding or DEFAULT_ENCODING
    try:
        return len(_encode(text, enc_name))
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

    The output is verified to fit the budget: the truncation marker itself costs
    tokens, so the head/tail halves shrink until the whole result fits. When the
    budget is smaller than the marker (a few tokens), the smallest representable
    form (1 head + marker + 1 tail token) is returned as a best effort.
    """
    enc_name = encoding or DEFAULT_ENCODING
    try:
        tokens = _encode(text, enc_name)
    except (ImportError, ValueError):
        budget = max_tokens * char_per_token
        if len(text) <= budget:
            return text
        half = budget // 2
        while half >= 1:
            out = _join_cut(text[:half], len(text) - 2 * half, "chars", text[len(text) - half :])
            if len(out) <= budget:
                return out
            half //= 2
        return _join_cut(text[:1], len(text) - 2, "chars", text[-1:])

    if len(tokens) <= max_tokens:
        return text
    enc = _get_encoder(enc_name)
    half = max_tokens // 2
    while half >= 1:
        out = _join_cut(
            enc.decode(tokens[:half]), len(tokens) - 2 * half, "tokens", enc.decode(tokens[len(tokens) - half :])
        )
        if len(_encode(out, enc_name)) <= max_tokens:
            return out
        half //= 2
    # Budget is smaller than the marker itself: best-effort minimal cut.
    return _join_cut(enc.decode(tokens[:1]), len(tokens) - 2, "tokens", enc.decode(tokens[len(tokens) - 1 :]))


def _join_cut(head: str, dropped: int, unit: str, tail: str) -> str:
    return f"{head}\n...[truncated ~{dropped} {unit}]...\n{tail}"


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
    unknown. The agent system prompt (`ModelRequest.instructions`) is counted once
    (see `render_message_texts`).
    """
    text = render_messages_to_text(messages, include_thinking=include_thinking)
    enc_name = encoding or DEFAULT_ENCODING
    try:
        return len(_encode(text, enc_name))
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
