"""Tests for tokenizer.py (§5.4)."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from pydantic_ai_extensions.context_compression.tokenizer import (
    DEFAULT_MAX_TOKENS,
    estimate_tokens,
    should_trigger,
)


def _msgs(n: int) -> list:
    return [ModelRequest(parts=[UserPromptPart(content=f"message number {i}")]) for i in range(n)]


def test_estimate_tokens_positive_and_monotonic():
    small = _msgs(1)
    big = _msgs(20)
    assert estimate_tokens(small) > 0
    assert estimate_tokens(big) > estimate_tokens(small)


def test_estimate_tokens_char_fallback_on_unknown_encoding():
    msgs = _msgs(3)
    # An unknown encoding makes tiktoken.get_encoding raise ValueError -> char fallback.
    tokens = estimate_tokens(msgs, encoding="not-a-real-encoding", char_per_token=4)
    text_len = sum(len(f"message number {i}") for i in range(3))
    assert tokens == max(1, text_len // 4)


def test_estimate_tokens_include_thinking_changes_count():
    from pydantic_ai.messages import ThinkingPart

    msgs = [ModelResponse(parts=[ThinkingPart(content="x" * 200), TextPart(content="y")])]
    with_thinking = estimate_tokens(msgs, include_thinking=True)
    without_thinking = estimate_tokens(msgs, include_thinking=False)
    assert with_thinking > without_thinking


def test_should_trigger_messages_spec():
    msgs = _msgs(5)
    assert should_trigger(msgs, ("messages", 5), None) is True
    assert should_trigger(msgs, ("messages", 6), None) is False


def test_should_trigger_tokens_spec():
    msgs = _msgs(5)
    # 5 short messages are well under 10 tokens of text? No -- they're ~30+ chars.
    big = should_trigger(msgs, ("tokens", 1), None)
    small = should_trigger(msgs, ("tokens", 100_000), None)
    assert big is True
    assert small is False


def test_should_trigger_fraction_uses_max_tokens():
    msgs = _msgs(5)
    # With max_tokens small enough, fraction triggers.
    assert should_trigger(msgs, ("fraction", 0.5), max_tokens=2) is True
    # With max_tokens huge, fraction does not trigger.
    assert should_trigger(msgs, ("fraction", 0.5), max_tokens=10_000_000) is False


def test_should_trigger_fraction_falls_back_to_default_max_tokens():
    msgs = _msgs(50)
    # max_tokens=None -> DEFAULT_MAX_TOKENS (128k); 50 short messages are far under 0.9*128k.
    assert should_trigger(msgs, ("fraction", 0.9), None) is False
    # Sanity: the default is what we expect.
    assert DEFAULT_MAX_TOKENS == 128_000


def test_should_trigger_invalid_spec_raises():
    with pytest.raises(ValueError):
        should_trigger(_msgs(1), ("bytes", 10), None)  # type: ignore[arg-type]
