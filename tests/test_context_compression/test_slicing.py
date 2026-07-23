"""Tests for slicing.py (§5.5) -- the safe-boundary + tool-pairing invariants."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_extensions.context_compression.slicing import (
    find_safe_split,
    is_safe_cutoff_point,
    partition_head,
)


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _call(name: str, call_id: str, args: dict | None = None) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args or {}, tool_call_id=call_id)])


def _return(name: str, call_id: str, content: str = "ok") -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name=name, content=content, tool_call_id=call_id)])


# --- is_safe_cutoff_point ---


def test_unsafe_when_call_in_prefix_return_in_recent():
    msgs = [_user("q"), _call("t", "c1"), _return("t", "c1"), _assistant("a"), _user("q2")]
    # k=2: prefix=[user, call], recent=[return, asst, user] -> call orphaned from return.
    assert is_safe_cutoff_point(msgs, 2) is False
    # k=3: prefix=[user, call, return], recent=[asst, user] -> pair intact.
    assert is_safe_cutoff_point(msgs, 3) is True
    # k=1: prefix=[user], recent=[call, return, asst, user] -> call+return both in recent.
    assert is_safe_cutoff_point(msgs, 1) is True


def test_boundaries_are_safe():
    msgs = [_user("q"), _call("t", "c1"), _return("t", "c1")]
    assert is_safe_cutoff_point(msgs, 0) is True  # empty prefix
    assert is_safe_cutoff_point(msgs, len(msgs)) is True  # empty recent


def test_native_tool_parts_paired_by_tool_call_id():
    msgs = [
        _user("q"),
        ModelResponse(parts=[NativeToolCallPart(tool_name="search", args={"q": "x"}, tool_call_id="n1")]),
        ModelRequest(parts=[NativeToolReturnPart(tool_name="search", content="result", tool_call_id="n1")]),
        _user("q2"),
    ]
    assert is_safe_cutoff_point(msgs, 2) is False  # native call in prefix, return in recent
    assert is_safe_cutoff_point(msgs, 3) is True


def test_retry_prompt_counts_as_return_side():
    msgs = [
        _user("q"),
        _call("t", "c1"),
        ModelRequest(parts=[RetryPromptPart(content="bad args", tool_name="t", tool_call_id="c1")]),
        _user("q2"),
    ]
    # k=2: call in prefix, retry (return-side) in recent -> unsafe.
    assert is_safe_cutoff_point(msgs, 2) is False
    # k=3: both in prefix -> safe.
    assert is_safe_cutoff_point(msgs, 3) is True


# --- find_safe_split ---


def test_messages_keep_returns_safe_cutoff():
    msgs = [_user("q"), _call("t", "c1"), _return("t", "c1"), _assistant("a"), _user("q2")]
    # keep last 2 messages: target = 5-2 = 3 (safe), so k=3.
    k = find_safe_split(msgs, ("messages", 2))
    assert k == 3
    recent = msgs[k:]
    assert len(recent) == 2
    assert recent[-1] is msgs[-1]  # current request retained


def test_messages_keep_walks_back_to_safe_point():
    msgs = [_user("q"), _call("t", "c1"), _return("t", "c1"), _user("q2")]
    # keep last 2: target = 4-2 = 2 -> unsafe (call in prefix, return in recent) -> walk back to 1.
    # keep_first_user_message=False so k=1 is allowed (otherwise k<=1 -> 0).
    k = find_safe_split(msgs, ("messages", 2), keep_first_user_message=False)
    assert k == 1
    assert is_safe_cutoff_point(msgs, k) is True
    # With keep_first_user_message=True (default), k=1 collapses to 0 (no meaningful prefix).
    assert find_safe_split(msgs, ("messages", 2)) == 0


def test_keep_first_user_message_returns_zero_when_prefix_too_small():
    msgs = [_user("q"), _assistant("a"), _user("q2")]
    # keep last 2: target = 3-2 = 1 -> keep_first_user_message and k<=1 -> 0.
    assert find_safe_split(msgs, ("messages", 2), keep_first_user_message=True) == 0
    # Without keeping first user message, k=1 is allowed.
    assert find_safe_split(msgs, ("messages", 2), keep_first_user_message=False) == 1


def test_messages_zero_means_minimum_safe_window():
    msgs = [_user("q"), _call("t", "c1"), _return("t", "c1"), _user("q2")]
    # keep ('messages', 0): no lower bound -> keep minimum safe window.
    k = find_safe_split(msgs, ("messages", 0))
    # target = n-1 = 3 (keep last 1). is_safe(3)? prefix=[user,call,return], recent=[user] -> safe.
    assert k == 3
    assert is_safe_cutoff_point(msgs, k) is True


def test_tokens_keep_respects_budget():
    msgs = [_user("x" * 50) for _ in range(10)]
    # Small budget: total tokens exceed it, so a non-trivial recent window is kept and k > 0.
    k = find_safe_split(msgs, ("tokens", 5))
    assert 0 < k < len(msgs)
    assert is_safe_cutoff_point(msgs, k) is True


def test_tokens_keep_returns_zero_when_history_under_budget():
    msgs = [_user("short"), _assistant("reply")]
    k = find_safe_split(msgs, ("tokens", 100_000))
    assert k == 0  # whole history under budget -> nothing to compress


def test_fraction_keep_uses_max_tokens():
    msgs = [_user("x" * 200) for _ in range(10)]
    # Small max_tokens -> small budget -> small recent window -> large cutoff.
    k_small_budget = find_safe_split(msgs, ("fraction", 0.5), max_tokens=100)
    # Huge max_tokens -> huge budget -> recent must be huge -> cutoff 0 (under budget).
    k_huge_budget = find_safe_split(msgs, ("fraction", 0.5), max_tokens=10_000_000)
    assert k_small_budget > k_huge_budget
    assert k_huge_budget == 0


def test_empty_messages_returns_zero():
    assert find_safe_split([], ("messages", 5)) == 0


@pytest.mark.parametrize(
    "msgs",
    [
        # call/return straddling various positions
        [_user("q"), _call("t", "c1"), _return("t", "c1"), _user("q2")],
        [_user("q"), _call("t", "c1"), _call("t", "c2"), _return("t", "c1"), _return("t", "c2"), _user("q2")],
        [
            _user("q"),
            _call("t", "c1"),
            _return("t", "c1"),
            _assistant("a"),
            _call("t", "c2"),
            _return("t", "c2"),
            _user("q2"),
        ],
    ],
    ids=["single-pair", "parallel-pairs", "two-turns"],
)
def test_property_no_orphaned_tool_call(msgs: list[ModelMessage]):
    """For any keep spec, find_safe_split never leaves a tool-call orphaned from its return."""
    for keep in [("messages", 1), ("messages", 2), ("messages", 3)]:
        k = find_safe_split(msgs, keep)
        if k == 0:
            continue  # nothing to compress
        assert is_safe_cutoff_point(msgs, k) is True, f"orphan at k={k} for keep={keep}"
        # recent always includes the current (last) request
        assert msgs[-1] in msgs[k:]


# --- partition_head ---


def test_partition_head_keeps_first_with_instructions():
    prefix = [
        ModelRequest(parts=[UserPromptPart(content="first")], instructions="sys"),
        _assistant("a"),
    ]
    head, body = partition_head(prefix, keep_first_user_message=True)
    assert len(head) == 1
    assert head[0].instructions == "sys"
    assert len(body) == 1


def test_partition_head_empty_when_not_kept():
    prefix = [ModelRequest(parts=[UserPromptPart(content="first")]), _assistant("a")]
    head, body = partition_head(prefix, keep_first_user_message=False)
    assert head == []
    assert len(body) == 2


def test_partition_head_empty_prefix():
    head, body = partition_head([], keep_first_user_message=True)
    assert head == [] and body == []
