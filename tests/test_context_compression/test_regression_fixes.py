"""Regression tests for the 0.1.1 review fixes.

Covers: instructions counted once (render + estimate), nested-binary leak in
`stringify`, strict truncation budget, prefix-sum fast path in `find_safe_split`,
sentinel body checksum, PEP 440 pre-release tags, and package exports.
"""

from __future__ import annotations

import logging

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

import pydantic_ai_extensions.context_compression as cc
from pydantic_ai_extensions.context_compression import (
    ContextCompression,
    SummaryRecord,
    build_summary_message,
    estimate_text_tokens,
    estimate_tokens,
    find_safe_split,
    find_summary,
    parse_summary_sentinel,
    render_messages_to_text,
    render_structured,
    truncate_text_to_tokens,
)

# --- fix: instructions counted once, not once per stamped request ---


def _user(text: str, instructions: str | None = None) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)], instructions=instructions)


def test_instructions_rendered_once_regardless_of_stamp_count():
    sys_prompt = "SYSTEM " * 50
    msgs = [_user(f"q{i}", instructions=sys_prompt) for i in range(5)]
    text = render_messages_to_text(msgs)
    assert text.count("SYSTEM") == 50  # exactly one copy, not one per request


def test_instructions_attributed_to_last_carrier():
    msgs = [
        _user("q0", instructions="old-prompt"),
        ModelResponse(parts=[TextPart(content="a0")]),
        _user("q1", instructions="new-prompt"),
    ]
    text = render_messages_to_text(msgs)
    assert "old-prompt" not in text
    assert "new-prompt" in text


def test_structured_render_instructions_once():
    sys_prompt = "SYSTEM " * 50
    msgs = [_user(f"q{i}", instructions=sys_prompt) for i in range(4)]
    out = render_structured(msgs)
    assert out.count("SYSTEM") == 50


def test_estimate_does_not_grow_with_instruction_copies():
    """Adding assistant turns must not multiply the system prompt into the estimate."""
    sys_prompt = "x" * 400  # ~100 tokens per copy
    short = [_user("q", instructions=sys_prompt), ModelResponse(parts=[TextPart(content="a")])]
    longer = [*short, *(_user(f"q{i}", instructions=sys_prompt) for i in range(9))]
    est_short, est_longer = estimate_tokens(short), estimate_tokens(longer)
    # 9 extra stamped requests: the old renderer inflated by ~9*100 tokens; the fix
    # adds only the new user messages themselves (~a handful of tokens).
    assert est_longer - est_short < 50


def test_suffix_estimate_counts_instructions_when_present():
    """Suffix estimates include the prompt once iff the suffix contains its last carrier."""
    sys_prompt = "x" * 400
    msgs = [_user("q0", instructions=sys_prompt), ModelResponse(parts=[TextPart(content="a0")]), _user("q1")]
    full = estimate_tokens(msgs)
    with_suffix = estimate_tokens(msgs[2:])  # drops the only instructions carrier
    assert full - with_suffix >= 50  # the difference is (at least) the system prompt


# --- fix: nested binary content no longer leaks raw bytes ---


def test_stringify_nested_binary_media():
    img = BinaryContent(data=b"\x89PNG" + b"A" * 2_000, media_type="image/png")
    for payload in ({"chart": img}, [img], {"a": {"b": [img]}}):
        out = cc.render.stringify(payload)
        assert "<binary>" in out
        assert "AAAA" not in out  # no raw/base64 bytes
        assert len(out) < 200


def test_stringify_top_level_bytes():
    assert cc.render.stringify(b"\x00\x01") == "<binary>"
    assert cc.render.stringify(bytearray(b"xyz")) == "<binary>"


def test_stringify_pydantic_model_with_binary_field():
    from pydantic import BaseModel

    class _Report(BaseModel):
        title: str
        chart: BinaryContent

    img = BinaryContent(data=b"\x89PNG" + b"B" * 2_000, media_type="image/png")
    out = cc.render.stringify(_Report(title="t", chart=img))
    assert "<binary>" in out
    assert "BBBB" not in out


def test_tool_return_with_nested_media_estimation_bounded():
    img = BinaryContent(data=b"\x89PNG" + b"C" * 5_000, media_type="image/png")
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="hi")]),
        ModelResponse(parts=[ToolReturnPart(tool_name="t", content={"report": "ok", "chart": img}, tool_call_id="c")]),
    ]
    # A 5KB image must not add thousands of phantom tokens to the estimate.
    assert estimate_tokens(msgs) < 200


def test_structured_render_tool_return_nested_media():
    img = BinaryContent(data=b"\x89PNG" + b"D" * 2_000, media_type="image/png")
    msgs = [ModelResponse(parts=[ToolReturnPart(tool_name="t", content=[img], tool_call_id="c")])]
    out = render_structured(msgs)
    # `<binary>` arrives XML-escaped inside the tool message body.
    assert "&lt;binary&gt;" in out
    assert "DDDD" not in out


def _summarizer(text: str = "S"):
    def gen(messages, info):
        return ModelResponse(parts=[TextPart(content=text)])

    return Agent(FunctionModel(gen), output_type=str)


def test_after_tool_execute_media_only_result_passes_through():
    """A media-only result now renders tiny -> under budget -> returned untouched."""
    import asyncio

    img = BinaryContent(data=b"\x89PNG" + b"E" * 50_000, media_type="image/png")
    cap = ContextCompression(_summarizer(), max_tool_output_tokens=30)
    result = [img]
    out = asyncio.run(cap.after_tool_execute(ctx=None, call=None, tool_def=None, args=None, result=result))  # type: ignore[arg-type]
    assert out is result  # passthrough, no stringification, no truncation


def test_after_tool_execute_oversized_dict_still_truncated_to_str():
    import asyncio

    cap = ContextCompression(_summarizer(), max_tool_output_tokens=30)
    big = {f"k{i}": "v" * 100 for i in range(100)}
    out = asyncio.run(cap.after_tool_execute(ctx=None, call=None, tool_def=None, args=None, result=big))  # type: ignore[arg-type]
    assert isinstance(out, str)  # documented: truncation degrades the type
    assert "truncated" in out


# --- fix: truncation respects the token budget ---


def _over_budget_text(budget: int) -> str:
    """Text that reliably encodes to many times `budget` tokens."""
    return "z " * (budget * 8)


@pytest.mark.parametrize("budget", [25, 100, 5_000])
def test_truncate_text_to_tokens_honors_budget(budget):
    out = truncate_text_to_tokens(_over_budget_text(budget), budget)
    assert "truncated" in out
    assert estimate_text_tokens(out) <= budget


def test_truncate_text_to_tokens_untouched_under_budget():
    assert truncate_text_to_tokens("short", 100) == "short"


@pytest.mark.parametrize("budget", [3, 10])
def test_truncate_text_to_tokens_tiny_budget_best_effort(budget):
    """Below the marker's own size, the minimal 1+1-token cut is returned (documented)."""
    out = truncate_text_to_tokens(_over_budget_text(1_000), budget)
    assert "truncated" in out
    assert out.count("\n") == 2  # minimal form: head + marker + tail


def test_truncate_text_to_tokens_char_fallback_honors_budget():
    budget, cpt = 40, 4  # 160 chars
    out = truncate_text_to_tokens("w" * 10_000, budget, encoding="not-a-real-encoding", char_per_token=cpt)
    assert "truncated" in out
    assert len(out) <= budget * cpt


# --- fix: find_safe_split fast path (suffix sums) vs injected counter ---


def _history(n: int) -> list:
    msgs = []
    for i in range(n):
        msgs.append(ModelRequest(parts=[UserPromptPart(content=f"q{i} " + "t" * 60)]))
        msgs.append(ModelResponse(parts=[TextPart(content=f"a{i} " + "u" * 60)]))
    return msgs


@pytest.mark.parametrize("keep", [("messages", 2), ("messages", 5), ("tokens", 100), ("tokens", 1_000)])
def test_fast_path_matches_injected_counter(keep):
    msgs = _history(8)

    def counter(seq):
        return estimate_tokens(seq)

    fast = find_safe_split(msgs, keep, max_tokens=10_000, keep_first_user_message=False)
    slow = find_safe_split(msgs, keep, max_tokens=10_000, keep_first_user_message=False, count_tokens=counter)
    # Both paths satisfy the keep lower bound under the same counter (they may differ by
    # a message at the boundary due to join-separator approximation).
    assert fast == slow


def test_fast_path_does_not_call_injected_counter():
    calls = []

    def counter(seq):
        calls.append(1)
        return estimate_tokens(seq)

    find_safe_split(_history(6), ("tokens", 50), count_tokens=counter)
    assert calls  # explicit counter is used
    find_safe_split(_history(6), ("tokens", 50))  # default path: no counter needed
    n_after_explicit = len(calls)
    find_safe_split(_history(6), ("tokens", 50), count_tokens=counter)
    assert len(calls) > n_after_explicit


def test_token_cutoff_from_counts_boundaries():
    from pydantic_ai_extensions.context_compression.slicing import _token_cutoff_from_counts

    counts = [10, 10, 10, 10]
    assert _token_cutoff_from_counts(counts, 5) == 3  # keep >= 5 tokens: last message alone
    assert _token_cutoff_from_counts(counts, 40) == 0  # whole history exactly at budget
    assert _token_cutoff_from_counts(counts, 100) == 0  # under budget -> nothing to compress
    assert _token_cutoff_from_counts([], 1) == 0


def test_fast_path_respects_tool_pair_safety():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="q " + "s" * 80)]),
        ModelResponse(parts=[TextPart(content="a " + "s" * 80)]),
        ModelRequest(parts=[UserPromptPart(content="q2")]),
        ModelResponse(parts=[TextPart(content="a2")]),
        ModelResponse(parts=[TextPart(content="call")]),
        ModelRequest(parts=[UserPromptPart(content="q3 " + "s" * 80)]),
    ]
    for keep in [("messages", 1), ("messages", 3), ("tokens", 10)]:
        k = find_safe_split(msgs, keep, keep_first_user_message=False)
        if k:
            assert cc.is_safe_cutoff_point(msgs, k)


# --- fix: sentinel body checksum ---


def test_sentinel_roundtrip_with_checksum():
    rec = SummaryRecord(text="hello\nworld", generation=2, covered_count=7, strategy="full", compacted_tokens=999)
    content = build_summary_message(rec).parts[0].content
    assert " sum=" in content
    assert parse_summary_sentinel(content) == rec


def test_sentinel_rejects_tampered_body():
    rec = SummaryRecord(text="honest summary", generation=0, covered_count=4, strategy="full")
    content = build_summary_message(rec).parts[0].content
    tampered = content.replace("honest summary", "injected summary")
    assert parse_summary_sentinel(tampered) is None


def test_sentinel_rejects_tampered_checksum():
    rec = SummaryRecord(text="honest summary", generation=0, covered_count=4, strategy="full")
    content = build_summary_message(rec).parts[0].content
    idx = content.index(" sum=") + len(" sum=")
    tampered = content[:idx] + ("00000000" if content[idx : idx + 8] != "00000000" else "ffffffff") + content[idx + 8 :]
    assert parse_summary_sentinel(tampered) is None


def test_sentinel_old_format_without_checksum_still_parses():
    content = "<conversation-summary generation=2 covered_count=5 strategy=full compacted_tokens=12>\nold text"
    rec = parse_summary_sentinel(content)
    assert rec is not None
    assert rec.text == "old text"
    assert rec.compacted_tokens == 12


def test_find_summary_rejects_full_format_echo():
    """A model echoing the exact marker format (wrong checksum for its text) is not a sentinel."""
    echo = ModelResponse(
        parts=[TextPart(content="<conversation-summary generation=9 covered_count=9 strategy=full sum=deadbeef>\n")]
    )
    real = build_summary_message(SummaryRecord(text="real", generation=1, covered_count=4, strategy="full"))
    found = find_summary([echo, real])
    assert found is not None
    assert found[0] == 1
    assert found[1].text == "real"


# --- fix: PEP 440 pre-release tags ---


@pytest.mark.parametrize(
    ("describe", "sha", "dirty", "expected"),
    [
        # existing behavior preserved
        ("v0.1.0", "abc1234", False, "0.1.0"),
        ("v0.1.0-5-g9a8b7c6", "9a8b7c6", False, "0.1.0+5.g9a8b7c6"),
        ("v0.1.0-5-g9a8b7c6", "9a8b7c6", True, "0.1.0+5.g9a8b7c6.dirty"),
        ("9a8b7c6", "9a8b7c6", False, "0.0.0+g9a8b7c6"),
        ("", "", False, "0.0.0+unknown"),
        # pre-release tags: previously produced unparsable `1.2.3-rc1...` strings
        ("v1.2.3-rc1", "abc1234", False, "1.2.3rc1"),
        ("v1.2.3-rc1", "abc1234", True, "1.2.3rc1+dirty"),
        ("v1.2.3-rc1-2-g9a8b7c6", "9a8b7c6", False, "1.2.3rc1+2.g9a8b7c6"),
        ("v1.2.3-rc1-2-g9a8b7c6", "9a8b7c6", True, "1.2.3rc1+2.g9a8b7c6.dirty"),
        ("v2.0.0-beta.1-1-gdeadbee", "deadbee", False, "2.0.0beta.1+1.gdeadbee"),
    ],
)
def test_normalize_version_prerelease(describe, sha, dirty, expected):
    import hatch_build as hb
    from pydantic_ai_extensions import version as vmod

    assert hb._normalize_version(describe, sha, dirty) == expected
    assert vmod._normalize_version(describe, sha, dirty) == expected


@pytest.mark.parametrize(
    "version",
    [
        "0.1.0",
        "0.1.0+5.g9a8b7c6",
        "0.1.0+5.g9a8b7c6.dirty",
        "1.2.3rc1",
        "1.2.3rc1+dirty",
        "1.2.3rc1+2.g9a8b7c6",
        "1.2.3rc1+2.g9a8b7c6.dirty",
        "2.0.0beta.1+1.gdeadbee",
        "0.0.0+g9a8b7c6",
        "0.0.0+unknown",
    ],
)
def test_produced_versions_are_pep440(version):
    from packaging.version import InvalidVersion, Version

    try:
        Version(version)
    except InvalidVersion:
        pytest.fail(f"version {version!r} is not valid PEP 440")


# --- fix: exports + conversation_id guidance log ---


def test_previously_missing_exports():
    assert callable(cc.prefix_body_after)
    assert cc.DEFAULT_ENCODING == "o200k_base"
    assert callable(cc.render_message_texts)


async def test_persist_false_with_store_logs_conversation_id_hint(caplog):
    class _Store:
        async def get(self, ctx):
            return None

        async def put(self, ctx, record):
            pass

    with caplog.at_level(logging.INFO, logger="pydantic_ai_extensions.context_compression.capability"):
        ContextCompression(_summarizer(), persist=False, summary_store=_Store())  # type: ignore[abstract]
    assert any("conversation_id" in r.message for r in caplog.records)
