"""Integration tests for ContextCompression (P3: persist=True, strategy='full')."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_extensions.context_compression import ContextCompression, parse_summary_sentinel
from pydantic_ai_extensions.context_compression.capability import _BLOCKED_MODEL_TYPES


def _summarizer(text: str = "SUMMARY."):
    """A summarizer agent backed by FunctionModel that always returns `text`."""

    def gen(messages, info):
        return ModelResponse(parts=[TextPart(content=text)])

    return Agent(FunctionModel(gen), output_type=str)


def _parent(cap: ContextCompression, reply: str = "ok"):
    def gen(messages, info):
        return ModelResponse(parts=[TextPart(content=reply)])

    return Agent(FunctionModel(gen), capabilities=[cap])


def _history_with_tool_pair() -> list:
    return [
        ModelRequest(parts=[UserPromptPart(content="q1")]),
        ModelResponse(parts=[TextPart(content="a1")]),
        ModelRequest(parts=[UserPromptPart(content="q2")]),
        ModelResponse(parts=[ToolCallPart(tool_name="t", args={}, tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="t", content="r", tool_call_id="c1")]),
        ModelResponse(parts=[TextPart(content="a2")]),
    ]


def _sentinel(messages: list) -> ModelResponse | None:
    for m in messages:
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, TextPart) and p.content.startswith("<conversation-summary"):
                    return m
    return None


async def test_compresses_when_threshold_exceeded():
    cap = ContextCompression(_summarizer("prior summary"), compress_threshold=("messages", 4), keep=("messages", 2))
    result = await _parent(cap).run("final q", message_history=_history_with_tool_pair())
    msgs = result.all_messages()
    sentinel = _sentinel(msgs)
    assert sentinel is not None
    # sentinel state lives in the TextPart content marker (not metadata), parsed via parse_summary_sentinel
    record = parse_summary_sentinel(sentinel.parts[0].content)
    assert record is not None
    assert record.strategy == "full"
    assert record.generation == 0
    assert "prior summary" in record.text


async def test_no_compression_below_threshold():
    cap = ContextCompression(_summarizer(), compress_threshold=("messages", 100), keep=("messages", 2))
    result = await _parent(cap).run("final q", message_history=_history_with_tool_pair())
    assert _sentinel(result.all_messages()) is None  # passthrough


async def test_tool_pair_preserved_in_recent_window():
    # keep the tool-call/return pair in the recent window (not summarized away).
    cap = ContextCompression(_summarizer("S"), compress_threshold=("messages", 3), keep=("messages", 4))
    result = await _parent(cap).run("final q", message_history=_history_with_tool_pair())
    msgs = result.all_messages()
    # The ToolCallPart (c1) and its ToolReturnPart (c1) must both be present post-compaction.
    call_ids = {
        p.tool_call_id for m in msgs if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)
    }
    return_ids = {
        p.tool_call_id for m in msgs if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)
    }
    assert call_ids  # the call survived
    assert call_ids == return_ids  # every surviving call has its return (no orphans)


async def test_summarizer_failure_degrades_to_no_compression():
    from pydantic_ai.exceptions import ModelHTTPError

    def bad_gen(messages, info):
        raise ModelHTTPError(status_code=500, model_name="fake-summarizer", body="boom")

    summarizer = Agent(FunctionModel(bad_gen), output_type=str)
    cap = ContextCompression(summarizer, compress_threshold=("messages", 4), keep=("messages", 2))
    result = await _parent(cap).run("final q", message_history=_history_with_tool_pair())
    # No sentinel -> compression was skipped gracefully, parent run unaffected.
    assert _sentinel(result.all_messages()) is None


async def test_summarizer_programming_error_propagates():
    """Bugs (not API failures) must surface, not silently disable compression (§5.9)."""

    def bad_gen(messages, info):
        raise RuntimeError("programming bug")

    summarizer = Agent(FunctionModel(bad_gen), output_type=str)
    cap = ContextCompression(summarizer, compress_threshold=("messages", 4), keep=("messages", 2))
    with pytest.raises(Exception, match=r"programming bug|boom"):
        await _parent(cap).run("final q", message_history=_history_with_tool_pair())


async def test_hybrid_generation_increments_and_resets():
    """Multi-run: generation increments across runs, and resets at full_reset_every."""
    cap = ContextCompression(
        _summarizer("SUMMARY"),
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        strategy="hybrid",
        full_reset_every=3,
        min_prefix=2,
        min_recompact_growth_tokens=0,  # disable cooldown: this test exercises the generation state machine
    )
    parent = _parent(cap)
    history = []
    for i in range(3):
        history.append(ModelRequest(parts=[UserPromptPart(content=f"q{i}")]))
        history.append(ModelResponse(parts=[TextPart(content=f"a{i}")]))
    gens: list[int | None] = []
    for turn in range(6):
        result = await parent.run(f"turn{turn}", message_history=history)
        msgs = result.all_messages()
        s = _sentinel(msgs)
        gens.append(parse_summary_sentinel(s.parts[0].content).generation if s else None)
        history = msgs
    # full_reset_every=3: full(0), incr(1), incr(2), incr(3), full(0), incr(1)
    assert gens == [0, 1, 2, 3, 0, 1]


# --- provider guard (§5.7) ---


@pytest.mark.skipif(not _BLOCKED_MODEL_TYPES, reason="no provider extras installed")
async def test_guard_rejects_blocked_model(monkeypatch):
    # Pick whichever blocked type is importable.
    try:
        from pydantic_ai.models.anthropic import AnthropicModel

        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        model = AnthropicModel("claude-sonnet-4-6")
        name = "AnthropicModel"
    except ImportError:
        from pydantic_ai.models.openai import OpenAIResponsesModel

        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        model = OpenAIResponsesModel("gpt-5.2")
        name = "OpenAIResponsesModel"
    cap = ContextCompression(_summarizer(), compress_threshold=("messages", 100))
    agent = Agent(model, capabilities=[cap])
    with pytest.raises(UserError, match="native compaction"):
        await agent.run("hi")  # for_run raises before any API call
    assert name in {t.__name__ for t in _BLOCKED_MODEL_TYPES}


@pytest.mark.skipif(not _BLOCKED_MODEL_TYPES, reason="no provider extras installed")
async def test_guard_allows_openai_chat_model(monkeypatch):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.tools import RunContext

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    cap = ContextCompression(_summarizer(), compress_threshold=("messages", 100))
    ctx = RunContext[Any](deps=None, model=OpenAIChatModel("gpt-5.2"), usage=RunUsage())  # type: ignore[arg-type]
    # for_run must NOT raise for OpenAIChatModel (no native compaction).
    out = await cap.for_run(ctx)
    assert out is cap


# --- constructor validation ---


def test_persist_true_with_store_raises():
    class _Store:
        async def get(self, ctx):
            return None

        async def put(self, ctx, record):
            pass

    with pytest.raises(UserError, match="summary_store"):
        ContextCompression(_summarizer(), persist=True, summary_store=_Store())  # type: ignore[arg-type]


def test_invalid_strategy_raises():
    with pytest.raises(UserError, match="strategy"):
        ContextCompression(_summarizer(), strategy="bogus")  # type: ignore[arg-type]


# --- after_tool_execute truncation (§5.10) ---


async def test_after_tool_execute_passthrough_when_disabled():
    cap = ContextCompression(_summarizer(), max_tool_output_tokens=None)
    out = await cap.after_tool_execute(ctx=None, call=None, tool_def=None, args=None, result="x" * 10_000)  # type: ignore[arg-type]
    assert out == "x" * 10_000


async def test_after_tool_execute_truncates_long_string():
    cap = ContextCompression(
        _summarizer(), max_tool_output_tokens=30, tool_output_head_lines=2, tool_output_tail_lines=2
    )
    result = "\n".join(f"line {i}" for i in range(50))
    out = await cap.after_tool_execute(ctx=None, call=None, tool_def=None, args=None, result=result)  # type: ignore[arg-type]
    assert isinstance(out, str)
    assert "truncated" in out
    assert "line 0" in out  # head retained
    assert "line 49" in out  # tail retained
    assert "line 25" not in out  # middle dropped


async def test_after_tool_execute_leaves_short_result_untouched():
    cap = ContextCompression(
        _summarizer(), max_tool_output_tokens=10_000, tool_output_head_lines=2, tool_output_tail_lines=2
    )
    out = await cap.after_tool_execute(ctx=None, call=None, tool_def=None, args=None, result="short")  # type: ignore[arg-type]
    assert out == "short"


async def test_after_tool_execute_binarycontent_untouched():
    from pydantic_ai.messages import BinaryContent

    cap = ContextCompression(_summarizer(), max_tool_output_tokens=1)
    binary = BinaryContent(data=b"\x89PNG", media_type="image/png")
    out = await cap.after_tool_execute(ctx=None, call=None, tool_def=None, args=None, result=binary)  # type: ignore[arg-type]
    assert out is binary


# --- persist=False (request-time only, §5.6) ---


async def test_persist_false_only_model_sees_compacted():
    """In persist=False mode, the model receives compacted messages but state retains the full history."""
    model_input: list | None = None

    def parent_fn(messages, info):
        nonlocal model_input
        model_input = list(messages)
        return ModelResponse(parts=[TextPart(content="ok")])

    cap = ContextCompression(_summarizer("S"), compress_threshold=("messages", 4), keep=("messages", 2), persist=False)
    parent = Agent(FunctionModel(parent_fn), capabilities=[cap])
    result = await parent.run("final q", message_history=_history_with_tool_pair())
    # model saw the compacted messages (fewer than original history + current)
    assert model_input is not None
    assert len(model_input) < 1 + len(_history_with_tool_pair())
    # state retains the full original history (persist=False: before_model_request is passthrough)
    assert len(result.all_messages()) == 1 + len(_history_with_tool_pair()) + 1  # history + current + response


async def test_persist_false_store_drives_incremental():
    """A mock SummaryStore enables incremental across runs in persist=False mode."""
    from pydantic_ai_extensions.context_compression.store import SummaryRecord as SR

    records: dict[str, SR] = {}

    class _Store:
        async def get(self, ctx):
            return records.get(ctx.conversation_id)

        async def put(self, ctx, record):
            records[ctx.conversation_id] = record

    store = _Store()  # type: ignore[abstract]
    cap = ContextCompression(
        _summarizer("SUMMARY"),
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        strategy="hybrid",
        full_reset_every=3,
        min_prefix=2,
        min_recompact_growth_tokens=0,  # disable cooldown: exercise incremental every run
        persist=False,
        summary_store=store,
    )
    parent = _parent(cap)
    history = _history_with_tool_pair()
    gens: list[int | None] = []
    for turn in range(5):
        result = await parent.run(f"turn{turn}", message_history=history)
        msgs = result.all_messages()
        s = _sentinel(msgs)
        gens.append(parse_summary_sentinel(s.parts[0].content).generation if s else None)
        history = msgs
    # persist=False + store: state retains full history (sentinel is only model-visible) so
    # all_messages() won't carry a sentinel; generations are always None on the state side.
    # The store is exercised via put/get (incremental when store has prior record).
    assert all(g is None for g in gens)
    assert len(records) > 0


# --- strategy='full' always resets generation (§5.3) ---


async def test_strategy_full_always_uses_generation_zero():
    """With strategy='full', every compaction starts fresh (generation 0)."""
    cap = ContextCompression(
        _summarizer("SUMMARY"),
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        strategy="full",
        min_prefix=2,
        min_recompact_growth_tokens=0,  # disable cooldown: every run should compact
    )
    parent = _parent(cap)
    history = []
    for i in range(3):
        history.append(ModelRequest(parts=[UserPromptPart(content=f"q{i}")]))
        history.append(ModelResponse(parts=[TextPart(content=f"a{i}")]))
    gens: list[int | None] = []
    for _ in range(4):
        result = await parent.run("next", message_history=history)
        msgs = result.all_messages()
        s = _sentinel(msgs)
        gens.append(parse_summary_sentinel(s.parts[0].content).generation if s else None)
        history = msgs
    # strategy='full': every compaction is a full re-summarize -> generation always 0
    assert all(g == 0 for g in gens if g is not None)


# --- token-based keep + threshold integration tests ---


async def test_keep_tokens_based_cutoff():
    """Token-based keep produces a non-trivial recent window."""
    cap = ContextCompression(
        _summarizer("S"),
        compress_threshold=("messages", 6),
        keep=("tokens", 1),  # tiny budget: keep ~1 token of recent
        min_prefix=2,
    )
    parent = _parent(cap)
    history = []
    for i in range(4):
        history.append(ModelRequest(parts=[UserPromptPart(content=f"q{i}")]))
        history.append(ModelResponse(parts=[TextPart(content=f"a{i}")]))
    result = await parent.run("final", message_history=history)
    msgs = result.all_messages()
    s = _sentinel(msgs)
    assert s is not None
    rec = parse_summary_sentinel(s.parts[0].content)
    assert rec is not None
    assert rec.strategy == "full"  # first compaction is always full
    # cutoff was non-trivial (prefix was summarized, recent window exists)
    assert rec.covered_count > 0


async def test_compress_threshold_tokens_based():
    """`compress_threshold=('tokens', N)` fires when N tokens are exceeded."""
    msgs = []
    for i in range(8):
        msgs.append(ModelRequest(parts=[UserPromptPart(content=f"q{i}")]))
        msgs.append(ModelResponse(parts=[TextPart(content=f"a{i}")]))
    cap = ContextCompression(
        _summarizer("S"),
        compress_threshold=("tokens", 5),  # tiny: 16 short messages exceed 5 tokens
        keep=("messages", 2),
        min_prefix=2,
    )
    parent = _parent(cap)
    result = await parent.run("final", message_history=msgs)
    s = _sentinel(result.all_messages())
    assert s is not None  # compression triggered
