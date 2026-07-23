"""Hardening tests: cooldown, streaming, serde round-trip, usage merge, store contract,
truncation fallbacks, constructor validation, sentinel parsing (§5.3-§5.10)."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from pydantic_ai_extensions.context_compression import (
    ContextCompression,
    SummaryRecord,
    build_summary_message,
    estimate_text_tokens,
    find_summary,
    parse_summary_sentinel,
)
from pydantic_ai_extensions.context_compression.slicing import find_safe_split


def _summarizer(text: str = "SUMMARY."):
    """A summarizer agent backed by FunctionModel that always returns `text."""

    def gen(messages, info):
        return ModelResponse(parts=[TextPart(content=text)])

    return Agent(FunctionModel(gen), output_type=str)


def _parent(cap: ContextCompression, reply: str = "ok"):
    def gen(messages, info):
        return ModelResponse(parts=[TextPart(content=reply)])

    return Agent(FunctionModel(gen), capabilities=[cap])


def _history(n_pairs: int = 3) -> list:
    msgs = []
    for i in range(n_pairs):
        msgs.append(ModelRequest(parts=[UserPromptPart(content=f"q{i}")]))
        msgs.append(ModelResponse(parts=[TextPart(content=f"a{i}")]))
    return msgs


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


# --- re-compaction cooldown (§5.3) ---


def _tool_loop_setup(cap: ContextCompression):
    """Parent agent that calls a tool for the first 4 steps, then answers."""
    steps = 0

    def parent_gen(messages, info):
        nonlocal steps
        steps += 1
        if steps <= 4:
            return ModelResponse(parts=[ToolCallPart(tool_name="noop", args={}, tool_call_id=f"c{steps}")])
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(FunctionModel(parent_gen), capabilities=[cap])

    @agent.tool_plain
    def noop() -> str:
        return "y" * 300  # big-ish return keeps the history above the tiny threshold

    def count_steps() -> int:
        return steps

    return agent, count_steps


def _big_history() -> list:
    msgs = []
    for i in range(4):
        msgs.append(ModelRequest(parts=[UserPromptPart(content=f"q{i} " + "z" * 100)]))
        msgs.append(ModelResponse(parts=[TextPart(content=f"a{i} " + "w" * 100)]))
    return msgs


async def test_cooldown_prevents_recompaction_in_tool_loop():
    """With the cooldown active, a tool-loop run compacts once, not once per step."""
    summarize_calls = 0

    def summarizer_gen(messages, info):
        nonlocal summarize_calls
        summarize_calls += 1
        return ModelResponse(parts=[TextPart(content="SUMMARY " + "x" * 200)])

    cap = ContextCompression(
        Agent(FunctionModel(summarizer_gen), output_type=str),
        compress_threshold=("tokens", 50),  # tiny: compacted history stays above it
        keep=("messages", 2),
        min_prefix=2,
        min_recompact_growth_tokens=10_000,  # per-step growth (~100 tokens) never reaches this
    )
    agent, count_steps = _tool_loop_setup(cap)
    await agent.run("start", message_history=_big_history())
    assert count_steps() == 5
    assert summarize_calls == 1  # only the first compaction; cooldown suppresses the rest


async def test_cooldown_disabled_recompacts_every_step():
    """min_recompact_growth_tokens=0 opts out: every above-threshold step re-compacts."""
    summarize_calls = 0

    def summarizer_gen(messages, info):
        nonlocal summarize_calls
        summarize_calls += 1
        return ModelResponse(parts=[TextPart(content="SUMMARY " + "x" * 200)])

    cap = ContextCompression(
        Agent(FunctionModel(summarizer_gen), output_type=str),
        compress_threshold=("tokens", 50),
        keep=("messages", 2),
        min_prefix=2,
        min_recompact_growth_tokens=0,
    )
    agent, count_steps = _tool_loop_setup(cap)
    await agent.run("start", message_history=_big_history())
    assert count_steps() == 5
    assert summarize_calls == 5  # one summarizer call per model step (pre-cooldown behavior)


async def test_cooldown_allows_recompaction_after_growth():
    """Once the history grows past the delta since the last compaction, it re-fires."""
    summarize_calls = 0

    def summarizer_gen(messages, info):
        nonlocal summarize_calls
        summarize_calls += 1
        return ModelResponse(parts=[TextPart(content="S" * 100)])

    cap = ContextCompression(
        Agent(FunctionModel(summarizer_gen), output_type=str),
        compress_threshold=("tokens", 50),
        keep=("messages", 2),
        min_prefix=2,
        min_recompact_growth_tokens=50,  # small delta: a couple of big turns exceed it
    )
    parent = _parent(cap)
    history = _big_history()
    for _ in range(4):
        result = await parent.run("turn " + "t" * 400, message_history=history)
        history = result.all_messages()
    assert summarize_calls >= 2  # growth exceeded the delta at least once -> re-compacted


# --- streaming (run_stream, both persist modes) ---


async def test_run_stream_persist_true_compacts():
    def parent_fn(messages, info):
        return ModelResponse(parts=[TextPart(content="ok")])

    async def parent_stream(messages, info):
        yield "ok"

    model = FunctionModel(function=parent_fn, stream_function=parent_stream)
    cap = ContextCompression(_summarizer("S"), compress_threshold=("messages", 4), keep=("messages", 2))
    agent = Agent(model, capabilities=[cap])
    async with agent.run_stream("final q", message_history=_history_with_tool_pair()) as result:
        output = await result.get_output()
    assert output == "ok"
    assert _sentinel(result.all_messages()) is not None


async def test_run_stream_persist_false():
    model_input: list | None = None

    def parent_fn(messages, info):
        return ModelResponse(parts=[TextPart(content="ok")])

    async def parent_stream(messages, info):
        nonlocal model_input
        model_input = list(messages)
        yield "ok"

    model = FunctionModel(function=parent_fn, stream_function=parent_stream)
    cap = ContextCompression(_summarizer("S"), compress_threshold=("messages", 4), keep=("messages", 2), persist=False)
    agent = Agent(model, capabilities=[cap])
    history = _history_with_tool_pair()
    async with agent.run_stream("final q", message_history=history) as result:
        await result.get_output()
    # model saw the compacted messages; state retains the full history
    assert model_input is not None
    assert len(model_input) < 1 + len(history)
    assert len(result.all_messages()) == 1 + len(history) + 1


# --- summarizer input is the render_structured product (§5.8) ---


async def test_summarizer_receives_structured_input():
    captured: list[str] = []

    def summarizer_gen(messages, info):
        captured.append(str(messages[-1].parts[0].content))
        return ModelResponse(parts=[TextPart(content="S")])

    cap = ContextCompression(
        Agent(FunctionModel(summarizer_gen), output_type=str),
        compress_threshold=("messages", 4),
        keep=("messages", 2),
    )
    await _parent(cap).run("final q", message_history=_history_with_tool_pair())
    assert len(captured) == 1
    prompt = captured[0]
    assert "<conversation-history>" in prompt
    assert '<message role="user">' in prompt
    assert "tool_call=" in prompt  # tool call rendered as a structured attribute


# --- sentinel serialization round-trip (persist=True durability) ---


async def test_sentinel_survives_serialization_roundtrip():
    cap = ContextCompression(
        _summarizer("S"),
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        strategy="hybrid",
        min_prefix=2,
        min_recompact_growth_tokens=0,
    )
    parent = _parent(cap)
    r1 = await parent.run("turn1", message_history=_history(3))
    # simulate service-side persistence: JSON round-trip via the framework's adapter
    blob = ModelMessagesTypeAdapter.dump_json(r1.all_messages())
    restored = ModelMessagesTypeAdapter.validate_json(blob)
    s = _sentinel(restored)
    assert s is not None
    rec = parse_summary_sentinel(s.parts[0].content)
    assert rec is not None
    assert rec.generation == 0
    assert rec.compacted_tokens is not None  # cooldown baseline persisted
    # second run with the restored history -> incremental (cross-process state recovery)
    r2 = await parent.run("turn2", message_history=list(restored))
    s2 = _sentinel(r2.all_messages())
    assert s2 is not None
    rec2 = parse_summary_sentinel(s2.parts[0].content)
    assert rec2 is not None
    assert rec2.generation == 1
    assert rec2.strategy == "incremental"


# --- summarizer usage merged into parent run (§5.9) ---


async def test_summarizer_usage_merged_into_parent():
    cap = ContextCompression(_summarizer("S"), compress_threshold=("messages", 4), keep=("messages", 2))
    result = await _parent(cap).run("final q", message_history=_history_with_tool_pair())
    assert result.usage.requests == 2  # parent request + summarizer request


# --- SummaryStore contract (persist=False, §5.6) ---


async def test_store_incremental_contract():
    from pydantic_ai_extensions.context_compression.store import SummaryRecord as SR

    records: dict[str, SR] = {}
    put_records: list[SR] = []
    gets = 0

    class _Store:
        async def get(self, ctx):
            nonlocal gets
            gets += 1
            return records.get(ctx.conversation_id)

        async def put(self, ctx, record):
            put_records.append(record)
            records[ctx.conversation_id] = record

    prompts: list[str] = []

    def summarizer_gen(messages, info):
        prompts.append(str(messages[-1].parts[0].content))
        return ModelResponse(parts=[TextPart(content=f"summary-v{len(prompts)}")])

    cap = ContextCompression(
        Agent(FunctionModel(summarizer_gen), output_type=str),
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        strategy="hybrid",
        min_prefix=2,
        min_recompact_growth_tokens=0,
        persist=False,
        summary_store=_Store(),  # type: ignore[abstract]
    )
    parent = _parent(cap)
    history = _history_with_tool_pair()
    for turn in range(3):
        result = await parent.run(f"turn{turn}", message_history=history)
        history = result.all_messages()
    assert gets >= 3
    assert len(put_records) == 3
    # first compaction is full, subsequent ones incremental merges
    assert put_records[0].strategy == "full"
    assert all(r.strategy == "incremental" for r in put_records[1:])
    # covered_count strictly increases (persist=False indexes the untouched history)
    covered = [r.covered_count for r in put_records]
    assert covered == sorted(covered) and len(set(covered)) == len(covered)
    # the incremental prompt carries the previous summary text
    assert any("Previous summary:" in p and "summary-v1" in p for p in prompts[1:])
    # the conversation was keyed by a single conversation_id across runs
    assert len(records) == 1


async def test_store_failure_does_not_block_run():
    class _BadStore:
        async def get(self, ctx):
            raise OSError("store down")

        async def put(self, ctx, record):
            raise OSError("store down")

    cap = ContextCompression(
        _summarizer("S"),
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        persist=False,
        summary_store=_BadStore(),  # type: ignore[abstract]
    )
    result = await _parent(cap).run("final q", message_history=_history_with_tool_pair())
    assert result.output == "ok"  # parent run unaffected


# --- truncation fallbacks (§5.10) ---


async def test_after_tool_execute_single_line_fallback():
    """A single-line huge output can't be line-truncated; token-precise cut applies."""
    cap = ContextCompression(
        _summarizer(), max_tool_output_tokens=20, tool_output_head_lines=2, tool_output_tail_lines=2
    )
    big = "x" * 100_000
    out = await cap.after_tool_execute(ctx=None, call=None, tool_def=None, args=None, result=big)  # type: ignore[arg-type]
    assert out != big
    assert "truncated" in out
    assert len(out) < 1_000
    assert estimate_text_tokens(out) <= 40  # budget + marker overhead


async def test_after_tool_execute_long_lines_fallback():
    """Head/tail line truncation that still exceeds the budget falls back to a token cut."""
    cap = ContextCompression(
        _summarizer(), max_tool_output_tokens=20, tool_output_head_lines=2, tool_output_tail_lines=2
    )
    text = "\n".join(["y" * 2_000] * 10)  # 4 kept lines would still be ~2000 tokens
    out = await cap.after_tool_execute(ctx=None, call=None, tool_def=None, args=None, result=text)  # type: ignore[arg-type]
    assert out != text
    assert "truncated" in out
    assert estimate_text_tokens(out) <= 40


async def test_max_summary_tokens_caps_summary():
    big_summary = "word " * 5_000

    def summarizer_gen(messages, info):
        return ModelResponse(parts=[TextPart(content=big_summary)])

    cap = ContextCompression(
        Agent(FunctionModel(summarizer_gen), output_type=str),
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        max_summary_tokens=50,
    )
    result = await _parent(cap).run("final q", message_history=_history_with_tool_pair())
    s = _sentinel(result.all_messages())
    assert s is not None
    rec = parse_summary_sentinel(s.parts[0].content)
    assert rec is not None
    assert estimate_text_tokens(rec.text) <= 60  # cap + marker overhead
    assert "truncated" in rec.text


# --- constructor validation ---


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_prefix": 0},
        {"full_reset_every": 0},
        {"char_per_token": 0},
        {"compress_threshold": ("fraction", 1.5)},
        {"compress_threshold": ("fraction", 0.0)},
        {"compress_threshold": ("bogus", 3)},
        {"keep": ("tokens", 0)},
        {"min_recompact_growth_tokens": -1},
        {"max_summary_tokens": 0},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(UserError):
        ContextCompression(_summarizer(), **kwargs)


# --- slicing clamp ---


def test_find_safe_split_clamps_oversized_keep():
    msgs = [ModelRequest(parts=[UserPromptPart(content=f"q{i}")]) for i in range(5)]
    # keep larger than the history, keep_first_user_message=False -> 0, never negative
    assert find_safe_split(msgs, ("messages", 100), keep_first_user_message=False) == 0


# --- empty incremental batch short-circuit (§5.3) ---


async def test_empty_new_batch_short_circuits():
    calls = 0

    def summarizer_gen(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart(content="S")])

    cap = ContextCompression(
        Agent(FunctionModel(summarizer_gen), output_type=str),
        compress_threshold=("messages", 3),
        keep=("messages", 3),
        strategy="hybrid",
        min_prefix=2,
        min_recompact_growth_tokens=0,
    )
    parent = _parent(cap)
    # history already holding a sentinel right after the head
    sentinel_msg = build_summary_message(
        SummaryRecord(text="old summary", generation=0, covered_count=2, strategy="full")
    )
    history = [
        ModelRequest(parts=[UserPromptPart(content="q0")]),
        sentinel_msg,
        ModelRequest(parts=[UserPromptPart(content="q1")]),
        ModelResponse(parts=[TextPart(content="a1")]),
    ]
    result = await parent.run("q2", message_history=history)
    # 5 messages, keep 3 -> k=2 -> prefix_body=[sentinel] -> new batch empty -> no-op
    assert calls == 0
    s = _sentinel(result.all_messages())
    assert s is not None
    rec = parse_summary_sentinel(s.parts[0].content)
    assert rec is not None and rec.generation == 0  # untouched


# --- sentinel parsing unit tests (§5.6) ---


def test_sentinel_roundtrip_with_compacted_tokens():
    rec = SummaryRecord(
        text="hello\nworld", generation=3, covered_count=10, strategy="incremental", compacted_tokens=1234
    )
    parsed = parse_summary_sentinel(build_summary_message(rec).parts[0].content)
    assert parsed == rec


def test_sentinel_old_format_without_compacted_tokens():
    content = "<conversation-summary generation=2 covered_count=5 strategy=full>\nold text"
    rec = parse_summary_sentinel(content)
    assert rec is not None
    assert rec.compacted_tokens is None
    assert rec.generation == 2
    assert rec.text == "old text"


def test_sentinel_echo_and_prefix_not_matched():
    # a marker quoted mid-text doesn't parse (anchored at content start)
    assert parse_summary_sentinel("I see <conversation-summary generation=9 covered_count=9 strategy=full>\nx") is None
    # startswith-only false positives are rejected by the strict parser via find_summary
    echo = ModelResponse(parts=[TextPart(content="<conversation-summary not-a-real-marker")])
    assert find_summary([echo]) is None


def test_find_summary_locates_real_sentinel_not_echo():
    real = build_summary_message(SummaryRecord(text="real", generation=1, covered_count=4, strategy="full"))
    echo = ModelResponse(parts=[TextPart(content="<conversation-summary bogus")])
    found = find_summary([echo, real])
    assert found is not None
    assert found[0] == 1
    assert found[1].text == "real"
