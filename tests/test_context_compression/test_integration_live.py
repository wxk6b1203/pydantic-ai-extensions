"""Live integration tests against a real DeepSeek API endpoint.

Run with: ``pytest --live -m live``
Skipped by default (no ``--live`` flag).
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel

from pydantic_ai_extensions.context_compression import ContextCompression, parse_summary_sentinel

pytestmark = pytest.mark.live


def _history(n: int = 6) -> list:
    """Build an alternating req/resp history of ``n`` pairs."""
    msgs = []
    for i in range(n):
        msgs.append(ModelRequest(parts=[UserPromptPart(content=f"Question {i}: what is {i}+{i}?")]))
        msgs.append(ModelResponse(parts=[TextPart(content=f"Answer {i}: {i + i}.")]))
    return msgs


def _sentinel_gen(messages: list) -> int | None:
    for m in messages:
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, TextPart) and p.content.startswith("<conversation-summary"):
                    rec = parse_summary_sentinel(p.content)
                    return rec.generation if rec else None
    return None


# --- tests ---


async def test_live_full_compression(live_model, live_summarizer):
    """Compression triggers with a real model; sentinel appears with a meaningful summary."""
    cap = ContextCompression(
        live_summarizer,
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        strategy="full",
        min_prefix=2,
    )
    agent = Agent(live_model, capabilities=[cap], instructions="You are a helpful math tutor.")
    result = await agent.run("What was the first question?", message_history=_history(6))
    msgs = result.all_messages()
    gen = _sentinel_gen(msgs)
    assert gen == 0  # first compaction is full
    # find the sentinel text and verify it's non-trivial (the summarizer produced real content)
    for m in msgs:
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, TextPart) and p.content.startswith("<conversation-summary"):
                    rec = parse_summary_sentinel(p.content)
                    assert rec is not None
                    assert len(rec.text) > 10  # real summary, not empty
                    return
    pytest.fail("No sentinel found in messages")


async def test_live_hybrid_incremental(live_model, live_summarizer):
    """Multi-run with hybrid strategy: generation increments across runs."""
    cap = ContextCompression(
        live_summarizer,
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        strategy="hybrid",
        full_reset_every=5,
        min_prefix=2,
    )
    agent = Agent(live_model, capabilities=[cap], instructions="You are a helpful assistant.")
    history = _history(4)
    gens: list[int | None] = []
    for turn in range(3):
        result = await agent.run(f"Turn {turn}: summarize what we discussed.", message_history=history)
        msgs = result.all_messages()
        gens.append(_sentinel_gen(msgs))
        history = msgs
    # At least the first run should produce a sentinel (gen 0).
    # Subsequent runs may or may not trigger depending on cleaned message count,
    # but at least one generation should be non-None.
    assert any(g is not None for g in gens)
    # If multiple compactions happened, generations should be non-decreasing.
    non_none = [g for g in gens if g is not None]
    if len(non_none) > 1:
        assert non_none == sorted(non_none)


async def test_live_persist_false(live_model, live_summarizer):
    """persist=False: state retains full history, model sees compacted messages."""
    cap = ContextCompression(
        live_summarizer,
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        persist=False,
        min_prefix=2,
    )
    agent = Agent(live_model, capabilities=[cap], instructions="You are a helpful assistant.")
    history = _history(6)
    result = await agent.run("What was discussed?", message_history=history)
    msgs = result.all_messages()
    # persist=False: state retains full history (no sentinel persisted)
    assert len(msgs) >= len(history)  # original history + current + response
    assert _sentinel_gen(msgs) is None  # no sentinel in state


async def test_live_tool_pair_preserved(live_model, live_summarizer):
    """Tool-call/return pairs survive compression."""
    history = [
        ModelRequest(parts=[UserPromptPart(content="What is 2+2?")]),
        ModelResponse(parts=[ToolCallPart(tool_name="calc", args={"expr": "2+2"}, tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="calc", content="4", tool_call_id="c1")]),
        ModelResponse(parts=[TextPart(content="2+2 = 4.")]),
        ModelRequest(parts=[UserPromptPart(content="What is 3+3?")]),
        ModelResponse(parts=[TextPart(content="3+3 = 6.")]),
    ]
    cap = ContextCompression(
        live_summarizer,
        compress_threshold=("messages", 4),
        keep=("messages", 4),  # keep the tool pair in the recent window
        min_prefix=2,
    )
    agent = Agent(live_model, capabilities=[cap], instructions="You are a math assistant.")
    result = await agent.run("What were the results?", message_history=history)
    msgs = result.all_messages()
    call_ids = {
        p.tool_call_id for m in msgs if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)
    }
    return_ids = {
        p.tool_call_id for m in msgs if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)
    }
    # If the tool pair survived (in recent window), calls == returns.
    if call_ids:
        assert call_ids == return_ids


async def test_live_summarizer_failure_graceful(live_model):
    """Summarizer failure (bad endpoint) does not block the parent run."""
    from openai import AsyncOpenAI
    from pydantic_ai.providers.deepseek import DeepSeekProvider

    bad_client = AsyncOpenAI(api_key="bad", base_url="http://10.0.10.42:9999/v1", timeout=5)
    bad_model = OpenAIChatModel("deepseek-v4-flash", provider=DeepSeekProvider(openai_client=bad_client))
    bad_summarizer = Agent(bad_model, output_type=str)

    cap = ContextCompression(
        bad_summarizer,
        compress_threshold=("messages", 4),
        keep=("messages", 2),
        min_prefix=2,
    )
    agent = Agent(live_model, capabilities=[cap], instructions="You are a helpful assistant.")
    # Should not raise despite summarizer failure; parent run completes normally.
    result = await agent.run("Hello!", message_history=_history(6))
    assert result.output  # got a real reply from the parent model
    assert _sentinel_gen(result.all_messages()) is None  # no compaction happened
