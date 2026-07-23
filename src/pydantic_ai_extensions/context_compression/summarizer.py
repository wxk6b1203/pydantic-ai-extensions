"""Summarizer prompts + helpers (§5.8 framework layer, §5.9 signatures).

The framework-layer prompts are fixed English templates (only the "guidance layer" --
the summarizer agent's `instructions` -- is user-configurable). `summarize_full` renders
the prefix via `render_structured` and runs the summarizer; `build_summary_message` wraps
the result as a sentinel-tagged `ModelResponse`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

from .render import render_structured
from .store import SummaryRecord

__all__ = [
    "build_summary_message",
    "find_summary",
    "merge_usage",
    "parse_summary_sentinel",
    "prefix_body_after",
    "summarize_full",
    "summarize_incremental",
]

# Sentinel marker on the summary part content. The state lives in content (not
# `ModelRequest.metadata`) because `_clean_message_history` drops metadata when merging
# consecutive same-role messages; part content survives merges and round-trips across runs.
# `compacted_tokens` is optional: markers written by older versions lack it (parsed as None,
# which skips the cooldown once and rewrites the field on the next compaction).
_SENTINEL_RE = re.compile(
    r"^<conversation-summary generation=(\d+) covered_count=(\d+) strategy=(\w+)"
    r"(?: compacted_tokens=(\d+))?>\n?(.*)$",
    re.DOTALL,
)

# Framework-layer (user-prompt) templates. Fixed; the summarizer's `instructions`
# (guidance layer) is the only user-configurable prompt entry point.
_FULL_PROMPT = (
    "The following is a structured conversation history (wrapped in <conversation-history>; "
    "each <message> is tagged with its role, tool calls/returns as structured attributes). "
    "Summarize it, preserving key facts, decisions and TODOs, omitting small talk and "
    "irrelevant detail:\n\n"
    "{history}"
)

_INCREMENTAL_PROMPT = (
    "Previous summary:\n{old}\n\n"
    "New structured content since the last summary:\n{new}\n\n"
    "Merge the new content into the previous summary and output the updated full summary "
    "(coherent prose, do not itemize)."
)


async def summarize_full(summarizer: Agent[Any, str], prefix_body: Sequence[ModelMessage]) -> AgentRunResult[str]:
    """Render `prefix_body` via `render_structured` and run the summarizer (full re-summarize).

    Returns the summarizer's `AgentRunResult` (`.output` is the summary text, `.usage`
    its token usage).
    """
    prompt = _FULL_PROMPT.format(history=render_structured(prefix_body))
    return await summarizer.run(prompt)


async def summarize_incremental(
    summarizer: Agent[Any, str], old_text: str, new_batch: Sequence[ModelMessage]
) -> AgentRunResult[str]:
    """Merge `new_batch` into the prior summary (`old_text`) and return the updated summary."""
    prompt = _INCREMENTAL_PROMPT.format(old=old_text, new=render_structured(new_batch))
    return await summarizer.run(prompt)


def parse_summary_sentinel(content: str) -> SummaryRecord | None:
    """Parse a `<conversation-summary ...>` marker; return a SummaryRecord or None.

    Strict: the content must *start* with the full marker format. Model responses that
    merely quote or echo a marker-looking string mid-text do not match.
    """
    m = _SENTINEL_RE.match(content)
    if not m:
        return None
    return SummaryRecord(
        text=m.group(5),
        generation=int(m.group(1)),
        covered_count=int(m.group(2)),
        strategy=m.group(3),  # type: ignore[arg-type]
        compacted_tokens=int(m.group(4)) if m.group(4) is not None else None,
    )


def find_summary(prefix_body: Sequence[ModelMessage]) -> tuple[int, SummaryRecord] | None:
    """Locate the summary message in `prefix_body` by strictly parsing each `TextPart`.

    Returns ``(index, record)`` of the *first* strictly-parsed sentinel. First-match is
    correct for the capability's own layout (the sentinel sits immediately after the
    preserved head); a looser ``startswith`` check would risk matching model echoes that
    don't carry the full marker format.
    """
    for i, m in enumerate(prefix_body):
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, TextPart):
                    rec = parse_summary_sentinel(p.content)
                    if rec is not None:
                        return i, rec
    return None


def prefix_body_after(prefix_body: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Messages after the old summary message in `prefix_body` (located by the sentinel marker).

    Used in `persist=True` incremental mode: the sentinel is a `ModelResponse` carrying a
    `<conversation-summary>` TextPart; the messages after it are the new batch since last summary.
    """
    found = find_summary(prefix_body)
    if found is None:
        return list(prefix_body)
    return list(prefix_body[found[0] + 1 :])


def build_summary_message(record: SummaryRecord) -> ModelResponse:
    """Build the sentinel summary as a single `ModelResponse` (assistant turn).

    The sentinel state (generation/covered_count/strategy) lives in the part *content*
    marker, not `ModelRequest.metadata`: `_clean_message_history` drops metadata when
    merging consecutive same-role messages, but part content survives -- so the sentinel
    round-trips across runs.
    """
    tokens_attr = f" compacted_tokens={record.compacted_tokens}" if record.compacted_tokens is not None else ""
    content = (
        f"<conversation-summary generation={record.generation} covered_count={record.covered_count} "
        f"strategy={record.strategy}{tokens_attr}>\n{record.text}"
    )
    return ModelResponse(parts=[TextPart(content=content)])


def merge_usage(ctx: Any, result: AgentRunResult[str]) -> None:
    """Fold the summarizer's usage into the parent run's `RunUsage`."""
    ctx.usage.incr(result.usage)
