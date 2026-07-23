"""SummaryRecord dataclass + SummaryStore protocol (§5.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic_ai.tools import RunContext

__all__ = ["SummaryRecord", "SummaryStore"]


@dataclass
class SummaryRecord:
    """State produced by one compression pass."""

    text: str
    """The summary body."""

    generation: int
    """Incremental-merge generation (reset to 0 on a full re-summarize)."""

    covered_count: int
    """Number of prefix messages covered by this summary. Used only in `persist=False`
    mode to locate the new batch; informational in `persist=True` (the sentinel position
    locates it)."""

    strategy: Literal["full", "incremental"]

    compacted_tokens: int | None = None
    """Estimated token count of the history *as it will be seen at the next check*,
    recorded at compaction time. Drives the re-compaction cooldown: compression only
    re-fires once the history has grown by at least `min_recompact_growth_tokens` since
    this baseline (prevents a summarizer call on every model step when the compacted
    history remains above the trigger threshold).

    `persist=True`: estimate of the compacted list (which becomes the new state).
    `persist=False`: estimate of the full source history (state is never compacted).
    `None` for records written by older versions -> cooldown is skipped once and the
    next compaction writes the field.
    """


class SummaryStore(Protocol):
    """Persist the running summary across runs (only needed for `persist=False`).

    Implementations key by `ctx.conversation_id` and must be concurrency-safe. Only
    `conversation_id` is used (not deps), so the protocol is non-generic.
    """

    async def get(self, ctx: RunContext[Any]) -> SummaryRecord | None: ...

    async def put(self, ctx: RunContext[Any], record: SummaryRecord) -> None: ...
