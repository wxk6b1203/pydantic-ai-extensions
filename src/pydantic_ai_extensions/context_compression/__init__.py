"""Context compression capability for Pydantic AI agents.

See `docs/context-compression.md` for the full design.
"""

from .capability import ContextCompression
from .render import render_messages_to_text, render_structured
from .slicing import find_safe_split, is_safe_cutoff_point, partition_head
from .store import SummaryRecord, SummaryStore
from .summarizer import (
    build_summary_message,
    find_summary,
    merge_usage,
    parse_summary_sentinel,
    summarize_full,
    summarize_incremental,
)
from .tokenizer import (
    DEFAULT_MAX_TOKENS,
    ContextSize,
    estimate_text_tokens,
    estimate_tokens,
    should_trigger,
    truncate_text_to_tokens,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "ContextCompression",
    "ContextSize",
    "SummaryRecord",
    "SummaryStore",
    "build_summary_message",
    "estimate_text_tokens",
    "estimate_tokens",
    "find_safe_split",
    "find_summary",
    "is_safe_cutoff_point",
    "merge_usage",
    "parse_summary_sentinel",
    "partition_head",
    "render_messages_to_text",
    "render_structured",
    "should_trigger",
    "summarize_full",
    "summarize_incremental",
    "truncate_text_to_tokens",
]
