"""ContextCompression capability (§5.1-§5.10).

`persist=True` + `persist=False` + `strategy='full'/'hybrid'` + `for_run` provider guard
+ `after_tool_execute` tool-output truncation. Includes the re-compaction cooldown
(`min_recompact_growth_tokens`, §5.3) that prevents a summarizer call on every model
step when the compacted history remains above the trigger threshold.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, ReinjectSystemPrompt
from pydantic_ai.exceptions import (
    ModelAPIError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import BinaryContent, ModelMessage
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import RunContext

from .render import stringify
from .slicing import find_safe_split, partition_head
from .store import SummaryRecord, SummaryStore
from .summarizer import (
    build_summary_message,
    find_summary,
    merge_usage,
    prefix_body_after,
    summarize_full,
    summarize_incremental,
)
from .tokenizer import (
    ContextSize,
    estimate_text_tokens,
    estimate_tokens,
    should_trigger,
    truncate_text_to_tokens,
)

__all__ = ["ContextCompression"]

logger = logging.getLogger(__name__)

# Failures expected from a summarizer *run*: provider API/connection errors (pydantic-ai
# maps both `APIStatusError` and `APIConnectionError` onto `ModelAPIError`), retries
# exhausted, usage limits, transport timeouts. Programming errors (KeyError,
# AttributeError, ...) deliberately propagate so bugs surface instead of silently
# disabling compression (§5.9).
_SUMMARIZER_EXCEPTIONS = (ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded, TimeoutError)


def _load_blocked_model_types() -> tuple[type, ...]:
    """Model classes that have native compaction (must not use this capability).

    Lazy-imported because the `anthropic` / `openai` extras are optional; if an extra
    isn't installed, that model class can't be in use anyway, so we skip it.
    """
    types: list[type] = []
    try:
        from pydantic_ai.models.anthropic import AnthropicModel

        types.append(AnthropicModel)
    except ImportError:
        pass
    try:
        from pydantic_ai.models.openai import OpenAIResponsesModel

        types.append(OpenAIResponsesModel)
    except ImportError:
        pass
    return tuple(types)


_BLOCKED_MODEL_TYPES = _load_blocked_model_types()


def _validate_context_size(name: str, spec: object) -> None:
    """Fail fast on malformed `compress_threshold` / `keep` specs (constructor time).

    Validates the *runtime* shape (callers may bypass type hints), so the value is
    inspected as `object` rather than trusted as `ContextSize`.
    """
    ok = False
    if isinstance(spec, tuple):
        pair = cast("tuple[Any, Any]", spec)
        if len(pair) == 2:
            kind, value = pair
            if kind == "messages":
                ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
            elif kind == "tokens":
                ok = isinstance(value, int) and not isinstance(value, bool) and value >= 1
            elif kind == "fraction":
                ok = isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < 1
    if not ok:
        raise UserError(
            f"invalid {name}: {spec!r}; expected ('messages', N>=0) | ('tokens', N>=1) | ('fraction', 0<F<1)"
        )


@dataclass(init=False)
class ContextCompression(AbstractCapability[Any]):
    """Provider-agnostic automatic context compression.

    When the estimated token count of the message history crosses `compress_threshold`,
    older messages are summarized (via `summarizer`) into a single summary, preserving a
    recent window and tool-call/return pairing integrity.

    `summarizer` must be a tool-less agent (`output_type=str`, default) -- otherwise it
    may call tools instead of producing a summary. Its `instructions` are the configurable
    compression-prompt entry point (see §5.8).
    """

    summarizer: Agent[Any, str]
    compress_threshold: ContextSize
    max_tokens: int | None
    encoding: str | None
    char_per_token: int
    include_thinking_in_estimate: bool
    keep: ContextSize
    keep_first_user_message: bool
    min_prefix: int
    strategy: Literal["full", "hybrid"]
    full_reset_every: int
    min_recompact_growth_tokens: int | None
    max_summary_tokens: int | None
    max_tool_output_tokens: int | None
    tool_output_head_lines: int
    tool_output_tail_lines: int
    persist: bool
    summary_store: SummaryStore | None

    def __init__(
        self,
        summarizer: Agent[Any, str],
        *,
        compress_threshold: ContextSize = ("fraction", 0.7),
        max_tokens: int | None = None,
        encoding: str | None = None,
        char_per_token: int = 4,
        include_thinking_in_estimate: bool = True,
        keep: ContextSize = ("messages", 6),
        keep_first_user_message: bool = True,
        min_prefix: int = 4,
        strategy: Literal["full", "hybrid"] = "hybrid",
        full_reset_every: int = 5,
        min_recompact_growth_tokens: int | None = 2_048,
        max_summary_tokens: int | None = None,
        max_tool_output_tokens: int | None = None,
        tool_output_head_lines: int = 5,
        tool_output_tail_lines: int = 5,
        persist: bool = True,
        summary_store: SummaryStore | None = None,
    ) -> None:
        if persist and summary_store is not None:
            raise UserError(
                "`summary_store` is only used with `persist=False`; with `persist=True` the "
                "state lives in the history sentinel and the store would be ignored."
            )
        if strategy not in ("full", "hybrid"):
            raise UserError(f"strategy must be 'full' or 'hybrid', got {strategy!r}")
        _validate_context_size("compress_threshold", compress_threshold)
        _validate_context_size("keep", keep)
        if min_prefix < 1:
            raise UserError(f"min_prefix must be >= 1, got {min_prefix}")
        if full_reset_every < 1:
            raise UserError(f"full_reset_every must be >= 1, got {full_reset_every}")
        if char_per_token < 1:
            raise UserError(f"char_per_token must be >= 1, got {char_per_token}")
        if min_recompact_growth_tokens is not None and min_recompact_growth_tokens < 0:
            raise UserError(
                f"min_recompact_growth_tokens must be >= 0 (0 disables the cooldown), got {min_recompact_growth_tokens}"
            )
        if max_summary_tokens is not None and max_summary_tokens < 1:
            raise UserError(f"max_summary_tokens must be >= 1, got {max_summary_tokens}")
        if not persist and summary_store is None:
            logger.info(
                "ContextCompression: persist=False without summary_store degrades to a full "
                "re-summarize on every compaction (no cross-run incremental state, §5.3)."
            )
        self.summarizer = summarizer
        self.compress_threshold = compress_threshold
        self.max_tokens = max_tokens
        self.encoding = encoding
        self.char_per_token = char_per_token
        self.include_thinking_in_estimate = include_thinking_in_estimate
        self.keep = keep
        self.keep_first_user_message = keep_first_user_message
        self.min_prefix = min_prefix
        self.strategy = strategy
        self.full_reset_every = full_reset_every
        self.min_recompact_growth_tokens = min_recompact_growth_tokens
        self.max_summary_tokens = max_summary_tokens
        self.max_tool_output_tokens = max_tool_output_tokens
        self.tool_output_head_lines = tool_output_head_lines
        self.tool_output_tail_lines = tool_output_tail_lines
        self.persist = persist
        self.summary_store = summary_store

    # --- spec / ordering / serialization ---

    def get_ordering(self) -> CapabilityOrdering | None:
        # Run after ReinjectSystemPrompt so the system prompt is reinjected first.
        return CapabilityOrdering(wrapped_by=[ReinjectSystemPrompt])

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # holds a summarizer (closure) -> not spec-serializable (cf. ProcessHistory)

    # --- provider guard (§5.7) ---

    async def for_run(self, ctx: RunContext[Any]) -> ContextCompression:
        model = ctx.model
        while isinstance(model, WrapperModel):
            model = model.wrapped
        if _BLOCKED_MODEL_TYPES and isinstance(model, _BLOCKED_MODEL_TYPES):
            raise UserError(
                f"ContextCompression does not support {type(model).__name__}: that provider has "
                "native compaction; use AnthropicCompaction / OpenAICompaction instead."
            )
        return self

    # --- hooks ---

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if not self.persist:
            return request_context  # persist=False handled by wrap_model_request
        compacted = await self._maybe_compact(ctx, request_context.messages)
        if compacted is not None:
            request_context.messages = compacted
        return request_context

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: Any,  # WrapModelRequestHandler
    ) -> Any:
        if self.persist:
            return await handler(request_context)  # already compacted in before_model_request
        compacted = await self._maybe_compact(ctx, request_context.messages)
        if compacted is not None:
            return await handler(replace(request_context, messages=compacted))
        return await handler(request_context)

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: Any,
        tool_def: Any,
        args: Any,
        result: Any,
    ) -> Any:
        """Truncate large tool returns at the source (§5.10).

        Line-based head/tail truncation first; if that can't fit the token budget
        (single-line or huge-line output), fall back to a token-precise middle cut.
        """
        if self.max_tool_output_tokens is None:
            return result
        if isinstance(result, BinaryContent):
            return result  # never stringify binary
        budget = self.max_tool_output_tokens
        text = stringify(result)
        if estimate_text_tokens(text, encoding=self.encoding, char_per_token=self.char_per_token) <= budget:
            return result
        lines = text.splitlines()
        head, tail = self.tool_output_head_lines, self.tool_output_tail_lines
        if len(lines) > head + tail:
            marker = f"...[truncated {len(lines) - head - tail} lines]..."
            candidate = "\n".join([*lines[:head], marker, *lines[-tail:]])
            if estimate_text_tokens(candidate, encoding=self.encoding, char_per_token=self.char_per_token) <= budget:
                return candidate
        # Line-based truncation cannot fit the budget -> token-precise middle cut.
        return truncate_text_to_tokens(text, budget, encoding=self.encoding, char_per_token=self.char_per_token)

    # --- core compaction ---

    def _estimate(self, messages: Sequence[ModelMessage]) -> int:
        return estimate_tokens(
            messages,
            encoding=self.encoding,
            char_per_token=self.char_per_token,
            include_thinking=self.include_thinking_in_estimate,
        )

    async def _maybe_compact(self, ctx: RunContext[Any], messages: Sequence[ModelMessage]) -> list[ModelMessage] | None:
        """Return the compacted message list, or None if no compression applies."""
        # Fast path (§5.4): a compressible prefix needs at least `min_prefix` messages
        # plus the current request; skip trigger/estimation entirely for short histories.
        if len(messages) < self.min_prefix + 1:
            return None
        if not should_trigger(
            messages,
            self.compress_threshold,
            self.max_tokens,
            encoding=self.encoding,
            char_per_token=self.char_per_token,
            include_thinking=self.include_thinking_in_estimate,
        ):
            return None
        k = find_safe_split(
            messages,
            self.keep,
            self.max_tokens,
            self.keep_first_user_message,
            count_tokens=self._estimate,
        )
        if k < self.min_prefix:
            return None
        head, prefix_body = partition_head(messages[:k], self.keep_first_user_message)
        existing = await self._load_existing_summary(ctx, prefix_body)
        # Cooldown (§5.3): the threshold is a trigger line, not a ceiling -- the compacted
        # history can stay above it, which would otherwise re-fire the summarizer on every
        # model step of a tool loop. Only re-compact once the history has grown by at
        # least `min_recompact_growth_tokens` since the last compaction baseline.
        if (
            existing is not None
            and self.min_recompact_growth_tokens
            and existing.compacted_tokens is not None
            and self._estimate(messages) - existing.compacted_tokens < self.min_recompact_growth_tokens
        ):
            return None
        do_full = existing is None or self.strategy == "full" or existing.generation >= self.full_reset_every
        try:
            if do_full:
                result = await summarize_full(self.summarizer, prefix_body)
                record = SummaryRecord(text=result.output, generation=0, covered_count=k, strategy="full")
            else:
                assert existing is not None  # do_full=False guarantees existing is not None
                new_batch = self._new_batch(existing, prefix_body, messages, k)
                if not new_batch:
                    # Growth is entirely inside the recent window; nothing new to merge.
                    return None
                result = await summarize_incremental(self.summarizer, existing.text, new_batch)
                record = SummaryRecord(
                    text=result.output,
                    generation=existing.generation + 1,
                    covered_count=k,
                    strategy="incremental",
                )
        except _SUMMARIZER_EXCEPTIONS:
            # Summarizer failed: degrade to no compression; don't block the parent run.
            logger.warning("context compression: summarizer failed; sending uncompacted history", exc_info=True)
            return None
        merge_usage(ctx, result)
        if self.max_summary_tokens is not None:
            record.text = truncate_text_to_tokens(
                record.text, self.max_summary_tokens, encoding=self.encoding, char_per_token=self.char_per_token
            )
        # Cooldown baseline: the token estimate of the history as it will be seen at the
        # next check -- the compacted list for persist=True (it becomes the new state),
        # the untouched full history for persist=False (state is never compacted).
        draft = [*head, build_summary_message(record), *messages[k:]]
        record.compacted_tokens = self._estimate(draft if self.persist else messages)
        summary_msg = build_summary_message(record)
        compacted = [*head, summary_msg, *messages[k:]]
        if not self.persist and self.summary_store is not None:
            try:
                await self.summary_store.put(ctx, record)
            except Exception:
                # User-supplied store failed: the model still gets the compacted view for
                # this request; only cross-run state is lost (next run degrades to full).
                logger.warning("context compression: summary_store.put failed; cross-run state lost", exc_info=True)
        return compacted

    async def _load_existing_summary(
        self, ctx: RunContext[Any], prefix_body: Sequence[ModelMessage]
    ) -> SummaryRecord | None:
        """Recover the prior summary: history sentinel (persist=True) or store (persist=False)."""
        if not self.persist:
            if self.summary_store is None:
                return None  # persist=False without store -> always full (degrade)
            try:
                return await self.summary_store.get(ctx)
            except Exception:
                logger.warning(
                    "context compression: summary_store.get failed; degrading to full re-summarize",
                    exc_info=True,
                )
                return None
        found = find_summary(prefix_body)
        return found[1] if found is not None else None

    def _new_batch(
        self,
        existing: SummaryRecord,
        prefix_body: Sequence[ModelMessage],
        messages: Sequence[ModelMessage],
        k: int,
    ) -> list[ModelMessage]:
        """Messages to merge incrementally since the last summary."""
        if self.persist:
            return prefix_body_after(prefix_body)
        # persist=False: history is never compacted, so index by covered_count.
        start = max(existing.covered_count, 1)
        return list(messages[start:k])
