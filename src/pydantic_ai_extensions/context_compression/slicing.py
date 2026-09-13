"""Safe-boundary slicing (§5.5).

The core invariant: a cutoff must never split a ``ToolCallPart`` from its matching
``ToolReturnPart`` / ``RetryPromptPart`` (matched by ``tool_call_id``). All three
part types carry ``tool_call_id`` (verified against pydantic-ai-slim 2.9.0, unchanged through 2.26.0).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)

from .render import render_message_texts
from .tokenizer import DEFAULT_MAX_TOKENS, ContextSize, estimate_text_tokens

__all__ = ["find_safe_split", "is_safe_cutoff_point", "partition_head"]

# Call-side parts (live in ModelResponse) and return-side parts (live in ModelRequest).
_CALL_TYPES = (ToolCallPart, NativeToolCallPart)
_RETURN_TYPES = (ToolReturnPart, NativeToolReturnPart, RetryPromptPart)


def _call_ids_in_prefix(messages: Sequence[ModelMessage], k: int) -> set[str]:
    ids: set[str] = set()
    for m in messages[:k]:
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, _CALL_TYPES):
                    ids.add(p.tool_call_id)
    return ids


def _return_ids_in_recent(messages: Sequence[ModelMessage], k: int) -> set[str]:
    ids: set[str] = set()
    for m in messages[k:]:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, _RETURN_TYPES):
                    ids.add(p.tool_call_id)
    return ids


def is_safe_cutoff_point(messages: Sequence[ModelMessage], k: int) -> bool:
    """A cutoff ``k`` (prefix=``messages[:k]``, recent=``messages[k:]``) is safe iff no
    ``ToolCallPart`` in prefix has its matching return in recent (by ``tool_call_id``).

    ``k == 0`` (empty prefix) and ``k == len(messages)`` (empty recent) are trivially
    safe; :func:`find_safe_split` keeps ``k`` within ``[0, n-1]`` so the current
    request is always retained.
    """
    return not (_call_ids_in_prefix(messages, k) & _return_ids_in_recent(messages, k))


def _per_message_token_counts(
    messages: Sequence[ModelMessage],
    *,
    encoding: str | None,
    char_per_token: int,
    include_thinking: bool,
) -> list[int]:
    """Token count of each message, rendered individually.

    Approximation note: the joined-history render counts ~1 separator token per
    message boundary that this per-message sum omits -- negligible for a cutoff
    heuristic, and it buys a single encoding pass instead of O(log n) full passes.
    """
    texts = render_message_texts(messages, include_thinking=include_thinking)
    return [estimate_text_tokens(t, encoding=encoding, char_per_token=char_per_token) for t in texts]


def _token_cutoff_from_counts(counts: Sequence[int], budget: int) -> int:
    """Largest cutoff where the recent-window token sum ``>= budget`` (0 if total < budget).

    ``keep`` is a *lower bound*: keep a recent window of *at least* ``budget`` tokens,
    as small as possible. ``counts[i]`` is the token count of ``messages[i]``; suffix
    sums let the binary search run on precomputed integers instead of re-encoding the
    suffix on every probe.
    """
    n = len(counts)
    if n == 0 or sum(counts) < budget:
        return 0
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + counts[i]
    lo, hi, ans = 0, n - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if suffix[mid] >= budget:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def _token_cutoff(
    messages: Sequence[ModelMessage], budget: int, count_tokens: Callable[[Sequence[ModelMessage]], int]
) -> int:
    """Largest cutoff where ``count_tokens(messages[cutoff:]) >= budget`` (``keep`` is a
    *lower bound*: keep a recent window of *at least* ``budget`` tokens, as small as
    possible). Returns 0 if the whole history is under budget (nothing to compress)."""
    n = len(messages)
    if n == 0 or count_tokens(messages) < budget:
        return 0
    lo, hi, ans = 0, n - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if count_tokens(messages[mid:]) >= budget:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def _keep_target(
    messages: Sequence[ModelMessage],
    keep: ContextSize,
    max_tokens: int | None,
    count_tokens: Callable[[Sequence[ModelMessage]], int],
) -> int:
    """Translate the ``keep`` spec into the ideal cutoff index (recent = messages[target:]).

    ``keep`` is a *lower bound*: the recent window ends up with at least the requested
    size (clamped by history length and tool-pair safety). Clamped to ``>= 0`` so a
    ``('messages', N)`` spec with ``N >= len(messages)`` cannot yield a negative index.
    """
    n = len(messages)
    match keep:
        case ("messages", count):
            # Keep at least 1 (the current request); count<=0 means no lower bound -> keep minimum.
            return max(n - max(count, 1), 0)
        case ("tokens", budget):
            return _token_cutoff(messages, budget, count_tokens)
        case ("fraction", frac):
            return _token_cutoff(messages, int((max_tokens or DEFAULT_MAX_TOKENS) * frac), count_tokens)
        case _:
            raise ValueError(f"Invalid keep: {keep!r}")


def find_safe_split(
    messages: Sequence[ModelMessage],
    keep: ContextSize,
    max_tokens: int | None = None,
    keep_first_user_message: bool = True,
    count_tokens: Callable[[Sequence[ModelMessage]], int] | None = None,
    *,
    encoding: str | None = None,
    char_per_token: int = 4,
    include_thinking: bool = True,
) -> int:
    """Find the largest safe cutoff ``k`` (prefix=``messages[:k]``, recent=``messages[k:]``)
    that respects the ``keep`` spec and never orphans a tool-call/return pair.

    Walks ``k`` backward from the ideal target until :func:`is_safe_cutoff_point` holds.
    Returns 0 if ``keep_first_user_message`` and the safe prefix would be <= 1 message
    (nothing meaningful to summarize).

    Token-based ``keep`` specs are resolved from per-message suffix sums (one encoding
    pass) unless a custom ``count_tokens`` is injected, in which case the counter is
    re-run per binary-search probe (exact but O(log n) full passes).
    """
    n = len(messages)
    if n == 0:
        return 0
    if count_tokens is None:
        counts = _per_message_token_counts(
            messages,
            encoding=encoding,
            char_per_token=char_per_token,
            include_thinking=include_thinking,
        )

        def target() -> int:
            match keep:
                case ("messages", count):
                    return max(n - max(count, 1), 0)
                case ("tokens", budget):
                    return _token_cutoff_from_counts(counts, budget)
                case ("fraction", frac):
                    return _token_cutoff_from_counts(counts, int((max_tokens or DEFAULT_MAX_TOKENS) * frac))
                case _:
                    raise ValueError(f"Invalid keep: {keep!r}")

    else:

        def target() -> int:
            return _keep_target(messages, keep, max_tokens, count_tokens)

    k = target()
    while k > 0 and not is_safe_cutoff_point(messages, k):
        k -= 1
    if keep_first_user_message and k <= 1:
        return 0
    return k


def partition_head(
    prefix: Sequence[ModelMessage], keep_first_user_message: bool
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    """Split ``prefix`` into ``(head, prefix_body)``.

    ``keep_first_user_message=True`` -> ``head=[prefix[0]]`` (preserves ``messages[0]``
    including its ``instructions`` field and first user part, verbatim); ``prefix_body``
    is the rest to be summarized. ``keep_first_user_message=False`` -> ``head=[]`` and the
    caller should pair with ``ReinjectSystemPrompt`` (``instructions`` may be lost).
    """
    if keep_first_user_message and prefix:
        return [prefix[0]], list(prefix[1:])
    return [], list(prefix)
