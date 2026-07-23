# 上下文自动压缩（Context Compression）设计

> 状态：已实现（P1–P6 + 评审加固轮 P7；对应 `pydantic-ai-slim >= 2.9.0`）
> 目标项目：`pydantic-ai-extensions`（greenfield 脚手架）
> 依赖框架：`pydantic-ai-slim`
> 适用场景：多 provider 通用（含 DeepSeek 等 OpenAI-Chat 兼容 provider），为长会话 agent 自动压缩历史上下文

---

## 1. 概述

### 1.1 目标

为使用 pydantic-ai 的服务提供一个**provider 无关、可复用**的上下文自动压缩能力，在长会话中按需把较旧的历史消息压缩为摘要，控制发往模型的 token 量，同时保留近期上下文与关键框架语义（工具调用配对、系统提示、会话 ID）。

### 1.2 非目标

- **不**替代 provider 原生 compaction（`OpenAICompaction` / `AnthropicCompaction`）。对使用 OpenAI Responses 或 Anthropic 的 agent，原生能力更优（服务端精确 token 计数、prompt cache 友好）；本能力面向**无原生 compaction 的 provider**（DeepSeek、Groq、Mistral、OpenAI-Chat 兼容家族等）或需要**统一跨 provider 行为**的场景。
- **不**做向量检索/长期记忆（memory）。仅做"滑动窗口 + 摘要"式的上下文内压缩。
- **不**修改 `pydantic-ai-slim` 核心（见 §4）。

### 1.3 用户已确认的关键决策

| # | 决策点 | 选择 |
|---|--------|------|
| 1 | 触发指标 | tiktoken 估算为主 + 字符数兜底 |
| 2 | 摘要策略 | 混合（增量合并 + 周期性全量重置） |
| 3 | 持久化 | 开关，两种都实现（写回存储 / 仅请求时） |

---

## 2. 背景与 pydantic-ai 现状

pydantic-ai 已提供三层上下文管理机制（源码核实）：

1. **Provider 原生 compaction 能力**
   - `OpenAICompaction`（`pydantic_ai/models/openai.py:3904`）：要求 `OpenAIResponsesModel`，非该类型直接抛 `UserError`。stateful（服务端 `context_management`）/ stateless（`/responses/compact` 端点）两模式。
   - `AnthropicCompaction`（`pydantic_ai/models/anthropic.py:2131`）：仅 Anthropic，服务端 `context_management` beta。
   - 产出 `CompactionPart`（`pydantic_ai/messages.py:1768`），**带 `provider_name`，只回传给同一 provider** → 跨 provider 不可用。
2. **`ProcessHistory` 能力**（`pydantic_ai/capabilities/process_history.py`）：把一个 `HistoryProcessor` 挂到 `before_model_request`，每次模型请求前可改写消息列表。
3. **`Hooks(before_model_request=...)` / `wrap_model_request=...`**：底层生命周期钩子，`ProcessHistory` 是前者的薄封装。

**关键约束（决定本设计）**：DeepSeek 在本框架中由 `DeepSeekProvider` + `OpenAIChatModel` 实现（`providers/deepseek.py:29`），是 Chat Completions 协议，**无 Responses API、无原生 compaction**；`OpenAICompaction` 不可用。因此多 provider 通用的压缩只能走第 2/3 层（客户端压缩，作用于归一化 `list[ModelMessage]`）。

---

## 3. 关键设计决策与理由

### 3.1 触发指标：tiktoken 估算 + 字符兜底

- `RunContext.usage`（`RunUsage`）是**当前 run 内累计**消耗，run 首次请求前接近 0，**不能**反映"历史现在多大"，不可作触发器。
- tiktoken 对 OpenAI/DeepSeek 系列编码（`o200k_base`）较准；对 Anthropic/Gemini 是近似。作为"触发阈值"够用（留足安全余量即可）。
- `tiktoken>=0.12.0` 已是 `pydantic-ai-slim` 的 `openai` 可选依赖（`pydantic_ai_slim/pyproject.toml:73`），本扩展显式依赖它合理。
- tiktoken 不可用时（未安装/不支持的编码）按 `字符数 / char_per_token`（默认 4）兜底。

### 3.2 摘要策略：混合（增量合并 + 周期性全量重置）

- **全量重摘要**：每次把整个待压缩前缀重新摘要。简单、质量稳，但超长会话越压越贵。
- **增量合并**：保留 running summary，只摘要"自上次摘要以来的新消息批次"，再合并进 running summary。省 token，但多轮合并后质量漂移。
- **混合**：默认增量合并；每 `full_reset_every`（默认 5）次增量后做一次全量重置，消除漂移。用 `generation` 计数器跟踪。

### 3.3 持久化开关：两种都实现

| 模式 | 实现钩子 | 行为 | 适用 |
|------|----------|------|------|
| 写回（`persist=True`，默认） | `before_model_request` | 压缩后消息**写回 state**（`_agent_graph.py:980` `ctx.state.message_history[:] = messages`）→ `result.all_messages()` 即压缩后历史 → 服务持久化即压缩后 | 默认；摘要状态天然随历史跨 run |
| 仅请求时（`persist=False`） | `wrap_model_request` | 用 `dataclasses.replace(request_context, messages=compacted)` 调 handler，**不写回 state** → `result.all_messages()` 保留完整原始历史，仅模型看到压缩后 | 需可回溯/审计原始历史、或想保留换策略余地 |

> **机制核实**：`before_model_request` 的返回值在 `_prepare_request` 中被写回 state（`_agent_graph.py:980`）；`wrap_model_request` 在写回**之后**执行（`_agent_graph.py:863`），其内调用 `handler(request_context)` 才真正发起 `model.request` / `model.request_stream`（`_agent_graph.py:680,700`）。因此 wrap 内替换 messages 不影响 state。两个钩子对 `agent.run` 与 `agent.run_stream` **都生效**（流式在同一个 wrap 内调 `request_stream`）。

---

## 4. 核心结论：是否需要改 pydantic-ai 核心

**整个特性可完全建立在 pydantic-ai 现有公共扩展 API 上，零核心改动。**

| 组件 | 实现方式 | 是否改核心 | 依据 |
|------|----------|-----------|------|
| 压缩能力本体 | 子类化 `AbstractCapability`，覆写 `before_model_request` / `wrap_model_request` | 否 | `capabilities/abstract.py:514,537`（两者默认 passthrough，可覆写） |
| 持久化"写回" | 覆写 `before_model_request` 改 `request_context.messages` | 否 | `_agent_graph.py:980` 自动写回 |
| 持久化"仅请求时" | 覆写 `wrap_model_request`，`replace(ctx, messages=...)` 调 handler | 否 | `_agent_graph.py:863,680`；`ModelRequestContext` 是 `@dataclass`（`models/__init__.py:180`） |
| 触发判定 | 自建 token 估算（tiktoken + 字符兜底） | 否 | 外部库；无现成 helper，自建 |
| 安全切分 | 用 `isinstance` 检查 `ToolCallPart`/`ToolReturnPart` 等 part 类型 | 否 | `messages.py` part 类型全公共 |
| 摘要生成 | 用独立 `Agent` 调 `summarizer.run(...)` | 否 | 公共 `Agent.run` |
| 压缩状态 sentinel | 编码在 `ModelResponse` 的 `TextPart` 内容标记中（`<conversation-summary generation=N ...>`）--`ModelRequest.metadata` 会被 `_clean_message_history` 合并时丢弃 | 否 | `messages.py:1678` `TextPart`；修正依据 `_agent_graph.py` `_clean_message_history` 合并逻辑 |
| 跨 run 摘要状态（仅请求时模式） | 用户实现 `SummaryStore` 协议，作**构造器参数**传入（按 `ctx.conversation_id` 自索引，非 deps 注入） | 否 | `RunContext.conversation_id`（`_run_context.py:87`）作 key |
| 排序（在 `ReinjectSystemPrompt` 之后） | 覆写 `get_ordering()` 返回 `CapabilityOrdering(wrapped_by=[ReinjectSystemPrompt])` | 否 | `capabilities/_ordering.py`；`abstract.py:232` |
| 系统提示补回 | 配套使用 `ReinjectSystemPrompt` 能力 | 否 | `capabilities/reinject_system_prompt.py` |
| Provider 守卫（禁与原生 compaction 叠加） | 覆写 `for_run`，解包 `WrapperModel` 后 `isinstance` 检查 `AnthropicModel`/`OpenAIResponsesModel` → 抛 `UserError` | 否 | `models/wrapper.py:66`、`models/anthropic.py:470`、`models/openai.py:1705`；`for_run` 默认 `return self`（`abstract.py:244`） |
| Spec 序列化 | `get_serialization_name` 返回 `None`（持 `summarizer` 闭包，不可 spec 序列化，同 `ProcessHistory`） | 否 | `abstract.py:217`；`capabilities/process_history.py:41`（参考） |
| 能力排序 | 覆写 `get_ordering()` 返回 `CapabilityOrdering(wrapped_by=[ReinjectSystemPrompt])` | 否 | `capabilities/_ordering.py`；用法见 `_deferred_capability_loader.py:71` |
| 工具输出截断 | 覆写 `after_tool_execute`，大工具返回按头/尾行截断（`BinaryContent` 不 stringify） | 否 | `abstract.py:644`；见 §5.10 |
| 触发/切分规格 | `ContextSize`（messages/tokens/fraction）统一表达 `compress_threshold` 与 `keep`；token 二分切点 + `tool_call_id` 配对 | 否 | `BaseToolCallPart`/`BaseToolReturnPart`（`tool_call_id`）；借鉴 summarization-pydantic-ai |

> 唯一"若未来更优"的核心增强点（**非本设计必需**）：在 `Model` 上增加 `count_tokens(messages)` 公共 API 以获得 provider 精确计数。当前用 tiktoken 估算即可，记为未来可选优化。

---

## 5. 架构

### 5.1 公共 API

模块：`pydantic_ai_extensions.context_compression`

```python
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, ReinjectSystemPrompt
from pydantic_ai.exceptions import UserError
from pydantic_ai._run_context import AgentDepsT


# 触发阈值 / 近期窗口的统一规格（借鉴 summarization-pydantic-ai）
# ('messages', N) 按消息条数；('tokens', N) 按 token 数；('fraction', F) 按 max_tokens 的比例
ContextSize = tuple[Literal['messages'], int] | tuple[Literal['tokens'], int] | tuple[Literal['fraction'], float]


@dataclass
class SummaryRecord:
    """一次压缩产出的摘要状态。"""
    text: str                       # 摘要正文
    generation: int                 # 增量合并代数（全量重置归零）
    covered_count: int              # 已被覆盖的前缀消息条数（仅 persist=False 用于定位新批次；persist=True 信息性，靠 sentinel 位置定位）
    strategy: Literal['full', 'incremental']
    compacted_tokens: int | None = None
    # 冷却基线（§5.3）：压缩时记录的"下次将看到的历史估算 token 数"。
    # persist=True 记压缩后列表的估算（即新 state）；persist=False 记原始完整历史的估算。
    # None 表示旧版 sentinel/store 记录（无此字段）→ 冷却跳过一次，下次压缩补写。


class SummaryStore(Protocol):
    """跨 run 持久化 running summary（仅 `persist=False` 模式需要）。
    按 `ctx.conversation_id` 索引；实现方自行保证并发安全。仅用 conversation_id，与 deps 类型无关，故非泛型。"""

    async def get(self, ctx: RunContext[Any]) -> SummaryRecord | None: ...
    async def put(self, ctx: RunContext[Any], record: SummaryRecord) -> None: ...


@dataclass(init=False)
class ContextCompression(AbstractCapability[AgentDepsT]):
    """Provider 无关的上下文自动压缩能力。

    `summarizer` 必须是**无工具**的 agent（`output_type=str`，默认）--否则它可能调工具而非产出摘要文本；
    其 `instructions` 即压缩 prompt 的可配入口（见 §5.8）。
    """

    def __init__(
        self,
        summarizer: Agent[Any, str],
        *,
        # 触发
        compress_threshold: ContextSize = ('fraction', 0.7),  # 何时触发：messages/tokens/fraction
        max_tokens: int | None = None,     # fraction 模式分母；None 取默认 128_000（profile 不暴露 context window，见 §5.4）
        encoding: str | None = None,        # None → o200k_base（OpenAI/DeepSeek 通用近似）
        char_per_token: int = 4,            # tiktoken 不可用时的兜底
        include_thinking_in_estimate: bool = True,  # 是否把 ThinkingPart 计入 token 估算（见 §5.4）
        # 切分
        keep: ContextSize = ('messages', 6),  # 近期窗口：messages/tokens/fraction；('messages', 0) 表示不设下限
        keep_first_user_message: bool = True,
        min_prefix: int = 4,                # 可压缩前缀不足此数则不压缩、返回 None（见 §5.3）
        # 策略
        strategy: Literal['full', 'hybrid'] = 'hybrid',
        full_reset_every: int = 5,          # hybrid：每 N 次增量后全量重置
        # 冷却与摘要上限（§5.3、§5.9）
        min_recompact_growth_tokens: int | None = 2_048,  # 距上次压缩基线至少增长这么多 token 才再压缩；0/None 关闭
        max_summary_tokens: int | None = None,  # 摘要长度上限，超出做 token 精确中段截断
        # 工具输出截断（见 §5.10，从源头控上下文，与压缩正交互补）
        max_tool_output_tokens: int | None = None,  # None 不截断
        tool_output_head_lines: int = 5,
        tool_output_tail_lines: int = 5,
        # 持久化
        persist: bool = True,
        summary_store: SummaryStore | None = None,   # 仅 persist=False 时使用
    ) -> None:
        if persist and summary_store is not None:
            raise UserError(
                '`summary_store` 仅在 `persist=False` 时使用；`persist=True` 的状态走历史 sentinel，store 会被忽略。'
            )
        # 构造期 fail-fast 校验（均抛 UserError）：ContextSize 规格合法
        # （messages N>=0 / tokens N>=1 / 0<fraction<1，含运行期形状检查）、
        # min_prefix>=1、full_reset_every>=1、char_per_token>=1、
        # min_recompact_growth_tokens>=0、max_summary_tokens>=1、strategy in ('full','hybrid')
        ...

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(wrapped_by=[ReinjectSystemPrompt])  # 排在 ReinjectSystemPrompt 之后（见 §7）

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # 持有 summarizer（含闭包），不可 spec 序列化（同 ProcessHistory）

    async def for_run(self, ctx): ...                 # Provider 守卫，见 §5.7
    async def before_model_request(self, ctx, request_context): ...   # 压缩，见 §5.2
    async def wrap_model_request(self, ctx, *, request_context, handler): ...  # 见 §5.2
    async def after_tool_execute(self, ctx, *, call, tool_def, args, result): ...  # 工具输出截断，见 §5.10
```

**使用示例**：

```python
from pydantic_ai import Agent
from pydantic_ai_extensions.context_compression import ContextCompression

summarizer = Agent('deepseek:deepseek-chat', instructions='精简总结技术讨论、已做决策与待办，忽略寒暄。')
# ↑ summarizer 的 instructions 即"压缩 prompt"的可配置入口（见 §5.8）
agent = Agent(
    'deepseek:deepseek-chat',
    capabilities=[ContextCompression(summarizer, compress_threshold=('tokens', 24_000))],
)
```

### 5.2 持久化开关：两个钩子的分工

```python
async def before_model_request(self, ctx, request_context):
    # 写回模式：在此压缩，框架会写回 state（_agent_graph.py:980）
    if not self.persist:
        return request_context                      # 仅请求时模式交给 wrap
    compacted = await self._maybe_compact(ctx, request_context.messages)
    if compacted is not None:
        request_context.messages = compacted
    return request_context

async def wrap_model_request(self, ctx, *, request_context, handler):
    # 仅请求时模式：替换发给模型的 messages，不动 state
    if self.persist:
        return await handler(request_context)       # 写回模式已在 before 里处理
    compacted = await self._maybe_compact(ctx, request_context.messages)
    if compacted is not None:
        return await handler(replace(request_context, messages=compacted))
    return await handler(request_context)
```

### 5.3 混合摘要引擎（状态机）

`_maybe_compact` 的核心逻辑：

```
if len(messages) < min_prefix + 1: return None          # 快路径：不可能存在可压缩前缀（见 §5.4）
if not should_trigger(messages, compress_threshold, max_tokens): return None  # 未超阈值，不动（见 §5.4）

k = find_safe_split(messages, keep, max_tokens, keep_first_user_message)  # token 二分切点 + tool_call_id 配对，见 §5.5
if k < min_prefix: return None                  # 可压缩前缀太小

head, prefix_body = partition_head(messages[:k], keep_first_user_message)
    # head = messages[0]（keep_first_user_message=True 时）：其 instructions 字段 + 首条 user part，整条原样保留
    # prefix_body = 其余待摘要消息（keep_first_user_message=False 时 messages[0] 也进摘要；
    #   instructions 由框架每次模型请求重新盖章，不随压缩丢失，见 §6.4）

existing = await load_existing_summary(ctx, prefix_body)
    # persist=True  → find_summary(prefix_body)：严格正则解析首个 sentinel（见 §5.6）
    # persist=False → await summary_store.get(ctx)（按 conversation_id；get 失败记日志并降级为 None）

# 冷却（§6.1）：阈值是触发线不是上限，压缩后历史可能仍在阈值之上，
# 若不加控制工具循环内每一步都会重压。距上次压缩基线增长不足 delta 时不再压缩（0/None 关闭）。
if existing and min_recompact_growth_tokens and existing.compacted_tokens is not None \
        and estimate_tokens(messages) - existing.compacted_tokens < min_recompact_growth_tokens:
    return None

do_full = (existing is None) or (strategy == 'full') or (existing.generation >= full_reset_every)

try:
    if do_full:
        result = await summarize_full(ctx, prefix_body)            # 见 §5.9；内部 await summarizer.run(...)
        record = SummaryRecord(result.output, generation=0, covered_count=k, strategy='full')
    else:
        new_batch = prefix_body_after(existing, prefix_body)       # 自上次覆盖点之后的新消息
        if not new_batch: return None                              # 增长全在近期窗口内，无新内容可合并
        result = await summarize_incremental(ctx, existing.text, new_batch)
        record = SummaryRecord(result.output, generation=existing.generation + 1,
                               covered_count=k, strategy='incremental')
except _SUMMARIZER_EXCEPTIONS:
    return None  # 摘要器失败：降级为不压缩、原历史发往模型，不阻断父 run（见 §5.9）

merge_usage(ctx, result)  # 把摘要器本次用量并入父 run 的 RunUsage（见 §5.9）
if max_summary_tokens is not None:
    record.text = truncate_text_to_tokens(record.text, max_summary_tokens)  # 摘要长度上限（§5.9）

# 冷却基线：persist=True 记压缩后列表的估算（即新 state）；persist=False 记原始完整历史的估算
record.compacted_tokens = estimate_tokens(compacted_draft if persist else messages)
summary_msg = build_summary_message(record)       # ModelResponse（sentinel 在 TextPart 内容标记中）
compacted = [*head, summary_msg, *messages[k:]]

if not persist: await summary_store.put(ctx, record)  # put 失败仅记日志：本次压缩仍生效，仅丢跨 run 状态
return compacted
```

- **`persist=True` 的增量检测**：压缩后历史头部为 `[head..., summary_resp(sentinel), ...新消息..., 近期窗口]`（sentinel 是单个 `ModelResponse`，见 §5.6）。下次触发时，在 `prefix_body` 中找到 sentinel，其后的消息即"自上次摘要以来的新批次"，增量合并后**替换**旧 sentinel（generation+1）；`generation >= full_reset_every` 时改为全量重置（generation 归零）。
- **`persist=False` 的增量检测**：历史不变，`SummaryRecord.covered_count` 单调递增；`new_batch = messages[existing.covered_count : k]`，合并后 `covered_count = k`。
- **`persist=False` 且未提供 `summary_store`**：降级为每次全量重摘要（无跨 run running summary）。文档与构造器明确提示此降级。

### 5.4 token 估算

```python
def estimate_tokens(messages: list[ModelMessage], *, encoding: str, char_per_token: int) -> int:
    text = render_messages_to_text(messages)
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding)     # 默认 'o200k_base'
        return len(enc.encode(text, disallowed_special=()))
    except (ImportError, ValueError):
        return max(1, len(text) // char_per_token)

def should_trigger(messages, compress_threshold: ContextSize, max_tokens: int | None) -> bool:
    """是否达到压缩触发线（ContextSize 三种规格）。estimate_tokens 用能力配置的 encoding/char_per_token。"""
    match compress_threshold:
        case ('messages', N):  return len(messages) >= N
        case ('tokens', N):    return estimate_tokens(messages) >= N
        case ('fraction', F):  return estimate_tokens(messages) >= int((max_tokens or 128_000) * F)
```

> **`max_tokens` 自动探测的局限**：pydantic-ai 的 model profile **不暴露 input context window**（profile 里只有 `max_tokens` 输出设置，见 `profiles/openai.py:171`），因此 `max_tokens` 无法从 profile 自动获取。当前设计：用户显式传 `max_tokens`；未传时 fraction 模式取默认 128_000 并告警。真正的自动探测需外部源（如 `genai-prices` 包或自建 model 注册表），记为未来增强。

> **估算不含工具 schema**：`estimate_tokens` 只渲染消息体（含 `ModelRequest.instructions`），不含每次请求随发的 tool definitions（`model_request_parameters`）。工具多的 agent 会系统性低估，`fraction` 阈值需为此留余量（默认 0.7 已含常规余量）。

`render_messages_to_text` 把各 part 渲染为纯文本（用于估算与送摘要器）：

| Part | 渲染 |
|------|------|
| `SystemPromptPart` | `content` |
| `UserPromptPart` | `content`（`str` 或序列中取 `str`/`TextContent`） |
| `TextPart` | `content` |
| `ThinkingPart` | `content`（受 `include_thinking_in_estimate` 控制；默认计入以提准估算，关闭则跳过） |
| `ToolCallPart` | `f"assistant called tool {tool_name} with {args}"` |
| `NativeToolCallPart` | 同上 |
| `ToolReturnPart` / `NativeToolReturnPart` | `f"tool {tool_name} returned {content}"` |
| `RetryPromptPart` | `f"tool {tool_name} retry: {content}"` |
| `InstructionPart` | `content` |
| `CompactionPart` | `content or "<encrypted compaction>"`（迁移自原生压缩时保留） |
| `FilePart` 等媒体 | `"<media>"`（不计入精确 token） |

> 上表描述的 `render_messages_to_text` 是**扁平散文渲染**，仅用于 token 估算。要点：
> - **另计 `ModelRequest.instructions` 字段**（agent 系统提示，非 part）--模型实际要为它付 token，漏算会低估；渲染时对每个 `ModelRequest` 先输出其 `instructions` 再输出各 part。
> - `UserPromptPart.content` 为序列时，`ImageUrl`/`AudioUrl`/`BinaryContent` 等非文本项渲染为 `<media>`（多模态有损，见 §6）。
> - §5.8 摘要输入用的是另一个渲染器 `render_structured`（XML-ish 结构标记，保留角色/工具调用/返回的结构边界），两者独立。
> - **快路径**：`_maybe_compact` 先按消息条数粗判（`len(messages) < min_prefix + 1` 直接返回 None——此时不可能存在长度 ≥ `min_prefix` 的可压缩前缀），免跑 tiktoken；仅长度可能超阈值时才调 `should_trigger`/`estimate_tokens`。注意 tokens/fraction 模式下，**每次模型请求**（含工具循环每步）仍会对当前历史做一次 render + tiktoken 编码（O（历史体量））；这是触发设计的固有成本，长会话下可接受，未来可按消息增量缓存计数优化。

### 5.5 安全边界切分（工具调用配对）

**硬约束**：裁剪/摘要不得把一个 `ToolCallPart`（在 `ModelResponse` 里）与其对应的 `ToolReturnPart`/`RetryPromptPart`（在下一个 `ModelRequest` 里）拆开，否则所有 provider 都会报错（见 pydantic-ai issue #2050）。

**规则**：消息序列形如 `Req, Resp, Req, Resp, ...`，末尾必为 `ModelRequest`（当前请求）。切点 `k` 必须保证**没有任何 `ToolCallPart` 与其返回 `ToolReturnPart` 被拆到两侧**--按 `tool_call_id` 精确配对判定（`ToolCallPart.tool_call_id` ↔ `ToolReturnPart.tool_call_id`，见 `messages.py` 的 `BaseToolCallPart`/`BaseToolReturnPart`）。这比"前缀必须以 `ModelRequest` 结尾"的结构式规则更灵活（允许切在无工具调用的 `ModelResponse` 之后），且是 token 二分切点所需的安全原语。

```python
def is_safe_cutoff_point(messages, k: int) -> bool:
    """切点 k（prefix=messages[:k], recent=messages[k:]）是否安全：
    不把任一 ToolCallPart 与其同 tool_call_id 的 ToolReturnPart 拆到两侧。"""
    prefix_calls = {p.tool_call_id for m in messages[:k] if isinstance(m, ModelResponse)
                    for p in m.parts if isinstance(p, ToolCallPart)}
    recent_returns = {p.tool_call_id for m in messages[k:] if isinstance(m, ModelRequest)
                      for p in m.parts if isinstance(p, ToolReturnPart)}
    return not (prefix_calls & recent_returns)       # 调用在 prefix、返回在 recent -> 悬空，不安全

def find_safe_split(messages, keep: ContextSize, max_tokens: int | None,
                    keep_first_user_message: bool, count_tokens=None) -> int:
    n = len(messages)
    target = _keep_target(messages, keep, max_tokens, count_tokens)  # 期望切点（见下）
    k = target
    while k > 0 and not is_safe_cutoff_point(messages, k):
        k -= 1                                       # 向前回退到安全边界（recent 只增不减）
    if keep_first_user_message and k <= 1:
        return 0                                     # 既要保留首条又没有可压缩前缀
    return k                                         # prefix = messages[:k], recent = messages[k:]

def _keep_target(messages, keep: ContextSize, max_tokens: int | None, count_tokens) -> int:
    """把 keep 规格转成期望切点 target。keep 是**下限**：保留侧至少为 keep 规格的量。
    返回值钳到 >= 0（('messages', N>=n) 不会产生负下标）。"""
    n = len(messages)
    match keep:
        case ('messages', N):  return max(n - max(N, 1), 0)      # N<=0 -> 不设下限（target=n-1，由 is_safe_cutoff_point 回退到最小安全窗）
        case ('tokens', N):    return _token_cutoff(messages, N, count_tokens)        # 二分：保留侧 token 数 >= N 的最大 cutoff
        case ('fraction', F):  return _token_cutoff(messages, int((max_tokens or 128_000) * F), count_tokens)
```

- `is_safe_cutoff_point` 按 `tool_call_id` 配对（`ToolCallPart`↔`ToolReturnPart`），比旧"前缀以 `ModelRequest` 结尾"更灵活（可切在纯文本 `ModelResponse` 之后），正确性不变。`RetryPromptPart` 视同返回处理（其带 `tool_call_id`，已核实）。
- `keep` 是**下限**：token 二分找"保留侧 token 数 ≥ N 的最大 cutoff"（保留至少 N tokens 的最小窗口；单条大消息可略超 N），回退到安全边界只会让 recent 更大、prefix 更小，正确性优先于压缩率。`('messages', 0)` 表示不设下限--仅保留安全边界强制的最小窗口（当前请求 + 维持配对所需，通常 2 条），最大化压缩。
- token 二分切点（`('tokens', N)` / `('fraction', F)`）：比消息数更贴合"按体量保留"；二分落点经 `is_safe_cutoff_point`（`tool_call_id` 配对）回退到安全点。
- `keep_first_user_message=True` 时，`messages[0]`（首条用户请求）原样保留在 head，不进摘要；`prefix_body = messages[1:k]`。head（req）+ sentinel（resp）角色交替、无连续同角色消息；即使出现连续 `ModelRequest`，也由框架 `_clean_message_history` 合并处理（仅合并同 instructions 的连续请求），各 provider 接受多轮 user。

### 5.6 Sentinel 机制

**实现中修正的设计缺陷**：最初设计把 sentinel 放在 `ModelRequest.metadata`。但 `_clean_message_history` 合并连续同角色消息时会**丢弃 metadata**（"we intentionally don't block merging when metadata differ, nor try to preserve them"），导致压缩后的 `head(req) + summary_req(req)` 合并后 sentinel metadata 丢失、跨 run 状态断裂。修正为：**sentinel 编码在摘要 part 的 *内容标记* 中**（合并时 part 内容存活），且摘要改为单个 `ModelResponse`（避免 req-req 合并）。

```python
def build_summary_message(record: SummaryRecord) -> ModelResponse:
    content = (
        f"<conversation-summary generation={record.generation}"
        f" covered_count={record.covered_count} strategy={record.strategy}"
        f" compacted_tokens={record.compacted_tokens}>\n{record.text}"  # compacted_tokens 非 None 时才带该属性
    )
    return ModelResponse(parts=[TextPart(content=content)])
```

- `persist=True`：sentinel 随历史持久化。`load_existing_summary` 经 `find_summary(prefix_body)` 扫描 `ModelResponse` 的 `TextPart`，用 `parse_summary_sentinel(content)` **严格解析**（正则锚定内容开头）：`<conversation-summary generation=N covered_count=K strategy=S compacted_tokens=T>` 后接正文。正则 `^<conversation-summary generation=(\d+) covered_count=(\d+) strategy=(\w+)(?: compacted_tokens=(\d+))?>\n?(.*)$`——`compacted_tokens` 可缺省以兼容旧版 sentinel；缺省解析为 `None`，冷却跳过一次、下次压缩补写。
- **判定统一为严格正则解析**：`find_summary` / `prefix_body_after` / `load_existing_summary` 都只认完整匹配（不再用 `startswith`），取第一个完整匹配的 sentinel（capability 布局中真实 sentinel 恒在 head 之后第一位，先于任何模型逐字引用）。模型在历史中见过 sentinel 后复述标记的概率低，且复述通常不满足"内容开头 + 完整属性格式"。
- `persist=False`：sentinel 不全量进 state，跨 run 状态改由 `SummaryStore` 承载（`parse_summary_sentinel` 仍用于 same-run sentinel 检测）。
- 标记也是轻量调试辅助；**判定以内容标记解析为准**（无 metadata 依赖）。

### 5.7 Provider 守卫（禁止与原生 compaction 叠加）

具备**原生上下文管理/compaction**的 provider 不应使用本能力——其原生机制更优（服务端精确计数、prompt cache 友好），叠加会互相破坏前缀与状态。因此在 `for_run`（每次 run 开始、钩子触发前调用一次）中做一次性守卫，fail-fast：

```python
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIResponsesModel

async def for_run(self, ctx):
    model = ctx.model
    while isinstance(model, WrapperModel):       # 解包 WrapperModel 包装层（逐层取 .wrapped）
        model = model.wrapped
    if isinstance(model, (AnthropicModel, OpenAIResponsesModel)):
        raise UserError(
            f'ContextCompression 不支持 {type(model).__name__}：该 provider 具备原生 compaction，'
            f'请改用 AnthropicCompaction / OpenAICompaction。'
        )
    return self                                  # 不缓存实例状态（见 §11.1 (Q4)）
```

- **允许**：`OpenAIChatModel`（DeepSeek/Groq/Mistral 等 OpenAI-Chat 兼容）及其他无原生 compaction 的 model。
- **禁止**：`AnthropicModel`、`OpenAIResponsesModel`。未来若新增带原生 compaction 的 model，应在此 `isinstance` 列表中补判。
- **"类似接口"**：其他带自有上下文预算/compaction 的 provider，若其 model 类不在上面列表中，框架无法自动识别；由调用方负责不与本能力叠加（含 `anthropic_task_budget.remaining` 等），并在自身服务文档标注。
- 守卫放 `for_run` 以"每 run 一次、尽早失败"；若该时机模型尚未最终解析，可下沉到 `_maybe_compact` 首次调用（isinstance + 解包成本极低）。

### 5.8 摘要 prompt 的构成

压缩用到的 prompt 分两层，**只有"指导层"可配置，"框架层"固定**：

| 层 | 内容 | 来源 | 是否可配 |
|----|------|------|---------|
| 指导层（system） | "摘要什么、保留什么、忽略什么、输出格式" | **`summarizer` agent 的 `instructions`** | 是（构造 summarizer 时设置） |
| 框架层（user） | 把历史装进任务模板 | 能力内部固定模板 | 否（结构化、正确性关键） |

即：**压缩 prompt 的可配置入口就是 `summarizer` agent 的 `instructions`**——pydantic-ai 的一等概念，无需在能力上另设参数（故不设 `summary_instructions`，避免两个竞争来源）。想要"每个 capability 不同摘要风格"，构造不同的 summarizer agent 即可。

#### 摘要输入：`render_structured` —— 结构化文本 + 声明式 prompt

把 `prefix_body` 渲染为**结构化标签文本**（保留角色/工具调用/返回的结构），不传 `message_history`——summarizer 的 `instructions` 作为 system prompt 正常生效。框架层 prompt 声明"这是一段结构化历史，而非普通文本"——这是个有效的 prompt 引导（nudge，非强制：LLM 收到的终究是文本），经验上让它更按结构而非逐字散文来理解：

渲染格式（`render_structured`，XML-ish 结构标记）：
```
<conversation-history>
<message role="user">SF 的天气？</message>
<message role="assistant" tool_call="get_weather" arguments='{"city":"SF"}' />
<message role="tool" tool_name="get_weather">{"temp":60,"unit":"F"}</message>
<message role="assistant">SF 现在 60°F。</message>
</conversation-history>
```

`render_structured` 对其余 part 的处理：`ThinkingPart` -> `<message role="assistant" thinking="...">`；`RetryPromptPart` -> `<message role="tool" tool_name="..." retry>`；`InstructionPart` -> `<message role="system">`；`CompactionPart` -> 取 `content` 作 `<message role="system" compaction>`（`content=None` 跳过）；多模态 `UserPromptPart` 中的 `ImageUrl` 等 -> `<message role="user"><media kind="image" /></message>`（有损，见 §6）。

- 全量框架层 prompt（user）：
  ```
  The following is a structured conversation history (wrapped in <conversation-history>; each <message> is tagged with its role, tool calls/returns as structured attributes). Summarize it, preserving key facts, decisions and TODOs, omitting small talk and irrelevant detail:

  {render_structured(prefix_body)}
  ```
- 增量框架层 prompt（user）：
  ```
  Previous summary:
  {existing.text}

  New structured content since the last summary:
  {render_structured(new_batch)}

  Merge the new content into the previous summary and output the updated full summary (coherent prose, do not itemize).
  ```
- `summarizer.run(structured_text)`：不传 `message_history` → summarizer 的 `instructions` 作为 system prompt 正常生效；`result.output`（str）即摘要正文。
- 优点：结构边界清晰（角色/工具调用/返回各自独立），LLM 更易按结构而非逐字散文理解；且保留该路径全部优势——任意模型可用、无需工具上下文、省 token、summarizer 的 `instructions` 正常生效。`render_structured` 与 §5.4 用于 token 估算的扁平 `render_messages_to_text` 是两个渲染器（后者仅作估算近似）。

> `result.output`（str）写入 `SummaryRecord.text` / sentinel；指导层（summarizer.instructions）始终是"保留什么/输出格式"的可配入口。

#### 为何不提供 `message_history` 模式（曾评估，暂不实现）

曾考虑把 `prefix_body` 作为真实 `message_history` 传入（保留原生 tool part），评估后暂不实现：

- 结构保真的边际收益已被 `render_structured` 覆盖（角色/工具调用/返回的结构边界已显式编码）；
- 该模式需合成 `SystemPromptPart(<summarizer.instructions>)` 才能让指导层送达，而 `instructions` 可为 str/callable/sequence，仅静态形态可支持，等同二等公民；
- 增量合并需伪造 `ModelResponse(TextPart(旧摘要))` 作为 history，构造别扭；
- 多数 provider 接受 history 中的 tool-call/return 而无需当前声明 tools，但个别严格 provider 可能拒绝（已知 caveat）。

若将来出现 `render_structured` 覆盖不了的具体场景（如重度多模态，`FilePart`/图片难以文本编码），再开 issue 评估。

### 5.9 运行时契约与辅助函数

#### 辅助函数签名（§5.3 引用，此处定义）

```python
def partition_head(prefix: list[ModelMessage], keep_first_user_message: bool) -> tuple[list[ModelMessage], list[ModelMessage]]:
    """(head, prefix_body)。keep_first_user_message=True -> head=[prefix[0]]（含 instructions 字段 + 首条 user part）；
    否则 head=[]（instructions 可能丢，须配 ReinjectSystemPrompt）。prefix_body = prefix[len(head):]。"""

async def load_existing_summary(ctx, prefix_body) -> SummaryRecord | None:
    """persist=True -> find_summary(prefix_body) 严格解析首个 sentinel，取摘要文本 + generation/covered_count/compacted_tokens。
    persist=False -> await summary_store.get(ctx)（get 失败记 warning 日志并返回 None，降级全量）。"""

def prefix_body_after(existing, prefix_body) -> list[ModelMessage]:
    """仅 persist=True 用：prefix_body 中旧 sentinel 之后的消息（靠 find_summary 严格解析定位）。
    persist=False 的增量在 _maybe_compact 内用 messages[existing.covered_count:k]，不用此函数。"""

async def summarize_full(ctx, prefix_body) -> AgentRunResult[str]:
    """render_structured(prefix_body) 作 user prompt（§5.8 全量模板），await summarizer.run(...)，返回 AgentRunResult。"""

async def summarize_incremental(ctx, old_text: str, new_batch) -> AgentRunResult[str]:
    """§5.8 增量模板（old_text + render_structured(new_batch)），await summarizer.run(...)，返回 AgentRunResult。"""

def build_summary_message(record: SummaryRecord) -> ModelResponse:
    """ModelResponse(TextPart("<conversation-summary generation=N covered_count=K strategy=S compacted_tokens=T>\\n摘要"))--sentinel 编码在内容标记中。"""

def merge_usage(ctx, result: AgentRunResult[str]) -> None:
    """把 result.usage 并入 ctx.usage（2.x 中 `AgentRunResult.usage` 是 property：`ctx.usage.incr(result.usage)`）。"""
```

#### 嵌套 `summarizer.run()` 的安全性

`_maybe_compact` 在父 run 的钩子内调 `summarizer.run()`，即父 run 内嵌套一个完整 agent run。pydantic-ai 用 contextvar（`set_current_run_context`）作用域 run 上下文，嵌套 run 在自己的 `with` 块内设置、退出即恢复，父 run 的 `current_run_context` 不受污染。**实现时需验证**：① 嵌套 run 的 contextvar、logfire/instrumentation span 不泄漏进父 run；② 摘要器是独立 agent 实例，不携带父 agent 的工具/依赖；③ 每次压缩触发一次额外模型调用，增加该轮延迟（见 §6）。

#### 失败处理

`summarizer.run()` 抛异常时，`_maybe_compact` catch 后 `return None`（§5.3）--降级为不压缩、原历史发往模型，**不阻断父 run**。catch 面已收窄为摘要器运行期异常：

```python
_SUMMARIZER_EXCEPTIONS = (ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded, TimeoutError)
```

pydantic-ai 已把 provider SDK 的 `APIStatusError`/`APIConnectionError` 统一映射到 `ModelAPIError`（`models/openai.py:193-198`），故该集合覆盖 API 错误、连接错误、重试耗尽、用量超限与传输超时；编程错误（`KeyError`/`AttributeError` 等）**故意不捕获**，让其暴露而非静默禁用压缩。失败时通过 stdlib `logging` 记 warning（含 traceback）。`summary_store.get`/`put` 是用户代码，各自宽捕获 + warning 日志：`get` 失败降级为全量重摘要；`put` 失败仅丢跨 run 状态，本次压缩仍生效。

#### 摘要长度上限（`max_summary_tokens`）

啰嗦的 summarizer 可能产出比 prefix 还长的"摘要"（压缩后反而膨胀）。设 `max_summary_tokens` 后，产出经 `truncate_text_to_tokens` 做 token 精确中段截断（tiktoken 可用时按 token 切，否则按 `tokens * char_per_token` 字符预算切），保证摘要本体不超预算。

#### `persist=False` 的两条约束

- **可观测性不对称**：wrap 内换 messages 调 handler，但写回 state 的 `last_request_context` 仍是**原始** messages。调试时"state=原始、模型收到=压缩后"是预期，不是 bug。
- **须加载完整连续历史**：`SummaryRecord.covered_count` 是历史中的消息索引；若服务只存/加载部分历史，索引失效。`persist=False` 要求每轮 `agent.run` 加载该会话**完整**历史（这也是该模式的本意--state 保留全量）。

### 5.10 工具输出截断（`after_tool_execute`）

从源头控制上下文体量，与压缩正交互补：大工具返回在入历史前先截断，减少压缩触发频率。覆写 `after_tool_execute(self, ctx, *, call, tool_def, args, result) -> result`（签名见 `capabilities/abstract.py:644`）：

```python
async def after_tool_execute(self, ctx, *, call, tool_def, args, result):
    if self.max_tool_output_tokens is None:
        return result
    if isinstance(result, BinaryContent):           # 二进制/多模态不 stringify，原样返回
        return result
    text = _stringify_tool_result(result)            # str/dict/pydantic model -> 文本
    if estimate_tokens_str(text) <= self.max_tool_output_tokens:
        return result
    lines = text.splitlines()
    head, tail = self.tool_output_head_lines, self.tool_output_tail_lines
    if len(lines) > head + tail:
        candidate = '\n'.join([*lines[:head], f'...[truncated {len(lines) - head - tail} lines]...', *lines[-tail:]])
        if estimate_tokens_str(candidate) <= self.max_tool_output_tokens:
            return candidate                         # 行级截断且已达标
    # 行级截断无法达标（单行大 JSON / 超长行）→ token 精确中段截断兜底
    return truncate_text_to_tokens(text, self.max_tool_output_tokens, ...)
    # 框架随后把返回值包成 ToolReturnPart 入历史
```

- `max_tool_output_tokens=None`（默认）不截断；设值后，工具结果估算 token 超出则先按头/尾行截断；若行级结果仍超预算、或行数不足以行截断（如单行大 JSON），兜底为 `truncate_text_to_tokens` 的 token 精确中段截断（tiktoken 可用时按 token 切，否则按 `tokens*char_per_token` 字符预算切），保证截后体量达标。
- **`BinaryContent` / 多模态结果不 stringify**（直接返回原 `result`），避免把二进制当文本截断。
- 仅截断工具**返回**（`ToolReturnPart` 来源），不动工具调用本身；可按 `tool_def` 返回类型决定是否截断。
- 与压缩的关系：截断是"预防性瘦身"（每步工具返回即时限长），压缩是"周期性汇总"（历史超阈值时摘要）。二者叠加--先截断控增量、再压缩控存量。
- 注意：`result: Any`，截断仅对可 stringify 内容有效；结构化/pydantic 结果截断后类型变 `str`（调用方需知悉）。

---

## 6. 边界情况与陷阱

1. **`before_model_request` 在多步工具循环里每次模型请求都触发**，且处理结果写回 state。阈值是触发线而非上限（§6.10），压缩后的历史仍可能在阈值之上——若不加控制，工具循环内**每一步**都会重压摘要（实测：单 run 6 步模型调用 → 6 次摘要调用，模型调用数与延迟翻倍）。因此引入**冷却机制**（`min_recompact_growth_tokens`，默认 2048）：每次压缩把"下次将看到的历史估算"记入 `SummaryRecord.compacted_tokens`（persist=True 写入 sentinel；persist=False 写入 store），仅当历史较该基线增长 ≥ delta 时才再次压缩。实测同场景 6 步 → 1 次摘要调用。`0`/`None` 关闭冷却（退回每步判定，仅建议测试用）。
2. **tool-call/return 配对**：见 §5.5。`find_safe_split` 保证不悬空。
3. **`CompactionPart` 跨 provider 不可用**：本能力**不产出** `CompactionPart`，只产出纯文本 `UserPromptPart`/`TextPart`，任何 provider 可读、可持久化、可跨 provider round-trip。若历史中残留过往原生压缩的 `CompactionPart`，`render` 时取其 `content`（Anthropic 可读）或忽略（OpenAI 加密，`content=None`），并在文档提示：跨 provider 前最好清洗。
4. **系统提示（区分 `instructions` 与 `SystemPromptPart`，两个概念）**：2.x 框架在**每次**模型请求前重新解析 agent 的 `instructions` 并盖章到当前请求（`_agent_graph.py:954-957`：`self.request.instructions = ...`），因此压缩删掉 `messages[0]` 的 `instructions` 字段**不会**丢系统提示——当前请求始终携带最新 instructions。`keep_first_user_message=False` 的真正代价是首条用户消息**内容**被并入摘要（有损但可控）。而 `SystemPromptPart`（静态系统提示 part）是历史消息的一部分，落在 `prefix_body` 内会被摘要掉，需配套 `ReinjectSystemPrompt`（它补回的正是 `SystemPromptPart`，不是 `instructions`）或调大 `keep` 规避。
5. **prompt cache 代价**：客户端重组历史会破坏 provider 的 prefix cache（OpenAI/Anthropic 均依赖前缀稳定）。这是客户端压缩相对 stateful 原生压缩的固有劣势，无原生可选时接受此代价。优化方向：摘要固定追加在 head 之后、recent 顺序不变，尽量保持前缀稳定。
6. **token 估算精度**：tiktoken 对非 OpenAI 系为近似。阈值留足安全余量（如 context window 的 60–70%）。
7. **并发**：`persist=False` + `SummaryStore` 时，store 实现须并发安全（多请求/多 run 可能并发读写同一 conversation）。`persist=True` 无此问题（状态在历史里，由调用方持久化逻辑串行化）。
8. **空/极短历史**：`min_prefix` 守卫；`find_safe_split` 返回 0 时不压缩。
9. **`new_messages()` 语义与持久化方式**：`persist=True` 会改写 state，但插入的 summary 消息**不会**出现在 `result.new_messages()` 中——框架只对 `messages[-1]` 和模型响应盖章 `run_id`（`fill_run_metadata`，`_agent_graph.py:992,1067`），capability 构造的 sentinel `run_id=None`，`_first_run_id_index` 从当前请求起算（实测确认）。因此 **`persist=True` 必须用 `result.all_messages()` 快照式持久化**；依赖 `new_messages()` 做增量追加持久化的服务会完整丢失压缩结果（库里保留未压缩历史，压缩每轮重算且 sentinel 永不落库），此类服务应改用 `persist=False` + `SummaryStore`。
10. **阈值是触发线，不是上限**：`compress_threshold` 仅决定"何时触发压缩"；压缩后 `summary + keep` 不保证 ≤ 阈值、更不保证 ≤ 模型 context window。`keep` 设过大（相对阈值/窗口）会使压缩失效。本能力是 best-effort，不保证装得下--需按模型窗口留足余量，必要时减小 `keep`。
11. **工具输出截断**：`after_tool_execute` 截断大工具返回（§5.10）；`BinaryContent` 不 stringify；截断后类型变 `str`，结构化结果调用方需知悉。
12. **token 二分切点**：`('tokens', N)`/`('fraction', F)` 按体量保留近期，比消息数更贴合；二分落点经 `is_safe_cutoff_point`（`tool_call_id` 配对）回退到安全点。

---

## 7. 与其他 pydantic-ai 能力的协同

| 能力 | 协同方式 |
|------|----------|
| `ReinjectSystemPrompt` | **强烈建议配套**。压缩可能移除含系统提示的头部，`ReinjectSystemPrompt` 在缺失时补回。排序：`ContextCompression` 声明 `wrapped_by=[ReinjectSystemPrompt]`（Reinject 先跑，压缩后跑、看到完整头部）。 |
| `OpenAICompaction` / `AnthropicCompaction` / `task_budget` | **禁止**与 `ContextCompression` 叠加。对 `AnthropicModel`/`OpenAIResponsesModel`，本能力在 `for_run` 直接抛 `UserError`（§5.7），应改用原生 `AnthropicCompaction`/`OpenAICompaction`。多 provider 统一行为时，仅对无原生 compaction 的 provider（DeepSeek 等 `OpenAIChatModel`）用本能力。 |
| `ProcessHistory`（其他用途，如隐私过滤） | 可组合。若过滤应在压缩前生效（压缩过滤后的结果），让 `ProcessHistory` 排在 `ContextCompression` 之前（`wrapped_by=[...]`）；反之亦然。 |
| `Hooks` | `ContextCompression` 即子类化 `AbstractCapability`，等价于声明式钩子；无需再用裸 `Hooks`。 |

> **⚠ Q6 强制约束**：具备原生上下文管理/compaction 的 provider（`AnthropicModel`、`OpenAIResponsesModel`）**不得**使用本能力——`for_run` 守卫会抛 `UserError`（§5.7）。对它们改用原生 `AnthropicCompaction` / `OpenAICompaction`，且不要叠加 `anthropic_task_budget.remaining`。本能力仅用于无原生 compaction 的 provider（DeepSeek/Groq/Mistral 等 `OpenAIChatModel`）。

---

## 8. 测试策略

遵循 pydantic-ai 测试规范（`tests/AGENTS.md`）：偏好集成测试、真实请求用录制/快照、用 `FunctionModel` 断言 provider 实际收到的消息形状。已实现用 ✅ 标注。

- **单元（纯函数）**：
  - ✅ `estimate_tokens`：tiktoken 路径 + ImportError 兜底路径。
  - ✅ `render_messages_to_text` / `render_structured`：各 part 类型覆盖。
  - ✅ `find_safe_split`：任意切分结果中 prefix 不含悬空 `ToolCallPart`、recent 内部自洽；`recent >= keep`；`keep_first_user_message` 行为；`('messages', N>=n)` 钳位到 0（不返回负下标）。
  - ✅ sentinel 解析：build/parse round-trip（含 `compacted_tokens`）、旧版无 `compacted_tokens` 格式兼容、模型 echo/不完整标记严格拒绝（`find_summary`）。
- **集成（`FunctionModel`）**：
  - ✅ `persist=True`：`result.all_messages()` 头部含 sentinel，近期窗口完整，无悬空工具调用。
  - ✅ `persist=False`：`result.all_messages()` 保留完整原始历史，而 `FunctionModel` 收到的是压缩后 messages。
  - ✅ 阈值未超时不压缩（passthrough）。
  - ✅ 摘要输入断言：summarizer 收到的 user prompt 是 `render_structured` 产物（含 `<conversation-history>`、`<message role="...">`、`tool_call=` 属性）。
  - ✅ `merge_usage`：压缩轮 `result.usage.requests == 2`（父请求 + 摘要器请求）。
  - ✅ sentinel 序列化 round-trip：`ModelMessagesTypeAdapter` JSON 往返后 sentinel 字段完整，恢复的历史驱动增量合并（跨进程持久化的关键依赖）。
- ✅ **混合状态机**：模拟连续多次触发，断言 `generation` 递增、到 `full_reset_every` 归零；增量合并 vs 全量重置的消息形状差异。
- ✅ **冷却机制**：工具循环内冷却开启时摘要器仅调用 1 次（每步增长 < delta）；`min_recompact_growth_tokens=0` 关闭时每步重压（回归对照）；增长超 delta 后再次触发。
- ✅ **空增量批次短路**：前缀仅含 sentinel 时不调摘要器、状态不变。
- **`SummaryStore`**：✅ get/put 时机、`covered_count` 单调递增、增量 prompt 含旧摘要正文；✅ get/put 抛异常不阻断父 run；✅ 无 store 时降级全量。
- **真实摘要器**：live 测试（`--live`，DeepSeek 端点）：压缩触发、混合增量、persist=False、工具配对、摘要器失败降级。⚠️ VCR 录制（pytest-recording + cassettes）仍未做，live 测试依赖环境变量中的 API key，记为后续改进。
- ✅ **流式**：`agent.run_stream` 下两种 persist 模式均生效（断言 wrap 路径与 state 形状）。
- ✅ **Provider 守卫**：`for_run` 对 `AnthropicModel`/`OpenAIResponsesModel` 抛 `UserError`；对 `OpenAIChatModel`（DeepSeek 等）放行；`persist=True`+`summary_store` 传入时构造器抛 `UserError`。
- ✅ **构造器校验**：非法 `ContextSize`（kind/fraction 越界/tokens<1）、`min_prefix<1`、`full_reset_every<1`、`char_per_token<1`、负冷却 delta、`max_summary_tokens<1` 均构造期抛 `UserError`。
- ✅ **触发规格**：`should_trigger` 对 messages/tokens/fraction 三种 `ContextSize` 判定正确；fraction 模式 `max_tokens=None` 走默认 128_000。
- ✅ **工具输出截断**：`after_tool_execute` 对超长 str 按头/尾行截断、`BinaryContent` 原样返回、未超阈值不动；✅ 单行/超长行兜底为 token 精确中段截断（截后达标）；✅ `max_summary_tokens` 摘要上限。
- ✅ **失败处理**：摘要器 `ModelHTTPError` 降级不阻断；编程错误（`RuntimeError`）故意上抛。

---

## 9. 项目脚手架（greenfield）

目标项目当前仅有空 `pyproject.toml`（`name="pydantic-ai-extensions"`, `requires-python=">=3.13"`, `dependencies=[]`）与空 `docs/`。需建立：

### 9.1 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pydantic-ai-extensions"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "pydantic-ai-slim>=2.9.0",
    "tiktoken>=0.12.0",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
    "inline-snapshot",
    "pytest-recording",
    "ruff",
    "pyright",
    "dirty-equals",
]

[tool.hatch.build.targets.wheel]
packages = ["src/pydantic_ai_extensions"]

[tool.ruff]
line-length = 120
target-version = "py313"

[tool.pyright]
pythonVersion = "3.13"
typeCheckingMode = "strict"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

> `pydantic-ai-slim` 版本约束锁定 `>=2.9.0`：除 `AbstractCapability` 钩子、`ModelRequestContext`、`CapabilityOrdering` 外，实现还用到 `InstructionPart`/`NativeToolCallPart`/`NativeToolReturnPart`/`CompactionPart`/`ReinjectSystemPrompt`/`RunContext.conversation_id`——经实测这些符号在 1.74.0 **全部缺失**（更低的下限会在 import 时直接失败）。2.9.0 是实现与测试实际验证的版本。

### 9.2 包布局

```
src/pydantic_ai_extensions/
├── __init__.py                    # 导出 ContextCompression, SummaryStore, SummaryRecord
└── context_compression/
    ├── __init__.py                # 子包公共 API（含 find_summary、truncate_text_to_tokens 等）
    ├── capability.py              # ContextCompression（钩子编排、冷却、校验、截断）
    ├── tokenizer.py               # estimate_tokens / estimate_text_tokens / truncate_text_to_tokens + should_trigger
    ├── render.py                  # render_messages_to_text（token 估算）+ render_structured（摘要输入）
    ├── slicing.py                 # find_safe_split, is_safe_cutoff_point, partition_head
    ├── summarizer.py              # summarize_full / summarize_incremental / build_summary_message / find_summary
    └── store.py                   # SummaryStore Protocol, SummaryRecord
tests/
└── test_context_compression/
    ├── test_tokenizer.py
    ├── test_slicing.py
    ├── test_render.py
    ├── test_capability_persist.py     # persist 两模式、混合状态机、provider 守卫、截断
    ├── test_capability_hardening.py   # 冷却、流式、serde round-trip、usage 合并、store 契约、校验、sentinel 解析
    └── test_integration_live.py       # --live 真实 DeepSeek 端点（默认跳过）
```

### 9.3 文档

- 本文件：`docs/context-compression.md`。
- 建议后续加 `mkdocs.yml`（MkDocs Material + `mkdocstrings-python`，对齐 pydantic-ai 风格）与 `docs/index.md`。

---

## 10. 实现阶段

P1–P6 已全部完成；P7 为评审后的加固轮（已完成）。

| 阶段 | 内容 | 产出 | 状态 |
|------|------|------|------|
| P1 | 项目脚手架：`pyproject.toml`、包结构、ruff/pyright/pytest 配置、CI 可跑 | 可 `make install` 的空包 | ✅ |
| P2 | 纯函数 + 完整单测：`tokenizer.py`（`estimate_tokens`）、`render.py`（估算用 `render_messages_to_text` + 摘要输入用 `render_structured`，两个渲染器）、`slicing.py`（`is_safe_cutoff_point`、`find_safe_split`、`_keep_target`、`_token_cutoff`、`partition_head`） | 纯函数 100% 覆盖 | ✅ |
| P3 | `ContextCompression` + `persist=True` + `strategy='full'` + `for_run` Provider 守卫 + `summarizer.py`（`summarize_full` 经 `render_structured` + §5.8 框架层 prompt、`build_summary_message`）。**仅 `structured_text` 路径，不实现 `message_history` 模式**；另含 `after_tool_execute` 工具输出截断（§5.10） | 单集成测试通过（含守卫拒绝 `AnthropicModel`/`OpenAIResponsesModel`；`FunctionModel` 断言摘要输入为 `render_structured` 产物） | ✅ |
| P4 | `strategy='hybrid'`（`summarize_incremental` 用 `render_structured(new_batch)` + 旧摘要合并、周期全量重置）、sentinel 检测 | 混合状态机测试通过（`generation` 递增/归零、增量 vs 全量形状） | ✅ |
| P5 | `persist=False`（`wrap_model_request` + `replace(ctx, messages=...)`）+ `SummaryStore` 协议 + 无 store 降级全量 | 两模式集成测试 + 流式（`run_stream`）测试通过 | ✅ |
| P6 | 排序（`wrapped_by=[ReinjectSystemPrompt]`）、与其他能力协同、边界情况、文档与示例 | 文档完善、CI 全绿 | ✅ |
| P7 | 评审加固：冷却机制（`min_recompact_growth_tokens` + sentinel `compacted_tokens`，§6.1）；`find_safe_split` 负下标钳位 + 构造器全量校验；截断 token 精确兜底（工具输出 + `max_summary_tokens`）；摘要器异常收窄 + 日志、store get/put 容错；空增量批次短路；sentinel 判定统一严格解析（`find_summary`）；`pydantic-ai-slim>=2.9.0` 版本下限；文档修正（§5.5 keep 语义、§6.4 instructions、§6.9 new_messages、§5.4 快路径/工具 schema） | §8 全部测试项 | ✅ |

---

## 11. 决策记录

### 11.1 已落实于正文的决策

- **Q2 `ThinkingPart` 是否计入估算** → 新增 `include_thinking_in_estimate: bool = True` 配置项控制（§5.1、§5.4）。默认计入（thinking 文本确实占用 token）；对"任务上下文"摘要价值低时可关闭以略低估，从而更晚触发压缩。
- **Q3 近期窗口单位** -> 采用 `ContextSize`（messages/tokens/fraction）统一表达 `keep`，并新增 `compress_threshold` 同规格 + `max_tokens`（借鉴 summarization-pydantic-ai）；`('messages', 0)` 表示**不设下限**--仅保留安全边界强制的最小窗口（当前请求 + 维持配对所需，通常 2 条），最大化压缩（§5.1、§5.5）。
- **Q5 前缀不足时的行为** → 可压缩前缀 `< min_prefix` 时**不压缩、返回 `None`**（保持原历史）。这会让首次压缩略延迟到前缀攒够，属预期；`compress_threshold` 与 `min_prefix` 的配比在真实会话上调参（§5.1、§5.3）。
- **Q6 与原生上下文预算机制的关系** → 对 `AnthropicModel` / `OpenAIResponsesModel`（具备原生 compaction 的 provider），`for_run` 直接抛 `UserError`，改用 `AnthropicCompaction` / `OpenAICompaction`；**不允许**叠加 `task_budget.remaining` 等（§5.7、§7）。
- **Q1 摘要器输入形态** → 仅 `structured_text`（`render_structured` 结构化标签文本 + 声明式 prompt）。曾评估 `message_history` 模式，暂不实现（理由见 §5.8）。
- **Q4 并发与 `for_run`** → **不在实例上缓存 running summary、不覆写 `for_run`**（默认 `return self` 共享实例，并发 run 写实例属性会数据竞争/跨会话串扰，跨 run 也不可靠）。`persist=True` 状态走历史 sentinel；`persist=False` 走 `SummaryStore`，无 store 则降级全量重摘要（§5.3、§5.6）。

---

## 12. 附录：源码依据

| 依据 | 位置 |
|------|------|
| `AbstractCapability.before_model_request` / `wrap_model_request`（默认 passthrough，可覆写） | `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py:514,537` |
| 处理后消息写回 state | `pydantic_ai_slim/pydantic_ai/_agent_graph.py:980` |
| `wrap_model_request` 在写回后执行，调 `model.request` / `request_stream` | `pydantic_ai_slim/pydantic_ai/_agent_graph.py:680,700,863` |
| `ModelRequestContext` 为 `@dataclass(kw_only=True)`（可 `replace`） | `pydantic_ai_slim/pydantic_ai/models/__init__.py:180` |
| `ModelRequest.metadata: dict[str,Any] \| None`（sentinel 落点） | `pydantic_ai_slim/pydantic_ai/messages.py:1629` |
| `RunContext.conversation_id` / `run_id` / `usage` / `deps` | `pydantic_ai_slim/pydantic_ai/_run_context.py:85,87,43,39` |
| `CapabilityOrdering`（`wrapped_by` 等排序原语） | `pydantic_ai_slim/pydantic_ai/capabilities/_ordering.py`、`abstract.py:232` |
| `ProcessHistory`（薄封装参考） | `pydantic_ai_slim/pydantic_ai/capabilities/process_history.py` |
| `ReinjectSystemPrompt` | `pydantic_ai_slim/pydantic_ai/capabilities/reinject_system_prompt.py` |
| `CompactionPart`（provider 锁定，不采用） | `pydantic_ai_slim/pydantic_ai/messages.py:1768` |
| `OpenAICompaction`（要求 `OpenAIResponsesModel`） / `AnthropicCompaction` | `pydantic_ai_slim/pydantic_ai/models/openai.py:3904` / `models/anthropic.py:2131` |
| `DeepSeekProvider` 搭配 `OpenAIChatModel`（无原生 compaction） | `pydantic_ai_slim/pydantic_ai/providers/deepseek.py:29` |
| `tiktoken` 已为 `pydantic-ai-slim` 可选依赖 | `pydantic_ai_slim/pyproject.toml:73` |
| tool-call/return 配对陷阱 | pydantic-ai issue #2050 |
| `instructions` 每次模型请求重新解析并盖章到当前请求（压缩删历史 instructions 字段不丢系统提示） | `pydantic_ai_slim/pydantic_ai/_agent_graph.py:954-957` |
| `new_messages()` 起算：`_first_run_id_index`（sentinel `run_id=None` 不计入）；`fill_run_metadata` 仅盖 `messages[-1]` 与模型响应 | `pydantic_ai_slim/pydantic_ai/_agent_graph.py:1779,992,1067`；`pydantic_ai_slim/pydantic_ai/_utils.py`（`fill_run_metadata`） |
| provider SDK 错误映射：`APIStatusError`→`ModelHTTPError`、`APIConnectionError`→`ModelAPIError` | `pydantic_ai_slim/pydantic_ai/models/openai.py:193-198` |
| 历史序列化 round-trip（sentinel 跨进程存活验证） | `pydantic_ai_slim/pydantic_ai/messages.py:2432`（`ModelMessagesTypeAdapter`） |
