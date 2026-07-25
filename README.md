# pydantic-ai-extensions

[Pydantic AI](https://ai.pydantic.dev/) 的社区扩展集。当前提供：

- **`ContextCompression`** — provider 无关的长会话上下文自动压缩能力：历史超阈值时把较旧消息压缩为摘要，保留近期窗口与工具调用配对完整性，控制发往模型的 token 量。

适用于**无原生 compaction 的 provider**（DeepSeek、Groq、Mistral 等 OpenAI-Chat 兼容家族）或需要统一跨 provider 行为的场景。对 `OpenAIResponsesModel` / `AnthropicModel`，框架原生 compaction 更优，本能力会在 `for_run` 直接拒绝（见下文"Provider 兼容性"）。

完整设计文档见 [`docs/context-compression.md`](docs/context-compression.md)。

## 安装

```bash
pip install pydantic-ai-extensions
```

要求：Python ≥ 3.13，`pydantic-ai-slim` ≥ 2.9.0，`tiktoken` ≥ 0.12.0（token 估算；不可用时按字符数兜底）。

## 快速开始

```python
from pydantic_ai import Agent
from pydantic_ai_extensions import ContextCompression

# 摘要器：必须是无工具的 agent（output_type=str），其 instructions 即压缩 prompt 的可配入口
summarizer = Agent(
    'deepseek:deepseek-chat',
    instructions='精简总结技术讨论、已做决策与待办，忽略寒暄。',
)

agent = Agent(
    'deepseek:deepseek-chat',
    capabilities=[
        ContextCompression(
            summarizer,
            compress_threshold=('fraction', 0.7),  # 估算 token 达 context window 70% 时触发
            max_tokens=64_000,                     # fraction 模式的分母（你的模型窗口）
            keep=('messages', 6),                  # 保留最近 6 条消息不压缩
        ),
    ],
)

result = agent.run_sync('...', message_history=history)
```

压缩发生时，较旧的历史被替换为一条带 `<conversation-summary ...>` 标记的摘要消息（sentinel），工具调用/返回按 `tool_call_id` 精确配对、绝不拆散。

## 特性

- **三种触发/保留规格**：`compress_threshold` 与 `keep` 均支持 `('messages', N)` / `('tokens', N)` / `('fraction', F)`（tiktoken 估算，字符数兜底）
- **混合摘要策略**：默认增量合并（保留 running summary，只摘要新批次）；每 `full_reset_every` 次增量后全量重置，消除质量漂移
- **冷却机制**：压缩后历史仍高于阈值时，仅在历史较上次基线再增长 `min_recompact_growth_tokens`（默认 2048）后才重新压缩——避免工具循环内每一步都调摘要器
- **两种持久化模式**：`persist=True`（压缩结果写回历史，默认）/ `persist=False`（仅模型看到压缩视图，原始历史不动，配 `SummaryStore` 跨 run 增量）
- **Provider 守卫**：自动识别并拒绝具备原生 compaction 的模型（解包 `WrapperModel` 后判定），fail-fast
- **工具输出截断**（可选）：`max_tool_output_tokens` 从源头限长大工具返回，行级头/尾截断 + token 精确中段截断兜底；`BinaryContent` 不受影响
- **摘要长度上限**（可选）：`max_summary_tokens` 防止啰嗦的摘要器让压缩后反而膨胀
- **失败降级**：摘要器 API/网络故障时自动降级为不压缩（记 warning 日志），不阻断父 run；编程错误则故意上抛
- **可组合**：声明 `wrapped_by=[ReinjectSystemPrompt]` 的标准 capability 排序，与 `ProcessHistory` 等能力共存

## 配置参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `summarizer` | （必填） | 无工具的摘要 agent（`Agent[Any, str]`），其 `instructions` 是摘要风格的配置入口 |
| `compress_threshold` | `('fraction', 0.7)` | 触发线：`('messages', N)` / `('tokens', N)` / `('fraction', F)`。注意是触发线不是上限（见下文注意事项） |
| `max_tokens` | `None` | fraction 模式的分母；`None` 取 128_000（框架 profile 不暴露 context window） |
| `encoding` | `None` | tiktoken 编码，`None` → `o200k_base`（OpenAI/DeepSeek 系近似） |
| `char_per_token` | `4` | tiktoken 不可用时的字符数兜底换算 |
| `include_thinking_in_estimate` | `True` | 是否把 `ThinkingPart` 计入估算 |
| `keep` | `('messages', 6)` | 近期窗口下限，同 `ContextSize` 规格；`('messages', 0)` 表示只保留安全边界强制的最小窗口 |
| `keep_first_user_message` | `True` | 首条用户消息原样保留、不进摘要 |
| `min_prefix` | `4` | 可压缩前缀不足此数则不压缩 |
| `strategy` | `'hybrid'` | `'full'`（每次全量重摘要）/ `'hybrid'`（增量 + 周期全量重置） |
| `full_reset_every` | `5` | hybrid：每 N 次增量后做一次全量重置 |
| `min_recompact_growth_tokens` | `2_048` | 冷却：距上次压缩基线至少增长这么多 token 才再压缩；`0`/`None` 关闭 |
| `max_summary_tokens` | `None` | 摘要长度上限，超出做 token 精确中段截断 |
| `max_tool_output_tokens` | `None` | 工具返回的 token 上限，`None` 不截断 |
| `tool_output_head_lines` / `tool_output_tail_lines` | `5` / `5` | 行级截断保留的头/尾行数 |
| `persist` | `True` | `True` 压缩写回历史；`False` 仅模型看到压缩视图 |
| `summary_store` | `None` | 仅 `persist=False` 使用；缺省时降级为每次全量重摘要 |

## 持久化与 `new_messages()` 的重要注意事项

- **`persist=True` 必须用 `result.all_messages()` 做快照式持久化**。压缩插入的摘要消息不带 `run_id`，**不会**出现在 `result.new_messages()` 中——依赖 `new_messages()` 增量追加持久化的服务会丢失压缩结果，应改用 `persist=False` + `SummaryStore`。
- **`persist=False` 要求每轮加载完整连续历史**（`covered_count` 是历史索引）；`SummaryStore` 按 `ctx.conversation_id` 索引，实现方需保证并发安全。
- **压缩阈值是触发线，不是上限**：压缩后 `summary + keep` 不保证 ≤ 阈值，更不保证 ≤ 模型窗口。`keep` 相对阈值/窗口设过大会使压缩失效，请按模型窗口留足余量。
- token 估算不含随请求发送的 **tool definitions**，工具多的 agent 会系统性低估，fraction 阈值请留余量。
- 摘要 sentinel 以纯文本 `TextPart` 存在于历史中（可读、可持久化、可跨 provider round-trip）；终端 UI 若直接渲染 `all_messages()` 会看到 `<conversation-summary ...>` 标记文本。

## Provider 兼容性

| Model 类 | 行为 |
|----------|------|
| `OpenAIChatModel`（DeepSeek/Groq/Mistral 等 OpenAI-Chat 兼容）及其他无原生 compaction 的 model | ✅ 可用 |
| `AnthropicModel` | ❌ `for_run` 抛 `UserError`，请改用 `AnthropicCompaction` |
| `OpenAIResponsesModel` | ❌ `for_run` 抛 `UserError`，请改用 `OpenAICompaction` |

与其他能力协同：建议配套 `ReinjectSystemPrompt`（本能力已声明排在其后）；**禁止**与原生 compaction 能力叠加；与 `ProcessHistory`（如隐私过滤）可组合，用排序声明控制先后。

## 开发

```bash
make install    # uv sync --extra dev
make test       # 单元/集成测试（FunctionModel，无外部调用）
make lint       # ruff check
make format     # ruff format + --fix
make typecheck  # pyright (strict)
make version    # 从当前 git 状态重新生成 _version.py（构建时也会自动生成）
make build      # 打 wheel + sdist 到 dist/（构建时自动烘焙 git 版本）

uv run pytest --live -m live   # 真实 API 测试（需 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL 环境变量，默认跳过）
```

## 版本信息（git 烘焙，零第三方依赖）

不发布到 PyPI 时，调用方仍需知道"我跑的到底是哪个 commit"。本包用 hatchling 的 **build hook**（`hatch_build.py`，纯 stdlib + `git` 二进制）在**每次构建时**把 git 状态烘焙进 `src/pydantic_ai_extensions/_version.py`（gitignored 构建产物）。无需 `hatch-vcs`/`setuptools-scm` 等第三方依赖，安装方也无需 git。

```python
import pydantic_ai_extensions as p

p.__version__        # 精确版本，如 '1.2.3' / '1.2.3+5.g9a8b7c6.dirty' / '0.0.0+g8d7220f.dirty'
p.__version_info__   # 发布段元组，如 (1, 2, 3)
p.__commit__         # 短 commit 哈希，如 '8d7220f'
p.__commit_full__    # 完整哈希
p.__branch__         # 分支名
p.__describe__       # 原始 git describe 输出
p.__dirty__          # 'true' / 'false'（工作区是否有未提交改动）
p.get_version()      # 等于 __version__
```

版本派生规则（PEP 440-ish）：

| git 状态 | `__version__` |
|----------|---------------|
| 干净标签 `v1.2.3` | `1.2.3` |
| 分支领先标签 5 个提交 | `1.2.3+5.g9a8b7c6` |
| 上述且工作区脏 | `1.2.3+5.g9a8b7c6.dirty` |
| 无版本标签（仅 commit） | `0.0.0+g8d7220f` |
| 无 git 且无已烘焙的 `_version.py`（如未构建过的 `.git`-less 源码副本） | `0.0.0+unknown` |

机制要点：

- **何时烘焙**：`uv build` / `uv sync`（editable 自装）/ `uv add <path|git>` / `pip install` 都会触发 build hook，写一次 `_version.py`。editable 安装或 `uv add git+...` 时 git 在场，烘焙出真实版本；从 sdist 构建 wheel 时无 git，hook 保留 sdist 阶段已烘焙的值（sdist 被 `force_include` 纳入了该文件），不会退化成 `unknown`。
- **包元数据 `version`** 是静态发布线（`0.1.0`，即 `pip show` 看到的）；`__version__` 才是含 commit 的**精确**标识。发版时 `git tag v0.1.0`，则两者一致。
- **fresh checkout 直接 import**（未先 build）：`_version.py` 不存在时，`version.py` 回退到运行时 `git describe`，所以在任意分支上直接跑也能拿到有意义的版本。
- 调用方排查"我用的哪个版本"：`python -c "import pydantic_ai_extensions as p; print(p.__version__, p.__commit__)"`。

## 引入方式（不发布到 PyPI）

- **同机协同开发**（editable，改源码即时生效）：
  ```bash
  uv add --editable /path/to/pydantic-ai-extensions
  ```
  ```toml
  [tool.uv.sources]
  pydantic-ai-extensions = { path = "/path/to/pydantic-ai-extensions", editable = true }
  ```

- **私有 Git**（部署 / 多机 / CI，需锁版本时推荐）：
  ```bash
  uv add "git+ssh://git@github.com/<owner>/pydantic-ai-extensions.git@v0.1.0"
  ```

- **构建 wheel 离线交付**：`make build` 后 `uv add ./dist/pydantic_ai_extensions-0.1.0-py3-none-any.whl`。

> 消费方若用 DeepSeek/OpenAI 兼容模型，需自行加 provider extra：`uv add "pydantic-ai-slim[openai]"`（本包的 `[anthropic,openai]` extra 仅用于自身测试，不传递）。

## 文档

- [`docs/context-compression.md`](docs/context-compression.md) — 完整设计文档：设计决策、架构、状态机、sentinel 机制、边界情况、测试策略
