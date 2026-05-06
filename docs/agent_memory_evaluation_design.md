# Agent Memory Evaluation Design

## 1. 目标

本项目的目标是构建一个统一的 agent memory 评测框架，用 LongMemEval 作为主要评测基准，对比不同 memory 系统在长期对话记忆任务上的效果。

第一阶段优先支持以下 memory backend：

- mem0
- A-mem
- MemoryOS

上层评测流程不关心各 memory 系统的内部实现，只要求它们能被包装成统一的 agent memory 接口。若某个 backend 需要额外配置、依赖服务或数据准备，应通过配置文件显式声明。

## 2. 非目标

当前阶段暂不处理以下内容：

- 不重写 mem0、A-mem、MemoryOS 的核心算法。
- 不要求三套 memory 系统内部存储结构一致。
- 不将 LongMemEval 改造成新的 benchmark。
- 不以交互式聊天应用为第一目标。
- 不在第一阶段追求完整生产级多租户、权限、监控能力。

## 3. 总体架构

```text
LongMemEval Dataset
        |
        v
Experiment Runner
        |
        v
Agent Runtime
        |
        +-- Memory Adapter
        |      +-- mem0 Adapter
        |      +-- A-mem Adapter
        |      +-- MemoryOS Adapter
        |
        +-- LLM Client
        |
        +-- Prompt / Context Builder
        |
        v
Prediction Writer
        |
        v
LongMemEval Evaluator
```

框架分为四层：

1. **Benchmark 层**：读取 LongMemEval 数据，按样本驱动实验。
2. **Agent 层**：模拟带长期记忆能力的 chat assistant。
3. **Memory Adapter 层**：把不同 memory 系统包装成统一接口。
4. **Evaluation 层**：输出 LongMemEval 兼容结果并调用官方评测脚本。

## 4. 核心设计原则

### 4.1 评测协议优先

不同 memory 系统可以有完全不同的内部实现，但必须遵守同一个评测协议：

1. 按时间顺序读入历史 session。
2. 将每个历史 session 注入 agent。
3. 允许 memory backend 在线写入、整理或更新记忆。
4. 在问题时间点提出 LongMemEval question。
5. agent 只能通过其 memory backend 和当前 question 回答。
6. 输出 `question_id` 和 `hypothesis`。

### 4.2 Backend 可替换

backend 的选择必须由配置决定，而不是写死在代码里。

示例：

```yaml
experiment:
  name: longmemeval_mem0_gpt4o
  dataset_path: LongMemEval/data/longmemeval_s_cleaned.json
  output_path: runs/mem0_gpt4o/predictions.jsonl

agent:
  llm:
    provider: openai
    model: gpt-4o
  memory:
    backend: mem0
    config_path: configs/memory/mem0.yaml
```

### 4.3 统一输入输出，不统一内部实现

Memory adapter 只负责翻译框架与具体 memory 系统之间的调用：

- 框架输入：session、turn、question、metadata
- 框架输出：retrieved memories、answer context、debug 信息
- backend 内部：由 mem0 / A-mem / MemoryOS 自行决定

## 5. 统一 Agent Memory 接口

建议定义一个最小接口：

```python
class MemoryBackend:
    def reset(self, sample_id: str) -> None:
        ...

    def ingest_session(self, session: "MemorySession") -> None:
        ...

    def retrieve(self, query: str, top_k: int, metadata: dict | None = None) -> list["MemoryItem"]:
        ...

    def build_context(self, query: str, retrieved: list["MemoryItem"]) -> str:
        ...

    def close(self) -> None:
        ...
```

其中：

- `reset`：每个 LongMemEval 样本开始前清空或切换隔离 namespace。
- `ingest_session`：按时间顺序喂入历史对话。
- `retrieve`：在回答问题前检索相关记忆。
- `build_context`：将检索结果转成 LLM prompt 可用文本。
- `close`：释放资源或写出 debug artifacts。

可选接口：

```python
class InspectableMemoryBackend(MemoryBackend):
    def export_state(self) -> dict:
        ...

    def get_debug_info(self) -> dict:
        ...
```

这些接口用于分析实验差异，但不应成为评测必需条件。

## 6. Canonical 数据结构

### 6.1 MemoryTurn

```python
@dataclass
class MemoryTurn:
    role: str
    content: str
    timestamp: str | None = None
    metadata: dict | None = None
```

### 6.2 MemorySession

```python
@dataclass
class MemorySession:
    session_id: str
    date: str | None
    turns: list[MemoryTurn]
    metadata: dict | None = None
```

### 6.3 MemoryItem

```python
@dataclass
class MemoryItem:
    id: str
    content: str
    score: float | None = None
    source_session_id: str | None = None
    metadata: dict | None = None
```

这些结构只服务于评测框架，不要求 backend 原生保存为相同格式。

## 7. LongMemEval 实验流程

单个样本的执行流程：

```text
for sample in dataset:
    backend.reset(sample.question_id)

    for session_id, date, session in chronological_history:
        memory_session = convert_to_memory_session(...)
        backend.ingest_session(memory_session)

    retrieved = backend.retrieve(sample.question, top_k)
    memory_context = backend.build_context(sample.question, retrieved)
    answer = agent.answer(question=sample.question, memory_context=memory_context)

    write_jsonl({
        "question_id": sample.question_id,
        "hypothesis": answer
    })
```

执行完成后，调用 LongMemEval 官方评测：

```bash
cd LongMemEval/src/evaluation
python evaluate_qa.py gpt-4o ../../../runs/<run_id>/predictions.jsonl ../../data/<dataset>.json
python print_qa_metrics.py gpt-4o ../../../runs/<run_id>/predictions.jsonl.log ../../data/<dataset>.json
```

## 8. Agent Runtime 设计

Agent runtime 保持简单，避免引入复杂工具调用。第一阶段只需要：

1. 接收 question。
2. 调用 memory backend 检索上下文。
3. 将 question 和 memory context 组合成 prompt。
4. 调用 LLM。
5. 输出 answer。

推荐 reader prompt 位于：

```text
agent_memory_eval/prompts.py
```

默认 system instructions：

```text
You are an answer reader for a LongMemEval long-term memory evaluation.

Answer the question using only the provided Memory Context.
Do not use outside knowledge, prior assumptions, or information that is not supported by the Memory Context.
If the Memory Context is empty, irrelevant, conflicting without a clear resolution, or insufficient, answer exactly: I don't know.
When relevant memories conflict, prefer the most recent memory that is clearly supported by the Memory Context.
Keep the answer concise and avoid explaining the evaluation process.
```

默认 user prompt 模板：

```text
Memory Context:
{memory_context}

Question:
{question}

Answer:
```

为了公平比较，默认所有 backend 使用同一个 reader LLM 和同一个回答 prompt。
该 reader prompt 只用于最终回答阶段，不用于 mem0 / A-mem / MemoryOS 的内部写入、抽取、演化或更新逻辑。

## 9. Backend 配置需求

### 9.1 mem0

预期能力：

- ingest session 时调用 `memory.add(...)`
- answer 前调用 `memory.search(...)`
- 支持按 `user_id` 或 metadata 隔离样本

可能需要配置：

```yaml
backend: mem0
user_id_template: "longmemeval_{question_id}"
top_k: 10
llm:
  provider: openai
  model: gpt-4o-mini
embedder:
  provider: openai
  model: text-embedding-3-small
vector_store:
  provider: chroma
  path: runs/vectorstores/mem0
```

注意事项：

- mem0 自身可能调用 LLM 做 memory extraction，因此需要单独配置 extraction LLM。
- 需要保证每个 LongMemEval 样本之间 memory 隔离。
- 若使用持久化 vector store，应在 `reset` 时创建独立 namespace 或清理旧数据。

### 9.2 A-mem

预期能力：

- ingest session 时将 session 内容转换为 note。
- 使用 `AgenticMemorySystem.add_note(...)` 写入。
- 使用 `search_agentic(...)` 检索。

可能需要配置：

```yaml
backend: amem
collection_template: "longmemeval_{question_id}"
top_k: 10
embedding_model: all-MiniLM-L6-v2
llm_backend: openai
llm_model: gpt-4o-mini
persist_directory: runs/vectorstores/amem
```

注意事项：

- A-mem 会生成 tags、context、keywords、links，写入成本可能高于普通向量检索。
- 需要确认每个样本是否能使用独立 Chroma collection。
- 若无法轻松清理 collection，建议每个 run 使用新的 persist directory。

### 9.3 MemoryOS

预期能力：

- ingest session 时写入短期记忆。
- MemoryOS 内部负责短期、中期、长期记忆流转。
- answer 前调用其 retriever 获取相关 memory。

可能需要配置：

```yaml
backend: memoryos
user_id_template: "longmemeval_{question_id}"
top_k: 10
storage_path: runs/vectorstores/memoryos
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
similarity_threshold: 0.35
```

注意事项：

- MemoryOS 有 short/mid/long-term 层级，需要确认 session ingest 后是否立即可检索。
- 若存在异步或批量更新机制，评测时需要显式触发 flush/update。
- 如果 MemoryOS 通过 MCP 或服务模式运行，需要额外声明服务启动命令。

## 10. 配置文件布局建议

```text
configs/
  experiments/
    longmemeval_mem0.yaml
    longmemeval_amem.yaml
    longmemeval_memoryos.yaml
  memory/
    mem0.yaml
    amem.yaml
    memoryos.yaml
  llm/
    openai_gpt4o.yaml

runs/
  <run_id>/
    config.resolved.yaml
    predictions.jsonl
    predictions.jsonl.log
    metrics.json
    retrieved_memories.jsonl
    debug/
```

## 11. 输出产物

每次实验至少产生：

- `predictions.jsonl`：LongMemEval 官方 evaluator 输入。
- `predictions.jsonl.log`：官方 evaluator 输出日志。
- `metrics.json`：聚合指标。
- `config.resolved.yaml`：完整实验配置快照。

建议额外保存：

- `retrieved_memories.jsonl`：每个问题检索到的 memory。
- `ingest_trace.jsonl`：每个 session 写入状态。
- `backend_debug.json`：backend 特定调试信息。
- `token_usage.jsonl`：每个样本的 memory 构建、检索和 reader prompt token 统计。
- `token_usage_summary.json`：当前 run 的 token 统计汇总。

`retrieved_memories.jsonl` 示例：

```json
{
  "question_id": "sample_001",
  "backend": "mem0",
  "query": "What restaurant did I say I liked?",
  "retrieved": [
    {
      "id": "memory_123",
      "content": "The user liked Sushi Nakazawa.",
      "score": 0.82,
      "source_session_id": "session_17"
    }
  ]
}
```

## 12. 公平性控制

为了让评测结果可解释，应固定以下变量：

- LongMemEval 数据版本。
- reader LLM。
- answer prompt。
- `top_k`。
- 是否允许 backend 使用额外 LLM 做 memory extraction。
- 是否允许 backend 使用 question date。
- 是否允许完整历史直接进入 answer prompt。

默认约束建议：

- 不允许将完整历史直接放入 answer prompt。
- 只允许 answer prompt 使用 backend 检索出的 memory context。
- 允许 backend 在 ingest 阶段使用自己的 extraction / summarization 逻辑。
- 每个 backend 必须使用同一份原始 session 输入。
- token 统计采用同一套框架侧计数逻辑，并 best-effort 捕获 backend 内部 LLM prompt。

## 13. 实验模式

### 13.1 Online Memory 模式

模拟真实 agent：

1. 历史 session 逐条进入。
2. backend 在线更新 memory。
3. 问题出现时进行检索和回答。

这是默认模式。

### 13.2 Offline Index 模式

一次性把所有 history 建索引，然后回答问题。

该模式适合和 LongMemEval 原始 retrieval baseline 对齐，但不完全等价于真实 agent memory。

### 13.3 Oracle 模式

只喂入 evidence sessions。

该模式用于确认 agent reader 和 memory adapter 是否工作正常，不应作为主结果。

## 14. 初始实现里程碑

### M1: 统一评测骨架

- 加载 LongMemEval JSON 数据。
- 逐样本读取 history session。
- 输出 LongMemEval 兼容 `predictions.jsonl`。
- 支持 no-memory baseline。

### M2: Memory Adapter 接入

- 接入 mem0 adapter。
- 接入 A-mem adapter。
- 接入 MemoryOS adapter。
- 支持 YAML 配置切换 backend。

### M3: 评测自动化

- 自动调用 LongMemEval evaluator。
- 生成 metrics summary。
- 保存 retrieved memories 和 ingest trace。

### M4: 对比实验

- 跑通 `longmemeval_oracle`。
- 跑通 `longmemeval_s_cleaned`。
- 对比 no-memory、mem0、A-mem、MemoryOS。

## 15. 待确认问题

以下问题已经形成第一阶段决策：

1. 第一阶段 reader LLM 使用统一 Responses-compatible 配置，由 `configs/llm/responses.yaml` 指定。
2. memory backend 的 extraction / organization LLM 第一阶段复用同一主 LLM，后续保留独立配置扩展点。
3. `top_k` 默认使用各 backend adapter 的默认值，需要公平对齐时再通过 `agent.top_k` 覆盖。
4. smoke test 优先使用 `longmemeval_oracle`。
5. MemoryOS 第一阶段使用 `MemoryOS/memoryos-pypi` 本地源码模式，不使用 MCP 服务模式。
6. retrieval recall 和 QA accuracy 的正式评估暂不接入，runner 先预留评估输出接口。

## 16. 推荐第一版决策

建议第一版采用如下决策：

- 默认实验模式：Online Memory。
- 默认数据：先用 `longmemeval_oracle` smoke test，再跑 `longmemeval_s_cleaned`。
- 默认 reader：统一 OpenAI Responses-compatible API，由 `configs/llm/responses.yaml` 配置。
- 默认 memory backend 配置：第一阶段 extraction / organization LLM 也复用同一个主 LLM，adapter 中保留后续单独切换接口。
- 默认隔离策略：每个 `question_id` 使用独立 namespace。
- 默认输出：`predictions.jsonl`、`metrics.json`、`retrieved_memories.jsonl`、`config.resolved.yaml`。

## 17. 当前实现方案

当前实现新增独立框架目录：

```text
agent_memory_eval/
  agent.py              # Agent runtime
  cli.py                # python -m agent_memory_eval
  config.py             # YAML + .env 加载
  llm.py                # OpenAI Responses API client
  llm_token_hooks.py    # backend 内部 LLM prompt token 捕获
  longmemeval.py        # LongMemEval 数据加载
  prompts.py            # LongMemEval reader prompt
  runner.py             # 实验执行
  token_usage.py        # token 统计
  backends/
    no_memory.py
    mem0_backend.py
    amem_backend.py
    memoryos_backend.py
```

配置文件位于：

```text
configs/
  llm/responses.yaml
  memory/no_memory.yaml
  memory/mem0.yaml
  memory/amem.yaml
  memory/memoryos.yaml
  experiments/longmemeval_oracle_*.yaml
  experiments/longmemeval_s_cleaned_*.yaml
  experiments/longmemeval_s_cleaned_smoke_*.yaml
```

统一 LLM 和 embedding 配置示例：

```yaml
provider: openai
model: qwen3.6-plus
api_key_env: LLM_API_KEY
base_url: https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1
temperature: 0.0
max_output_tokens: 1200
extra_body:
  enable_thinking: true
embedding:
  provider: dashscope
  model: text-embedding-v4
  api_key_env: EMBEDDING_API_KEY
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```


如果使用 DashScope Qwen 的 OpenAI-compatible Responses 接口，最小 `.env` 可以是：

```text
LLM_API_KEY=你的DashScopeKey
EMBEDDING_API_KEY=你的DashScopeKey
```

对应统一配置文件为：

```text
configs/llm/responses.yaml
```

其中 base URL 已配置为：

```text
https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1
```

并通过 `extra_body.enable_thinking: true` 启用 Qwen thinking 模式。框架会跳过 `reasoning` 输出项，只把最终 `message` 文本写入 `hypothesis`。

LLM 和 embedding 可以独立配置 URL 和 key：

```yaml
provider: openai
model: qwen3.6-plus
api_key_env: LLM_API_KEY
base_url: https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1
embedding:
  provider: dashscope
  model: text-embedding-v4
  api_key_env: EMBEDDING_API_KEY
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 17.1 MemoryOS 方案选择

第一阶段建议使用 **MemoryOS 本地源码库 adapter**，即 `MemoryOS/memoryos-pypi`。

原因：

- 与 mem0、A-mem 一样在同一 Python 进程内执行，实验日志和异常更容易统一。
- 不需要额外 MCP 服务启动、端口管理和进程生命周期管理。
- 更容易保证每个 `question_id` 使用独立 user namespace。
- 后续如果要比较 MCP 服务模式，可以新增 `memoryos_mcp` backend，而不影响本地源码 adapter。

### 17.2 默认 top_k

若 `agent.top_k` 未显式配置，则使用各 backend adapter 默认值：

- no-memory: `0`
- mem0: `20`，对应 mem0 `search` 默认值。
- A-mem: `5`，对应 `search_agentic(..., k=5)` 默认值。
- MemoryOS: `7`，对应 `retrieval_queue_capacity` 默认值。

如需公平固定 top_k，可在实验 YAML 中添加：

```yaml
agent:
  top_k: 10
```

### 17.3 Smoke Test

第一阶段优先跑 `longmemeval_oracle`：

```bash
python -m agent_memory_eval validate configs/experiments/longmemeval_oracle_no_memory.yaml
python -m agent_memory_eval run configs/experiments/longmemeval_oracle_no_memory.yaml --limit 1 --dry-run
python -m agent_memory_eval validate configs/experiments/longmemeval_oracle_no_memory.yaml
python -m agent_memory_eval run configs/experiments/longmemeval_oracle_no_memory.yaml --limit 1
```

PowerShell 也可使用：

```powershell
.\scripts\run_agent_memory_eval.ps1 -Config configs/experiments/longmemeval_oracle_no_memory.yaml -Limit 1
```

正式接入 backend 后可替换配置：

```bash
python -m agent_memory_eval run configs/experiments/longmemeval_oracle_mem0.yaml --limit 1
python -m agent_memory_eval run configs/experiments/longmemeval_oracle_amem.yaml --limit 1
python -m agent_memory_eval run configs/experiments/longmemeval_oracle_memoryos.yaml --limit 1
```

当前版本会生成 `metrics.json` 占位文件，但暂不调用 LongMemEval evaluator。评估接口已在 runner 输出中预留。

同时提供基于 `longmemeval_s_cleaned` 第一条样本裁切出的完整 haystack smoke fixture：

```text
LongMemEval/data/longmemeval_s_cleaned_smoke_1.json
```

该 fixture 保留 1 个 question 的完整 53 个历史 session，可用于验证完整 haystack 写入、检索和最终 reader 回答链路。

对应配置：

```bash
python -m agent_memory_eval run configs/experiments/longmemeval_s_cleaned_smoke_no_memory.yaml
python -m agent_memory_eval run configs/experiments/longmemeval_s_cleaned_smoke_mem0.yaml
python -m agent_memory_eval run configs/experiments/longmemeval_s_cleaned_smoke_amem.yaml
python -m agent_memory_eval run configs/experiments/longmemeval_s_cleaned_smoke_memoryos.yaml
```

主评测配置：

```bash
python -m agent_memory_eval run configs/experiments/longmemeval_s_cleaned_no_memory.yaml
python -m agent_memory_eval run configs/experiments/longmemeval_s_cleaned_mem0.yaml
python -m agent_memory_eval run configs/experiments/longmemeval_s_cleaned_amem.yaml
python -m agent_memory_eval run configs/experiments/longmemeval_s_cleaned_memoryos.yaml
```

### 17.4 Token Usage

runner 会额外写出：

```text
runs/<run_id>/
  token_usage.jsonl
  token_usage_summary.json
```

统计字段：

```text
build_input_tokens        # 送入 memory 构建阶段的原始 session 文本 token
build_llm_prompt_tokens   # 可捕获的 memory 内部 LLM prompt token
build_tokens              # build_input_tokens + build_llm_prompt_tokens
query_input_tokens        # memory 检索 query token
query_llm_prompt_tokens   # 可捕获的检索阶段内部 LLM prompt token
reader_prompt_tokens      # 最终 reader prompt token
query_tokens              # query_input_tokens + query_llm_prompt_tokens + reader_prompt_tokens
total_tokens              # build_tokens + query_tokens
```

注意：

- 统计器优先使用 `tiktoken`，无法匹配模型时回退到 `cl100k_base`，再失败时使用 regex fallback。
- 内部 LLM prompt 捕获是 best-effort，覆盖当前 adapter 已知调用点：Mem0 `generate_response`、A-MEM `get_completion`、MemoryOS `chat_completion`。
- 该统计用于不同 backend 的可比较分析，不保证等同于供应商账单里的精确 token usage。

### 17.5 当前验证状态

当前已经验证：

- `python -m compileall agent_memory_eval`
- `python -m agent_memory_eval validate configs/experiments/longmemeval_oracle_no_memory.yaml`
- `python -m agent_memory_eval validate configs/experiments/longmemeval_oracle_mem0.yaml`
- `python -m agent_memory_eval validate configs/experiments/longmemeval_oracle_amem.yaml`
- `python -m agent_memory_eval validate configs/experiments/longmemeval_oracle_memoryos.yaml`
- `python -m agent_memory_eval run configs/experiments/longmemeval_oracle_no_memory.yaml --limit 1`
- `python -m agent_memory_eval validate configs/experiments/longmemeval_s_cleaned_smoke_mem0.yaml`
- `python -m agent_memory_eval run configs/experiments/longmemeval_s_cleaned_smoke_no_memory.yaml`

`no_memory` run 已成功写出：

```text
runs/longmemeval_oracle_no_memory/
  config.resolved.yaml
  predictions.jsonl
  retrieved_memories.jsonl
  ingest_trace.jsonl
  backend_debug.jsonl
  token_usage.jsonl
  token_usage_summary.json
  metrics.json
```

其中 `metrics.json` 当前为：

```json
{
  "status": "not_evaluated"
}
```

这是预期行为，因为正式 LongMemEval evaluator 尚未接入。

### 17.6 自研 Memory Backend

后续自研 memory 算法应作为新的 backend adapter 接入，而不是改动 runner、LongMemEval loader 或 agent runtime。

接入指引见：

```text
docs/custom_memory_backend_guide.md
```
