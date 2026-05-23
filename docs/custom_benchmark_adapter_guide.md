# 自定义 Benchmark Adapter 与 Memory 架构接入指南

Agent_memory_eval的 runner 只依赖统一 benchmark adapter，不直接依赖具体数据集。当前内置 adapter：

```text
agent_memory_eval/benchmarks/longmemeval.py
agent_memory_eval/benchmarks/locomo.py
```

后续接入 LOCOMO、自定义长期对话集或任务型 agent benchmark 时，应新增 benchmark adapter，而不是修改 `agent_memory_eval/runner.py`。

如果后续接入新的 memory 架构，也应通过 `agent_memory_eval/backends/*_backend.py` 编写薄 adapter，而不是把核心 memory 算法直接写进评测框架。推荐做法是：memory 算法作为独立 Python 包维护，`agent_memory_eval` 只负责把 benchmark 统一样本格式转换成该算法的 `reset/add/search` 调用。

## 1. Adapter 契约

所有 benchmark adapter 继承：

```python
from agent_memory_eval.benchmarks.base import BenchmarkAdapter
```

必须实现：

```python
class MyBenchmark(BenchmarkAdapter):
    benchmark_name = "my_benchmark"

    def validate(self) -> list[str]:
        ...

    def load_samples(self, limit: int | None = None) -> list[BenchmarkSample]:
        ...

    def prediction_record(self, sample: BenchmarkSample, answer: str) -> dict:
        ...

    def evaluate(self, *, predictions_path: Path, run_dir: Path, progress: bool = True) -> dict:
        ...
```

其中 `evaluate` 可以返回真实指标，也可以返回：

```python
{
    "status": "not_evaluated",
    "benchmark": self.benchmark_name,
}
```

## 2. 统一样本格式

adapter 需要把原始 benchmark 数据转换成：

```python
from agent_memory_eval.models import BenchmarkSample, MemorySession, MemoryTurn
```

最小字段：

```python
BenchmarkSample(
    question_id="sample_001",
    question_type="...",
    question="...",
    answer="...",
    question_date=None,
    sessions=[
        MemorySession(
            session_id="session_1",
            date="2025-01-01 10:00",
            turns=[
                MemoryTurn(role="user", content="..."),
                MemoryTurn(role="assistant", content="..."),
            ],
        )
    ],
    raw=raw_record,
    benchmark="my_benchmark",
)
```

Memory backend 只会看到 `MemorySession` 和最终 `question`，因此换 benchmark 不需要改 backend。

### 2.1 数据边界

benchmark adapter 只做数据集格式转换，不应包含 memory 策略：

- 可以做：读取数据、标准化 session/turn/question/answer、调用 benchmark 官方 evaluator。
- 不应做：摘要、检索、事实抽取、memory 写入策略、针对某个 backend 的特殊字段拼接。

memory backend 只做 memory 架构适配，不应包含 benchmark 私有逻辑：

- 可以做：把 `MemorySession` 切成算法需要的写入单位，调用独立 memory 包，返回 `MemoryItem`。
- 不应做：读取原始 benchmark 文件、调用 benchmark evaluator、根据 LongMemEval/LOCOMO 的答案字段作弊。

这个边界保证后续换 benchmark 时不用改 memory backend，换 memory 架构时也不用改 benchmark adapter。

## 3. Memory 架构接入规范

新增 memory 架构时，优先遵守以下规范。

### 3.1 核心算法独立于 agent_memory_eval

自研 memory 算法建议单独开源，例如：

```text
self_memory/
  self_memory/
    __init__.py
    memory.py
    retriever.py
    storage.py
    config.py
  pyproject.toml
  README.md
```

`agent_memory_eval` 中只保留薄 adapter：

```text
agent_memory_eval/backends/self_memory_backend.py
configs/memory/self_memory.yaml
```

adapter 只负责协议转换、配置读取、token 统计、debug 信息，不承载核心算法逻辑。

### 3.2 Backend 契约

所有 memory backend 继承：

```python
from agent_memory_eval.backends.base import MemoryBackend
```

必须实现：

```python
class MyMemoryBackend(MemoryBackend):
    backend_name = "my_memory"
    default_top_k = 10

    def reset(self, sample_id: str) -> None:
        ...

    def ingest_session(self, session: MemorySession) -> None:
        ...

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata: dict | None = None,
    ) -> list[MemoryItem]:
        ...
```

runner 的调用顺序固定为：

```python
backend.reset(sample.question_id)
for session in sample.sessions:
    backend.ingest_session(session)
retrieved = backend.retrieve(sample.question, top_k=agent.top_k)
context = backend.build_context(sample.question, retrieved)
```

因此 backend 必须保证每个 `sample_id` 的状态隔离，不能让不同问题、不同样本之间的 memory 相互污染，除非某个实验明确声明要测跨样本长期记忆。

### 3.3 写入粒度规范

memory backend 应显式声明写入粒度：

```yaml
backend: my_memory
ingest_granularity: pair  # 可选: native, session, pair, turn
```

推荐含义：

| 粒度 | 含义 | 适用场景 |
|---|---|---|
| `native` | 按算法原生方式写入 | 完整 memory 系统对比 |
| `session` | 一个 session 作为一条写入单位 | 会话级摘要、archival memory |
| `pair` | user/assistant 相邻 pair 作为写入单位 | 公平基线、对话交互记忆 |
| `turn` | 每个 turn 作为一条写入单位 | 事实检索、A-MEM 风格 note |

默认建议：

- 完整系统对比：使用该方法的原生粒度，并在配置和结果中记录。
- 公平 chunking baseline：优先使用 `pair`。
- A-MEM 当前默认：`turn`，因为其 benchmark 复现通常逐 turn 写入 note。

如果 backend 支持多粒度，切分逻辑应放在 adapter 层，不要改独立 memory 包的核心 API。每次写入建议在 token/debug metadata 中记录：

```text
chunk_type
chunk_index
session_id
timestamp/date
role 或 has_user_input/has_assistant_response
```

### 3.4 检索返回规范

`retrieve` 必须返回统一 `MemoryItem`：

```python
MemoryItem(
    id="...",
    content="...",
    score=0.82,
    source_session_id="session_1",
    metadata={...},
)
```

规范：

- `content` 是最终给 reader LLM 的证据文本，应可读、可溯源。
- `score` 如果后端没有相似度可以为 `None`，不要伪造。
- `source_session_id` 能填则填，方便排查检索是否命中正确会话。
- `metadata` 可以保留 backend 私有字段，但不要依赖 benchmark 答案。

如需自定义 reader 上下文格式，可以覆写 `build_context`，但应保持简洁，避免把大量无关 memory 塞进上下文造成不公平。

### 3.5 配置规范

每个 backend 应有独立配置：

```text
configs/memory/my_memory.yaml
```

示例：

```yaml
backend: my_memory
repo_path: selfMemory
ingest_granularity: pair
storage_path: runs/vectorstores/my_memory

llm_model: null          # 默认继承 suite.agent.llm.model
api_key_env: LLM_API_KEY
base_url_env: LLM_BASE_URL

embedding_model: text-embedding-3-small
embedding_api_key_env: EMBEDDING_API_KEY
embedding_base_url_env: EMBEDDING_BASE_URL
```

规范：

- API key 不写死在 YAML 中，优先使用 `*_env` 从 `.env` 或环境变量读取。
- `base_url` / `base_url_env` 都应支持，方便 OpenAI-compatible 服务切换。
- 存储路径必须按 backend/sample/run 隔离，避免复用旧向量库污染结果。
- suite 内 inline 配置可以覆盖 memory YAML，用于 ablation。

### 3.6 注册规范

在 `agent_memory_eval/backends/factory.py` 注册：

```python
if backend == "my_memory":
    from .my_memory_backend import MyMemoryBackend

    return MyMemoryBackend(config, llm_config)
```

然后在 suite 中加入：

```yaml
suite:
  backends:
    - name: my_memory
      config_path: configs/memory/my_memory.yaml
```

backend 名称统一使用小写 snake_case，并保持三处一致：

```text
configs/memory/my_memory.yaml -> backend: my_memory
configs/suites/*.yaml         -> name: my_memory
factory.py                    -> if backend == "my_memory"
```

### 3.7 评测公平性规范

为了让不同 memory 架构的 benchmark 结果可解释，应遵守：

- 不使用 benchmark 的 gold answer、evidence label、question type 做写入或检索增强。
- 不在 backend 中读取原始 dataset 文件，只消费 `MemorySession` 和 `question`。
- 每个 sample 前调用 `reset(sample_id)` 后应清空或切换 namespace。
- 明确记录 `top_k`、写入粒度、reader LLM、memory LLM、embedding model。
- 如果某个方法原生会生成摘要、图结构、长期/短期分层，应作为该方法能力保留；如果做公平 chunking ablation，则单独建 suite，不和 native 结果混在一起。

推荐维护两类 suite：

```text
native suite              # 各 memory 方法按原生接入方式运行
controlled_chunking suite # 所有可控 backend 使用统一 turn/pair/session 粒度
```

### 3.8 日志与可观测性

backend 应尽量记录：

- build token：每次写入 memory 的原始文本。
- query token：每次检索 query。
- retrieved context token：最终给 reader 的 memory context。
- backend debug：memory 数量、粒度、namespace、存储路径。

可以通过 backend 内部的 `self.token_usage.record_build(...)`、`record_memory_query(...)` 和 `get_debug_info()` 暴露这些信息。运行目录中重点检查：

```text
config.resolved.yaml
ingest_trace.jsonl
retrievals.jsonl
backend_debug.jsonl
token_usage.json
metrics.json
```

### 3.9 最小验证流程

接入新 memory backend 后，建议按顺序验证：

```powershell
python -m agent_memory_eval validate configs\suites\longmemeval_smoke.yaml --backend my_memory
python -m agent_memory_eval run configs\suites\longmemeval_smoke.yaml --backend my_memory --dry-run
python -m agent_memory_eval run configs\suites\longmemeval_smoke.yaml --backend my_memory --limit 1 --no-eval
python -m agent_memory_eval run configs\suites\locomo.yaml --backend my_memory --limit 5
```

检查点：

- `config.resolved.yaml` 中 backend 配置符合预期。
- `backend_debug.jsonl` 中 memory 数量随写入增加。
- `retrievals.jsonl` 中返回的 memory 与问题相关。
- `token_usage.json` 中 build/query/reader token 没有异常爆炸。
- 多次运行不会复用上一轮的脏存储。

## 4. 注册 Benchmark Adapter

在 `agent_memory_eval/benchmarks/factory.py` 添加：

```python
if name == "my_benchmark":
    from .my_benchmark import MyBenchmark

    return MyBenchmark(config, root)
```

## 5. Suite 配置

推荐格式：

```yaml
suite:
  name: my_benchmark
  benchmark:
    name: my_benchmark
    dataset_path: data/my_benchmark.json
    evaluation:
      enabled: true

  agent:
    llm_config_path: configs/llm/responses.yaml
    top_k: 10

  backends:
    - name: mem0
      config_path: configs/memory/mem0.yaml
    - name: amem
      config_path: configs/memory/amem.yaml

  run:
    run_dir_template: runs/{suite}_{backend}
```

运行：

```powershell
python -m agent_memory_eval validate configs\suites\my_benchmark.yaml
python -m agent_memory_eval run configs\suites\my_benchmark.yaml --backend mem0
```

## 6. LongMemEval Adapter 现状

LongMemEval adapter 已接入官方 QA evaluator：

```text
LongMemEval/src/evaluation/evaluate_qa.py
LongMemEval/src/evaluation/print_qa_metrics.py
```

启用评估时，运行目录会生成：

```text
predictions.jsonl
predictions.jsonl.eval-results-gpt-4o
evaluation.stdout.txt
evaluation.stderr.txt
qa_metrics.txt
metrics.json
```

`metrics.json` 包含：

```text
overall_accuracy
task_averaged_accuracy
abstention_accuracy
per_task
```

开发 smoke 配置可以设置：

```yaml
suite:
  benchmark:
    name: longmemeval
    evaluation:
      enabled: false
```

## 7. LOCOMO Adapter 现状

LOCOMO adapter 读取：

```text
locomo/data/locomo10.json
```

一个 LOCOMO conversation 内有多个 QA。adapter 会把每个 QA 展开成一个 `BenchmarkSample`，并让这些 QA 共享同一组 conversation sessions。`question_id` 格式为：

```text
<sample_id>::qa_<index>
```

评估不调用 judge LLM，而是使用 LOCOMO QA 的本地 F1 规则。输出：

```text
locomo_eval.json
locomo_predictions.json
metrics.json
```

运行：

```powershell
python -m agent_memory_eval run configs\suites\locomo.yaml --backend mem0 --limit 10
```
