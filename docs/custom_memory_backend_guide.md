# 自研 Memory Backend 接入指南

本文档说明如何把自研 memory 算法接入 `agent_memory_eval`，并通过同一套 suite 同时跑 LongMemEval、LOCOMO 或后续新增 benchmark。

核心思路：**自研 memory 算法应作为独立 Python 包开发和开源，`agent_memory_eval` 只保留一个很薄的评测 adapter**。benchmark adapter 会把不同数据集统一转换成 `MemorySession` / `MemoryTurn` / `BenchmarkSample`，`agent_memory_eval` backend adapter 只负责把这些结构翻译成你的独立 memory 包的 `add/search/reset` 等 API。

如果你要接入的是新数据集或新 benchmark，而不是新 memory 架构，请看：

```text
docs/custom_benchmark_adapter_guide.md
```

推荐边界：

```text
your_memory/                 # 独立开源项目，别人可以单独 pip install 使用
  your_memory/
    memory.py
    schema.py
    config.py
    ...

agent_memory_eval/  # 你的本地测评框架
  backends/
    your_memory_backend.py    # thin adapter，只做格式转换和评测日志
```

## 1. 需要改哪些文件

在 `agent_memory_eval` 中新增或修改：

```text
agent_memory_eval/backends/<your_backend>_backend.py
agent_memory_eval/backends/factory.py
configs/memory/<your_backend>.yaml
configs/suites/*.yaml
```

在独立 memory 项目中维护：

```text
your_memory/
  your_memory/
    memory.py
    schema.py
    config.py
    llms/
    embeddings/
    stores/
    algorithms/
  pyproject.toml
  README.md
```

不要修改：

```text
agent_memory_eval/runner.py
agent_memory_eval/suite_runner.py
agent_memory_eval/agent.py
agent_memory_eval/benchmarks/*.py
LongMemEval/ 或 locomo/ 原始代码
```

## 2. 调用生命周期

每个 benchmark 样本都会走同一个流程：

```text
backend.reset(question_id)

for session in chronological_sessions:
    backend.ingest_session(session)

retrieved = backend.retrieve(question, top_k)
memory_context = backend.build_context(question, retrieved)
answer = reader_llm(memory_context, question)

write predictions / retrieved_memories / debug / metrics
```

你的 backend 最少实现：

```text
reset
ingest_session
retrieve
```

必要时重写：

```text
build_context
get_debug_info
close
```

## 3. 推荐对外 API

独立 memory 包建议暴露一个稳定主入口，例如：

```python
from your_memory import Memory

memory = Memory.from_config("configs/default.yaml")
memory.reset(namespace="alice")
memory.add(messages, user_id="alice", metadata={"session_id": "S1"})
results = memory.search("What does Alice prefer?", user_id="alice", top_k=5)
```

`agent_memory_eval` 不应该依赖你的内部算法模块，只依赖这种稳定 API。这样别人使用 `your_memory` 时不需要安装或理解 `agent_memory_eval`。

## 4. agent_memory_eval Adapter 实现

所有 backend 继承：

```python
from agent_memory_eval.backends.base import MemoryBackend
```

下面示例展示 thin adapter 的形态：真实算法在 `your_memory` 包里，adapter 只做转换。

```python
from __future__ import annotations

from typing import Any

from .base import MemoryBackend, session_to_text
from ..models import MemoryItem, MemorySession


class MyMemoryBackend(MemoryBackend):
    backend_name = "my_memory"
    default_top_k = 10

    def __init__(self, config: dict[str, Any], llm_config: dict[str, Any]):
        super().__init__()
        from your_memory import Memory

        self.config = config
        self.llm_config = llm_config
        self.sample_id: str | None = None
        self.memory = Memory.from_config(config)

    def reset(self, sample_id: str) -> None:
        super().reset(sample_id)
        self.sample_id = sample_id
        self.namespace = _safe_namespace(f"{self.backend_name}_{sample_id}")
        self.memory.reset(namespace=self.namespace)

    def ingest_session(self, session: MemorySession) -> None:
        text = session_to_text(session)
        self.token_usage.record_build(
            text,
            event="my_memory.ingest_session",
            metadata={"session_id": session.session_id, "turn_count": len(session.turns)},
        )
        self.memory.add(
            text,
            namespace=self.namespace,
            metadata={"session_id": session.session_id, "date": session.date, **session.metadata},
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        k = top_k if top_k is not None else self.default_top_k
        self.token_usage.record_memory_query(
            query,
            event="my_memory.retrieve",
            metadata={"top_k": k},
        )
        results = self.memory.search(query, namespace=self.namespace, top_k=k)
        items: list[MemoryItem] = []
        for idx, result in enumerate(results):
            items.append(
                MemoryItem(
                    id=str(result.get("id", f"my_memory_{idx}")),
                    content=str(result.get("content", "")),
                    score=_float_or_none(result.get("score")),
                    source_session_id=(result.get("metadata") or {}).get("session_id"),
                    metadata=result,
                )
            )
        return items

    def get_debug_info(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "sample_id": self.sample_id,
            "stats": self.memory.stats(namespace=self.namespace),
        }


def _safe_namespace(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
```

如果你的独立包当前还没有稳定 API，可以先在 adapter 中写很薄的一层兼容代码，但不要把核心算法长期留在 `agent_memory_eval/backends` 里。

## 5. 注册 Backend

在 `agent_memory_eval/backends/factory.py` 中添加：

```python
if backend == "my_memory":
    from .my_memory_backend import MyMemoryBackend

    return MyMemoryBackend(config, llm_config)
```

`backend` 名称应和 YAML 中一致。

## 6. Memory 配置文件

新增：

```text
configs/memory/my_memory.yaml
```

示例：

```yaml
backend: my_memory
package_path: ../your_memory
storage_path: runs/vectorstores/my_memory
default_top_k: 10
index:
  type: hybrid
  use_bm25: true
  use_embedding: true
```

建议：

- 算法超参数放在 YAML，不要写死在代码里。
- 持久化路径放 `runs/` 下，避免污染源码目录。
- key 放 `.env`，不要写进 YAML。
- 如果独立包尚未安装，可以在 adapter 里读取 `package_path` 并临时加入 `sys.path`；正式发布后建议改为 `pip install -e ../your_memory`。

## 7. 加入 Suite

把 backend 加到要对比的 suite：

```yaml
suite:
  backends:
    - name: my_memory
      config_path: configs/memory/my_memory.yaml
```

例如同时加入：

```text
configs/suites/longmemeval_smoke.yaml
configs/suites/longmemeval_s_cleaned.yaml
configs/suites/locomo.yaml
```

统一覆盖 `top_k`：

```yaml
suite:
  agent:
    top_k: 10
```

只对某个 backend 覆盖参数：

```yaml
suite:
  backends:
    - name: my_memory
      config_path: configs/memory/my_memory.yaml
      default_top_k: 20
      index:
        use_bm25: false
```

inline 参数会覆盖 `configs/memory/my_memory.yaml` 中的同名字段。

## 8. 输入数据结构

### MemorySession

```python
@dataclass
class MemorySession:
    session_id: str
    date: str | None
    turns: list[MemoryTurn]
    metadata: dict
```

### MemoryTurn

```python
@dataclass
class MemoryTurn:
    role: str
    content: str
    timestamp: str | None
    metadata: dict
```

不同 benchmark 的细节会被 adapter 放进 `metadata`。例如：

- LongMemEval turn metadata 包含 `source_session_id`
- LOCOMO turn metadata 包含 `speaker`、`dia_id`、`source_session_id`

### MemoryItem

`retrieve` 必须返回：

```python
@dataclass
class MemoryItem:
    id: str
    content: str
    score: float | None
    source_session_id: str | None
    metadata: dict
```

`content` 会进入最终 reader prompt，应尽量简洁、可读、和问题相关。

## 9. 使用 LLM 和 Embedding

suite 里的：

```yaml
agent:
  llm_config_path: configs/llm/responses.yaml
```

会加载 reader LLM 配置，并传给 backend 的 `__init__(config, llm_config)`。

读取主 LLM：

```python
from ..config import env_value

llm_api_key = env_value(llm_config.get("api_key_env", "LLM_API_KEY"))
llm_base_url = llm_config.get("base_url")
llm_model = llm_config.get("model")
```

读取 embedding：

```python
embedding_config = llm_config.get("embedding", {})
embedding_api_key = env_value(embedding_config.get("api_key_env", "EMBEDDING_API_KEY"))
embedding_base_url = embedding_config.get("base_url")
embedding_model = embedding_config.get("model")
```

如果你的算法需要单独的 extraction LLM，可以放在 `configs/memory/my_memory.yaml`：

```yaml
extraction_llm:
  model: qwen3.6-plus
  api_key_env: LLM_API_KEY
  base_url: https://...
```

## 10. 样本隔离

每个 `question_id` 必须隔离 memory 状态。

推荐：

```python
def reset(self, sample_id: str) -> None:
    super().reset(sample_id)
    self.sample_id = sample_id
    self.namespace = f"{self.backend_name}_{sample_id}"
```

如果使用持久化 vector store：

- 每个样本使用独立 collection / namespace。
- 或 `reset` 时清理当前 namespace。
- 不要让不同样本共享 memory，否则会信息泄漏。

注意 LOCOMO 的 `question_id` 形如：

```text
conv-26::qa_0
```

如果底层数据库不支持 `:`，需要在 namespace 中做安全转义。

## 11. build_context

默认格式：

```text
[1 source=S1 score=0.8123] memory content
[2 source=S2] memory content
```

如果你的 memory 有结构化内容，可以重写：

```python
def build_context(self, query: str, retrieved: list[MemoryItem]) -> str:
    ...
```

建议：

- 控制长度，避免把无关内容塞给 reader。
- 保留 source/session 信息，便于 debug。
- 对 profile / episodic / semantic / graph memory 分区展示。

## 12. Token 和 Debug

推荐记录：

```python
self.token_usage.record_build(...)
self.token_usage.record_memory_query(...)
self.token_usage.record_build_llm_prompt(...)
self.token_usage.record_query_llm_prompt(...)
```

runner 会写出：

```text
token_usage.jsonl
token_usage_summary.json
backend_debug.jsonl
```

`get_debug_info` 可返回：

```python
def get_debug_info(self) -> dict[str, Any]:
    return {
        "memory_count": self.memory.count(namespace=self.namespace),
        "namespace": self.namespace,
    }
```

## 13. Smoke Test

开发一个新 backend 后，建议按顺序：

```powershell
python -m agent_memory_eval validate configs\suites\longmemeval_smoke.yaml --backend my_memory
python -m agent_memory_eval run configs\suites\longmemeval_smoke.yaml --backend my_memory --dry-run
python -m agent_memory_eval run configs\suites\longmemeval_smoke.yaml --backend my_memory --limit 1 --no-eval
```

再跑 LOCOMO 小样本：

```powershell
python -m agent_memory_eval validate configs\suites\locomo.yaml --backend my_memory
python -m agent_memory_eval run configs\suites\locomo.yaml --backend my_memory --limit 5
```

检查：

```text
runs/<suite>_<backend>/
  config.resolved.yaml
  predictions.jsonl
  retrieved_memories.jsonl
  ingest_trace.jsonl
  backend_debug.jsonl
  token_usage.jsonl
  token_usage_summary.json
  metrics.json
```

重点看：

- `retrieved_memories.jsonl` 是否有结果。
- `memory_context` 是否可读。
- `source_session_id` 是否能追溯。
- `metrics.json` 是否 evaluated / skipped 符合预期。

## 14. 推荐开发里程碑

M1：最小可跑

- 独立包实现 `Memory.from_config/add/search/reset`
- `agent_memory_eval` adapter 实现 `reset/ingest_session/retrieve`
- 跑通 `longmemeval_smoke`

M2：真实索引

- 加 embedding / keyword / hybrid index
- 支持 `top_k`
- 保存 score
- 输出 debug 信息

M3：记忆加工

- extraction
- summary
- entity linking
- temporal metadata
- profile / semantic / episodic 分层

M4：正式对比

- 跑 `longmemeval_s_cleaned`
- 跑 `locomo`
- 对比 `no_memory`、`mem0`、`amem`、`memoryos`
- 分析 retrieval failure 和 token cost
- 准备独立包 README、examples、pyproject，确保脱离 `agent_memory_eval` 可用

## 15. 常见错误

### 15.1 不隔离样本

不同 `question_id` 共享同一 index 会造成信息泄漏。

### 15.2 返回原始数据库对象

`retrieve` 必须返回 `list[MemoryItem]`，不要直接返回 Chroma/Qdrant/Neo4j 原始对象。

### 15.3 content 太长

`MemoryItem.content` 会进入 reader prompt。过长会增加成本和噪声。

### 15.4 忘记调用 `super().reset`

`super().reset(sample_id)` 会重置 token tracker。忘记调用会污染 token 统计。

### 15.5 backend 名称不一致

这三个地方应一致：

```text
configs/memory/my_memory.yaml -> backend: my_memory
configs/suites/*.yaml -> name: my_memory
agent_memory_eval/backends/factory.py -> if backend == "my_memory"
```

### 15.6 把核心算法写进评测 adapter

`agent_memory_eval/backends/my_memory_backend.py` 应该只是适配层。核心算法、数据结构、存储和 provider 抽象应放在独立 `your_memory` 包里。

## 16. 文件命名建议

假设算法名为 `HyperMemory`：

```text
../hyper_memory/
  hyper_memory/
    memory.py
    schema.py
    config.py
  pyproject.toml

agent_memory_eval/backends/hyper_memory_backend.py  # thin adapter
configs/memory/hyper_memory.yaml
configs/suites/longmemeval_smoke.yaml
configs/suites/longmemeval_s_cleaned.yaml
configs/suites/locomo.yaml
runs/longmemeval_smoke_hyper_memory/
```

backend 名称建议用小写 snake_case：

```yaml
backend: hyper_memory
```

## 17. 对外发布建议

独立 memory 包建议结构：

```text
your_memory/
  your_memory/
    memory.py
    schema.py
    config.py
    llms/
    embeddings/
    stores/
    algorithms/
    adapters/
      hypermemo.py
      langchain.py
  configs/
  examples/
  tests/
  pyproject.toml
```

推荐外部 API 保持简单：

```python
from your_memory import Memory

memory = Memory.from_config("configs/default.yaml")
memory.add(messages, user_id="alice")
results = memory.search("What does Alice prefer?", user_id="alice", top_k=5)
```

Adapter 可以作为单独文件保留在评测仓库里；如果希望开源包也自带适配器，可以放在：

```text
your_memory/adapters/hypermemo.py
```

但这个适配器应是可选依赖，不应让普通用户安装 `agent_memory_eval` 才能使用你的 memory 包。
