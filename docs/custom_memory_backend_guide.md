# 自研 Memory Backend 接入指南

本文档用于指导后续将自研 memory 算法接入 HyperMemo 评测框架。目标是让新算法可以和 `mem0`、`A-mem`、`MemoryOS` 一样，通过配置切换，并使用同一套 LongMemEval runner 进行 smoke test 和后续评测。

## 1. 接入原则

自研 memory 算法应作为新的 **Memory Backend Adapter** 接入。

不要修改以下模块：

- `agent_memory_eval/runner.py`
- `agent_memory_eval/longmemeval.py`
- `agent_memory_eval/agent.py`
- LongMemEval 原始数据和官方评测代码

应新增或修改：

- `agent_memory_eval/backends/<your_backend>_backend.py`
- `agent_memory_eval/backends/factory.py`
- `configs/memory/<your_backend>.yaml`
- `configs/experiments/longmemeval_oracle_<your_backend>.yaml`

## 2. 框架调用生命周期

LongMemEval 中每个样本的执行顺序如下：

```text
backend.reset(question_id)

for session in chronological_sessions:
    backend.ingest_session(session)

retrieved = backend.retrieve(question, top_k)
memory_context = backend.build_context(question, retrieved)
answer = reader_llm(memory_context, question)

write predictions / retrieved_memories / debug
```

你的 memory 算法只需要实现 `reset`、`ingest_session`、`retrieve`，必要时重写 `build_context`。

## 3. 必须实现的接口

所有 backend 都继承：

```python
from agent_memory_eval.backends.base import MemoryBackend
```

最小实现：

```python
from __future__ import annotations

from typing import Any

from .base import MemoryBackend, session_to_text
from ..models import MemoryItem, MemorySession


class MyMemoryBackend(MemoryBackend):
    backend_name = "my_memory"
    default_top_k = 10

    def __init__(self, config: dict[str, Any], llm_config: dict[str, Any]):
        self.config = config
        self.llm_config = llm_config
        self.sample_id: str | None = None
        self.store = None

    def reset(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.store = {}

    def ingest_session(self, session: MemorySession) -> None:
        text = session_to_text(session)
        self.store[session.session_id] = {
            "text": text,
            "date": session.date,
            "metadata": session.metadata,
        }

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        k = top_k if top_k is not None else self.default_top_k
        results = []
        for session_id, payload in list(self.store.items())[:k]:
            results.append(
                MemoryItem(
                    id=session_id,
                    content=payload["text"],
                    score=None,
                    source_session_id=session_id,
                    metadata=payload,
                )
            )
        return results
```

## 4. 数据结构说明

### 4.1 MemorySession

`ingest_session` 收到的是统一格式：

```python
@dataclass
class MemorySession:
    session_id: str
    date: str | None
    turns: list[MemoryTurn]
    metadata: dict
```

LongMemEval 的 `haystack_sessions` 已经被转换为 `MemorySession`，并按照时间顺序注入。

### 4.2 MemoryTurn

```python
@dataclass
class MemoryTurn:
    role: str
    content: str
    timestamp: str | None
    metadata: dict
```

通常 `role` 为 `user` 或 `assistant`。

### 4.3 MemoryItem

`retrieve` 需要返回：

```python
@dataclass
class MemoryItem:
    id: str
    content: str
    score: float | None
    source_session_id: str | None
    metadata: dict
```

`content` 会进入最终 reader LLM 的 memory context，因此应是可读的、对回答有帮助的文本。

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
storage_path: runs/vectorstores/my_memory
default_top_k: 10
index:
  type: hybrid
  use_bm25: true
  use_embedding: true
```

建议把算法超参数全部放在这里，不要写死在代码中。

## 7. 实验配置文件

新增：

```text
configs/experiments/longmemeval_oracle_my_memory.yaml
```

示例：

```yaml
experiment:
  name: longmemeval_oracle_my_memory
  dataset_path: LongMemEval/data/longmemeval_oracle.json
  run_dir: runs/longmemeval_oracle_my_memory
  limit: 1

agent:
  llm_config_path: configs/llm/responses.yaml
  memory:
    config_path: configs/memory/my_memory.yaml
```

如果想统一覆盖 top_k：

```yaml
agent:
  top_k: 10
```

## 8. 使用 LLM 和 Embedding

统一 LLM 配置位于：

```text
configs/llm/responses.yaml
```

adapter 初始化时会收到：

```python
def __init__(self, config: dict[str, Any], llm_config: dict[str, Any]):
    ...
```

其中：

- `config`：来自 `configs/memory/my_memory.yaml`
- `llm_config`：来自 `configs/llm/responses.yaml`

如需读取主 LLM key：

```python
from ..config import env_value

llm_api_key = env_value(llm_config.get("api_key_env", "LLM_API_KEY"))
llm_base_url = llm_config.get("base_url")
```

如需读取 embedding key：

```python
embedding_config = llm_config.get("embedding", {})
embedding_api_key = env_value(embedding_config.get("api_key_env", "EMBEDDING_API_KEY"))
embedding_base_url = embedding_config.get("base_url")
embedding_model = embedding_config.get("model")
```

注意：

- 第一阶段 reader LLM 和 backend extraction LLM 默认共用主 LLM。
- 如果你的算法需要单独 extraction LLM，可先在 `configs/memory/my_memory.yaml` 中加字段，后续再扩展通用配置规范。

## 9. Namespace 和隔离

每个 LongMemEval 样本都应隔离 memory 状态。

推荐做法：

```python
def reset(self, sample_id: str) -> None:
    self.sample_id = sample_id
    self.namespace = f"longmemeval_{sample_id}"
```

如果使用持久化存储，建议：

- 每个 `question_id` 使用独立 collection / namespace。
- 或每次 `reset` 清理当前 namespace。
- 不要让不同样本共享 memory，否则评测会污染。

## 10. build_context 的选择

默认 `MemoryBackend.build_context` 会把 `MemoryItem` 格式化为：

```text
[1 source=session_x score=0.8123] memory content
[2 source=session_y] memory content
```

如果你的 memory 有结构化内容，例如 profile、episodic memory、semantic memory、graph links，可以重写：

```python
def build_context(self, query: str, retrieved: list[MemoryItem]) -> str:
    ...
```

建议输出保持简洁，避免把过多无关内容塞给 reader LLM。

## 11. Debug 输出

可选实现：

```python
def get_debug_info(self) -> dict[str, Any]:
    return {
        "memory_count": len(self.store),
        "namespace": self.namespace,
    }
```

runner 会写入：

```text
backend_debug.jsonl
```

这对分析不同 memory 算法很重要。

## 12. Smoke Test 流程

开发一个新 backend 后，建议按顺序执行：

```powershell
python -m agent_memory_eval validate configs/experiments/longmemeval_oracle_my_memory.yaml
python -m agent_memory_eval run configs/experiments/longmemeval_oracle_my_memory.yaml --limit 1 --dry-run
python -m agent_memory_eval run configs/experiments/longmemeval_oracle_my_memory.yaml --limit 1
```

检查输出：

```text
runs/longmemeval_oracle_my_memory/
  predictions.jsonl
  retrieved_memories.jsonl
  ingest_trace.jsonl
  backend_debug.jsonl
  metrics.json
```

重点检查：

- `retrieved_memories.jsonl` 是否有结果。
- `source_session_id` 是否能追溯到 LongMemEval session。
- `memory_context` 是否可读。
- `predictions.jsonl` 是否为合法 JSONL。

## 13. 推荐开发里程碑

### M1: 最小可跑版本

- 实现 `reset`
- 实现 `ingest_session`
- 实现简单 `retrieve`
- 跑通 `longmemeval_oracle` 单样本

### M2: 加入真实索引

- 增加 embedding 或 keyword index
- 支持 top_k
- 保存 score
- 写出 debug 信息

### M3: 加入记忆加工

- extraction
- summary
- entity linking
- temporal metadata
- profile / semantic / episodic 分层

### M4: 对齐评测

- 接入 `longmemeval_s_cleaned`
- 分析 retrieval failure
- 接入正式 LongMemEval evaluator
- 记录不同超参数的实验配置

## 14. 常见错误

### 14.1 不隔离样本

如果不同 `question_id` 共享同一个 index，会导致信息泄漏。必须在 `reset` 中隔离。

### 14.2 retrieve 返回原始对象

`retrieve` 必须返回 `list[MemoryItem]`，不要直接返回向量库或数据库原始结果。

### 14.3 content 太长

`MemoryItem.content` 会进入 reader prompt。过长会带来成本和噪声，建议在 adapter 内控制长度或摘要。

### 14.4 忘记处理空检索

如果没有检索结果，返回空列表即可。默认 context 会变成：

```text
No relevant memories retrieved.
```

## 15. 文件命名建议

假设算法名为 `HyperMemory`：

```text
agent_memory_eval/backends/hyper_memory_backend.py
configs/memory/hyper_memory.yaml
configs/experiments/longmemeval_oracle_hyper_memory.yaml
runs/longmemeval_oracle_hyper_memory/
```

backend 名称建议使用小写 snake_case：

```yaml
backend: hyper_memory
```

## 16. 对外发布时的代码组织建议

本节仅供参考。后续如果要把自研 memory 算法作为独立项目发布，最终目录结构和模块边界应以实际算法开发需要为准，不必强行套用本文结构。

综合当前参考项目：

- `mem0` 更适合作为对外 SDK、配置系统、provider 抽象和插件化工程组织的参考。
- `A-mem` 更适合作为研究型核心算法代码的简洁性参考。
- `MemoryOS` 更适合作为 short / mid / long-term 分层记忆思想的模块化参考。

建议优先采用 **mem0 风格的对外接口与包组织**，同时吸收 **A-mem 的算法可读性**，如果算法本身包含多层记忆，再参考 **MemoryOS 的分层模块**。

一个可参考的发布结构：

```text
your_memory/
  your_memory/
    __init__.py
    memory.py              # 对外主入口：Memory
    schema.py              # MemoryItem / Session / Turn
    config.py              # 配置加载
    prompts.py
    llms/                  # LLM provider 封装
    embeddings/            # embedding provider 封装
    stores/                # vector / graph / kv store 封装
    algorithms/            # 核心 memory 算法
    adapters/
      hypermemo.py         # 接入本评测框架
      langchain.py         # 可选生态 adapter
    utils/
  examples/
    quickstart.py
    longmemeval_smoke.py
  configs/
    default.yaml
    dashscope.yaml
  tests/
  docs/
  pyproject.toml
  README.md
```

推荐对外 API 保持简单：

```python
from your_memory import Memory

memory = Memory.from_config("configs/default.yaml")

memory.add(messages, user_id="alice")
results = memory.search("What does Alice prefer?", user_id="alice", top_k=5)
```

组织原则：

- 对外只暴露稳定的 `Memory` 主入口，不要求用户理解内部算法细节。
- 核心算法放在 `algorithms/`，尽量保持可读、可复现、便于写实验。
- LLM、embedding、存储后端使用 provider 抽象，便于后续替换。
- key 放 `.env`，模型名、base URL、top_k、存储路径放 YAML。
- 如果算法包含短期、中期、长期记忆，可拆成明确模块，但不要让调用方必须直接操作这些模块。
- 发布时建议同时提供 HyperMemo adapter，使算法可以直接接入 LongMemEval runner。

