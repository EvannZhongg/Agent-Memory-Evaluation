# 自定义 Benchmark Adapter 接入指南

`agent_memory_eval` 的 runner 只依赖统一 benchmark adapter，不直接依赖具体数据集。当前内置 adapter：

```text
agent_memory_eval/benchmarks/longmemeval.py
agent_memory_eval/benchmarks/locomo.py
```

后续接入 LOCOMO、自定义长期对话集或任务型 agent benchmark 时，应新增 benchmark adapter，而不是修改 `agent_memory_eval/runner.py`。

本文档只说明 **benchmark adapter** 怎么接入。自研 memory 架构、backend adapter、写入粒度、检索返回格式等规范见：

```text
docs/custom_memory_backend_guide.md
```

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

memory backend 只做 memory 架构适配，不应包含 benchmark 私有逻辑。具体规范见 `docs/custom_memory_backend_guide.md`。

这个边界保证后续换 benchmark 时不用改 memory backend，换 memory 架构时也不用改 benchmark adapter。

## 3. 注册 Adapter

在 `agent_memory_eval/benchmarks/factory.py` 添加：

```python
if name == "my_benchmark":
    from .my_benchmark import MyBenchmark

    return MyBenchmark(config, root)
```

## 4. Suite 配置

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

## 5. LongMemEval Adapter 现状

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

## 6. LOCOMO Adapter 现状

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
