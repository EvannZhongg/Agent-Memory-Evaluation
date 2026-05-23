# Agent Memory Evaluation Design

当前默认 benchmark 是 LongMemEval，但 runner 不直接依赖 LongMemEval；benchmark 由 adapter 接入，memory 系统由 backend adapter 接入。

## 1. 当前架构

```text
Suite YAML
  |
  v
Suite Runner
  |
  +-- expands backend matrix
  |
  v
Experiment Runner
  |
  +-- Benchmark Adapter
  |     +-- LongMemEval Adapter
  |
  +-- Agent Runtime
  |     +-- Memory Backend
  |     +-- Reader LLM
  |
  v
Run Artifacts
```

核心原则：

- suite 描述 benchmark、agent 公共参数和 backend 矩阵。
- benchmark adapter 负责加载样本、写 prediction 格式、执行 evaluator。
- memory backend 只实现 `reset`、`ingest_session`、`retrieve`、可选 `build_context`。
- runner 只编排数据流，不关心 benchmark 或 memory 系统内部实现。

## 2. 配置布局

```text
configs/
  llm/
    responses.yaml
  memory/
    no_memory.yaml
    mem0.yaml
    amem.yaml
    memoryos.yaml
  suites/
    longmemeval_smoke.yaml
    longmemeval_oracle.yaml
    longmemeval_s_cleaned.yaml
```

suite 示例：

```yaml
suite:
  name: longmemeval_s_cleaned
  benchmark:
    name: longmemeval
    dataset_path: LongMemEval/data/longmemeval_s_cleaned.json
    evaluation:
      enabled: true
      metric_model: gpt-4o
      api_key_env: OPENAI_API_KEY
      base_url_env: OPENAI_BASE_URL

  agent:
    llm_config_path: configs/llm/responses.yaml
    top_k: 10

  backends:
    - name: no_memory
      config_path: configs/memory/no_memory.yaml
    - name: mem0
      config_path: configs/memory/mem0.yaml

  run:
    run_dir_template: runs/{suite}_{backend}
```

## 3. 运行流程

单个样本流程：

```text
backend.reset(question_id)
for session in chronological_sessions:
    backend.ingest_session(session)

retrieved = backend.retrieve(question, top_k)
memory_context = backend.build_context(question, retrieved)
answer = reader_llm(memory_context, question)
benchmark.prediction_record(sample, answer)
```

suite 流程：

```text
load suite
for backend in suite.backends:
    expand resolved experiment config
    run experiment
write runs/<suite>_summary.json
```

## 4. CLI

```powershell
python -m agent_memory_eval validate configs\suites\longmemeval_smoke.yaml
python -m agent_memory_eval run configs\suites\longmemeval_smoke.yaml
python -m agent_memory_eval run configs\suites\longmemeval_s_cleaned.yaml --backend mem0
python -m agent_memory_eval run configs\suites\longmemeval_s_cleaned.yaml --limit 10 --no-eval
```

## 5. 输出

每个 backend run 写入独立目录：

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

suite 级别额外写入：

```text
runs/<suite>_summary.json
```

## 6. 扩展点

新增 memory backend：

```text
agent_memory_eval/backends/<backend>_backend.py
configs/memory/<backend>.yaml
configs/suites/*.yaml -> suite.backends 添加该 backend
```

新增 benchmark：

```text
agent_memory_eval/benchmarks/<benchmark>.py
agent_memory_eval/benchmarks/factory.py 注册 adapter
configs/suites/<benchmark>.yaml
```

详细说明见：

```text
docs/custom_memory_backend_guide.md
docs/custom_benchmark_adapter_guide.md
```
