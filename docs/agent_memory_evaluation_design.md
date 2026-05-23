# Agent Memory Evaluation Design

`agent_memory_eval` 是一个 memory benchmark harness。它负责统一 benchmark 输入、memory backend 调用、reader LLM 问答、评估输出和 token/debug 统计；它不试图统一各 memory 系统的内部算法。

当前内置 benchmark adapter：

```text
agent_memory_eval/benchmarks/longmemeval.py
agent_memory_eval/benchmarks/locomo.py
```

当前内置 memory backend：

```text
agent_memory_eval/backends/no_memory.py
agent_memory_eval/backends/mem0_backend.py
agent_memory_eval/backends/amem_backend.py
agent_memory_eval/backends/memoryos_backend.py
```

## 1. 设计目标

- 用同一套 `BenchmarkSample / MemorySession / MemoryTurn` 协议接入不同 benchmark。
- 用同一套 `MemoryBackend / MemoryItem` 协议接入不同 memory 架构。
- 保留 Mem0、A-MEM、MemoryOS 等方法的原生写入、抽取、演化和检索逻辑。
- 统一最终 reader prompt、运行目录、prediction 格式、metrics、token 统计和 debug 文件。
- 支持 suite 级 backend matrix，方便一次运行同一 benchmark 下的多种 memory backend。
- 让自研 memory 算法可以独立开源，`agent_memory_eval` 只作为本地测评框架使用。

非目标：

- 不在 runner 中写 benchmark 私有逻辑。
- 不在 benchmark adapter 中写 memory 策略。
- 不把自研 memory 核心算法长期放进 `agent_memory_eval/backends`。
- 不为了统一接口而破坏各 memory 方法的原生行为。

## 2. 架构

```text
Suite YAML
  |
  v
Suite Runner
  |
  +-- expand backend matrix
  |
  v
Experiment Runner
  |
  +-- Benchmark Adapter
  |     +-- LongMemEval
  |     +-- LOCOMO
  |
  +-- Agent Runtime
  |     +-- Memory Backend
  |     +-- Reader LLM
  |
  +-- Benchmark Evaluator
  |
  v
Run Artifacts
```

职责边界：

- `suite.py`：把 suite YAML 展开成每个 backend 的 resolved experiment config。
- `suite_runner.py`：遍历 backend matrix，运行实验，并汇总 suite 级结果。
- `runner.py`：编排单个 backend run 的样本循环、写文件、调用 evaluator。
- `benchmarks/*`：加载数据集、转换统一样本、写 prediction record、执行 evaluator。
- `backends/*`：把统一样本转成具体 memory 系统的写入/检索调用。
- `agent.py`：执行 retrieve -> build_context -> reader LLM answer。
- `llm.py`：reader LLM 的 OpenAI-compatible Responses API 客户端。

## 3. 配置布局

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
    locomo.yaml
```

suite 是主入口。一个 suite 描述：

- 使用哪个 benchmark。
- benchmark 数据路径和 evaluator 配置。
- reader LLM 配置。
- 要对比的 backend matrix。
- run_dir 命名规则。

示例：

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
    - name: amem
      config_path: configs/memory/amem.yaml

  run:
    run_dir_template: runs/{suite}_{backend}
```

## 4. 数据模型

benchmark adapter 输出统一结构：

```text
BenchmarkSample
  question_id
  question_type
  question
  answer
  question_date
  sessions: list[MemorySession]
  raw
  benchmark

MemorySession
  session_id
  date
  turns: list[MemoryTurn]
  metadata

MemoryTurn
  role
  content
  timestamp
  metadata
```

memory backend 检索返回统一结构：

```text
MemoryItem
  id
  content
  score
  source_session_id
  metadata
```

原则：

- benchmark adapter 可以把原始字段放进 `raw` / `metadata`，但不能做 memory 策略。
- memory backend 只能依赖 `MemorySession` 和当前 `question`，不能读取 benchmark 原始答案或 evaluator 私有字段。
- `MemoryItem.content` 会进入 reader prompt，应保持可读、可追溯、不要过长。

## 5. 运行流程

单个样本：

```text
backend.reset(question_id)

for session in chronological_sessions:
    backend.ingest_session(session)

retrieved = backend.retrieve(question, top_k)
memory_context = backend.build_context(question, retrieved)
answer = reader_llm(memory_context, question)

benchmark.prediction_record(sample, answer)
```

单个 backend run：

```text
create benchmark adapter
load samples
create memory backend
for sample in samples:
    run sample flow
write predictions/retrievals/debug/token files
benchmark.evaluate(predictions_path, run_dir)
write metrics.json
```

suite run：

```text
load suite YAML
for backend in suite.backends:
    expand config
    run backend experiment
write runs/<suite>_summary.json
```

## 6. 评估策略与粒度策略

后续实验统一遵守：

```text
Evaluator follows benchmark.
Granularity follows memory method for native comparison.
Shared granularity is only for ablation, not main leaderboard.
```

也就是：

- **主评估标准归 benchmark**：LongMemEval 使用 LongMemEval QA evaluator；LOCOMO 使用 LOCOMO QA/F1 评估逻辑；未来新 benchmark 使用该 benchmark 自己定义的 evaluator。
- **主榜写入粒度归 memory 架构**：Mem0、A-MEM、MemoryOS、自研 memory 都按各自原生策略或 adapter 默认策略构建 memory。
- **统一粒度只做消融**：如果要比较 `turn` / `pair` / `session` chunking 的影响，应单独建 controlled suite，不和 native 主结果混在一起。

memory 方法自带评估可以作为诊断指标，但不能作为跨方法主排行榜指标。可作为辅助分析的内容包括：

- 检索命中率或内部 retrieval score。
- summary / profile / knowledge extraction 的中间质量。
- memory 数量、压缩率、更新次数。
- token cost、构建时间、存储大小。
- 失败案例和 evidence 命中情况。

主结果命名建议：

```text
LongMemEval-native
LOCOMO-native
```

消融结果命名建议：

```text
LongMemEval-pair-controlled
LongMemEval-turn-controlled
LOCOMO-pair-controlled
```

## 7. Memory 写入粒度

写入粒度由 memory backend adapter 决定，不由 benchmark adapter 决定。

常见粒度：

```text
native   # 按方法原生方式写入
session  # 整个 session 作为一条写入单位
pair     # user/assistant pair 作为写入单位
turn     # 每个 turn 作为一条写入单位
```

当前状态：

- Mem0：session messages batch；内部再抽取 memory。
- A-MEM：默认 `ingest_granularity: turn`，可配置 `session` / `pair`。
- MemoryOS：pair-level `user_input / agent_response`。
- NoMemory：不写入。

建议：

- 方法原生对比使用 native 或当前 backend 默认粒度。
- 公平 chunking ablation 单独建 suite，不和 native 结果混合解释。
- 每个 backend 应在 config/debug/token metadata 中记录写入粒度。

## 8. 输出文件

每个 backend run 写入：

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

LongMemEval evaluator 启用时额外写入：

```text
predictions.jsonl.eval-results-gpt-4o
evaluation.stdout.txt
evaluation.stderr.txt
qa_metrics.txt
```

LOCOMO evaluator 启用时额外写入：

```text
locomo_eval.json
locomo_predictions.json
```

suite 级汇总：

```text
runs/<suite>_summary.json
```

## 9. Token 成本口径

后续评估必须统一 token 成本口径。主结果除了 benchmark 指标外，还应报告总开销和均摊开销。

### 9.1 统计目标

统一统计以下 token：

```text
llm_input_tokens       # 所有 LLM prompt/input token
llm_output_tokens      # 所有 LLM completion/output token
embedding_input_tokens # 所有送入 embedding 模型的文本 token
embedding_output_tokens# embedding 输出规模；如无法按 token 计，可记录 vector_count / dimension
total_tokens           # llm_input + llm_output + embedding_input，必要时另列 embedding_output_units
```

按阶段拆分：

```text
build_*   # memory 构建阶段：抽取、总结、演化、写入前 embedding
query_*   # memory 检索阶段：query rewrite、检索前 embedding、rerank
reader_*  # 最终回答阶段：reader prompt 和 reader output
eval_*    # benchmark evaluator / judge LLM；默认单独统计，不并入 agent memory 成本
```

构建开销和查询开销都必须单独计算：

```text
build_total_tokens        # build_llm_input + build_llm_output + build_embedding_input
memory_query_total_tokens # query_llm_input + query_llm_output + query_embedding_input
reader_total_tokens       # reader_llm_input + reader_llm_output
query_total_tokens        # memory_query_total_tokens + reader_total_tokens
agent_total_tokens        # build_total_tokens + query_total_tokens
```

其中 `memory_query_total_tokens` 只表示 memory 系统在回答前的检索/重写/rerank/embedding 成本；`query_total_tokens` 表示一次 benchmark 问答的完整查询侧成本，包含 memory query 和最终 reader LLM。embedding output 不并入 token 总和，单独用 `*_embedding_vector_count` / `*_embedding_dimension` 记录规模。

建议最终输出字段：

```text
build_llm_input_tokens
build_llm_output_tokens
build_embedding_input_tokens
build_embedding_vector_count
build_total_tokens

query_llm_input_tokens
query_llm_output_tokens
query_embedding_input_tokens
query_embedding_vector_count
memory_query_total_tokens

reader_llm_input_tokens
reader_llm_output_tokens
reader_total_tokens

query_total_tokens
agent_total_tokens
eval_llm_input_tokens
eval_llm_output_tokens
eval_total_tokens
```

其中：

- `agent_total_tokens` 用于比较 memory 架构本身的端到端成本。
- `build_total_tokens` 用于比较 memory 构建/写入成本。
- `query_total_tokens` 用于比较回答一次问题时的检索和 reader 成本。
- `eval_total_tokens` 用于记录 judge 成本，但不应影响 memory 方法对比。
- 如果某个 backend 内部无法拿到真实 provider usage，使用统一 tokenizer 做 best-effort 估算，并在 `token_counter` / `token_usage_notes` 中标注。

### 9.2 均摊指标

每个 run 至少汇总：

```text
total_build_tokens
total_memory_query_tokens
total_reader_tokens
total_query_tokens
total_agent_tokens

avg_build_tokens_per_sample
avg_build_tokens_per_session
avg_build_tokens_per_turn
avg_query_tokens_per_sample
avg_query_tokens_per_answer
avg_agent_tokens_per_sample
avg_agent_tokens_per_session
avg_agent_tokens_per_turn
avg_agent_tokens_per_answer

total_llm_input_tokens
total_llm_output_tokens
total_embedding_input_tokens
total_embedding_vector_count
```

定义：

- `per_sample`：除以 benchmark sample 数。
- `per_session`：除以所有写入的 session 数。
- `per_turn`：除以所有写入的 turn 数。
- `per_answer`：除以最终回答次数，通常等于 sample 数。
- `avg_build_tokens_per_turn` 是衡量 memory 写入效率的核心字段。
- `avg_query_tokens_per_answer` 是衡量回答阶段成本的核心字段。

这些均摊字段用于回答：

- 哪个 memory 方法总成本更低。
- 哪个方法每轮对话写入成本更高。
- 哪个方法把成本花在 build、memory query、reader 还是 evaluator。
- 准确率提升是否值得额外 token 成本。

### 9.3 当前实现状态

当前 `TokenUsageTracker` 已有 best-effort 字段：

```text
build_input_tokens
build_llm_prompt_tokens
query_input_tokens
query_llm_prompt_tokens
retrieved_context_tokens
reader_prompt_tokens
build_tokens
query_tokens
total_tokens
```

这些字段主要统计输入文本和可捕获的内部 LLM prompt。后续需要补齐：

- LLM output/completion token。
- embedding input token。
- embedding vector count / dimension。
- build/query/reader 的 total 与 average 聚合字段。
- evaluator/judge LLM token 单独统计。
- `avg_*_per_sample/session/turn/answer` 汇总字段。

实现建议：

- 所有 backend adapter 在调用独立 memory 包前后记录 build/query 的输入、LLM prompt、LLM output、embedding 文本。
- reader LLM client 优先读取 provider 返回的 usage；没有 usage 时用统一 tokenizer 估算。
- embedding provider wrapper 记录每次 embedding 的输入文本 token、文本条数、vector 维度。
- `token_usage.jsonl` 保存 sample 级明细，`token_usage_summary.json` 保存 build/query/reader/eval 的 totals 和 averages。
- suite summary 中至少保留 `total_build_tokens`、`total_query_tokens`、`agent_total_tokens`、`avg_build_tokens_per_turn`、`avg_query_tokens_per_answer` 和 `avg_agent_tokens_per_sample`，方便横向比较。

## 10. 开发规范

### 10.1 总体边界

- runner 只编排流程，不写 benchmark 特例，不写 memory 特例。
- benchmark adapter 只处理数据集和 evaluator。
- memory backend 只处理 memory 系统适配。
- reader prompt 只在最终回答阶段使用，不干预各 memory 系统内部 prompt。

### 10.2 新 benchmark

新增文件：

```text
agent_memory_eval/benchmarks/<benchmark>.py
agent_memory_eval/benchmarks/factory.py
configs/suites/<benchmark>.yaml
```

必须提供：

```text
validate
load_samples
prediction_record
evaluate
```

详细规范见：

```text
docs/custom_benchmark_adapter_guide.md
```

### 10.3 新 memory backend

新增文件：

```text
agent_memory_eval/backends/<backend>_backend.py
agent_memory_eval/backends/factory.py
configs/memory/<backend>.yaml
configs/suites/*.yaml
```

自研 memory 算法应放在独立 Python 包，例如：

```text
C-HyperMem/
  c_hypermem/
  pyproject.toml
```

`agent_memory_eval/backends/<backend>_backend.py` 只做 thin adapter。

详细规范见：

```text
docs/custom_memory_backend_guide.md
```

### 10.4 配置规范

- API key 不写进 YAML，使用 `.env` 和 `*_env`。
- `base_url` 和 `base_url_env` 都应优先支持。
- 存储路径默认放在 `runs/` 或 `runs/vectorstores/`。
- suite 内 inline config 可覆盖 `configs/memory/*.yaml`，用于小范围 ablation。
- `config.resolved.yaml` 必须能复现实验配置。

### 10.5 公平性规范

- backend 不得读取 gold answer、evidence label 或 question type 来增强检索。
- 每个 `question_id` 必须隔离 memory namespace / collection / storage。
- 不同 backend 使用相同 reader LLM、相同 question、相同 `top_k` 进行主结果对比。
- 主评估使用 benchmark evaluator，不使用 memory 方法自带 evaluator 作为跨方法主分数。
- native 主榜按 memory 方法原生粒度或 adapter 默认粒度构建 memory。
- 方法原生能力和 controlled chunking ablation 分开报告。
- controlled chunking suite 可以统一 `turn` / `pair` / `session` 粒度，但只能作为消融分析。
- 报告结果时同时给出 accuracy/F1、LLM input/output token、embedding token、平均每轮成本、retrieved context 大小和失败案例。

### 10.6 Git 与本地依赖

建议提交：

```text
agent_memory_eval/
configs/
docs/
scripts/
requirements.txt
README.md
LongMemEval/data/longmemeval_s_cleaned_smoke_1.json
```

LOCOMO 当前可随仓库提交和更新；如果后续体积膨胀，再迁移到外部数据下载流程。

## 11. 推荐开发流程

常规代码变更：

```powershell
python -m agent_memory_eval validate configs\suites\longmemeval_smoke.yaml
python -m agent_memory_eval run configs\suites\longmemeval_smoke.yaml --backend no_memory --limit 1 --no-eval
```

涉及 benchmark adapter：

```powershell
python -m agent_memory_eval validate configs\suites\locomo.yaml
python -m agent_memory_eval run configs\suites\locomo.yaml --backend no_memory --limit 5
```

涉及 memory backend：

```powershell
python -m agent_memory_eval validate configs\suites\longmemeval_smoke.yaml --backend <backend>
python -m agent_memory_eval run configs\suites\longmemeval_smoke.yaml --backend <backend> --limit 1 --no-eval
python -m agent_memory_eval run configs\suites\locomo.yaml --backend <backend> --limit 5
```

正式对比：

```powershell
python -m agent_memory_eval run configs\suites\longmemeval_s_cleaned.yaml
python -m agent_memory_eval run configs\suites\locomo.yaml
```

## 12. 后续 Roadmap

优先级建议：

1. 稳定 benchmark adapter 层：LongMemEval / LOCOMO 的输入、prediction、metrics 保持一致。
2. 稳定 memory backend 层：所有 backend 输出 `MemoryItem`、debug、统一 token usage。
3. 增加 native suite 和 controlled chunking suite，避免公平性解释混淆。
4. 为自研 memory 架构提供 thin adapter，并保持核心包独立开源。
5. 增加结果分析脚本：按 question type、retrieval hit、LLM/embedding token cost、失败原因聚合。
6. 增加 smoke CI：至少 validate suite、导入 backend、跑 no_memory 小样本。

## 13. 文档分工

```text
README.md
  快速开始、运行命令、数据和 git 策略

docs/agent_memory_evaluation_design.md
  总体架构、数据流、开发规范、roadmap

docs/custom_benchmark_adapter_guide.md
  新 benchmark / dataset adapter 接入

docs/custom_memory_backend_guide.md
  新 memory backend / 自研 memory 架构接入
```
