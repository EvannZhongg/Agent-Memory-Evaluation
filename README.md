# Agent Memory Evaluation

一个用于评测 agent memory 系统的统一实验框架。它使用 LongMemEval 作为统一输入协议，对比不同 memory backend 在长期对话记忆任务中的写入、检索、回答和 token 成本。

当前支持的 backend：

- `none`：无记忆 baseline
- `mem0`：本地源码 adapter，对应 `mem0/`
- `amem`：本地源码 adapter，对应 `A-mem/`
- `memoryos`：本地源码 adapter，对应 `MemoryOS/memoryos-pypi/`

核心原则是：**统一外部评测协议，不统一 memory 系统内部实现**。Mem0 / A-MEM / MemoryOS 的内部写入、抽取、演化、更新 prompt 保持各自原样；只统一最终 reader prompt、数据流、输出格式和统计口径。

## 项目结构

```text
agent_memory_eval/      # 统一评测框架
  benchmarks/           # benchmark adapter；当前支持 LongMemEval / LOCOMO
configs/
  llm/responses.yaml    # 统一 reader LLM 与 embedding 配置
  memory/               # 各 memory backend 配置
  suites/               # benchmark suite / backend matrix 配置
docs/                   # 设计和接入文档
scripts/                # PowerShell helper
LongMemEval/            # benchmark 代码；全量数据默认不入库
mem0/                   # mem0 源码
A-mem/                  # A-MEM 源码
MemoryOS/               # MemoryOS 源码
runs/                   # 实验输出；不入库
```

本仓库定位为 **memory benchmark harness**。以下内容只建议本地保留，不建议提交到 git：

```text
C-HyperMem/                 # 自研 memory 架构，应作为独立仓库/包维护
Memory-in-the-LLM-Era-main/ # 外部复现仓库，仅用于对照分析
参考文献/                   # 论文笔记和本地资料
runs/                       # 实验输出
```

仓库保留一个小型 LongMemEval 完整 haystack fixture：

```text
LongMemEval/data/longmemeval_s_cleaned_smoke_1.json
```

全量 `longmemeval_s_cleaned.json`、`longmemeval_m_cleaned.json`、`longmemeval_oracle.json` 较大，默认被 `.gitignore` 忽略，需要本地下载到 `LongMemEval/data/`。

LOCOMO suite 默认读取：

```text
locomo/data/locomo10.json
```

`locomo/` 可以随仓库提交和更新；如果后续数据体积继续增大，再单独把大文件迁到外部数据下载流程。

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

各 memory backend 依赖单独安装：

```powershell
pip install -e .\mem0
pip install -e .\A-mem
pip install -r .\MemoryOS\memoryos-pypi\requirements.txt
```

如果后续接入自己的 memory 架构，例如 `C-HyperMem`，建议在独立目录或独立仓库中安装：

```powershell
pip install -e ..\C-HyperMem
```

`agent_memory_eval` 中只保留一个薄 adapter 和 `configs/memory/*.yaml`，不要把自研 memory 包源码直接提交到本评测仓库。

## API Key

在项目根目录创建 `.env`：

```text
LLM_API_KEY=你的LLM服务Key
EMBEDDING_API_KEY=你的Embedding服务Key
OPENAI_API_KEY=LongMemEval官方QA evaluator使用的OpenAI Key
OPENAI_BASE_URL=https://api.openai.com/v1
```

统一配置在：

```text
configs/llm/responses.yaml
```

当前默认使用 DashScope 的 OpenAI-compatible Responses 接口。reader LLM 和 memory backend 的 extraction/organization LLM 第一阶段默认复用这份配置，后续可以在 adapter 层扩展为独立配置。

LongMemEval 官方 QA evaluator 默认使用 `gpt-4o` 作为 judge，并读取 `OPENAI_API_KEY` 和可选的 `OPENAI_BASE_URL`。可以在 suite 配置的 `suite.benchmark.evaluation` 中修改 `metric_model`、`base_url`、`base_url_env` 或关闭评估。

## 评测流程

单个 benchmark 样本的执行流程：

```text
backend.reset(question_id)
for session in haystack_sessions 按时间顺序:
    backend.ingest_session(session)

retrieved = backend.retrieve(question)
memory_context = backend.build_context(retrieved)
answer = reader_llm(question, memory_context)
write predictions.jsonl
benchmark.evaluate(predictions.jsonl) -> metrics.json
```

最终 reader prompt 位于：

```text
agent_memory_eval/prompts.py
```

它只用于最终问答阶段，不用于 Mem0 / A-MEM / MemoryOS 的内部 memory prompt。

## Suite 配置

benchmark 和 backend 矩阵通过 suite 配置。一个 suite 描述“哪个 benchmark、哪些 backend、怎么输出 run 目录”：

```yaml
suite:
  name: longmemeval_s_cleaned_mem0
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

开发 smoke suite 默认设置 `evaluation.enabled: false`，避免每次调 adapter 都调用 LLM judge。如果只想对某个 suite 写死 judge endpoint，也可以用：

```yaml
suite:
  benchmark:
    evaluation:
      base_url: https://your-openai-compatible-endpoint/v1
```

当前内置 suite：

```text
configs/suites/longmemeval_smoke.yaml
configs/suites/longmemeval_oracle.yaml
configs/suites/longmemeval_s_cleaned.yaml
configs/suites/locomo.yaml
```

## 运行示例

校验配置：

```powershell
python -m agent_memory_eval validate configs\suites\longmemeval_smoke.yaml
```

跑 smoke suite 的全部 backend：

```powershell
python -m agent_memory_eval run configs\suites\longmemeval_smoke.yaml
```

只跑单个 backend：

```powershell
python -m agent_memory_eval run configs\suites\longmemeval_s_cleaned.yaml --backend mem0
```

临时限制样本数或关闭评估：

```powershell
python -m agent_memory_eval run configs\suites\longmemeval_s_cleaned.yaml --limit 10 --no-eval
```

运行 LOCOMO：

```powershell
python -m agent_memory_eval validate configs\suites\locomo.yaml
python -m agent_memory_eval run configs\suites\locomo.yaml --backend mem0 --limit 10
```

LOCOMO adapter 使用本地 F1 规则评估，不调用额外 judge LLM。

## 输出文件

每次运行会在 `runs/<run_dir>/` 下写出：

```text
config.resolved.yaml       # 本次实验解析后的完整配置
predictions.jsonl          # LongMemEval evaluator 输入
retrieved_memories.jsonl   # 每个问题检索到的 memory
ingest_trace.jsonl         # 每个 session 写入记录
backend_debug.jsonl        # backend 调试信息
token_usage.jsonl          # 每个样本的 token 统计
token_usage_summary.json   # token 统计汇总
metrics.json               # benchmark adapter 产出的评测指标或跳过/失败状态
evaluation.stdout.txt      # LongMemEval evaluate_qa.py stdout，启用评估时生成
evaluation.stderr.txt      # LongMemEval evaluate_qa.py stderr，启用评估时生成
qa_metrics.txt             # LongMemEval print_qa_metrics.py 输出，启用评估时生成
```

每次 suite 运行还会写出：

```text
runs/<suite>_summary.json  # 每个 backend 的 run_dir、accuracy 和 token 汇总
```

LongMemEval 评估启用时还会生成：

```text
predictions.jsonl.eval-results-gpt-4o
```

`metrics.json` 会包含 `overall_accuracy`、`task_averaged_accuracy`、`abstention_accuracy` 和 `per_task` 分组结果。

LOCOMO 评估启用时会额外生成：

```text
locomo_eval.json
locomo_predictions.json
```

`metrics.json` 会包含 `overall_f1` 和 `per_category` 分组结果。

## Benchmark Adapter

runner 不再直接依赖 LongMemEval loader，而是通过 `agent_memory_eval/benchmarks/` 下的 adapter 访问 benchmark。

新增 benchmark 时实现：

```text
agent_memory_eval/benchmarks/<benchmark>.py
```

并在 `agent_memory_eval/benchmarks/factory.py` 注册。adapter 需要提供：

```python
validate()
load_samples(limit)
prediction_record(sample, answer)
evaluate(predictions_path, run_dir)
```

只要 adapter 输出统一的 `BenchmarkSample.sessions/question/question_id`，现有 memory backend 和 agent runtime 都不用改。

token 统计采用统一口径，既报告总开销，也报告构建开销、查询开销和均摊开销：

```text
build_llm_input_tokens      # memory 构建阶段 LLM input/prompt
build_llm_output_tokens     # memory 构建阶段 LLM output/completion
build_embedding_input_tokens
build_total_tokens          # 写入、抽取、总结、演化、写入前 embedding 的总 token

query_llm_input_tokens      # memory query/rewrite/rerank LLM input
query_llm_output_tokens     # memory query/rewrite/rerank LLM output
query_embedding_input_tokens
memory_query_total_tokens   # memory 检索侧总 token

reader_llm_input_tokens     # 最终回答 LLM input
reader_llm_output_tokens    # 最终回答 LLM output
reader_total_tokens

query_total_tokens          # memory_query_total_tokens + reader_total_tokens
agent_total_tokens          # build_total_tokens + query_total_tokens

eval_llm_input_tokens       # benchmark evaluator / judge LLM，单独统计
eval_llm_output_tokens
eval_total_tokens
```

`token_usage_summary.json` 和 suite summary 应同时包含总量和均摊指标，例如：

```text
total_build_tokens
total_memory_query_tokens
total_reader_tokens
total_query_tokens
total_agent_tokens
avg_build_tokens_per_turn
avg_query_tokens_per_answer
avg_agent_tokens_per_sample
avg_agent_tokens_per_turn
```

token 统计是框架侧的可比较口径，并 best-effort 捕获三套 memory 内部 LLM prompt。若某个库内部使用异步、子进程或绕过当前 LLM 方法，可能无法得到精确 billing usage；这种情况需要在 `token_usage_notes` 中标注。

## 数据说明

`longmemeval_oracle.json` 和 `longmemeval_s_cleaned.json` 字段格式一致：

```text
question_id
question_type
question
answer
question_date
haystack_session_ids
haystack_dates
haystack_sessions
answer_session_ids
```

区别是：

- `oracle` 只包含 evidence sessions，适合 adapter 快速验证。
- `s_cleaned` 包含完整 haystack history，适合最终效果评估。
- `longmemeval_s_cleaned_smoke_1.json` 是从 `s_cleaned` 裁切出的单样本完整 haystack，用于跑通测试。

## Git 与本地依赖策略

建议提交到 git 的内容：

```text
agent_memory_eval/
configs/
docs/
scripts/
requirements.txt
README.md
.gitignore
LongMemEval/data/longmemeval_s_cleaned_smoke_1.json
```

默认不提交的内容：

```text
.env
.venv/
runs/
C-HyperMem/
Memory-in-the-LLM-Era-main/
参考文献/
LongMemEval/data/longmemeval_*.json  # smoke fixture 除外
各类 vectorstores / chroma / sqlite / cache
```

当前仓库中 `mem0/`、`A-mem/`、`MemoryOS/`、`LongMemEval/` 已作为本地源码 adapter 依赖存在。如果后续想进一步瘦身，可以改成“外部 clone + pip install -e”的模式，并用 `git rm --cached` 将这些第三方源码从版本库移出；`.gitignore` 中已经预留了对应注释规则。

## GitHub 上传

目标仓库：

```text
https://github.com/EvannZhongg/Agent-Memory-Evaluation.git
```

项目克隆：
```
git clone https://github.com/EvannZhongg/Agent-Memory-Evaluation.git
cd Agent-Memory-Evaluation
```

如果本地已经初始化过 git，只需要：

```powershell
git status

git add .
git commit -m "你的提交说明"

git pull --rebase origin main
git push origin main
```

## 下载 LongMemEval 数据集

```powershell
cd LongMemEval
hf download xiaowu0162/longmemeval-cleaned `
  longmemeval_oracle.json `
  --repo-type dataset `
  --local-dir data
```

## 文档

详细设计见：

```text
docs/agent_memory_evaluation_design.md
docs/custom_benchmark_adapter_guide.md
docs/custom_memory_backend_guide.md
```
