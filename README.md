# HyperMemo Agent Memory Evaluation

HyperMemo 是一个用于评测 agent memory 系统的统一实验框架。它使用 LongMemEval 作为统一输入协议，对比不同 memory backend 在长期对话记忆任务中的写入、检索、回答和 token 成本。

当前支持的 backend：

- `none`：无记忆 baseline
- `mem0`：本地源码 adapter，对应 `mem0/`
- `amem`：本地源码 adapter，对应 `A-mem/`
- `memoryos`：本地源码 adapter，对应 `MemoryOS/memoryos-pypi/`

核心原则是：**统一外部评测协议，不统一 memory 系统内部实现**。Mem0 / A-MEM / MemoryOS 的内部写入、抽取、演化、更新 prompt 保持各自原样；HyperMemo 只统一最终 reader prompt、数据流、输出格式和统计口径。

## 项目结构

```text
agent_memory_eval/      # 统一评测框架
configs/
  experiments/          # LongMemEval 实验配置
  llm/responses.yaml    # 统一 reader LLM 与 embedding 配置
  memory/               # 各 memory backend 配置
docs/                   # 设计和接入文档
scripts/                # PowerShell helper
LongMemEval/            # benchmark 代码；全量数据默认不入库
mem0/                   # mem0 源码
A-mem/                  # A-MEM 源码
MemoryOS/               # MemoryOS 源码
runs/                   # 实验输出，已被 .gitignore 忽略
```

仓库保留一个小型完整 haystack fixture：

```text
LongMemEval/data/longmemeval_s_cleaned_smoke_1.json
```

全量 `longmemeval_s_cleaned.json`、`longmemeval_m_cleaned.json`、`longmemeval_oracle.json` 较大，默认被 `.gitignore` 忽略，需要本地下载到 `LongMemEval/data/`。

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

## API Key

在项目根目录创建 `.env`：

```text
LLM_API_KEY=你的LLM服务Key
EMBEDDING_API_KEY=你的Embedding服务Key
```

统一配置在：

```text
configs/llm/responses.yaml
```

当前默认使用 DashScope 的 OpenAI-compatible Responses 接口。reader LLM 和 memory backend 的 extraction/organization LLM 第一阶段默认复用这份配置，后续可以在 adapter 层扩展为独立配置。

## 评测流程

单个 LongMemEval 样本的执行流程：

```text
backend.reset(question_id)
for session in haystack_sessions 按时间顺序:
    backend.ingest_session(session)

retrieved = backend.retrieve(question)
memory_context = backend.build_context(retrieved)
answer = reader_llm(question, memory_context)
write predictions.jsonl
```

最终 reader prompt 位于：

```text
agent_memory_eval/prompts.py
```

它只用于最终问答阶段，不用于 Mem0 / A-MEM / MemoryOS 的内部 memory prompt。

## 实验配置

完整 `longmemeval_s_cleaned` 主评测配置：

```text
configs/experiments/longmemeval_s_cleaned_no_memory.yaml
configs/experiments/longmemeval_s_cleaned_mem0.yaml
configs/experiments/longmemeval_s_cleaned_amem.yaml
configs/experiments/longmemeval_s_cleaned_memoryos.yaml
```

单样本完整 haystack smoke 配置：

```text
configs/experiments/longmemeval_s_cleaned_smoke_no_memory.yaml
configs/experiments/longmemeval_s_cleaned_smoke_mem0.yaml
configs/experiments/longmemeval_s_cleaned_smoke_amem.yaml
configs/experiments/longmemeval_s_cleaned_smoke_memoryos.yaml
```

Oracle smoke 配置：

```text
configs/experiments/longmemeval_oracle_no_memory.yaml
configs/experiments/longmemeval_oracle_mem0.yaml
configs/experiments/longmemeval_oracle_amem.yaml
configs/experiments/longmemeval_oracle_memoryos.yaml
```

## 运行示例

校验配置：

```powershell
python -m agent_memory_eval validate configs\experiments\longmemeval_s_cleaned_smoke_mem0.yaml
```

跑单样本完整 haystack smoke：

```powershell
python -m agent_memory_eval run configs\experiments\longmemeval_s_cleaned_smoke_mem0.yaml
```

跑完整主集：

```powershell
python -m agent_memory_eval run configs\experiments\longmemeval_s_cleaned_mem0.yaml
python -m agent_memory_eval run configs\experiments\longmemeval_s_cleaned_amem.yaml
python -m agent_memory_eval run configs\experiments\longmemeval_s_cleaned_memoryos.yaml
```

PowerShell helper：

```powershell
.\scripts\run_agent_memory_eval.ps1 -Config configs\experiments\longmemeval_s_cleaned_smoke_mem0.yaml
```

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
metrics.json               # 当前为评估占位文件
```

token 统计字段：

```text
build_input_tokens        # 送入 memory 构建阶段的原始文本 token
build_llm_prompt_tokens   # 可捕获的 memory 内部 LLM prompt token
build_tokens              # 构建总 token
query_input_tokens        # memory 检索 query token
query_llm_prompt_tokens   # 可捕获的检索阶段内部 LLM prompt token
reader_prompt_tokens      # 最终 reader prompt token
query_tokens              # 查询/回答总 token
total_tokens              # build + query
```

token 统计是框架侧的可比较口径，并 best-effort 捕获三套 memory 内部 LLM prompt。若某个库内部使用异步、子进程或绕过当前 LLM 方法，可能无法得到精确 billing usage。

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

## GitHub 首次上传

目标仓库：

```text
https://github.com/EvannZhongg/HyperMemo.git
```

在项目根目录执行：

```powershell
git init
git branch -M main
git add .
git commit -m "Initial HyperMemo agent memory evaluation framework"
git remote add origin https://github.com/EvannZhongg/HyperMemo.git
git push -u origin main
```

如果本地已经初始化过 git，只需要：

```powershell
git remote add origin https://github.com/EvannZhongg/HyperMemo.git
git branch -M main
git push -u origin main
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
docs/custom_memory_backend_guide.md
```
