from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .agent import AgentRuntime
from .backends import create_backend
from .config import load_experiment_config, resolve_path
from .llm import OpenAIResponsesClient
from .longmemeval import load_dataset


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    root = Path(config["_meta"]["root"])
    experiment = config.get("experiment", {})
    dataset_path = experiment.get("dataset_path")
    if not dataset_path:
        errors.append("experiment.dataset_path is required")
    elif not resolve_path(dataset_path, root).exists():
        errors.append(f"Dataset not found: {resolve_path(dataset_path, root)}")

    agent = config.get("agent", {})
    llm = agent.get("llm", {})
    if llm.get("provider", "openai") != "openai":
        errors.append("Only OpenAI Responses-compatible provider is supported in phase 1")
    if not llm.get("model"):
        errors.append("agent.llm.model is required")
    if not agent.get("memory", {}).get("backend"):
        errors.append("agent.memory.backend is required")
    return errors


def run_experiment(config_path: str | Path, *, limit: int | None = None, dry_run: bool = False) -> Path:
    config = load_experiment_config(config_path)
    errors = validate_config(config)
    if errors:
        raise ValueError("\n".join(errors))
    if dry_run:
        return Path(config["_meta"]["config_path"])

    root = Path(config["_meta"]["root"])
    experiment = config["experiment"]
    run_id = experiment.get("run_id") or _default_run_id(experiment.get("name", "experiment"))
    run_dir = resolve_path(experiment.get("run_dir", f"runs/{run_id}"), root)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_path(experiment["dataset_path"], root)
    samples = load_dataset(dataset_path, limit=limit or experiment.get("limit"))

    resolved_config_path = run_dir / "config.resolved.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    llm_config = config["agent"]["llm"]
    llm = OpenAIResponsesClient.from_dict(llm_config)
    memory_config = dict(config["agent"]["memory"])
    memory_config.setdefault("root", str(root))
    backend = create_backend(memory_config, llm_config)
    backend.set_tokenizer_model(config["agent"].get("tokenizer_model") or llm_config.get("model"))
    top_k = config["agent"].get("top_k")
    agent = AgentRuntime(
        backend=backend,
        llm=llm,
        top_k=top_k,
        instructions=config["agent"].get("instructions") or AgentRuntime.__dataclass_fields__["instructions"].default,
    )

    predictions_path = run_dir / "predictions.jsonl"
    retrieved_path = run_dir / "retrieved_memories.jsonl"
    ingest_trace_path = run_dir / "ingest_trace.jsonl"
    backend_debug_path = run_dir / "backend_debug.jsonl"
    token_usage_path = run_dir / "token_usage.jsonl"
    token_summaries: list[dict[str, Any]] = []

    with predictions_path.open("w", encoding="utf-8") as pred_f, retrieved_path.open(
        "w", encoding="utf-8"
    ) as ret_f, ingest_trace_path.open("w", encoding="utf-8") as trace_f, backend_debug_path.open(
        "w", encoding="utf-8"
    ) as debug_f, token_usage_path.open("w", encoding="utf-8") as token_f:
        for sample in samples:
            backend.reset(sample.question_id)
            for session in sample.sessions:
                backend.ingest_session(session)
                _write_jsonl(
                    trace_f,
                    {
                        "question_id": sample.question_id,
                        "session_id": session.session_id,
                        "date": session.date,
                        "turn_count": len(session.turns),
                        "backend": backend.backend_name,
                    },
                )

            answer, retrieved, memory_context = agent.answer(sample.question)
            _write_jsonl(pred_f, {"question_id": sample.question_id, "hypothesis": answer})
            _write_jsonl(
                ret_f,
                {
                    "question_id": sample.question_id,
                    "backend": backend.backend_name,
                    "query": sample.question,
                    "top_k": top_k if top_k is not None else backend.default_top_k,
                    "memory_context": memory_context,
                    "retrieved": [
                        {
                            "id": item.id,
                            "content": item.content,
                            "score": item.score,
                            "source_session_id": item.source_session_id,
                            "metadata": item.metadata,
                        }
                        for item in retrieved
                    ],
                },
            )
            _write_jsonl(
                debug_f,
                {
                    "question_id": sample.question_id,
                    "backend": backend.backend_name,
                    "debug": backend.get_debug_info(),
                },
            )
            token_usage = {
                "question_id": sample.question_id,
                "backend": backend.backend_name,
                **backend.get_token_usage(),
            }
            token_summaries.append(token_usage)
            _write_jsonl(token_f, token_usage)
            backend.close()

    token_usage_summary = _summarize_token_usage(token_summaries)
    (run_dir / "token_usage_summary.json").write_text(
        json.dumps(token_usage_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics_stub = {
        "status": "not_evaluated",
        "message": "Evaluation interface is reserved. Run LongMemEval evaluator later.",
        "predictions_path": str(predictions_path),
        "dataset_path": str(dataset_path),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_stub, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def _summarize_token_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    fields = [
        "build_input_tokens",
        "build_llm_prompt_tokens",
        "build_tokens",
        "query_input_tokens",
        "query_llm_prompt_tokens",
        "memory_query_tokens",
        "retrieved_context_tokens",
        "reader_prompt_tokens",
        "query_tokens",
        "total_tokens",
    ]
    summary: dict[str, Any] = {
        "sample_count": len(rows),
        "backend": rows[0].get("backend"),
        "tokenizer_model": rows[0].get("tokenizer_model"),
        "token_counter": rows[0].get("token_counter"),
        "totals": {},
        "averages": {},
    }
    for field in fields:
        total = sum(int(row.get(field) or 0) for row in rows)
        summary["totals"][field] = total
        summary["averages"][field] = total / len(rows)
    return summary


def _write_jsonl(file_obj, payload: dict[str, Any]) -> None:
    file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")
    file_obj.flush()


def _default_run_id(name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return f"{safe_name}_{stamp}"
