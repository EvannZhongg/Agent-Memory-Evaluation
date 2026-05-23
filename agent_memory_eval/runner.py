from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .agent import AgentRuntime
from .backends import create_backend
from .benchmarks import create_benchmark
from .config import load_experiment_config, resolve_path
from .llm import OpenAIResponsesClient


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    root = Path(config["_meta"]["root"])
    experiment = config.get("experiment", {})
    try:
        benchmark_config = _benchmark_config(experiment)
    except ValueError as exc:
        benchmark_config = {}
        errors.append(str(exc))
    if not benchmark_config.get("dataset_path"):
        errors.append("experiment.benchmark.dataset_path is required")
    else:
        try:
            benchmark = create_benchmark(benchmark_config, root)
            errors.extend(benchmark.validate())
        except ValueError as exc:
            errors.append(str(exc))

    agent = config.get("agent", {})
    llm = agent.get("llm", {})
    if llm.get("provider", "openai") != "openai":
        errors.append("Only OpenAI Responses-compatible provider is supported in phase 1")
    if not llm.get("model"):
        errors.append("agent.llm.model is required")
    if not agent.get("memory", {}).get("backend"):
        errors.append("agent.memory.backend is required")
    return errors


def run_experiment(
    config_path: str | Path,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> Path:
    config = load_experiment_config(config_path)
    return run_resolved_experiment(config, limit=limit, dry_run=dry_run, progress=progress)


def run_resolved_experiment(
    config: dict[str, Any],
    *,
    limit: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> Path:
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
    progress = progress and bool(experiment.get("progress", True))

    benchmark = create_benchmark(_benchmark_config(experiment), root)
    samples = benchmark.load_samples(limit=limit or experiment.get("limit"))
    if progress:
        print(
            f"[run] benchmark={benchmark.benchmark_name} "
            f"backend={config['agent']['memory'].get('backend')} "
            f"samples={len(samples)} dataset={benchmark.config.get('dataset_path')}",
            flush=True,
        )

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
        for sample_index, sample in enumerate(samples, start=1):
            if progress:
                print(
                    f"[sample {sample_index}/{len(samples)}] reset question_id={sample.question_id} "
                    f"sessions={len(sample.sessions)}",
                    flush=True,
                )
            backend.reset(sample.question_id)
            for session_index, session in enumerate(sample.sessions, start=1):
                if progress:
                    print(
                        f"[sample {sample_index}/{len(samples)}] ingest "
                        f"{session_index}/{len(sample.sessions)} "
                        f"session_id={session.session_id} turns={len(session.turns)}",
                        flush=True,
                    )
                _write_jsonl(
                    trace_f,
                    {
                        "event": "ingest_started",
                        "question_id": sample.question_id,
                        "session_id": session.session_id,
                        "date": session.date,
                        "turn_count": len(session.turns),
                        "backend": backend.backend_name,
                    },
                )
                backend.ingest_session(session)
                _write_jsonl(
                    trace_f,
                    {
                        "event": "ingest_completed",
                        "question_id": sample.question_id,
                        "session_id": session.session_id,
                        "date": session.date,
                        "turn_count": len(session.turns),
                        "backend": backend.backend_name,
                    },
                )

            if progress:
                print(f"[sample {sample_index}/{len(samples)}] retrieve+answer", flush=True)
            answer, retrieved, memory_context = agent.answer(sample.question)
            _write_jsonl(pred_f, benchmark.prediction_record(sample, answer))
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
            if progress:
                print(
                    f"[sample {sample_index}/{len(samples)}] completed "
                    f"retrieved={len(retrieved)} build_tokens={token_usage.get('build_tokens')} "
                    f"query_tokens={token_usage.get('query_tokens')}",
                    flush=True,
                )

    token_usage_summary = _summarize_token_usage(token_summaries)
    (run_dir / "token_usage_summary.json").write_text(
        json.dumps(token_usage_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics_stub = benchmark.evaluate(
        predictions_path=predictions_path,
        run_dir=run_dir,
        progress=progress,
    )
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


def _benchmark_config(experiment: dict[str, Any]) -> dict[str, Any]:
    benchmark = dict(experiment.get("benchmark") or {})
    if "name" not in benchmark:
        raise ValueError("experiment.benchmark.name is required")
    return benchmark
