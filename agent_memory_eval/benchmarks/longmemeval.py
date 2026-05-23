from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .base import BenchmarkAdapter
from ..config import env_value, resolve_path
from ..models import BenchmarkSample, MemorySession, MemoryTurn


LONGMEMEVAL_QA_TYPES = [
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]
SUPPORTED_METRIC_MODELS = {"gpt-4o", "gpt-4o-mini", "llama-3.1-70b-instruct"}


class LongMemEvalBenchmark(BenchmarkAdapter):
    benchmark_name = "longmemeval"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.dataset_path.exists():
            errors.append(f"Dataset not found: {self.dataset_path}")
        evaluation = self.config.get("evaluation", {})
        if evaluation.get("enabled", True):
            metric_model = str(evaluation.get("metric_model", "gpt-4o"))
            if metric_model not in SUPPORTED_METRIC_MODELS:
                errors.append(
                    "LongMemEval evaluate_qa.py supports metric_model values: "
                    f"{sorted(SUPPORTED_METRIC_MODELS)}; got {metric_model!r}"
                )
            base_url = evaluation.get("base_url")
            if base_url is not None and not str(base_url).strip():
                errors.append("LongMemEval evaluation.base_url cannot be empty")
            evaluator_dir = self.evaluator_dir
            if not (evaluator_dir / "evaluate_qa.py").exists():
                errors.append(f"LongMemEval QA evaluator not found: {evaluator_dir / 'evaluate_qa.py'}")
            if not (evaluator_dir / "print_qa_metrics.py").exists():
                errors.append(f"LongMemEval metrics printer not found: {evaluator_dir / 'print_qa_metrics.py'}")
        return errors

    def load_samples(self, limit: int | None = None) -> list[BenchmarkSample]:
        return load_dataset(self.dataset_path, limit=limit)

    def prediction_record(self, sample: BenchmarkSample, answer: str) -> dict[str, Any]:
        return {"question_id": sample.question_id, "hypothesis": answer}

    def evaluate(
        self,
        *,
        predictions_path: Path,
        run_dir: Path,
        progress: bool = True,
    ) -> dict[str, Any]:
        evaluation = self.config.get("evaluation", {})
        if not evaluation.get("enabled", True):
            return {
                "status": "skipped",
                "benchmark": self.benchmark_name,
                "message": "Evaluation disabled by benchmark.evaluation.enabled=false.",
                "predictions_path": str(predictions_path),
                "dataset_path": str(self.dataset_path),
            }

        metric_model = str(evaluation.get("metric_model", "gpt-4o"))
        env = os.environ.copy()
        api_key_env = evaluation.get("api_key_env")
        if api_key_env:
            api_key = env_value(str(api_key_env))
            if api_key:
                env["OPENAI_API_KEY"] = api_key
        base_url = evaluation.get("base_url")
        if base_url:
            env["OPENAI_BASE_URL"] = str(base_url)
            env["OPENAI_API_BASE"] = str(base_url)
        else:
            base_url_env = evaluation.get("base_url_env")
            if base_url_env:
                resolved_base_url = env_value(str(base_url_env))
                if resolved_base_url:
                    env["OPENAI_BASE_URL"] = resolved_base_url
                    env["OPENAI_API_BASE"] = resolved_base_url
        organization_env = evaluation.get("organization_env")
        if organization_env:
            organization = env_value(str(organization_env))
            if organization:
                env["OPENAI_ORGANIZATION"] = organization

        evaluator_dir = self.evaluator_dir
        eval_cmd = [
            sys.executable,
            "evaluate_qa.py",
            metric_model,
            str(predictions_path),
            str(self.dataset_path),
        ]
        if progress:
            print(f"[eval] running LongMemEval QA evaluator metric_model={metric_model}", flush=True)
        eval_result = subprocess.run(
            eval_cmd,
            cwd=evaluator_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        (run_dir / "evaluation.stdout.txt").write_text(eval_result.stdout, encoding="utf-8")
        (run_dir / "evaluation.stderr.txt").write_text(eval_result.stderr, encoding="utf-8")

        evaluated_predictions_path = Path(f"{predictions_path}.eval-results-{metric_model}")
        if eval_result.returncode != 0:
            return {
                "status": "failed",
                "benchmark": self.benchmark_name,
                "stage": "evaluate_qa",
                "returncode": eval_result.returncode,
                "metric_model": metric_model,
                "predictions_path": str(predictions_path),
                "dataset_path": str(self.dataset_path),
                "stdout_path": str(run_dir / "evaluation.stdout.txt"),
                "stderr_path": str(run_dir / "evaluation.stderr.txt"),
                "stderr_tail": eval_result.stderr[-2000:],
            }

        metrics = summarize_longmemeval_qa(evaluated_predictions_path, self.dataset_path, metric_model)
        metrics.update(
            {
                "status": "evaluated",
                "benchmark": self.benchmark_name,
                "metric_model": metric_model,
                "predictions_path": str(predictions_path),
                "evaluated_predictions_path": str(evaluated_predictions_path),
                "dataset_path": str(self.dataset_path),
                "stdout_path": str(run_dir / "evaluation.stdout.txt"),
                "stderr_path": str(run_dir / "evaluation.stderr.txt"),
            }
        )

        printer_cmd = [
            sys.executable,
            "print_qa_metrics.py",
            str(evaluated_predictions_path),
            str(self.dataset_path),
        ]
        printer_result = subprocess.run(
            printer_cmd,
            cwd=evaluator_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        (run_dir / "qa_metrics.txt").write_text(printer_result.stdout, encoding="utf-8")
        if printer_result.returncode != 0:
            metrics["printer_status"] = "failed"
            metrics["printer_stderr"] = printer_result.stderr[-2000:]
        else:
            metrics["printer_status"] = "ok"
            metrics["qa_metrics_path"] = str(run_dir / "qa_metrics.txt")
        return metrics

    @property
    def dataset_path(self) -> Path:
        return resolve_path(self.config["dataset_path"], self.root)

    @property
    def evaluator_dir(self) -> Path:
        evaluator_dir = self.config.get("evaluator_dir", "LongMemEval/src/evaluation")
        return resolve_path(evaluator_dir, self.root)


def load_dataset(path: str | Path, limit: int | None = None) -> list[BenchmarkSample]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    samples: list[BenchmarkSample] = []
    for raw in raw_samples[:limit]:
        session_ids = raw.get("haystack_session_ids", [])
        dates = raw.get("haystack_dates", [])
        sessions = raw.get("haystack_sessions", [])

        memory_sessions: list[MemorySession] = []
        for idx, turns in enumerate(sessions):
            session_id = str(session_ids[idx]) if idx < len(session_ids) else f"session_{idx}"
            date = str(dates[idx]) if idx < len(dates) else None
            memory_turns = [_turn_from_raw(turn, timestamp=date, session_id=session_id) for turn in turns]
            memory_sessions.append(
                MemorySession(
                    session_id=session_id,
                    date=date,
                    turns=memory_turns,
                    metadata={
                        "question_id": raw.get("question_id"),
                        "question_date": raw.get("question_date"),
                        "session_index": idx,
                    },
                )
            )
        memory_sessions.sort(key=lambda session: _date_sort_key(session.date))

        samples.append(
            BenchmarkSample(
                question_id=str(raw.get("question_id")),
                question_type=str(raw.get("question_type", "")),
                question=str(raw.get("question", "")),
                answer=raw.get("answer"),
                question_date=raw.get("question_date"),
                sessions=memory_sessions,
                raw=raw,
                benchmark="longmemeval",
            )
        )
    return samples


def iter_samples(path: str | Path, limit: int | None = None) -> Iterable[BenchmarkSample]:
    yield from load_dataset(path, limit=limit)


def summarize_longmemeval_qa(
    evaluated_predictions_path: str | Path,
    dataset_path: str | Path,
    metric_model: str | None = None,
) -> dict[str, Any]:
    evaluated_rows = [json.loads(line) for line in Path(evaluated_predictions_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    references = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    ref_by_id = {entry["question_id"]: entry for entry in references}

    type2acc: dict[str, list[int]] = {task_type: [] for task_type in LONGMEMEVAL_QA_TYPES}
    all_acc: list[int] = []
    abstention_acc: list[int] = []
    skipped = 0
    model_name = None

    for row in evaluated_rows:
        question_id = row.get("question_id")
        ref = ref_by_id.get(question_id)
        if not ref:
            skipped += 1
            continue
        label_info = row.get("autoeval_label", {})
        if model_name is None:
            model_name = label_info.get("model")
        value = 1 if label_info.get("label") else 0
        question_type = ref.get("question_type")
        type2acc.setdefault(question_type, []).append(value)
        all_acc.append(value)
        if str(question_id).endswith("_abs"):
            abstention_acc.append(value)

    per_task = {
        task_type: {
            "accuracy": _mean(values),
            "count": len(values),
        }
        for task_type, values in type2acc.items()
    }
    task_values = [values for values in type2acc.values() if values]
    return {
        "metric_model": metric_model,
        "autoeval_model": model_name,
        "sample_count": len(evaluated_rows),
        "matched_count": len(all_acc),
        "skipped_count": skipped,
        "overall_accuracy": _mean(all_acc),
        "task_averaged_accuracy": _mean([_mean(values) for values in task_values]),
        "abstention_accuracy": _mean(abstention_acc),
        "abstention_count": len(abstention_acc),
        "per_task": per_task,
    }


def _turn_from_raw(raw: dict, timestamp: str | None, session_id: str) -> MemoryTurn:
    return MemoryTurn(
        role=str(raw.get("role", "")),
        content=str(raw.get("content", "")),
        timestamp=timestamp,
        metadata={k: v for k, v in raw.items() if k not in {"role", "content"}}
        | {"source_session_id": session_id},
    )


def _date_sort_key(value: str | None) -> tuple[int, str]:
    if not value:
        return (1, "")
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return (0, datetime.strptime(value, fmt).isoformat())
        except ValueError:
            continue
    return (0, value)


def _mean(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
