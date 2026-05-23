from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runner import run_resolved_experiment, validate_config
from .suite import expand_suite_configs, load_suite_config


def validate_suite_config(
    config_path: str | Path,
    *,
    backend_filter: list[str] | None = None,
    limit: int | None = None,
    no_eval: bool = False,
) -> list[str]:
    suite_config = load_suite_config(config_path)
    errors: list[str] = []
    try:
        experiment_configs = expand_suite_configs(
            suite_config,
            backend_filter=backend_filter,
            limit=limit,
            no_eval=no_eval,
        )
    except ValueError as exc:
        return [str(exc)]

    for experiment_config in experiment_configs:
        backend = experiment_config["_meta"].get("backend_name")
        for error in validate_config(experiment_config):
            errors.append(f"[{backend}] {error}")
    return errors


def run_suite(
    config_path: str | Path,
    *,
    backend_filter: list[str] | None = None,
    limit: int | None = None,
    no_eval: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> list[Path]:
    suite_config = load_suite_config(config_path)
    experiment_configs = expand_suite_configs(
        suite_config,
        backend_filter=backend_filter,
        limit=limit,
        no_eval=no_eval,
    )

    run_dirs: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    for experiment_config in experiment_configs:
        backend = str(experiment_config["_meta"]["backend_name"])
        if progress:
            print(f"[suite] running backend={backend}", flush=True)
        run_dir = run_resolved_experiment(
            experiment_config,
            dry_run=dry_run,
            progress=progress,
        )
        run_dirs.append(run_dir)
        summary_rows.append(_summary_row(experiment_config, run_dir, dry_run=dry_run))

    if not dry_run and summary_rows:
        summary_path = _suite_summary_path(experiment_configs[0])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress:
            print(f"[suite] summary={summary_path}", flush=True)
    return run_dirs


def _summary_row(config: dict[str, Any], run_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "suite": config["_meta"].get("suite_name"),
        "backend": config["_meta"].get("backend_name"),
        "run_dir": str(run_dir),
        "dry_run": dry_run,
    }
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row["metrics_status"] = metrics.get("status")
            row["metric"] = metrics.get("metric")
            row["overall_accuracy"] = metrics.get("overall_accuracy")
            row["overall_f1"] = metrics.get("overall_f1")
            row["task_averaged_accuracy"] = metrics.get("task_averaged_accuracy")
            row["primary_score"] = (
                metrics.get("overall_accuracy")
                if metrics.get("overall_accuracy") is not None
                else metrics.get("overall_f1")
            )
            row["total_tokens"] = _total_tokens(run_dir)
        except json.JSONDecodeError:
            row["metrics_status"] = "invalid_json"
    return row


def _total_tokens(run_dir: Path) -> int | None:
    token_summary_path = run_dir / "token_usage_summary.json"
    if not token_summary_path.exists():
        return None
    try:
        token_summary = json.loads(token_summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return (token_summary.get("totals") or {}).get("total_tokens")


def _suite_summary_path(config: dict[str, Any]) -> Path:
    root = Path(config["_meta"]["root"])
    suite_name = config["_meta"]["suite_name"]
    return root / "runs" / f"{suite_name}_summary.json"
