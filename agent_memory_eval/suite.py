from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .config import deep_merge, load_dotenv, load_yaml, resolve_path


def load_suite_config(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    root_path = Path(root or Path.cwd()).resolve()
    load_dotenv(root_path / ".env")

    config_path = resolve_path(path, root_path)
    config = load_yaml(config_path)
    if "suite" not in config:
        raise ValueError(f"Suite config must contain a top-level 'suite' key: {config_path}")
    config["_meta"] = {
        "root": str(root_path),
        "config_path": str(config_path),
    }
    return config


def expand_suite_configs(
    suite_config: dict[str, Any],
    *,
    backend_filter: list[str] | None = None,
    limit: int | None = None,
    no_eval: bool = False,
) -> list[dict[str, Any]]:
    root = Path(suite_config["_meta"]["root"])
    suite = suite_config["suite"]
    suite_name = suite.get("name")
    if not suite_name:
        raise ValueError("suite.name is required")

    backend_names = set(backend_filter or [])
    backends = suite.get("backends") or []
    if not isinstance(backends, list) or not backends:
        raise ValueError("suite.backends must be a non-empty list")

    configs: list[dict[str, Any]] = []
    for backend_entry in backends:
        if not isinstance(backend_entry, dict):
            raise ValueError("Each suite.backends entry must be a mapping")
        backend_name = backend_entry.get("name") or backend_entry.get("backend")
        if not backend_name:
            raise ValueError("Each suite.backends entry requires name")
        if backend_names and backend_name not in backend_names:
            continue
        configs.append(
            _build_experiment_config(
                suite_config,
                backend_entry,
                limit=limit,
                no_eval=no_eval,
            )
        )

    if backend_names and not configs:
        raise ValueError(f"No suite backend matched filter: {sorted(backend_names)}")
    return configs


def _build_experiment_config(
    suite_config: dict[str, Any],
    backend_entry: dict[str, Any],
    *,
    limit: int | None,
    no_eval: bool,
) -> dict[str, Any]:
    root = Path(suite_config["_meta"]["root"])
    suite = suite_config["suite"]
    suite_name = suite["name"]
    backend_name = backend_entry.get("name") or backend_entry.get("backend")

    benchmark = copy.deepcopy(suite.get("benchmark") or {})
    if no_eval:
        benchmark.setdefault("evaluation", {})["enabled"] = False

    run_config = suite.get("run") or {}
    run_dir_template = run_config.get("run_dir_template", "runs/{suite}_{backend}")
    run_dir = run_dir_template.format(
        suite=suite_name,
        backend=backend_name,
        benchmark=benchmark.get("name", "benchmark"),
    )

    agent = copy.deepcopy(suite.get("agent") or {})
    llm_ref = agent.pop("llm_config_path", None)
    if llm_ref:
        llm_config = load_yaml(resolve_path(llm_ref, root))
        agent["llm"] = deep_merge(llm_config, agent.get("llm", {}))

    memory = _load_memory_config(backend_entry, root)
    memory.setdefault("backend", backend_name)
    agent["memory"] = memory

    experiment_limit = limit if limit is not None else run_config.get("limit") or suite.get("limit")
    experiment: dict[str, Any] = {
        "name": f"{suite_name}_{backend_name}",
        "benchmark": benchmark,
        "run_dir": run_dir,
    }
    if experiment_limit is not None:
        experiment["limit"] = experiment_limit
    if "progress" in run_config:
        experiment["progress"] = run_config["progress"]

    return {
        "experiment": experiment,
        "agent": agent,
        "_meta": {
            "root": str(root),
            "config_path": suite_config["_meta"]["config_path"],
            "suite_name": suite_name,
            "backend_name": backend_name,
        },
    }


def _load_memory_config(backend_entry: dict[str, Any], root: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    config_path = backend_entry.get("config_path")
    if config_path:
        config = load_yaml(resolve_path(config_path, root))

    inline = {
        key: copy.deepcopy(value)
        for key, value in backend_entry.items()
        if key not in {"name", "backend", "config_path"}
    }
    return deep_merge(config, inline)
