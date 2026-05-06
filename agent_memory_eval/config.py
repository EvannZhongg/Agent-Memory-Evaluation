from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise ConfigError(f"Config file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {yaml_path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_experiment_config(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    root_path = Path(root or Path.cwd()).resolve()
    load_dotenv(root_path / ".env")

    config_path = resolve_path(path, root_path)
    config = load_yaml(config_path)

    agent = config.setdefault("agent", {})
    llm_ref = agent.get("llm_config_path")
    if llm_ref:
        llm_config = load_yaml(resolve_path(llm_ref, root_path))
        agent["llm"] = deep_merge(llm_config, agent.get("llm", {}))

    memory = agent.setdefault("memory", {})
    memory_ref = memory.get("config_path")
    if memory_ref:
        memory_config = load_yaml(resolve_path(memory_ref, root_path))
        agent["memory"] = deep_merge(memory_config, memory)

    config["_meta"] = {
        "root": str(root_path),
        "config_path": str(config_path),
    }
    return config


def env_value(name: str | None, default: str | None = None) -> str | None:
    if not name:
        return default
    return os.environ.get(name, default)

