from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter


def create_benchmark(config: dict[str, Any], root: Path) -> BenchmarkAdapter:
    name = str(config.get("name") or "longmemeval").lower()
    if name in {"longmemeval", "long_mem_eval", "lme"}:
        from .longmemeval import LongMemEvalBenchmark

        return LongMemEvalBenchmark(config, root)
    if name in {"locomo", "loco"}:
        from .locomo import LocomoBenchmark

        return LocomoBenchmark(config, root)
    raise ValueError(f"Unknown benchmark adapter: {name}")
