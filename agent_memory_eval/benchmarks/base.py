from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import BenchmarkSample


class BenchmarkAdapter(ABC):
    benchmark_name = "base"

    def __init__(self, config: dict[str, Any], root: Path):
        self.config = config
        self.root = root

    @abstractmethod
    def validate(self) -> list[str]:
        pass

    @abstractmethod
    def load_samples(self, limit: int | None = None) -> list[BenchmarkSample]:
        pass

    @abstractmethod
    def prediction_record(self, sample: BenchmarkSample, answer: str) -> dict[str, Any]:
        pass

    def evaluate(
        self,
        *,
        predictions_path: Path,
        run_dir: Path,
        progress: bool = True,
    ) -> dict[str, Any]:
        return {
            "status": "not_evaluated",
            "benchmark": self.benchmark_name,
            "message": f"Benchmark '{self.benchmark_name}' does not provide an evaluator.",
            "predictions_path": str(predictions_path),
        }
