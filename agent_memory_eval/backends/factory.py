from __future__ import annotations

from typing import Any

from .base import MemoryBackend
from .no_memory import NoMemoryBackend


def create_backend(config: dict[str, Any], llm_config: dict[str, Any]) -> MemoryBackend:
    backend = config.get("backend", "none")
    if backend in {"none", "no_memory"}:
        return NoMemoryBackend()
    if backend == "mem0":
        from .mem0_backend import Mem0Backend

        return Mem0Backend(config, llm_config)
    if backend in {"amem", "a-mem"}:
        from .amem_backend import AMemBackend

        return AMemBackend(config, llm_config)
    if backend == "memoryos":
        from .memoryos_backend import MemoryOSBackend

        return MemoryOSBackend(config, llm_config)
    raise ValueError(f"Unknown memory backend: {backend}")

