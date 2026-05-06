from __future__ import annotations

from typing import Any

from .base import MemoryBackend
from ..models import MemoryItem, MemorySession


class NoMemoryBackend(MemoryBackend):
    backend_name = "none"
    default_top_k = 0

    def __init__(self) -> None:
        super().__init__()

    def reset(self, sample_id: str) -> None:
        super().reset(sample_id)
        self.sample_id = sample_id

    def ingest_session(self, session: MemorySession) -> None:
        return None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        return []
