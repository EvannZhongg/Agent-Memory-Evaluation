from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import MemoryItem, MemorySession
from ..token_usage import TokenUsageTracker


class MemoryBackend(ABC):
    backend_name = "base"
    default_top_k = 10

    def __init__(self) -> None:
        self.token_usage = TokenUsageTracker()

    @abstractmethod
    def reset(self, sample_id: str) -> None:
        self.token_usage = TokenUsageTracker(self.token_usage.tokenizer_model)

    @abstractmethod
    def ingest_session(self, session: MemorySession) -> None:
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        pass

    def build_context(self, query: str, retrieved: list[MemoryItem]) -> str:
        if not retrieved:
            return "No relevant memories retrieved."
        lines = []
        for idx, item in enumerate(retrieved, start=1):
            source = f" source={item.source_session_id}" if item.source_session_id else ""
            score = f" score={item.score:.4f}" if item.score is not None else ""
            lines.append(f"[{idx}{source}{score}] {item.content}")
        return "\n".join(lines)

    def get_debug_info(self) -> dict[str, Any]:
        return {}

    def get_token_usage(self) -> dict[str, Any]:
        return self.token_usage.to_dict()

    def set_tokenizer_model(self, model: str | None) -> None:
        self.token_usage.tokenizer_model = model

    def close(self) -> None:
        pass


def session_to_text(session: MemorySession) -> str:
    header = f"Session {session.session_id}"
    if session.date:
        header += f" at {session.date}"
    body = "\n".join(f"{turn.role}: {turn.content}" for turn in session.turns)
    return f"{header}\n{body}"
