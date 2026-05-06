from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import MemoryBackend, session_to_text
from ..llm_token_hooks import record_method_prompt_tokens
from ..models import MemoryItem, MemorySession
from ..pathing import add_repo_path


class AMemBackend(MemoryBackend):
    backend_name = "amem"
    default_top_k = 5

    def __init__(self, config: dict[str, Any], llm_config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.llm_config = llm_config
        self.root = Path(config.get("root", ".")).resolve()
        self.repo_path = add_repo_path(self.root, config.get("repo_path", "A-mem"))
        self.memory = None
        self.sample_id: str | None = None

    def reset(self, sample_id: str) -> None:
        from agentic_memory.memory_system import AgenticMemorySystem

        super().reset(sample_id)
        self.sample_id = sample_id
        self.memory = AgenticMemorySystem(
            model_name=self.config.get("embedding_model", "all-MiniLM-L6-v2"),
            llm_backend=self.config.get("llm_backend", "openai"),
            llm_model=self.config.get("llm_model") or self.llm_config.get("model", "gpt-4o-mini"),
            evo_threshold=int(self.config.get("evo_threshold", 100)),
            api_key=None,
        )

    def ingest_session(self, session: MemorySession) -> None:
        assert self.memory is not None
        note_text = session_to_text(session)
        self.token_usage.record_build(
            note_text,
            event="amem.add_note.session",
            metadata={
                "session_id": session.session_id,
                "date": session.date,
                "turn_count": len(session.turns),
            },
        )
        llm = getattr(getattr(self.memory, "llm_controller", None), "llm", None)
        with record_method_prompt_tokens(
            llm,
            "get_completion",
            self.token_usage,
            phase="build",
            event="amem.internal_llm_prompt",
        ):
            self.memory.add_note(
                note_text,
                time=session.date,
                category=self.config.get("category", "LongMemEvalSession"),
                tags=["longmemeval", session.session_id],
            )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        assert self.memory is not None
        k = top_k if top_k is not None else self.default_top_k
        self.token_usage.record_memory_query(
            query,
            event="amem.search_agentic.query",
            metadata={"top_k": k},
        )
        llm = getattr(getattr(self.memory, "llm_controller", None), "llm", None)
        with record_method_prompt_tokens(
            llm,
            "get_completion",
            self.token_usage,
            phase="query",
            event="amem.internal_llm_prompt",
        ):
            results = self.memory.search_agentic(query, k=k)
        items: list[MemoryItem] = []
        for idx, raw in enumerate(results or []):
            items.append(
                MemoryItem(
                    id=str(raw.get("id", f"amem_{idx}")),
                    content=str(raw.get("content") or raw.get("memory") or ""),
                    score=_float_or_none(raw.get("score") or raw.get("distance")),
                    source_session_id=raw.get("session_id"),
                    metadata=raw,
                )
            )
        return items

    def get_debug_info(self) -> dict[str, Any]:
        if not self.memory:
            return {}
        return {"memory_count": len(getattr(self.memory, "memories", {}))}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
