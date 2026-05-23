from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .base import MemoryBackend, session_to_text
from ..llm_token_hooks import record_method_prompt_tokens
from ..models import MemoryItem, MemorySession, MemoryTurn
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
        self.ingest_granularity = str(config.get("ingest_granularity", "turn")).lower()
        if self.ingest_granularity not in {"turn", "pair", "session"}:
            raise ValueError(
                "A-MEM ingest_granularity must be one of: turn, pair, session "
                f"(got {self.ingest_granularity!r})"
            )
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
        llm = getattr(getattr(self.memory, "llm_controller", None), "llm", None)
        for chunk in _iter_note_chunks(session, self.ingest_granularity):
            self.token_usage.record_build(
                chunk["content"],
                event=f"amem.add_note.{self.ingest_granularity}",
                metadata={
                    "session_id": session.session_id,
                    "date": session.date,
                    "turn_count": len(session.turns),
                    **chunk["metadata"],
                },
            )
            with record_method_prompt_tokens(
                llm,
                "get_completion",
                self.token_usage,
                phase="build",
                event="amem.internal_llm_prompt",
            ):
                self.memory.add_note(
                    chunk["content"],
                    time=chunk.get("timestamp") or session.date,
                    category=self.config.get("category", "A-MemNote"),
                    tags=["memory_eval", self.ingest_granularity, session.session_id],
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
        return {
            "memory_count": len(getattr(self.memory, "memories", {})),
            "ingest_granularity": self.ingest_granularity,
        }


def _iter_note_chunks(session: MemorySession, granularity: str) -> Iterable[dict[str, Any]]:
    if granularity == "session":
        yield {
            "content": session_to_text(session),
            "timestamp": session.date,
            "metadata": {"chunk_type": "session", "chunk_index": 0},
        }
        return

    if granularity == "turn":
        for index, turn in enumerate(session.turns):
            yield {
                "content": _turn_to_text(turn),
                "timestamp": turn.timestamp or session.date,
                "metadata": {
                    "chunk_type": "turn",
                    "chunk_index": index,
                    "role": turn.role,
                    "timestamp": turn.timestamp,
                },
            }
        return

    for index, (user_turn, assistant_turn) in enumerate(_session_pairs(session)):
        user_text = user_turn.content if user_turn else ""
        assistant_text = assistant_turn.content if assistant_turn else ""
        yield {
            "content": f"user: {user_text}\nassistant: {assistant_text}",
            "timestamp": (user_turn.timestamp if user_turn else None)
            or (assistant_turn.timestamp if assistant_turn else None)
            or session.date,
            "metadata": {
                "chunk_type": "pair",
                "chunk_index": index,
                "has_user_input": bool(user_text),
                "has_assistant_response": bool(assistant_text),
            },
        }


def _turn_to_text(turn: MemoryTurn) -> str:
    speaker = turn.metadata.get("speaker") or turn.role
    return f"Speaker {speaker} says : {turn.content}"


def _session_pairs(session: MemorySession) -> list[tuple[MemoryTurn | None, MemoryTurn | None]]:
    pairs: list[tuple[MemoryTurn | None, MemoryTurn | None]] = []
    pending_user: MemoryTurn | None = None
    for turn in session.turns:
        if turn.role == "user":
            if pending_user is not None:
                pairs.append((pending_user, None))
            pending_user = turn
        elif turn.role == "assistant":
            if pending_user is None:
                pairs.append((None, turn))
            else:
                pairs.append((pending_user, turn))
                pending_user = None
    if pending_user is not None:
        pairs.append((pending_user, None))
    return pairs


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
