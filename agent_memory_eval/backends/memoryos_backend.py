from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import MemoryBackend
from ..config import env_value
from ..llm_token_hooks import record_method_prompt_tokens
from ..models import MemoryItem, MemorySession
from ..pathing import add_repo_path


class MemoryOSBackend(MemoryBackend):
    backend_name = "memoryos"
    default_top_k = 7

    def __init__(self, config: dict[str, Any], llm_config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.llm_config = llm_config
        self.root = Path(config.get("root", ".")).resolve()
        self.repo_path = add_repo_path(self.root, config.get("repo_path", "MemoryOS/memoryos-pypi"))
        self.memory = None
        self.sample_id: str | None = None
        self._last_retrieval: list[MemoryItem] = []

    def reset(self, sample_id: str) -> None:
        from memoryos import Memoryos

        super().reset(sample_id)
        self.sample_id = sample_id
        storage_path = Path(self.config.get("storage_path", "runs/vectorstores/memoryos")).resolve()
        self.memory = Memoryos(
            user_id=self._user_id(),
            openai_api_key=self.config.get("api_key")
            or env_value(self.config.get("api_key_env"))
            or env_value(self.llm_config.get("api_key_env", "LLM_API_KEY")),
            openai_base_url=self.config.get("base_url")
            or env_value(self.config.get("base_url_env"))
            or self.llm_config.get("base_url")
            or env_value(self.llm_config.get("base_url_env", "LLM_BASE_URL")),
            data_storage_path=str(storage_path),
            assistant_id=self.config.get("assistant_id", "longmemeval_assistant"),
            short_term_capacity=int(self.config.get("short_term_capacity", 10)),
            mid_term_capacity=int(self.config.get("mid_term_capacity", 2000)),
            long_term_knowledge_capacity=int(self.config.get("long_term_knowledge_capacity", 100)),
            retrieval_queue_capacity=int(self.config.get("retrieval_queue_capacity", self.default_top_k)),
            mid_term_heat_threshold=float(self.config.get("mid_term_heat_threshold", 5.0)),
            mid_term_similarity_threshold=float(self.config.get("mid_term_similarity_threshold", 0.6)),
            llm_model=self.config.get("llm_model") or self.llm_config.get("model", "gpt-4o-mini"),
            embedding_model_name=self.config.get("embedding_model", "all-MiniLM-L6-v2"),
            embedding_model_kwargs=self.config.get("embedding_model_kwargs"),
        )

    def ingest_session(self, session: MemorySession) -> None:
        assert self.memory is not None
        pairs = _session_pairs(session)
        for user_input, assistant_response in pairs:
            pair_text = f"Session {session.session_id}"
            if session.date:
                pair_text += f" at {session.date}"
            pair_text += f"\nuser: {user_input}\nassistant: {assistant_response}"
            self.token_usage.record_build(
                pair_text,
                event="memoryos.add_memory.pair",
                metadata={
                    "session_id": session.session_id,
                    "date": session.date,
                    "has_user_input": bool(user_input),
                    "has_assistant_response": bool(assistant_response),
                },
            )
            with record_method_prompt_tokens(
                getattr(self.memory, "client", None),
                "chat_completion",
                self.token_usage,
                phase="build",
                event="memoryos.internal_llm_prompt",
            ):
                self.memory.add_memory(
                    user_input=user_input,
                    agent_response=assistant_response,
                    timestamp=session.date,
                    meta_data={"session_id": session.session_id, "question_id": self.sample_id},
                )
        if bool(self.config.get("force_mid_term_analysis_after_session", False)):
            with record_method_prompt_tokens(
                getattr(self.memory, "client", None),
                "chat_completion",
                self.token_usage,
                phase="build",
                event="memoryos.internal_llm_prompt",
            ):
                self.memory.force_mid_term_analysis()

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
            event="memoryos.retrieve_context.query",
            metadata={"top_k": k},
        )
        if bool(self.config.get("force_mid_term_analysis_before_retrieve", True)):
            with record_method_prompt_tokens(
                getattr(self.memory, "client", None),
                "chat_completion",
                self.token_usage,
                phase="build",
                event="memoryos.internal_llm_prompt",
            ):
                self.memory.force_mid_term_analysis()

        with record_method_prompt_tokens(
            getattr(self.memory, "client", None),
            "chat_completion",
            self.token_usage,
            phase="query",
            event="memoryos.internal_llm_prompt",
        ):
            result = self.memory.retriever.retrieve_context(user_query=query, user_id=self._user_id())
        items: list[MemoryItem] = []
        for idx, page in enumerate(result.get("retrieved_pages", []) or []):
            content = (
                f"User: {page.get('user_input', '')}\n"
                f"Assistant: {page.get('agent_response', '')}\n"
                f"Time: {page.get('timestamp', '')}\n"
                f"Overview: {page.get('meta_info', '')}"
            )
            items.append(
                MemoryItem(
                    id=str(page.get("id") or page.get("page_id") or f"memoryos_page_{idx}"),
                    content=content,
                    score=_float_or_none(page.get("score")),
                    source_session_id=page.get("session_id"),
                    metadata=page,
                )
            )

        for idx, entry in enumerate(result.get("retrieved_user_knowledge", []) or []):
            items.append(
                MemoryItem(
                    id=str(entry.get("id") or f"memoryos_user_knowledge_{idx}"),
                    content=str(entry.get("knowledge", "")),
                    score=_float_or_none(entry.get("score")),
                    metadata=entry,
                )
            )

        self._last_retrieval = items
        return items[:k]

    def _user_id(self) -> str:
        template = self.config.get("user_id_template", "longmemeval_{question_id}")
        return template.format(question_id=self.sample_id)

    def get_debug_info(self) -> dict[str, Any]:
        if not self.memory:
            return {}
        try:
            return self.memory.get_memory_stats()
        except Exception:
            return {"last_retrieval_count": len(self._last_retrieval)}


def _session_pairs(session: MemorySession) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for turn in session.turns:
        if turn.role == "user":
            if pending_user is not None:
                pairs.append((pending_user, ""))
            pending_user = turn.content
        elif turn.role == "assistant":
            if pending_user is None:
                pairs.append(("", turn.content))
            else:
                pairs.append((pending_user, turn.content))
                pending_user = None
    if pending_user is not None:
        pairs.append((pending_user, ""))
    return pairs

def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
