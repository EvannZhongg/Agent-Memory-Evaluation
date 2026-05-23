from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import MemoryBackend
from .base import session_to_text
from ..llm_token_hooks import record_method_prompt_tokens
from ..config import env_value
from ..models import MemoryItem, MemorySession
from ..pathing import add_repo_path


class Mem0Backend(MemoryBackend):
    backend_name = "mem0"
    default_top_k = 20

    def __init__(self, config: dict[str, Any], llm_config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.llm_config = llm_config
        self.root = Path(config.get("root", ".")).resolve()
        self.repo_path = add_repo_path(self.root, config.get("repo_path", "mem0"))
        self.memory = None
        self.sample_id: str | None = None

    def reset(self, sample_id: str) -> None:
        from mem0 import Memory

        super().reset(sample_id)
        self.sample_id = sample_id
        memory_config = self._build_memory_config(sample_id)
        self.memory = Memory.from_config(memory_config)

    def ingest_session(self, session: MemorySession) -> None:
        assert self.memory is not None
        messages = [{"role": turn.role, "content": turn.content} for turn in session.turns]
        self.token_usage.record_build(
            session_to_text(session),
            event="mem0.add.session",
            metadata={
                "session_id": session.session_id,
                "date": session.date,
                "turn_count": len(session.turns),
            },
        )
        with record_method_prompt_tokens(
            getattr(self.memory, "llm", None),
            "generate_response",
            self.token_usage,
            phase="build",
            event="mem0.internal_llm_prompt",
        ):
            self.memory.add(
                messages,
                user_id=self._user_id(),
                metadata={
                    "question_id": self.sample_id,
                    "session_id": session.session_id,
                    "session_date": session.date,
                },
                infer=bool(self.config.get("infer", True)),
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
            event="mem0.search.query",
            metadata={"top_k": k},
        )
        with record_method_prompt_tokens(
            getattr(self.memory, "llm", None),
            "generate_response",
            self.token_usage,
            phase="query",
            event="mem0.internal_llm_prompt",
        ):
            result = self.memory.search(
                query,
                top_k=k,
                filters={"user_id": self._user_id()},
                threshold=float(self.config.get("threshold", 0.1)),
                rerank=bool(self.config.get("rerank", False)),
            )
        raw_items = result.get("results", result) if isinstance(result, dict) else result
        items: list[MemoryItem] = []
        for idx, raw in enumerate(raw_items or []):
            metadata = raw.get("metadata") if isinstance(raw, dict) else {}
            items.append(
                MemoryItem(
                    id=str(raw.get("id", f"mem0_{idx}")),
                    content=str(raw.get("memory") or raw.get("content") or raw.get("text") or ""),
                    score=_float_or_none(raw.get("score")),
                    source_session_id=(metadata or {}).get("session_id") or raw.get("session_id"),
                    metadata=raw,
                )
            )
        return items

    def _user_id(self) -> str:
        template = self.config.get("user_id_template", "longmemeval_{question_id}")
        return template.format(question_id=self.sample_id)

    def _build_memory_config(self, sample_id: str) -> dict[str, Any]:
        raw = dict(self.config.get("mem0_config") or {})
        collection_template = self.config.get("collection_template")

        if not raw:
            embedding_config = self.llm_config.get("embedding", {})
            llm_base_url = self.llm_config.get("base_url") or env_value(self.llm_config.get("base_url_env"))
            llm_chat_base_url = (
                self.config.get("chat_base_url")
                or env_value(self.config.get("chat_base_url_env"))
                or self.llm_config.get("chat_base_url")
                or _responses_to_chat_base_url(llm_base_url)
                or llm_base_url
            )
            embedding_base_url = embedding_config.get("base_url") or env_value(embedding_config.get("base_url_env"))
            llm_api_key = env_value(self.llm_config.get("api_key_env", "LLM_API_KEY"))
            embedding_api_key = env_value(embedding_config.get("api_key_env", "EMBEDDING_API_KEY")) or llm_api_key
            raw = {
                "llm": {
                    "provider": self.config.get("llm_provider", "openai"),
                    "config": {
                        "model": self.config.get("llm_model") or self.llm_config.get("model", "gpt-4o-mini"),
                        "api_key": llm_api_key,
                        "openai_base_url": llm_chat_base_url,
                    },
                },
                "embedder": {
                    "provider": self.config.get("embedder_provider", "openai"),
                    "config": {
                        "model": self.config.get("embedding_model")
                        or embedding_config.get("model", "text-embedding-3-small"),
                        "api_key": embedding_api_key,
                        "openai_base_url": embedding_base_url,
                    },
                },
                "vector_store": {
                    "provider": self.config.get("vector_store_provider", "chroma"),
                    "config": {
                        "collection_name": collection_template.format(question_id=sample_id)
                        if collection_template
                        else f"longmemeval_{sample_id}",
                        "path": str(Path(self.config.get("storage_path", "runs/vectorstores/mem0")).resolve()),
                    },
                },
            }
        if collection_template:
            raw.setdefault("vector_store", {}).setdefault("config", {})["collection_name"] = (
                collection_template.format(question_id=sample_id)
            )
        return raw


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _responses_to_chat_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    dashscope_responses = "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
    if base_url.rstrip("/") == dashscope_responses:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    marker = "/api/v2/apps/protocols/compatible-mode/v1"
    if marker in base_url:
        return base_url.replace(marker, "/compatible-mode/v1")
    return None
