from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str | None, *, model: str | None = None) -> int:
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        encoding = _encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return len(_TOKEN_PATTERN.findall(text))


def count_messages_tokens(messages: list[dict[str, Any]], *, model: str | None = None) -> int:
    text = "\n".join(f"{message.get('role', '')}: {message.get('content', '')}" for message in messages)
    return count_tokens(text, model=model)


@dataclass
class TokenUsageTracker:
    tokenizer_model: str | None = None
    build_input_tokens: int = 0
    build_llm_prompt_tokens: int = 0
    query_input_tokens: int = 0
    query_llm_prompt_tokens: int = 0
    retrieved_context_tokens: int = 0
    reader_prompt_tokens: int = 0
    build_events: list[dict[str, Any]] = field(default_factory=list)
    query_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def build_tokens(self) -> int:
        return self.build_input_tokens + self.build_llm_prompt_tokens

    @property
    def memory_query_tokens(self) -> int:
        return self.query_input_tokens

    @property
    def query_tokens(self) -> int:
        return self.query_input_tokens + self.query_llm_prompt_tokens + self.reader_prompt_tokens

    @property
    def total_tokens(self) -> int:
        return self.build_tokens + self.query_tokens

    def record_build(self, text: str, *, event: str, metadata: dict[str, Any] | None = None) -> int:
        tokens = count_tokens(text, model=self.tokenizer_model)
        self.build_input_tokens += tokens
        self.build_events.append({"event": event, "tokens": tokens, "metadata": metadata or {}})
        return tokens

    def record_build_llm_prompt(self, text: str, *, event: str, metadata: dict[str, Any] | None = None) -> int:
        tokens = count_tokens(text, model=self.tokenizer_model)
        self.build_llm_prompt_tokens += tokens
        self.build_events.append({"event": event, "tokens": tokens, "metadata": metadata or {}})
        return tokens

    def record_memory_query(self, text: str, *, event: str, metadata: dict[str, Any] | None = None) -> int:
        tokens = count_tokens(text, model=self.tokenizer_model)
        self.query_input_tokens += tokens
        self.query_events.append({"event": event, "tokens": tokens, "metadata": metadata or {}})
        return tokens

    def record_query_llm_prompt(self, text: str, *, event: str, metadata: dict[str, Any] | None = None) -> int:
        tokens = count_tokens(text, model=self.tokenizer_model)
        self.query_llm_prompt_tokens += tokens
        self.query_events.append({"event": event, "tokens": tokens, "metadata": metadata or {}})
        return tokens

    def record_retrieved_context(self, text: str, *, event: str, metadata: dict[str, Any] | None = None) -> int:
        tokens = count_tokens(text, model=self.tokenizer_model)
        self.retrieved_context_tokens += tokens
        self.query_events.append({"event": event, "tokens": tokens, "metadata": metadata or {}})
        return tokens

    def record_reader_prompt(self, text: str, *, event: str, metadata: dict[str, Any] | None = None) -> int:
        tokens = count_tokens(text, model=self.tokenizer_model)
        self.reader_prompt_tokens += tokens
        self.query_events.append({"event": event, "tokens": tokens, "metadata": metadata or {}})
        return tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_model": self.tokenizer_model,
            "token_counter": token_counter_name(self.tokenizer_model),
            "build_input_tokens": self.build_input_tokens,
            "build_llm_prompt_tokens": self.build_llm_prompt_tokens,
            "build_tokens": self.build_tokens,
            "query_input_tokens": self.query_input_tokens,
            "query_llm_prompt_tokens": self.query_llm_prompt_tokens,
            "memory_query_tokens": self.memory_query_tokens,
            "retrieved_context_tokens": self.retrieved_context_tokens,
            "reader_prompt_tokens": self.reader_prompt_tokens,
            "query_tokens": self.query_tokens,
            "total_tokens": self.total_tokens,
            "build_events": self.build_events,
            "query_events": self.query_events,
        }


def _encoding_for_model(model: str | None):
    import tiktoken  # type: ignore

    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            pass
    return tiktoken.get_encoding("cl100k_base")


def token_counter_name(model: str | None) -> str:
    try:
        encoding = _encoding_for_model(model)
        return f"tiktoken:{encoding.name}"
    except Exception:
        return "regex_fallback"
