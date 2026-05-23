from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryTurn:
    role: str
    content: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySession:
    session_id: str
    date: str | None
    turns: list[MemoryTurn]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryItem:
    id: str
    content: str
    score: float | None = None
    source_session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSample:
    question_id: str
    question_type: str
    question: str
    answer: str | None
    question_date: str | None
    sessions: list[MemorySession]
    raw: dict[str, Any] = field(default_factory=dict)
    benchmark: str = "unknown"


LongMemEvalSample = BenchmarkSample
