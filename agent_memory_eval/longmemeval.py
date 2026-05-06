from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Iterable

from .models import LongMemEvalSample, MemorySession, MemoryTurn


def _turn_from_raw(raw: dict, timestamp: str | None, session_id: str) -> MemoryTurn:
    return MemoryTurn(
        role=str(raw.get("role", "")),
        content=str(raw.get("content", "")),
        timestamp=timestamp,
        metadata={
            k: v
            for k, v in raw.items()
            if k not in {"role", "content"}
        }
        | {"source_session_id": session_id},
    )


def load_dataset(path: str | Path, limit: int | None = None) -> list[LongMemEvalSample]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    samples: list[LongMemEvalSample] = []
    for raw in raw_samples[:limit]:
        session_ids = raw.get("haystack_session_ids", [])
        dates = raw.get("haystack_dates", [])
        sessions = raw.get("haystack_sessions", [])

        memory_sessions: list[MemorySession] = []
        for idx, turns in enumerate(sessions):
            session_id = str(session_ids[idx]) if idx < len(session_ids) else f"session_{idx}"
            date = str(dates[idx]) if idx < len(dates) else None
            memory_turns = [
                _turn_from_raw(turn, timestamp=date, session_id=session_id)
                for turn in turns
            ]
            memory_sessions.append(
                MemorySession(
                    session_id=session_id,
                    date=date,
                    turns=memory_turns,
                    metadata={
                        "question_id": raw.get("question_id"),
                        "question_date": raw.get("question_date"),
                        "session_index": idx,
                    },
                )
            )
        memory_sessions.sort(key=lambda session: _date_sort_key(session.date))

        samples.append(
            LongMemEvalSample(
                question_id=str(raw.get("question_id")),
                question_type=str(raw.get("question_type", "")),
                question=str(raw.get("question", "")),
                answer=raw.get("answer"),
                question_date=raw.get("question_date"),
                sessions=memory_sessions,
                raw=raw,
            )
        )
    return samples


def iter_samples(path: str | Path, limit: int | None = None) -> Iterable[LongMemEvalSample]:
    yield from load_dataset(path, limit=limit)


def _date_sort_key(value: str | None) -> tuple[int, str]:
    if not value:
        return (1, "")
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return (0, datetime.strptime(value, fmt).isoformat())
        except ValueError:
            continue
    return (0, value)
