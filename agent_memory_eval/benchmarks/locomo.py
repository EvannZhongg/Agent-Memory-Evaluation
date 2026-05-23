from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter
from ..config import resolve_path
from ..models import BenchmarkSample, MemorySession, MemoryTurn


LOCOMO_CATEGORY_LABELS = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


class LocomoBenchmark(BenchmarkAdapter):
    benchmark_name = "locomo"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.dataset_path.exists():
            errors.append(f"Dataset not found: {self.dataset_path}")
        return errors

    def load_samples(self, limit: int | None = None) -> list[BenchmarkSample]:
        return load_dataset(self.dataset_path, limit=limit)

    def prediction_record(self, sample: BenchmarkSample, answer: str) -> dict[str, Any]:
        return {
            "question_id": sample.question_id,
            "sample_id": sample.raw.get("sample_id"),
            "qa_index": sample.raw.get("qa_index"),
            "category": sample.raw.get("qa", {}).get("category"),
            "hypothesis": answer,
        }

    def evaluate(
        self,
        *,
        predictions_path: Path,
        run_dir: Path,
        progress: bool = True,
    ) -> dict[str, Any]:
        evaluation = self.config.get("evaluation", {})
        if not evaluation.get("enabled", True):
            return {
                "status": "skipped",
                "benchmark": self.benchmark_name,
                "message": "Evaluation disabled by benchmark.evaluation.enabled=false.",
                "predictions_path": str(predictions_path),
                "dataset_path": str(self.dataset_path),
            }

        prediction_key = str(evaluation.get("prediction_key", "hypermemo_prediction"))
        predictions = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        refs = _qa_reference_map(self.dataset_path)
        rows: list[dict[str, Any]] = []
        nested: dict[str, dict[str, Any]] = {}
        category_scores: dict[int, list[float]] = defaultdict(list)
        skipped = 0

        for pred in predictions:
            question_id = pred.get("question_id")
            ref = refs.get(question_id)
            if ref is None:
                skipped += 1
                continue
            hypothesis = str(pred.get("hypothesis", ""))
            score = locomo_qa_score(hypothesis, ref["answer"], int(ref["category"]))
            category = int(ref["category"])
            category_scores[category].append(score)
            row = {
                "question_id": question_id,
                "sample_id": ref["sample_id"],
                "qa_index": ref["qa_index"],
                "category": category,
                "category_label": LOCOMO_CATEGORY_LABELS.get(category, f"category_{category}"),
                "question": ref["question"],
                "answer": ref["answer"],
                "hypothesis": hypothesis,
                "f1": score,
            }
            rows.append(row)

            sample = nested.setdefault(ref["sample_id"], {"sample_id": ref["sample_id"], "qa": []})
            qa = dict(ref["qa"])
            qa[prediction_key] = hypothesis
            qa[f"{prediction_key}_f1"] = round(score, 3)
            sample["qa"].append(qa)

        eval_path = run_dir / "locomo_eval.json"
        nested_path = run_dir / "locomo_predictions.json"
        eval_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        nested_path.write_text(json.dumps(list(nested.values()), ensure_ascii=False, indent=2), encoding="utf-8")

        per_category = {
            str(category): {
                "label": LOCOMO_CATEGORY_LABELS.get(category, f"category_{category}"),
                "count": len(scores),
                "f1": _mean(scores),
            }
            for category, scores in sorted(category_scores.items())
        }
        return {
            "status": "evaluated",
            "benchmark": self.benchmark_name,
            "metric": "locomo_f1",
            "sample_count": len(rows),
            "skipped_count": skipped,
            "overall_f1": _mean([row["f1"] for row in rows]),
            "per_category": per_category,
            "predictions_path": str(predictions_path),
            "evaluated_predictions_path": str(eval_path),
            "locomo_predictions_path": str(nested_path),
            "dataset_path": str(self.dataset_path),
        }

    @property
    def dataset_path(self) -> Path:
        return resolve_path(self.config["dataset_path"], self.root)


def load_dataset(path: str | Path, limit: int | None = None) -> list[BenchmarkSample]:
    dataset_path = Path(path)
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples: list[BenchmarkSample] = []
    for record in records:
        sessions = _conversation_sessions(record)
        sample_id = str(record.get("sample_id", f"sample_{len(samples)}"))
        for qa_index, qa in enumerate(record.get("qa", [])):
            question_id = f"{sample_id}::qa_{qa_index}"
            samples.append(
                BenchmarkSample(
                    question_id=question_id,
                    question_type=str(qa.get("category", "")),
                    question=str(qa.get("question", "")),
                    answer=str(qa.get("answer", "")),
                    question_date=None,
                    sessions=sessions,
                    raw={
                        "sample_id": sample_id,
                        "qa_index": qa_index,
                        "qa": qa,
                    },
                    benchmark="locomo",
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def locomo_qa_score(prediction: str, answer: Any, category: int) -> float:
    answer_text = str(answer)
    if category == 3:
        answer_text = answer_text.split(";")[0].strip()

    if category in {2, 3, 4}:
        return f1_score(prediction, answer_text)
    if category == 1:
        return multi_answer_f1(prediction, answer_text)
    if category == 5:
        lowered = prediction.lower()
        return 1.0 if "no information available" in lowered or "not mentioned" in lowered else 0.0
    return f1_score(prediction, answer_text)


def _conversation_sessions(record: dict[str, Any]) -> list[MemorySession]:
    conversation = record.get("conversation") or {}
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    sessions: list[MemorySession] = []
    for session_key in _sorted_session_keys(conversation):
        session_id = session_key.replace("session_", "S")
        date = conversation.get(f"{session_key}_date_time")
        turns = [
            _turn_from_dialog(dialog, timestamp=date, session_id=session_id, speaker_a=speaker_a, speaker_b=speaker_b)
            for dialog in conversation.get(session_key, [])
        ]
        sessions.append(MemorySession(session_id=session_id, date=date, turns=turns, metadata={"sample_id": record.get("sample_id")}))
    return sessions


def _turn_from_dialog(
    dialog: dict[str, Any],
    *,
    timestamp: str | None,
    session_id: str,
    speaker_a: str | None,
    speaker_b: str | None,
) -> MemoryTurn:
    speaker = str(dialog.get("speaker", "speaker"))
    text = str(dialog.get("text", ""))
    if dialog.get("blip_caption"):
        text += f"\n[shared image caption: {dialog['blip_caption']}]"
    return MemoryTurn(
        role=_speaker_role(speaker, speaker_a, speaker_b),
        content=f"{speaker}: {text}",
        timestamp=timestamp,
        metadata={
            "speaker": speaker,
            "dia_id": dialog.get("dia_id"),
            "source_session_id": session_id,
            **{key: value for key, value in dialog.items() if key not in {"speaker", "text"}},
        },
    )


def _qa_reference_map(path: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        sample_id = str(record.get("sample_id"))
        for qa_index, qa in enumerate(record.get("qa", [])):
            question_id = f"{sample_id}::qa_{qa_index}"
            refs[question_id] = {
                "question_id": question_id,
                "sample_id": sample_id,
                "qa_index": qa_index,
                "qa": qa,
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "category": qa.get("category"),
            }
    return refs


def _sorted_session_keys(conversation: dict[str, Any]) -> list[str]:
    def session_number(key: str) -> int:
        try:
            return int(key.split("_", 1)[1])
        except (IndexError, ValueError):
            return 10**9

    return sorted(
        [
            key
            for key, value in conversation.items()
            if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list)
        ],
        key=session_number,
    )


def _speaker_role(speaker: str, speaker_a: str | None, speaker_b: str | None) -> str:
    if speaker_a and speaker == speaker_a:
        return "user"
    if speaker_b and speaker == speaker_b:
        return "assistant"
    return "user"


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text)).lower().replace(",", "")
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [_stem(token) for token in normalize_answer(prediction).split()]
    ground_truth_tokens = [_stem(token) for token in normalize_answer(ground_truth).split()]
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    predictions = [part.strip() for part in str(prediction).split(",")]
    ground_truths = [part.strip() for part in str(ground_truth).split(",")]
    if not ground_truths:
        return 0.0
    return sum(max(f1_score(pred, gt) for pred in predictions) for gt in ground_truths) / len(ground_truths)


def _stem(token: str) -> str:
    try:
        from nltk.stem import PorterStemmer

        return PorterStemmer().stem(token)
    except Exception:
        return token


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
