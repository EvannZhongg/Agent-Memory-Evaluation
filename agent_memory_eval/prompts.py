from __future__ import annotations

LONGMEMEVAL_READER_INSTRUCTIONS = """You are an answer reader for long-term memory evaluation.

Answer the question using only the provided Memory Context.
Do not use outside knowledge, prior assumptions, or information that is not supported by the Memory Context.
If the Memory Context is empty, irrelevant, conflicting without a clear resolution, or insufficient, answer exactly: I don't know.
When relevant memories conflict, prefer the most recent memory that is clearly supported by the Memory Context.
Keep the answer concise and avoid explaining the evaluation process.
"""


def build_reader_prompt(*, memory_context: str, question: str) -> str:
    return (
        f"Memory Context:\n{memory_context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
