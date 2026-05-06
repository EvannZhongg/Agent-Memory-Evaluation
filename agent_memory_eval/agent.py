from __future__ import annotations

from dataclasses import dataclass

from .backends.base import MemoryBackend
from .llm import OpenAIResponsesClient
from .models import MemoryItem
from .prompts import LONGMEMEVAL_READER_INSTRUCTIONS, build_reader_prompt


DEFAULT_INSTRUCTIONS = LONGMEMEVAL_READER_INSTRUCTIONS


@dataclass
class AgentRuntime:
    backend: MemoryBackend
    llm: OpenAIResponsesClient
    top_k: int | None = None
    instructions: str = DEFAULT_INSTRUCTIONS

    def answer(self, question: str) -> tuple[str, list[MemoryItem], str]:
        retrieved = self.backend.retrieve(question, top_k=self.top_k)
        memory_context = self.backend.build_context(question, retrieved)
        self.backend.token_usage.record_retrieved_context(
            memory_context,
            event="reader.memory_context",
            metadata={"retrieved_count": len(retrieved)},
        )
        prompt = build_reader_prompt(memory_context=memory_context, question=question)
        reader_input = f"{self.instructions}\n\n{prompt}" if self.instructions else prompt
        self.backend.token_usage.record_reader_prompt(
            reader_input,
            event="reader.prompt",
            metadata={"retrieved_count": len(retrieved)},
        )
        answer = self.llm.generate(prompt, instructions=self.instructions)
        return answer, retrieved, memory_context
