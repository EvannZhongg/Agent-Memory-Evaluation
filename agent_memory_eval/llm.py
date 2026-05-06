from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import env_value


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "LLM_API_KEY"
    base_url_env: str = "LLM_BASE_URL"
    base_url: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = 1200
    extra_body: dict[str, Any] | None = None


class OpenAIResponsesClient:
    def __init__(self, config: LLMConfig):
        if config.provider != "openai":
            raise ValueError("Only OpenAI Responses-compatible LLMs are supported in phase 1.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use OpenAI Responses API.") from exc

        api_key = env_value(config.api_key_env)
        base_url = config.base_url or env_value(config.base_url_env)
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.config = config
        self.api_key_env = config.api_key_env

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenAIResponsesClient":
        allowed = set(LLMConfig.__dataclass_fields__.keys())
        return cls(LLMConfig(**{key: value for key, value in data.items() if key in allowed}))

    def generate(self, prompt: str, *, instructions: str | None = None) -> str:
        request: dict[str, Any] = {
            "model": self.config.model,
            "input": prompt,
        }
        if instructions:
            request["instructions"] = instructions
        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            request["max_output_tokens"] = self.config.max_output_tokens
        if self.config.extra_body:
            request["extra_body"] = self.config.extra_body

        try:
            response = self.client.responses.create(**request)
        except Exception as exc:
            raise RuntimeError(
                f"Responses API call failed for model '{self.config.model}'. "
                f"Check API key env '{self.api_key_env}' and base_url configuration."
            ) from exc
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "reasoning":
                continue
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()
