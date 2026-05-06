from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .token_usage import TokenUsageTracker


@contextmanager
def record_method_prompt_tokens(
    obj: Any,
    method_name: str,
    tracker: TokenUsageTracker,
    *,
    phase: str,
    event: str,
) -> Iterator[None]:
    if obj is None or not hasattr(obj, method_name):
        yield
        return

    original = getattr(obj, method_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        text = _prompt_text_from_call(args, kwargs)
        if text:
            metadata = {"method": method_name, "phase": phase}
            if phase == "build":
                tracker.record_build_llm_prompt(text, event=event, metadata=metadata)
            else:
                tracker.record_query_llm_prompt(text, event=event, metadata=metadata)
        return original(*args, **kwargs)

    setattr(obj, method_name, wrapped)
    try:
        yield
    finally:
        setattr(obj, method_name, original)


def _prompt_text_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    messages = kwargs.get("messages")
    if messages is None and args and isinstance(args[0], list):
        messages = args[0]
    if messages is not None:
        return _messages_to_text(messages)

    prompt = kwargs.get("prompt")
    if prompt is None and args and isinstance(args[0], str):
        prompt = args[0]
    return str(prompt or "")


def _messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages)
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            parts.append(f"{message.get('role', '')}: {message.get('content', '')}")
        else:
            parts.append(str(message))
    return "\n".join(parts)
