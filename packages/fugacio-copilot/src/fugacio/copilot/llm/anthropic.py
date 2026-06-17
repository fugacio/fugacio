"""Anthropic (Claude) Messages-API adapter for the Fugacio copilot.

Translates the neutral `Message` / tool schemas
to Anthropic's Messages format (system prompt hoisted to a top-level argument,
``tool_use`` / ``tool_result`` content blocks) and parses tool calls back out.
The ``anthropic`` SDK is imported lazily, so importing this module never requires
the dependency; install it with the ``llm`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fugacio.copilot.llm.base import (
    ChatResponse,
    JsonDict,
    Message,
    ToolCall,
    anthropic_tools,
)


def _to_anthropic_message(m: Message) -> JsonDict:
    """Convert a neutral message to an Anthropic message dict (non-system)."""
    if m.role == "assistant" and m.tool_calls:
        blocks: list[JsonDict] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        blocks.extend(
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            for tc in m.tool_calls
        )
        return {"role": "assistant", "content": blocks}
    if m.role == "tool":
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            ],
        }
    return {"role": m.role, "content": m.content}


class AnthropicProvider:
    """An `LLMProvider` backed by the Anthropic API.

    Args:
        model: Claude model name (e.g. ``"claude-3-5-sonnet-latest"``).
        client: An existing ``anthropic.Anthropic`` client; one is created if omitted.
        api_key: API key passed to a freshly created client.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        *,
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self.client = client
        else:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "AnthropicProvider requires the 'anthropic' package; install the "
                    "'llm' extra: pip install 'fugacio-copilot[llm]'"
                ) from exc
            self.client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JsonDict] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Call the Messages API and parse the reply into a `ChatResponse`."""
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [_to_anthropic_message(m) for m in messages if m.role != "system"]
        kwargs: JsonDict = {
            "model": self.model,
            "messages": convo,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = anthropic_tools(tools)
        resp = self.client.messages.create(**kwargs)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        return ChatResponse(content="".join(text_parts), tool_calls=tuple(calls), raw=resp)
