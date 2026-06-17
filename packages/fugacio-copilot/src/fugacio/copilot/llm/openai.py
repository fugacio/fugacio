"""OpenAI chat-completions adapter for the Fugacio copilot.

Translates the neutral `Message` / tool schemas
to the OpenAI chat-completions wire format and parses tool calls back out. The
``openai`` SDK is imported lazily inside the constructor, so importing this
module never requires the dependency; install it with the ``llm`` extra.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from fugacio.copilot.llm.base import ChatResponse, JsonDict, Message, ToolCall, openai_tools


def _to_openai_message(m: Message) -> JsonDict:
    """Convert a neutral message to an OpenAI chat-completions message dict."""
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ],
        }
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    return {"role": m.role, "content": m.content}


class OpenAIProvider:
    """An `LLMProvider` backed by the OpenAI API.

    Args:
        model: Chat model name (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
        client: An existing ``openai.OpenAI`` client; one is created if omitted.
        api_key: API key passed to a freshly created client.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self.client = client
        else:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "OpenAIProvider requires the 'openai' package; install the "
                    "'llm' extra: pip install 'fugacio-copilot[llm]'"
                ) from exc
            self.client = openai.OpenAI(api_key=api_key)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JsonDict] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Call chat-completions and parse the reply into a `ChatResponse`."""
        kwargs: JsonDict = {
            "model": self.model,
            "messages": [_to_openai_message(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        calls = tuple(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (getattr(choice, "tool_calls", None) or [])
        )
        return ChatResponse(content=choice.content or "", tool_calls=calls, raw=resp)
