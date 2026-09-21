"""OpenAI chat-completions adapter for the Fugacio copilot.

Translates the neutral `Message` / tool schemas to the OpenAI chat-completions
wire format and parses tool calls and the finish reason back out. The
``openai`` SDK is imported lazily inside the constructor, so importing this
module never requires the dependency; install it with the ``llm`` extra.

The provider caps output with ``max_completion_tokens`` (reasoning models
reject ``max_tokens``) and sends ``temperature`` only when a caller passes one.
Tool arguments that aren't a valid JSON object don't raise: they come back as a
`ToolCall.invalid` call, which the agent loops answer with an error result so
the model can resend it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from fugacio.copilot.llm.base import ChatResponse, JsonDict, Message, ToolCall, openai_tools

#: Chat-completions ``finish_reason`` values mapped to the neutral stop reasons.
_STOP_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _to_openai_message(m: Message) -> JsonDict | None:
    """Convert a neutral message to a chat-completions message dict.

    Returns ``None`` for an assistant turn with neither text nor tool calls,
    which has nothing to send.
    """
    if m.role == "assistant":
        if not m.tool_calls:
            return {"role": "assistant", "content": m.content} if m.content else None
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
        # Tool messages have no error flag, so a failure is marked in the text.
        content = f"ERROR: {m.content}" if m.is_error else m.content
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": content}
    return {"role": m.role, "content": m.content}


def _parse_tool_call(tc: Any) -> ToolCall:
    """Parse one tool call, marking arguments that aren't a JSON object."""
    raw = tc.function.arguments or "{}"
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ToolCall.invalid(tc.id, tc.function.name, raw, f"invalid JSON: {exc}")
    if not isinstance(arguments, dict):
        return ToolCall.invalid(tc.id, tc.function.name, raw, "arguments must be a JSON object")
    return ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)


class OpenAIProvider:
    """An `LLMProvider` backed by the OpenAI API.

    Args:
        model: Chat model name.
        client: An existing ``openai.OpenAI`` client; one is created if omitted.
        api_key: API key passed to a freshly created client.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
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
        temperature: float | None = None,
        max_tokens: int = 16000,
    ) -> ChatResponse:
        """Call chat-completions and parse the reply into a `ChatResponse`.

        Args:
            messages: The conversation so far.
            tools: Engine tool schemas the model may call.
            temperature: Sampling temperature. ``None`` (the default) omits it,
                as reasoning models require.
            max_tokens: Per-reply output token cap, sent as
                ``max_completion_tokens``.

        Returns:
            The parsed reply. A ``"refusal"`` reply (a content-filter stop or a
            model refusal) carries no content, and a ``"max_tokens"`` reply
            keeps its text but drops its tool calls.
        """
        converted = (_to_openai_message(m) for m in messages)
        kwargs: JsonDict = {
            "model": self.model,
            "messages": [m for m in converted if m is not None],
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        message = choice.message
        stop_reason: str | None = _STOP_REASONS.get(choice.finish_reason, choice.finish_reason)
        if getattr(message, "refusal", None):
            stop_reason = "refusal"
        if stop_reason == "refusal":
            return ChatResponse(stop_reason=stop_reason, raw=resp)
        calls: tuple[ToolCall, ...] = ()
        if stop_reason != "max_tokens":  # the token cap may have truncated arguments
            calls = tuple(
                _parse_tool_call(tc) for tc in (getattr(message, "tool_calls", None) or ())
            )
        return ChatResponse(
            content=message.content or "", tool_calls=calls, stop_reason=stop_reason, raw=resp
        )
