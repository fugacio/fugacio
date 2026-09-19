"""Anthropic (Claude) Messages-API adapter for the Fugacio copilot.

Translates the neutral `Message` / tool schemas to Anthropic's Messages format:
the system prompt is hoisted to the top-level argument, the tool results that
answer one assistant turn travel together in a single user message of
``tool_result`` blocks, and each replayed assistant turn leads with its
``thinking`` / ``redacted_thinking`` blocks, verbatim, before its text and
``tool_use`` blocks. Replies are parsed back with their stop reason. The
``anthropic`` SDK is imported lazily, so importing this module never requires
the dependency; install it with the ``llm`` extra.

Recent Claude models, including the default Claude Opus 5, reject sampling
parameters, so the provider sends ``temperature`` only when a caller passes
one. By default it also enables
Anthropic's server-side refusal fallbacks: when a model's safety classifiers
decline a request, the API reruns it on Anthropic's recommended fallback model
within the same call.
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

#: Beta header that enables the scalar ``fallbacks="default"`` request form.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: Reply block types kept in `ChatResponse.provider_blocks` and replayed verbatim.
_REPLAYED_BLOCKS = frozenset({"thinking", "redacted_thinking"})


def _to_anthropic_messages(messages: Sequence[Message]) -> list[JsonDict]:
    """Convert neutral non-system messages to Anthropic message dicts.

    Consecutive tool results become one user message of ``tool_result`` blocks.
    An assistant turn with neither text nor tool calls is skipped, since the API
    rejects empty assistant content.
    """
    convo: list[JsonDict] = []
    results: list[JsonDict] = []
    for m in messages:
        if m.role == "tool":
            result: JsonDict = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
            }
            if m.is_error:
                result["is_error"] = True
            results.append(result)
            continue
        if results:
            convo.append({"role": "user", "content": results})
            results = []
        if m.role != "assistant":
            convo.append({"role": m.role, "content": m.content})
        elif m.content or m.tool_calls:
            blocks: list[JsonDict] = list(m.provider_blocks)
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks.extend(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in m.tool_calls
            )
            convo.append({"role": "assistant", "content": blocks})
    if results:
        convo.append({"role": "user", "content": results})
    return convo


def _block_dict(block: Any) -> JsonDict:
    """Convert a ``thinking`` or ``redacted_thinking`` reply block to a request dict.

    SDK blocks are pydantic models, dumped with their API field names. Other
    objects, such as test doubles, are read attribute by attribute.
    """
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dict(dump(by_alias=True, exclude_none=True))
    if block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    return {"type": "redacted_thinking", "data": block.data}


def _parse_response(resp: Any) -> ChatResponse:
    """Parse a Messages-API reply, checking its stop reason before its content."""
    stop_reason = getattr(resp, "stop_reason", None)
    if stop_reason == "refusal":
        # A declined request has no usable content: a refusal before any output
        # has none, and partial output before a refusal must be discarded.
        return ChatResponse(stop_reason=stop_reason, raw=resp)
    blocks = list(resp.content or ())
    # After a mid-output fallback, the declining model's reasoning and tool calls
    # before the final ``fallback`` block are neither replayed nor run. Its text
    # stays: the fallback model continued from it.
    boundary = max(
        (i for i, block in enumerate(blocks) if getattr(block, "type", None) == "fallback"),
        default=-1,
    )
    text: list[str] = []
    calls: list[ToolCall] = []
    replayed: list[JsonDict] = []
    for i, block in enumerate(blocks):
        kind = getattr(block, "type", None)
        if kind == "text":
            text.append(block.text)
        elif i <= boundary:
            continue
        elif kind in _REPLAYED_BLOCKS:
            replayed.append(_block_dict(block))
        elif kind == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    if stop_reason == "max_tokens":
        calls = []  # the token cap may have truncated a call's input
    return ChatResponse(
        content="".join(text),
        tool_calls=tuple(calls),
        stop_reason=stop_reason,
        provider_blocks=tuple(replayed),
        raw=resp,
    )


class AnthropicProvider:
    """An `LLMProvider` backed by the Anthropic Messages API.

    Args:
        model: Claude model id.
        client: An existing ``anthropic.Anthropic`` client; one is created if omitted.
        api_key: API key passed to a freshly created client.
        fallbacks: Server-side refusal-fallback mode. With the default,
            ``"default"``, requests go to the beta Messages endpoint with the
            `FALLBACK_BETA` header, and the API routes a declined request to
            Anthropic's recommended fallback model for the refusal category.
            ``None`` disables fallbacks and uses the standard endpoint, as
            platforms without server-side fallbacks (Amazon Bedrock, Vertex AI,
            and Microsoft Foundry) require.
    """

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        client: Any | None = None,
        api_key: str | None = None,
        fallbacks: str | None = "default",
    ) -> None:
        self.model = model
        self.fallbacks = fallbacks
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
        temperature: float | None = None,
        max_tokens: int = 16000,
    ) -> ChatResponse:
        """Call the Messages API and parse the reply into a `ChatResponse`.

        Args:
            messages: The conversation so far; system messages become the
                top-level system prompt.
            tools: Engine tool schemas the model may call.
            temperature: Sampling temperature. ``None`` (the default) omits it,
                as recent Claude models, including Claude Opus 5, require.
            max_tokens: Per-reply output token cap, thinking included.

        Returns:
            The parsed reply. A ``"refusal"`` reply carries no content, and a
            ``"max_tokens"`` reply keeps its text but drops its tool calls.
        """
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        kwargs: JsonDict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": _to_anthropic_messages([m for m in messages if m.role != "system"]),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = anthropic_tools(tools)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.fallbacks is None:
            resp = self.client.messages.create(**kwargs)
        else:
            # ``extra_body`` keeps this working on SDK releases that predate a
            # typed ``fallbacks`` argument; the wire request is identical.
            resp = self.client.beta.messages.create(
                betas=[FALLBACK_BETA], extra_body={"fallbacks": self.fallbacks}, **kwargs
            )
        return _parse_response(resp)
