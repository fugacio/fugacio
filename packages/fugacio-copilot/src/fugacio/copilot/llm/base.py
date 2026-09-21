"""Provider-neutral chat/tool-calling types for the Fugacio copilot.

The copilot talks to language models through a tiny, vendor-independent surface:
a `Message` list goes in (with the tool schemas the engine exposes), a
`ChatResponse` comes back (free text and/or one or more `ToolCall` requests,
plus a normalized stop reason). Concrete providers (`OpenAIProvider`,
`AnthropicProvider`, or the deterministic `MockProvider` used in tests)
implement the single-method `LLMProvider` protocol by translating to and from
their own wire formats. Nothing here imports a vendor SDK, so the copilot is
importable with no LLM dependency installed; the SDK is only needed when you
actually construct a real provider.

Normalized stop reasons are ``"end_turn"`` (finished), ``"tool_use"`` (wants
tool results), ``"max_tokens"`` (cut off at the token cap), ``"refusal"``
(declined; its content is never an answer), ``"pause_turn"`` (paused; resend
the conversation to resume), and ``"stop_sequence"``. A provider passes any
other value through unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

JsonDict = dict[str, Any]

#: Argument key holding the raw text of tool arguments that couldn't be parsed.
INVALID_ARGUMENTS = "__invalid_arguments__"
#: Argument key holding the parse error for unparseable tool arguments.
ARGUMENTS_ERROR = "__error__"


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool.

    Attributes:
        id: Provider-assigned call id (echoed back with the result).
        name: Tool name (must exist in the registry).
        arguments: Parsed JSON arguments for the tool. When the provider
            couldn't parse them, they hold the `INVALID_ARGUMENTS` and
            `ARGUMENTS_ERROR` markers instead (see `invalid`).
    """

    id: str
    name: str
    arguments: JsonDict = field(default_factory=dict)

    @classmethod
    def invalid(cls, call_id: str, name: str, raw: str, error: str) -> ToolCall:
        """A call whose raw arguments couldn't be parsed into a JSON object.

        The agent loops don't run such a call. They answer it with an error
        result quoting ``raw`` and ``error``, so the model can resend it.

        Args:
            call_id: Provider-assigned call id.
            name: Requested tool name.
            raw: The argument text exactly as the model produced it.
            error: Why the text isn't a valid JSON object.

        Returns:
            A `ToolCall` whose arguments carry the parse-failure markers.
        """
        return cls(
            id=call_id, name=name, arguments={INVALID_ARGUMENTS: raw, ARGUMENTS_ERROR: error}
        )


@dataclass(frozen=True)
class Message:
    """One turn in a chat transcript.

    Attributes:
        role: ``"system"``, ``"user"``, ``"assistant"`` or ``"tool"``.
        content: Text content (may be empty for an assistant tool-call turn).
        tool_calls: Tool calls requested by an assistant turn.
        tool_call_id: For a ``"tool"`` turn, the id of the call it answers.
        name: For a ``"tool"`` turn, the tool name (some providers want it).
        is_error: For a ``"tool"`` turn, whether the call failed.
        provider_blocks: For an assistant turn, the opaque provider blocks
            (`ChatResponse.provider_blocks`) its provider echoes back verbatim.
    """

    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    provider_blocks: tuple[Any, ...] = ()

    @classmethod
    def system(cls, content: str) -> Message:
        """A system instruction message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """A user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: Sequence[ToolCall] = (),
        provider_blocks: Sequence[Any] = (),
    ) -> Message:
        """An assistant message, optionally requesting tool calls.

        Args:
            content: The assistant's text.
            tool_calls: Tool calls the assistant requested.
            provider_blocks: The reply's `ChatResponse.provider_blocks`, kept so
                the provider can echo them on the next request.

        Returns:
            The assistant `Message`.
        """
        return cls(
            role="assistant",
            content=content,
            tool_calls=tuple(tool_calls),
            provider_blocks=tuple(provider_blocks),
        )

    @classmethod
    def tool(
        cls, content: str, tool_call_id: str, name: str | None = None, *, is_error: bool = False
    ) -> Message:
        """A tool-result message answering a specific tool call.

        Args:
            content: The serialized tool result.
            tool_call_id: Id of the call this message answers.
            name: The tool name.
            is_error: Whether the call failed (``content`` then explains why).

        Returns:
            The tool-result `Message`.
        """
        return cls(
            role="tool", content=content, tool_call_id=tool_call_id, name=name, is_error=is_error
        )


@dataclass(frozen=True)
class ChatResponse:
    """A model's reply: free-text content and/or requested tool calls.

    Check `stop_reason` before trusting the content: a ``"refusal"`` reply
    carries no usable content, and a ``"max_tokens"`` reply was cut off.

    Attributes:
        content: The assistant's text (the final answer when there are no calls).
        tool_calls: Any tool calls the model wants executed before continuing.
        stop_reason: Why the model stopped, normalized as described in the
            module docstring, or ``None`` when unknown.
        provider_blocks: Opaque provider content blocks (for example, Anthropic
            ``thinking`` and ``redacted_thinking`` blocks) that must be echoed
            verbatim when this reply is replayed; pass them to
            `Message.assistant`.
        raw: The provider's raw response object, for debugging (not portable).
    """

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str | None = None
    provider_blocks: tuple[Any, ...] = ()
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model requested at least one tool call."""
        return len(self.tool_calls) > 0


@runtime_checkable
class LLMProvider(Protocol):
    """A function-calling chat model.

    Implementations translate the neutral `Message` / tool-schema inputs to
    their own API and parse the reply into a `ChatResponse`.
    """

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JsonDict] = (),
        temperature: float | None = None,
        max_tokens: int = 16000,
    ) -> ChatResponse:
        """Return the model's reply to ``messages`` with ``tools`` available.

        Args:
            messages: The conversation so far.
            tools: Engine tool schemas the model may call.
            temperature: Sampling temperature. ``None`` (the default) omits the
                parameter, which models that reject sampling parameters require.
            max_tokens: Per-reply output token cap.

        Returns:
            The parsed `ChatResponse`.
        """
        ...


def openai_tools(schemas: Sequence[JsonDict]) -> list[JsonDict]:
    """Wrap engine tool schemas in OpenAI's ``{"type": "function", ...}`` envelope."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in schemas
    ]


def anthropic_tools(schemas: Sequence[JsonDict]) -> list[JsonDict]:
    """Map engine tool schemas to Anthropic's ``{"name", "description", "input_schema"}``."""
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["parameters"],
        }
        for s in schemas
    ]
