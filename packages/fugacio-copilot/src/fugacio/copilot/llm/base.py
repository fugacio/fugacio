"""Provider-neutral chat/tool-calling types for the Fugacio copilot.

The copilot talks to language models through a tiny, vendor-independent surface:
a :class:`Message` list goes in (with the tool schemas the engine exposes), a
:class:`ChatResponse` comes back (either free text or one or more
:class:`ToolCall` requests). Concrete providers -- :class:`OpenAIProvider`,
:class:`AnthropicProvider`, or the deterministic :class:`MockProvider` used in
tests -- implement the single-method :class:`LLMProvider` protocol by translating
to and from their own wire formats. Nothing here imports a vendor SDK, so the
copilot is importable with no LLM dependency installed; the SDK is only needed
when you actually construct a real provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool.

    Attributes:
        id: Provider-assigned call id (echoed back with the result).
        name: Tool name (must exist in the registry).
        arguments: Parsed JSON arguments for the tool.
    """

    id: str
    name: str
    arguments: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """One turn in a chat transcript.

    Attributes:
        role: ``"system"``, ``"user"``, ``"assistant"`` or ``"tool"``.
        content: Text content (may be empty for an assistant tool-call turn).
        tool_calls: Tool calls requested by an assistant turn.
        tool_call_id: For a ``"tool"`` turn, the id of the call it answers.
        name: For a ``"tool"`` turn, the tool name (some providers want it).
    """

    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        """A system instruction message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """A user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: Sequence[ToolCall] = ()) -> Message:
        """An assistant message, optionally requesting tool calls."""
        return cls(role="assistant", content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None) -> Message:
        """A tool-result message answering a specific tool call."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass(frozen=True)
class ChatResponse:
    """A model's reply: free-text content and/or requested tool calls.

    Attributes:
        content: The assistant's text (the final answer when there are no calls).
        tool_calls: Any tool calls the model wants executed before continuing.
        raw: The provider's raw response object, for debugging (not portable).
    """

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model requested at least one tool call."""
        return len(self.tool_calls) > 0


@runtime_checkable
class LLMProvider(Protocol):
    """A function-calling chat model.

    Implementations translate the neutral :class:`Message` / tool-schema inputs to
    their own API and parse the reply into a :class:`ChatResponse`.
    """

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JsonDict] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Return the model's reply to ``messages`` with ``tools`` available."""
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
