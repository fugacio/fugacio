"""A deterministic in-memory provider for testing the agent loop without an API.

:class:`MockProvider` plays back either a fixed list of :class:`ChatResponse`
objects (one per ``chat`` call) or a callable that decides the reply from the
running message list. It records every call it receives, so tests can assert on
how the agent drove the conversation. This keeps the whole tool-calling loop --
plan, call tool, feed the result back, answer -- fully exercisable in the fast,
hermetic unit suite.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from fugacio.copilot.llm.base import ChatResponse, JsonDict, Message

Script = list[ChatResponse] | Callable[[list[Message]], ChatResponse]


@dataclass
class MockProvider:
    """A scripted :class:`~fugacio.copilot.llm.base.LLMProvider`.

    Attributes:
        script: Either a list of replies returned in order, or a function
            ``(messages) -> ChatResponse`` evaluated on each call.
        calls: Recorded ``(messages, tools)`` for each ``chat`` invocation.
    """

    script: Script
    calls: list[tuple[list[Message], list[JsonDict]]] = field(default_factory=list)
    _index: int = 0

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JsonDict] = (),
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Return the next scripted reply (or evaluate the script callable)."""
        self.calls.append((list(messages), list(tools)))
        if callable(self.script):
            return self.script(list(messages))
        if self._index >= len(self.script):
            # Exhausted script: end the conversation gracefully.
            return ChatResponse(content="(mock provider: script exhausted)")
        reply = self.script[self._index]
        self._index += 1
        return reply
