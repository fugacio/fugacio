"""A minimal tool-calling agent loop for the Fugacio design copilot.

The agent is intentionally model-agnostic: the *planner* (the component that, given
a goal and the available tool schemas, decides the next tool call or the final
answer) is injected. This keeps the control loop fully testable with a scripted
planner, while a real LLM planner -- OpenAI, Anthropic, or any function-calling
model -- drops straight in behind the optional ``llm`` extra.

A planner is a callable ``(goal, tool_schemas, transcript) -> decision`` where
``decision`` is either ``{"tool": name, "arguments": {...}}`` to call a tool or
``{"final_answer": text}`` to stop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fugacio.copilot.tools import ToolSpec, call_tool, default_registry, tool_schemas

JsonDict = dict[str, Any]
Planner = Callable[[str, list[JsonDict], list[JsonDict]], JsonDict]


@dataclass(frozen=True)
class AgentResult:
    """Outcome of an agent run.

    Attributes:
        answer: The final natural-language (or structured) answer.
        transcript: The ordered list of tool calls and their results.
    """

    answer: str
    transcript: list[JsonDict] = field(default_factory=list)


def run_agent(
    goal: str,
    planner: Planner,
    *,
    registry: dict[str, ToolSpec] | None = None,
    max_steps: int = 6,
) -> AgentResult:
    """Run the plan/act loop until the planner returns a final answer.

    Args:
        goal: The natural-language design goal.
        planner: Decision function (see module docstring); inject an LLM here.
        registry: Tool registry to expose (defaults to :func:`default_registry`).
        max_steps: Maximum number of tool calls before giving up.

    Returns:
        An :class:`AgentResult` with the final answer and the full transcript.
    """
    registry = default_registry() if registry is None else registry
    schemas = tool_schemas(registry)
    transcript: list[JsonDict] = []
    for _ in range(max_steps):
        decision = planner(goal, schemas, transcript)
        if "final_answer" in decision:
            return AgentResult(answer=str(decision["final_answer"]), transcript=transcript)
        name = decision["tool"]
        arguments = decision.get("arguments", {})
        result = call_tool(name, arguments, registry)
        transcript.append({"tool": name, "arguments": arguments, "result": result})
    return AgentResult(answer="step budget exhausted before a final answer", transcript=transcript)
