"""Chemical-engineering design copilot for Fugacio (depends on ``fugacio.sim``).

The copilot exposes the differentiable engine to a language-model design agent
through:

* a **tool registry** (:func:`default_registry`, :func:`tool_schemas`,
  :func:`call_tool`) of deterministic, JSON-in/JSON-out engineering calculations;
* a model-agnostic **agent loop** (:func:`run_agent`) whose planner is injected,
  so a real LLM plugs in behind the optional ``llm`` extra;
* human-readable **reports** (:func:`summarize_bubble_point`).
"""

from fugacio.copilot.agent import AgentResult, Planner, run_agent
from fugacio.copilot.report import summarize_bubble_point
from fugacio.copilot.tools import (
    ToolSpec,
    call_tool,
    default_registry,
    tool_schemas,
)

__all__ = [
    "AgentResult",
    "Planner",
    "ToolSpec",
    "call_tool",
    "default_registry",
    "run_agent",
    "summarize_bubble_point",
    "tool_schemas",
]

__version__ = "0.0.1"
