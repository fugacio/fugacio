"""Chemical-engineering design copilot for Fugacio (depends on ``fugacio.sim``).

The copilot exposes the differentiable engine to a language-model design agent
through:

* a **tool registry** (:func:`default_registry`, :func:`tool_schemas`,
  :func:`call_tool`) of deterministic, JSON-in/JSON-out engineering calculations
  spanning properties, unit operations, distillation, reactors, optimization,
  design specs, and economics;
* a vendor-neutral **LLM provider layer** (:class:`OpenAIProvider`,
  :class:`AnthropicProvider`, and the test :class:`MockProvider`) behind the
  optional ``llm`` extra;
* a model-agnostic **agent loop** (:func:`run_agent`) plus a real multi-turn
  function-calling loop (:func:`run_llm_agent`), with planner adapters
  (:func:`llm_planner`, :func:`heuristic_planner`);
* human-readable **reports** (:func:`summarize_bubble_point`, and the richer
  markdown summaries in :mod:`fugacio.copilot.report`).
"""

from fugacio.copilot.agent import (
    DEFAULT_SYSTEM_PROMPT,
    AgentResult,
    Planner,
    heuristic_planner,
    llm_planner,
    run_agent,
    run_llm_agent,
)
from fugacio.copilot.llm import (
    AnthropicProvider,
    ChatResponse,
    LLMProvider,
    Message,
    MockProvider,
    OpenAIProvider,
    ToolCall,
)
from fugacio.copilot.report import (
    stream_table,
    summarize_bubble_point,
    summarize_economics,
    summarize_heat_integration,
    summarize_lqr_design,
    summarize_mpc_simulation,
    summarize_optimization,
    summarize_pid_tuning,
    summarize_transcript,
)
from fugacio.copilot.tools import (
    ToolSpec,
    call_tool,
    default_registry,
    tool_schemas,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AgentResult",
    "AnthropicProvider",
    "ChatResponse",
    "LLMProvider",
    "Message",
    "MockProvider",
    "OpenAIProvider",
    "Planner",
    "ToolCall",
    "ToolSpec",
    "call_tool",
    "default_registry",
    "heuristic_planner",
    "llm_planner",
    "run_agent",
    "run_llm_agent",
    "stream_table",
    "summarize_bubble_point",
    "summarize_economics",
    "summarize_heat_integration",
    "summarize_lqr_design",
    "summarize_mpc_simulation",
    "summarize_optimization",
    "summarize_pid_tuning",
    "summarize_transcript",
    "tool_schemas",
]

__version__ = "0.0.1"
