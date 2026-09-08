"""A design-agent loop whose completion is grounded in computed case artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fugacio.copilot.agent import _safe_call
from fugacio.copilot.case_tools import DesignSession
from fugacio.copilot.llm.base import LLMProvider, Message
from fugacio.copilot.tools import tool_schemas
from fugacio.sim.cases import CaseWorkspace

DESIGN_SYSTEM_PROMPT = (
    "You are Fugacio's process-design copilot. Build or load a portable case, "
    "use dimensioned quantities, run it, inspect acceptance and evidence, and "
    "perform studies when required by the user's goal. Call case_format for the "
    "supported vocabulary. Finish only by calling submit_design with recorded "
    "metric names from an accepted run of the current case revision. Numerical "
    "convergence isn't physical acceptance or empirical qualification. Never "
    "claim a whole plant is empirically qualified. Plain text is planning input "
    "and cannot serve as the final engineering answer. Tool errors are failed "
    "actions; fix their cause before submitting. Only the explicit workspace "
    "and registered tool operations are available."
)


@dataclass(frozen=True)
class DesignAgentResult:
    """Deterministic report and structured evidence from a submitted design."""

    answer: str
    design: dict[str, Any] | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "budget"
    steps: int = 0


def run_design_agent(
    goal: str,
    provider: LLMProvider,
    *,
    workspace: CaseWorkspace,
    max_steps: int = 12,
    max_calls: int = 40,
    max_tokens: int = 4096,
) -> DesignAgentResult:
    """Plan and compute a process design; ignore unsupported final-text claims.

    The general ``run_llm_agent`` remains suitable for explanatory questions.
    This stricter loop requires an accepted artifact and generates its answer
    from recorded values. Both model-turn and tool-call budgets are bounded.
    """
    if not 1 <= max_steps <= 100 or not 1 <= max_calls <= 200:
        raise ValueError("invalid design-agent budgets")
    session = DesignSession(workspace)
    registry = session.registry()
    schemas = tool_schemas(registry)
    messages = [Message.system(DESIGN_SYSTEM_PROMPT), Message.user(goal)]
    transcript: list[dict[str, Any]] = []
    count = 0
    for step in range(max_steps):
        reply = provider.chat(messages, tools=schemas, temperature=0.0, max_tokens=max_tokens)
        messages.append(Message.assistant(reply.content, reply.tool_calls))
        if not reply.has_tool_calls:
            transcript.append(
                {"event": "unverified_text", "content": reply.content, "accepted": False}
            )
            messages.append(
                Message.user(
                    "No design has been submitted. Compute an accepted current-case "
                    "run and call submit_design; text alone cannot establish "
                    "engineering results."
                )
            )
            continue
        for call in reply.tool_calls:
            if count >= max_calls:
                return DesignAgentResult(
                    "Tool-call budget exhausted without a submitted design.",
                    transcript=transcript,
                    steps=step + 1,
                )
            result = _safe_call(call.name, call.arguments, registry)
            count += 1
            transcript.append({"tool": call.name, "arguments": call.arguments, "result": result})
            messages.append(Message.tool(json.dumps(result, allow_nan=False), call.id, call.name))
        # Process the entire requested tool batch first: a later case update
        # invalidates an earlier submission in the same model response.
        if session.pending is not None:
            design = session.pending
            return DesignAgentResult(design["report"], design, transcript, "submitted", step + 1)
    return DesignAgentResult(
        "Turn budget exhausted without a submitted design.", transcript=transcript, steps=max_steps
    )
