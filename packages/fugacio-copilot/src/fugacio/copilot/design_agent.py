"""A design-agent loop whose completion is grounded in computed case artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fugacio.copilot.agent import _is_tool_error, _safe_call
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

#: Answers for model stop reasons that end the loop without a submitted design.
_STOPPED_ANSWERS = {
    "refusal": "The model declined the request (stop reason: refusal), so no design was submitted.",
    "max_tokens": (
        "A model reply hit the max_tokens limit before it finished (stop reason: "
        "max_tokens), so no design was submitted. Raise max_tokens and retry."
    ),
}


@dataclass(frozen=True)
class DesignAgentResult:
    """Deterministic report and structured evidence from a submitted design.

    Attributes:
        answer: The deterministic design report, or why no design was submitted.
        design: The submitted design and its evidence, if one was submitted.
        transcript: Tool calls with their arguments and results, plus events
            such as ``unverified_text`` that record model text never accepted
            as an answer.
        stop_reason: ``"submitted"``, ``"budget"``, ``"refusal"`` (the model
            declined), or ``"max_tokens"`` (a reply hit the token cap).
        steps: Number of model turns taken.
    """

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
    max_tokens: int = 16000,
) -> DesignAgentResult:
    """Plan and compute a process design; ignore unsupported final-text claims.

    The general ``run_llm_agent`` remains suitable for explanatory questions.
    This stricter loop requires an accepted artifact and generates its answer
    from recorded values. Both model-turn and tool-call budgets are bounded. A
    refused or truncated reply ends the loop with that stop reason and no
    design; its content is recorded as an unaccepted event, and its tool calls
    never run. A paused reply is resent so the model can resume it. No
    temperature is sent, so the provider's sampling defaults apply.
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
        reply = provider.chat(messages, tools=schemas, max_tokens=max_tokens)
        reason = reply.stop_reason or ""
        if reason in _STOPPED_ANSWERS:
            transcript.append({"event": reason, "content": reply.content, "accepted": False})
            return DesignAgentResult(
                _STOPPED_ANSWERS[reason], transcript=transcript, stop_reason=reason, steps=step + 1
            )
        messages.append(Message.assistant(reply.content, reply.tool_calls, reply.provider_blocks))
        if not reply.has_tool_calls:
            if reason == "pause_turn":
                continue  # resending the conversation resumes the paused turn
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
            messages.append(
                Message.tool(
                    json.dumps(result, allow_nan=False),
                    call.id,
                    call.name,
                    is_error=_is_tool_error(result),
                )
            )
        # Process the entire requested tool batch first: a later case update
        # invalidates an earlier submission in the same model response.
        if session.pending is not None:
            design = session.pending
            return DesignAgentResult(design["report"], design, transcript, "submitted", step + 1)
    return DesignAgentResult(
        "Turn budget exhausted without a submitted design.", transcript=transcript, steps=max_steps
    )
