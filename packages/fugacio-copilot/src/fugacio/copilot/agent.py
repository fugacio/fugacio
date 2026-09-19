"""Tool-calling agent loops for the Fugacio design copilot.

Two complementary loops share one tool registry:

* `run_agent`: the original *model-agnostic* loop. A **planner** callable
  ``(goal, tool_schemas, transcript) -> decision`` decides the next tool call or
  the final answer, where ``decision`` is ``{"tool": name, "arguments": {...}}``
  or ``{"final_answer": text}``, optionally with a ``"stop_reason"`` that
  explains an early stop. This keeps the control flow fully testable with a
  scripted planner and lets any decision policy drop in.

* `run_llm_agent`: a real multi-turn function-calling loop over an
  `LLMProvider` (OpenAI, Anthropic, or the test `MockProvider`). It maintains
  the full message history with tool-call ids and provider blocks, executes
  every requested tool (validating arguments and capturing errors so the model
  can self-correct), feeds the JSON results back with failures flagged, and
  returns when the model answers in plain text. A refused or truncated reply
  ends the loop with an explicit outcome instead of an answer.

`llm_planner` bridges the two: it turns a provider into a planner for the
simple loop. A deterministic `heuristic_planner` is provided for tests and
offline use.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fugacio.copilot.llm.base import (
    ARGUMENTS_ERROR,
    INVALID_ARGUMENTS,
    LLMProvider,
    Message,
    ToolCall,
)
from fugacio.copilot.tools import ToolSpec, call_tool, default_registry, tool_schemas
from fugacio.thermo.diagnostics import ConvergenceError

JsonDict = dict[str, Any]
Planner = Callable[[str, list[JsonDict], list[JsonDict]], JsonDict]

#: Default system prompt grounding the model as a chemical-engineering copilot.
DEFAULT_SYSTEM_PROMPT = (
    "You are Fugacio, an expert chemical-process design copilot. You answer "
    "engineering questions by calling the provided tools, which run a rigorous, "
    "differentiable thermodynamics and flowsheet engine. Prefer computing with "
    "tools over estimating from memory. Use SI units (kelvin, pascal, mol/s, "
    "watts, dollars). When you have enough information, give a concise, "
    "quantitative final answer that cites the numbers the tools returned. "
    "Treat errors and failed convergence reports as failed calculations; explain "
    "the reported cause and adjust the inputs before using those results."
)

#: Answers for model stop reasons that end a loop without a usable reply.
_STOPPED_ANSWERS = {
    "refusal": (
        "The model declined the request (stop reason: refusal); its reply was "
        "discarded, so there's no answer."
    ),
    "max_tokens": (
        "The model's reply hit the max_tokens limit before it finished (stop "
        "reason: max_tokens); raise max_tokens and retry."
    ),
}
#: Resends allowed for a paused turn within one `llm_planner` decision.
_MAX_PAUSE_RESUMES = 4
#: Keys of the error envelope `_safe_call` returns for a failed tool call.
_ERROR_KEYS = frozenset({"error", "context", "report"})


@dataclass(frozen=True)
class AgentResult:
    """Outcome of an agent run.

    Attributes:
        answer: The final natural-language (or structured) answer.
        transcript: Ordered tool calls with their arguments and results.
        stop_reason: Why the loop ended: ``"answer"``, ``"budget"``,
            ``"refusal"`` (the model declined, so there's no answer), or
            ``"max_tokens"`` (a reply hit the token cap before it finished).
        steps: Number of planner / model turns taken.
    """

    answer: str
    transcript: list[JsonDict] = field(default_factory=list)
    stop_reason: str = "answer"
    steps: int = 0


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
        planner: Decision function (see module docstring); inject an LLM via
            `llm_planner`, or pass a scripted/heuristic planner. A final
            decision's optional ``"stop_reason"`` becomes the result's.
        registry: Tool registry to expose (defaults to `default_registry`).
        max_steps: Maximum number of tool calls before giving up.

    Returns:
        An `AgentResult` with the final answer and the full transcript.
    """
    registry = default_registry() if registry is None else registry
    schemas = tool_schemas(registry)
    transcript: list[JsonDict] = []
    for step in range(max_steps):
        decision = planner(goal, schemas, transcript)
        if "final_answer" in decision:
            return AgentResult(
                answer=str(decision["final_answer"]),
                transcript=transcript,
                stop_reason=str(decision.get("stop_reason", "answer")),
                steps=step + 1,
            )
        name = decision["tool"]
        arguments = decision.get("arguments", {})
        result = call_tool(name, arguments, registry)
        transcript.append({"tool": name, "arguments": arguments, "result": result})
    return AgentResult(
        answer="step budget exhausted before a final answer",
        transcript=transcript,
        stop_reason="budget",
        steps=max_steps,
    )


def run_llm_agent(
    goal: str,
    provider: LLMProvider,
    *,
    registry: dict[str, ToolSpec] | None = None,
    system: str = DEFAULT_SYSTEM_PROMPT,
    max_steps: int = 8,
    max_tokens: int = 16000,
) -> AgentResult:
    """Drive a real function-calling LLM through the tool registry to an answer.

    Maintains the full chat history with tool-call ids and provider blocks,
    executes each requested tool (validating arguments and turning errors into a
    JSON error result, flagged as an error, that the model can recover from),
    and loops until the model replies with plain text or the step budget is
    exhausted. A ``"refusal"`` or ``"max_tokens"`` reply ends the loop with that
    stop reason: its content never becomes the answer, and its tool calls never
    run. A ``"pause_turn"`` reply is resent so the model can resume it. No
    temperature is sent, so the provider's sampling defaults apply.

    Args:
        goal: The user's natural-language request.
        provider: Any `LLMProvider`.
        registry: Tool registry (defaults to `default_registry`).
        system: System prompt; defaults to `DEFAULT_SYSTEM_PROMPT`.
        max_steps: Maximum model turns.
        max_tokens: Per-turn token cap (thinking included, where it applies).

    Returns:
        An `AgentResult`.
    """
    registry = default_registry() if registry is None else registry
    schemas = tool_schemas(registry)
    messages: list[Message] = [Message.system(system), Message.user(goal)]
    transcript: list[JsonDict] = []

    for step in range(max_steps):
        reply = provider.chat(messages, tools=schemas, max_tokens=max_tokens)
        reason = reply.stop_reason or ""
        if reason in _STOPPED_ANSWERS:
            return AgentResult(
                answer=_STOPPED_ANSWERS[reason],
                transcript=transcript,
                stop_reason=reason,
                steps=step + 1,
            )
        if not reply.has_tool_calls and reason != "pause_turn":
            return AgentResult(
                answer=reply.content,
                transcript=transcript,
                stop_reason="answer",
                steps=step + 1,
            )
        # A paused turn without tool calls resumes when the conversation is resent.
        messages.append(Message.assistant(reply.content, reply.tool_calls, reply.provider_blocks))
        for call in reply.tool_calls:
            result = _safe_call(call.name, call.arguments, registry)
            transcript.append({"tool": call.name, "arguments": call.arguments, "result": result})
            messages.append(
                Message.tool(
                    json.dumps(result), call.id, call.name, is_error=_is_tool_error(result)
                )
            )

    return AgentResult(
        answer="step budget exhausted before a final answer",
        transcript=transcript,
        stop_reason="budget",
        steps=max_steps,
    )


def _safe_call(name: str, arguments: JsonDict, registry: dict[str, ToolSpec]) -> JsonDict:
    """Execute a tool, returning a structured ``{"error": ...}`` result on failure.

    Surfacing the error to the model (rather than raising) lets the agent recover
    from a hallucinated tool name, unparseable or missing arguments, or a failed
    calculation on the next turn. `_is_tool_error` recognizes these results.
    """
    if name not in registry:
        return {"error": f"unknown tool {name!r}; available: {sorted(registry)}"}
    if INVALID_ARGUMENTS in arguments:
        return {
            "error": (
                f"arguments for {name!r} aren't a JSON object "
                f"({arguments.get(ARGUMENTS_ERROR)}): {arguments[INVALID_ARGUMENTS]!r}; "
                "resend the call with a JSON object that matches the tool's schema"
            )
        }
    missing = _missing_required(registry[name], arguments)
    if missing:
        return {"error": f"missing required arguments for {name!r}: {missing}"}
    from fugacio.thermo.acceptance import PhysicalAcceptanceError

    try:
        return call_tool(name, arguments, registry)
    except (ConvergenceError, PhysicalAcceptanceError) as exc:
        return {"error": str(exc), "context": exc.context, "report": exc.report.to_dict()}
    except Exception as exc:  # report any tool failure back to the model
        return {"error": f"{type(exc).__name__}: {exc}"}


def _is_tool_error(result: JsonDict) -> bool:
    """Whether ``result`` is the error envelope `_safe_call` returns for a failed call."""
    return "error" in result and result.keys() <= _ERROR_KEYS


def _missing_required(spec: ToolSpec, arguments: JsonDict) -> list[str]:
    """Names of required schema parameters absent from ``arguments``."""
    required = spec.parameters.get("required", [])
    return [key for key in required if key not in arguments]


def llm_planner(
    provider: LLMProvider,
    *,
    system: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 16000,
) -> Planner:
    """Adapt an `LLMProvider` into a `Planner` for `run_agent`.

    On each call it reconstructs the conversation from the goal and transcript,
    asks the model for the next step, and maps the first requested tool call to a
    ``{"tool", "arguments"}`` decision (or the text reply to ``final_answer``).
    A paused reply is resent so the model can resume it. A refused or truncated
    reply, or one still paused after the resends, becomes a ``final_answer``
    that explains the stop, with its ``stop_reason``. No temperature is sent, so
    the provider's sampling defaults apply.
    """

    def planner(goal: str, schemas: list[JsonDict], transcript: list[JsonDict]) -> JsonDict:
        messages = _rebuild_messages(goal, transcript, system)
        reply = provider.chat(messages, tools=schemas, max_tokens=max_tokens)
        for _ in range(_MAX_PAUSE_RESUMES):
            if reply.stop_reason != "pause_turn" or reply.has_tool_calls:
                break
            messages.append(Message.assistant(reply.content, (), reply.provider_blocks))
            reply = provider.chat(messages, tools=schemas, max_tokens=max_tokens)
        reason = reply.stop_reason or ""
        if reason in _STOPPED_ANSWERS:
            return {"final_answer": _STOPPED_ANSWERS[reason], "stop_reason": reason}
        if reply.has_tool_calls:
            call = reply.tool_calls[0]
            return {"tool": call.name, "arguments": call.arguments}
        if reason == "pause_turn":
            return {
                "final_answer": "The model kept pausing its turn without finishing it.",
                "stop_reason": reason,
            }
        return {"final_answer": reply.content}

    return planner


def _rebuild_messages(goal: str, transcript: list[JsonDict], system: str) -> list[Message]:
    """Reconstruct a chat history (with synthetic call ids) from a flat transcript."""
    messages: list[Message] = [Message.system(system), Message.user(goal)]
    for i, entry in enumerate(transcript):
        call_id = f"call_{i}"
        messages.append(
            Message.assistant(
                "", [ToolCall(id=call_id, name=entry["tool"], arguments=entry["arguments"])]
            )
        )
        result = entry["result"]
        messages.append(
            Message.tool(
                json.dumps(result), call_id, entry["tool"], is_error=_is_tool_error(result)
            )
        )
    return messages


def heuristic_planner(rules: Sequence[tuple[str, JsonDict]], *, default_answer: str) -> Planner:
    """A deterministic keyword planner: fire the first rule whose keyword is in the goal.

    Each rule is ``(keyword, decision)``; the first ``keyword`` found in the goal
    (case-insensitive) triggers its ``decision`` once. After its tool runs (or if
    no rule matches), the planner returns ``default_answer``. Useful for offline
    demos and tests without an LLM.
    """

    def planner(goal: str, schemas: list[JsonDict], transcript: list[JsonDict]) -> JsonDict:
        if transcript:
            return {"final_answer": default_answer}
        lowered = goal.lower()
        for keyword, decision in rules:
            if keyword.lower() in lowered:
                return decision
        return {"final_answer": default_answer}

    return planner
