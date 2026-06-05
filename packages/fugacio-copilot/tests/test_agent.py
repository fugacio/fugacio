"""The model-agnostic agent loop, driven by a scripted (deterministic) planner."""

from typing import Any

from fugacio.copilot import AgentResult, run_agent


def _scripted_planner(script: list[dict[str, Any]]):
    """Return a planner that emits each scripted decision in order."""
    calls = {"n": 0}

    def planner(goal: str, schemas: list[dict], transcript: list[dict]) -> dict:
        decision = script[calls["n"]]
        calls["n"] += 1
        return decision

    return planner


def test_agent_runs_tools_then_answers() -> None:
    planner = _scripted_planner(
        [
            {
                "tool": "saturation_pressure",
                "arguments": {"component": "propane", "temperature": 300.0},
            },
            {"final_answer": "Propane saturation pressure computed."},
        ]
    )
    result = run_agent("What is the saturation pressure of propane at 300 K?", planner)
    assert isinstance(result, AgentResult)
    assert result.answer == "Propane saturation pressure computed."
    assert len(result.transcript) == 1
    assert result.transcript[0]["tool"] == "saturation_pressure"
    assert result.transcript[0]["result"]["psat_pa"] > 0.0


def test_agent_passes_tool_results_into_transcript() -> None:
    seen: dict[str, Any] = {}

    def planner(goal: str, schemas: list[dict], transcript: list[dict]) -> dict:
        if not transcript:
            return {"tool": "list_components", "arguments": {}}
        seen["transcript"] = transcript
        return {"final_answer": "done"}

    run_agent("list components", planner)
    assert "water" in seen["transcript"][0]["result"]["components"]


def test_agent_respects_step_budget() -> None:
    def planner(goal: str, schemas: list[dict], transcript: list[dict]) -> dict:
        return {"tool": "list_components", "arguments": {}}

    result = run_agent("loop forever", planner, max_steps=3)
    assert "budget" in result.answer
    assert len(result.transcript) == 3
