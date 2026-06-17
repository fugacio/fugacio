"""Real multi-turn tool-calling loop, exercised with a deterministic mock provider.

Verifies that the LLM agent calls the requested tool, feeds the JSON result back
into the conversation, and returns the model's final text, and that bad tool
calls are turned into recoverable error results rather than crashing the loop.
"""

from fugacio.copilot import run_llm_agent
from fugacio.copilot.agent import llm_planner, run_agent
from fugacio.copilot.llm import ChatResponse, MockProvider, ToolCall, openai_tools
from fugacio.copilot.tools import tool_schemas


def test_run_llm_agent_calls_tool_then_answers() -> None:
    provider = MockProvider(
        script=[
            ChatResponse(
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="saturation_pressure",
                        arguments={"component": "propane", "temperature": 300.0},
                    ),
                )
            ),
            ChatResponse(content="Propane Psat at 300 K is about 9.97 bar."),
        ]
    )
    result = run_llm_agent("Psat of propane at 300 K?", provider)
    assert result.stop_reason == "answer"
    assert "propane" in result.answer.lower()
    assert len(result.transcript) == 1
    assert result.transcript[0]["tool"] == "saturation_pressure"
    assert result.transcript[0]["result"]["psat_pa"] > 0.0
    # The tool result was fed back as a tool-role message before the final turn.
    second_call_messages = provider.calls[1][0]
    assert any(m.role == "tool" for m in second_call_messages)


def test_run_llm_agent_recovers_from_bad_tool_call() -> None:
    provider = MockProvider(
        script=[
            ChatResponse(tool_calls=(ToolCall(id="c1", name="does_not_exist", arguments={}),)),
            ChatResponse(content="Recovered and answered."),
        ]
    )
    result = run_llm_agent("trigger an error", provider)
    assert "error" in result.transcript[0]["result"]
    assert result.answer == "Recovered and answered."


def test_run_llm_agent_reports_missing_required_arguments() -> None:
    provider = MockProvider(
        script=[
            ChatResponse(tool_calls=(ToolCall(id="c1", name="saturation_pressure", arguments={}),)),
            ChatResponse(content="done"),
        ]
    )
    result = run_llm_agent("bad args", provider)
    assert "missing required" in result.transcript[0]["result"]["error"]


def test_run_llm_agent_respects_step_budget() -> None:
    # Always asks for a tool, never answers -> budget exhausts.
    provider = MockProvider(
        script=lambda messages: ChatResponse(
            tool_calls=(ToolCall(id="c", name="list_components", arguments={}),)
        )
    )
    result = run_llm_agent("loop", provider, max_steps=3)
    assert result.stop_reason == "budget"
    assert len(result.transcript) == 3


def test_llm_planner_bridges_into_simple_loop() -> None:
    provider = MockProvider(
        script=[
            ChatResponse(tool_calls=(ToolCall(id="c1", name="list_components", arguments={}),)),
            ChatResponse(content="There are many components."),
        ]
    )
    result = run_agent("list components", llm_planner(provider))
    assert result.answer == "There are many components."
    assert result.transcript[0]["tool"] == "list_components"


def test_openai_tool_envelope_shape() -> None:
    wrapped = openai_tools(tool_schemas())
    assert wrapped[0]["type"] == "function"
    assert "name" in wrapped[0]["function"]
    assert "parameters" in wrapped[0]["function"]
