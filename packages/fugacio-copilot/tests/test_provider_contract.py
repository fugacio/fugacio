"""Provider contract for current models: request shapes, replay rules, and stop reasons.

The provider SDKs aren't installed in the test environment, so these tests pass
duck-typed fake clients that record each request's keyword arguments and replay
scripted responses built from plain objects.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fugacio.copilot import (
    DesignAgentResult,
    run_design_agent,
    run_llm_agent,
    summarize_transcript,
)
from fugacio.copilot.agent import llm_planner, run_agent
from fugacio.copilot.llm import (
    ARGUMENTS_ERROR,
    INVALID_ARGUMENTS,
    AnthropicProvider,
    ChatResponse,
    Message,
    MockProvider,
    OpenAIProvider,
    ToolCall,
)
from fugacio.copilot.tools import ToolSpec
from fugacio.sim.cases import CaseWorkspace

SCHEMA: dict[str, Any] = {
    "name": "echo",
    "description": "Echo text back.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}
THINKING = {"type": "thinking", "thinking": "", "signature": "sig"}


def echo_registry() -> tuple[dict[str, ToolSpec], list[str]]:
    """A one-tool registry plus the list of texts its tool actually echoed."""
    ran: list[str] = []

    def echo(text: str) -> dict[str, Any]:
        ran.append(text)
        return {"echo": text}

    spec = ToolSpec(SCHEMA["name"], SCHEMA["description"], SCHEMA["parameters"], echo)
    return {"echo": spec}, ran


# -- Anthropic fakes ---------------------------------------------------------


class FakeAnthropic:
    """Records ``messages.create`` and ``beta.messages.create`` calls; replays replies."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.messages = SimpleNamespace(create=self._endpoint("messages"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._endpoint("beta")))

    def _endpoint(self, name: str) -> Callable[..., Any]:
        def create(**kwargs: Any) -> Any:
            self.requests.append((name, copy.deepcopy(kwargs)))
            return self.replies.pop(0)

        return create


class SdkBlock(SimpleNamespace):
    """A stand-in for an SDK pydantic content block (converted with ``model_dump``)."""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return dict(vars(self))


def claude(*blocks: Any, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool_use(call_id: str, name: str = "echo", **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def thinking(signature: str) -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking="", signature=signature)


FALLBACK = SimpleNamespace(type="fallback")


# -- OpenAI fakes ------------------------------------------------------------


class FakeOpenAI:
    """Records ``chat.completions.create`` calls and replays replies."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs))
        return self.replies.pop(0)


def gpt(
    content: str | None = None,
    *tool_calls: Any,
    finish_reason: str = "stop",
    refusal: str | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls) or None, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def function_call(call_id: str, arguments: str, name: str = "echo") -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, type="function", function=function)


# -- Anthropic request shape -------------------------------------------------


def test_anthropic_defaults_to_opus_5_with_fallbacks_and_no_sampling_parameters() -> None:
    client = FakeAnthropic(claude(text("Hello.")))
    reply = AnthropicProvider(client=client).chat(
        [Message.system("Be brief."), Message.user("Hi")], tools=[SCHEMA]
    )
    endpoint, request = client.requests[0]
    assert endpoint == "beta"
    assert request["model"] == "claude-opus-5"
    assert request["max_tokens"] == 16000
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["extra_body"] == {"fallbacks": "default"}
    assert not {"temperature", "top_p", "top_k"} & request.keys()
    assert request["system"] == "Be brief."
    assert request["tools"] == [
        {"name": "echo", "description": "Echo text back.", "input_schema": SCHEMA["parameters"]}
    ]
    assert request["messages"] == [{"role": "user", "content": "Hi"}]
    assert (reply.content, reply.stop_reason) == ("Hello.", "end_turn")


def test_anthropic_without_fallbacks_uses_the_standard_endpoint() -> None:
    client = FakeAnthropic(claude(text("Hello.")))
    AnthropicProvider(client=client, fallbacks=None).chat([Message.user("Hi")])
    endpoint, request = client.requests[0]
    assert endpoint == "messages"
    assert not {"betas", "extra_body", "temperature", "top_p", "top_k"} & request.keys()
    assert (request["model"], request["max_tokens"]) == ("claude-opus-5", 16000)


@pytest.mark.parametrize("fallbacks", ["default", None])
def test_anthropic_forwards_an_explicit_temperature(fallbacks: str | None) -> None:
    client = FakeAnthropic(claude(text("Hello.")))
    provider = AnthropicProvider("claude-sonnet-4-6", client=client, fallbacks=fallbacks)
    provider.chat([Message.user("Hi")], temperature=0.3, max_tokens=512)
    _, request = client.requests[0]
    assert (request["temperature"], request["max_tokens"]) == (0.3, 512)
    assert request["model"] == "claude-sonnet-4-6"


# -- Anthropic replay rules --------------------------------------------------


def test_anthropic_groups_one_turns_tool_results_into_one_user_message() -> None:
    client = FakeAnthropic(claude(text("Done.")))
    AnthropicProvider(client=client).chat(
        [
            Message.user("Go"),
            Message.assistant("", [ToolCall("c1", "echo", {"text": "a"}), ToolCall("c2", "x")]),
            Message.tool('{"echo": "a"}', "c1", "echo"),
            Message.tool('{"error": "unknown tool"}', "c2", "x", is_error=True),
            Message.user("Thanks"),
        ]
    )
    messages = client.requests[0][1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "user"]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": '{"echo": "a"}'},
        {
            "type": "tool_result",
            "tool_use_id": "c2",
            "content": '{"error": "unknown tool"}',
            "is_error": True,
        },
    ]


def test_llm_agent_sends_a_turns_results_together_and_flags_the_failed_call() -> None:
    registry, ran = echo_registry()
    client = FakeAnthropic(
        claude(tool_use("c1", text="hi"), tool_use("c2", name="missing"), stop_reason="tool_use"),
        claude(text("Echoed hi.")),
    )
    result = run_llm_agent("Echo hi", AnthropicProvider(client=client), registry=registry)
    assert (result.answer, result.stop_reason, ran) == ("Echoed hi.", "answer", ["hi"])
    followup = client.requests[1][1]["messages"]
    assert [m["role"] for m in followup] == ["user", "assistant", "user"]
    ok, failed = followup[2]["content"]
    assert ok["tool_use_id"] == "c1" and "is_error" not in ok
    assert json.loads(ok["content"]) == {"echo": "hi"}
    assert failed["tool_use_id"] == "c2" and failed["is_error"] is True
    assert "unknown tool" in json.loads(failed["content"])["error"]


def test_thinking_blocks_are_replayed_verbatim_before_text_and_tool_use() -> None:
    registry, _ = echo_registry()
    client = FakeAnthropic(
        claude(
            SdkBlock(type="thinking", thinking="", signature="sig-1"),
            SimpleNamespace(type="redacted_thinking", data="opaque"),
            text("Let me check."),
            tool_use("c1", text="hi"),
            stop_reason="tool_use",
        ),
        claude(text("Done.")),
    )
    run_llm_agent("Echo hi", AnthropicProvider(client=client), registry=registry)
    assistant = client.requests[1][1]["messages"][1]
    assert assistant == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig-1"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "c1", "name": "echo", "input": {"text": "hi"}},
        ],
    }


def test_blocks_before_the_final_fallback_block_are_dropped() -> None:
    client = FakeAnthropic(
        claude(
            thinking("declined-1"),
            tool_use("c0", text="early"),
            text("Partial. "),
            FALLBACK,
            thinking("declined-2"),
            SimpleNamespace(type="redacted_thinking", data="declined-3"),
            tool_use("c1", text="middle"),
            FALLBACK,
            thinking("kept"),
            text("Final."),
            tool_use("c2", text="late"),
            stop_reason="tool_use",
        ),
        claude(text("Done.")),
    )
    provider = AnthropicProvider(client=client)
    reply = provider.chat([Message.user("Go")])
    assert reply.provider_blocks == ({"type": "thinking", "thinking": "", "signature": "kept"},)
    assert [call.id for call in reply.tool_calls] == ["c2"]
    assert reply.content == "Partial. Final."
    provider.chat(
        [
            Message.user("Go"),
            Message.assistant(reply.content, reply.tool_calls, reply.provider_blocks),
            Message.tool("{}", "c2", "echo"),
        ]
    )
    replayed = client.requests[1][1]["messages"][1]["content"]
    assert [block["type"] for block in replayed] == ["thinking", "text", "tool_use"]
    assert replayed[0]["signature"] == "kept" and replayed[2]["id"] == "c2"


def test_empty_assistant_text_is_never_sent() -> None:
    client = FakeAnthropic(claude(text("OK.")))
    AnthropicProvider(client=client).chat(
        [
            Message.user("First"),
            Message.assistant(""),
            Message.assistant("", (), (THINKING,)),
            Message.user("Second"),
            Message.assistant("", [ToolCall("c1", "echo", {"text": "a"})], (THINKING,)),
            Message.tool("{}", "c1", "echo"),
        ]
    )
    messages = client.requests[0][1]["messages"]
    assert [m["role"] for m in messages] == ["user", "user", "assistant", "user"]
    assert [block["type"] for block in messages[2]["content"]] == ["thinking", "tool_use"]


def test_design_agent_never_sends_an_empty_assistant_turn(tmp_path: Path) -> None:
    client = FakeAnthropic(claude(), claude(stop_reason="refusal"))
    result = run_design_agent(
        "Design a heater", AnthropicProvider(client=client), workspace=CaseWorkspace(tmp_path)
    )
    assert result.stop_reason == "refusal"
    followup = client.requests[1][1]["messages"]
    assert [m["role"] for m in followup] == ["user", "user"]  # goal, then the nudge


# -- Anthropic stop reasons --------------------------------------------------


def test_anthropic_refusal_discards_content_and_max_tokens_drops_calls() -> None:
    client = FakeAnthropic(
        claude(
            thinking("s"), text("Partial answer"), tool_use("c1", text="x"), stop_reason="refusal"
        ),
        claude(thinking("s"), text("Cut off"), tool_use("c2", text="x"), stop_reason="max_tokens"),
    )
    provider = AnthropicProvider(client=client)
    refused = provider.chat([Message.user("Go")])
    assert refused.stop_reason == "refusal"
    assert (refused.content, refused.tool_calls, refused.provider_blocks) == ("", (), ())
    truncated = provider.chat([Message.user("Go")])
    assert truncated.stop_reason == "max_tokens"
    assert truncated.content == "Cut off" and not truncated.has_tool_calls


def test_llm_agent_resends_a_paused_anthropic_turn() -> None:
    client = FakeAnthropic(
        claude(thinking("p"), text("Searching."), stop_reason="pause_turn"),
        claude(text("Answer.")),
    )
    result = run_llm_agent("Go", AnthropicProvider(client=client), registry=echo_registry()[0])
    assert (result.answer, result.stop_reason, result.steps) == ("Answer.", "answer", 2)
    resent = client.requests[1][1]["messages"]
    assert resent[-1] == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "p"},
            {"type": "text", "text": "Searching."},
        ],
    }


# -- Agent loops: stop reasons, provider blocks, and error flags ------------


@pytest.mark.parametrize("reason", ["refusal", "max_tokens"])
def test_llm_agent_stops_explicitly_on_refusal_and_max_tokens(reason: str) -> None:
    registry, ran = echo_registry()
    provider = MockProvider(
        [
            ChatResponse(
                content="Claimed answer",
                tool_calls=(ToolCall("c1", "echo", {"text": "x"}),),
                stop_reason=reason,
            ),
            ChatResponse(content="unreachable"),
        ]
    )
    result = run_llm_agent("Go", provider, registry=registry)
    assert (result.stop_reason, result.steps) == (reason, 1)
    assert "Claimed answer" not in result.answer and reason in result.answer
    assert result.transcript == [] and ran == []
    assert len(provider.calls) == 1


def test_llm_agent_resends_a_paused_turn() -> None:
    provider = MockProvider(
        [
            ChatResponse(
                content="Still working.", stop_reason="pause_turn", provider_blocks=(THINKING,)
            ),
            ChatResponse(content="Done.", stop_reason="end_turn"),
        ]
    )
    result = run_llm_agent("Go", provider, registry=echo_registry()[0])
    assert (result.answer, result.stop_reason, result.steps) == ("Done.", "answer", 2)
    resent = provider.calls[1][0]
    assert [m.role for m in resent] == ["system", "user", "assistant"]
    assert resent[-1] == Message.assistant("Still working.", (), (THINKING,))


def test_llm_agent_keeps_provider_blocks_and_flags_failed_calls() -> None:
    provider = MockProvider(
        [
            ChatResponse(
                tool_calls=(ToolCall("c1", "echo", {"text": "a"}), ToolCall("c2", "echo", {})),
                stop_reason="tool_use",
                provider_blocks=(THINKING,),
            ),
            ChatResponse(content="Done."),
        ]
    )
    run_llm_agent("Go", provider, registry=echo_registry()[0])
    assistant, ok, failed = provider.calls[1][0][-3:]
    assert assistant.provider_blocks == (THINKING,)
    assert (ok.is_error, failed.is_error) == (False, True)
    assert "missing required" in json.loads(failed.content)["error"]


@pytest.mark.parametrize("reason", ["refusal", "max_tokens"])
def test_design_agent_stops_explicitly_on_refusal_and_max_tokens(
    tmp_path: Path, reason: str
) -> None:
    claim = "Verified design costing $123."
    provider = MockProvider(
        [
            ChatResponse(
                content=claim,
                tool_calls=(ToolCall("s", "submit_design", {"run_id": "r", "metrics": ["duty"]}),),
                stop_reason=reason,
            )
        ]
    )
    result = run_design_agent("Design a heater", provider, workspace=CaseWorkspace(tmp_path))
    assert (result.stop_reason, result.design, result.steps) == (reason, None, 1)
    assert "$123" not in result.answer and reason in result.answer
    assert result.transcript == [{"event": reason, "content": claim, "accepted": False}]


def test_design_agent_resends_a_paused_turn_without_a_nudge(tmp_path: Path) -> None:
    provider = MockProvider(
        [
            ChatResponse(content="Planning.", stop_reason="pause_turn"),
            ChatResponse(stop_reason="refusal"),
        ]
    )
    result = run_design_agent("Design a heater", provider, workspace=CaseWorkspace(tmp_path))
    resent = provider.calls[1][0]
    assert [m.role for m in resent] == ["system", "user", "assistant"]
    assert resent[-1].content == "Planning."
    assert (result.stop_reason, result.steps) == ("refusal", 2)
    assert [entry["event"] for entry in result.transcript] == ["refusal"]


def test_design_agent_keeps_provider_blocks_and_flags_failed_calls(tmp_path: Path) -> None:
    provider = MockProvider(
        [
            ChatResponse(
                tool_calls=(ToolCall("c1", "list_cases", {}), ToolCall("c2", "no_such_tool", {})),
                stop_reason="tool_use",
                provider_blocks=(THINKING,),
            ),
            ChatResponse(stop_reason="refusal"),
        ]
    )
    run_design_agent("Design a heater", provider, workspace=CaseWorkspace(tmp_path))
    assistant, ok, failed = provider.calls[1][0][-3:]
    assert assistant.provider_blocks == (THINKING,)
    assert (ok.is_error, failed.is_error) == (False, True)


def test_agent_loops_send_no_temperature_and_a_16000_token_cap(tmp_path: Path) -> None:
    provider = MockProvider([ChatResponse(content="Answer.")])
    run_llm_agent("Go", provider, registry=echo_registry()[0])
    run_design_agent("Go", provider, workspace=CaseWorkspace(tmp_path), max_steps=1)
    run_agent("Go", llm_planner(provider), registry=echo_registry()[0])
    assert provider.call_kwargs == [{"temperature": None, "max_tokens": 16000}] * 3


def test_llm_planner_reports_a_refusal_instead_of_answering() -> None:
    provider = MockProvider([ChatResponse(content="Partial", stop_reason="refusal")])
    result = run_agent("Go", llm_planner(provider), registry=echo_registry()[0])
    assert result.stop_reason == "refusal" and "Partial" not in result.answer


def test_llm_planner_resumes_a_paused_turn() -> None:
    provider = MockProvider(
        [
            ChatResponse(content="Working.", stop_reason="pause_turn"),
            ChatResponse(content="Done.", stop_reason="end_turn"),
        ]
    )
    result = run_agent("Go", llm_planner(provider), registry=echo_registry()[0])
    assert (result.answer, result.stop_reason) == ("Done.", "answer")
    assert provider.calls[1][0][-1] == Message.assistant("Working.")


# -- OpenAI ------------------------------------------------------------------


def test_openai_uses_max_completion_tokens_and_no_default_temperature() -> None:
    client = FakeOpenAI(gpt("Hi."), gpt("Hi again."))
    provider = OpenAIProvider(client=client)
    reply = provider.chat([Message.user("Hi")], tools=[SCHEMA])
    provider.chat([Message.user("Hi")], temperature=0.2)
    first, second = client.requests
    assert first["model"] == "gpt-5-mini"
    assert first["max_completion_tokens"] == 16000
    assert not {"max_tokens", "temperature"} & first.keys()
    assert first["tools"][0]["function"]["name"] == "echo"
    assert second["temperature"] == 0.2
    assert (reply.content, reply.stop_reason) == ("Hi.", "end_turn")


@pytest.mark.parametrize(
    ("finish_reason", "stop_reason"),
    [
        ("stop", "end_turn"),
        ("tool_calls", "tool_use"),
        ("length", "max_tokens"),
        ("content_filter", "refusal"),
        ("function_call", "function_call"),
    ],
)
def test_openai_maps_finish_reasons(finish_reason: str, stop_reason: str) -> None:
    client = FakeOpenAI(gpt("Text", finish_reason=finish_reason))
    assert OpenAIProvider(client=client).chat([Message.user("Hi")]).stop_reason == stop_reason


def test_openai_refusal_and_truncation_are_not_trusted() -> None:
    client = FakeOpenAI(
        gpt(None, refusal="I can't help with that."),
        gpt("Filtered", finish_reason="content_filter"),
        gpt("Partial", function_call("c1", '{"text": "a"'), finish_reason="length"),
    )
    provider = OpenAIProvider(client=client)
    for _ in range(2):
        refused = provider.chat([Message.user("Hi")])
        assert (refused.stop_reason, refused.content, refused.tool_calls) == ("refusal", "", ())
    truncated = provider.chat([Message.user("Hi")])
    assert truncated.stop_reason == "max_tokens" and not truncated.has_tool_calls


def test_openai_marks_arguments_that_arent_a_json_object() -> None:
    client = FakeOpenAI(
        gpt(
            None,
            function_call("c1", '{"text": "hi"'),
            function_call("c2", "[1, 2]"),
            function_call("c3", ""),
            finish_reason="tool_calls",
        )
    )
    malformed, listed, empty = OpenAIProvider(client=client).chat([Message.user("Hi")]).tool_calls
    assert malformed.arguments[INVALID_ARGUMENTS] == '{"text": "hi"'
    assert "invalid JSON" in malformed.arguments[ARGUMENTS_ERROR]
    assert listed.arguments[INVALID_ARGUMENTS] == "[1, 2]"
    assert empty.arguments == {}


def test_openai_malformed_tool_arguments_become_an_error_result() -> None:
    registry, ran = echo_registry()
    client = FakeOpenAI(
        gpt(None, function_call("c1", '{"text": "hi"'), finish_reason="tool_calls"),
        gpt("Resent correctly.", finish_reason="stop"),
    )
    result = run_llm_agent("Echo hi", OpenAIProvider(client=client), registry=registry)
    assert (result.answer, ran) == ("Resent correctly.", [])
    error = result.transcript[0]["result"]["error"]
    assert "invalid JSON" in error and '{"text": "hi"' in error
    tool_message = client.requests[1]["messages"][-1]
    assert (tool_message["role"], tool_message["tool_call_id"]) == ("tool", "c1")
    assert tool_message["content"].startswith("ERROR: ")
    assert json.loads(tool_message["content"].removeprefix("ERROR: ")) == {"error": error}


def test_openai_marks_failed_tool_results_and_skips_empty_assistant_turns() -> None:
    client = FakeOpenAI(gpt("Done."))
    OpenAIProvider(client=client).chat(
        [
            Message.user("Go"),
            Message.assistant(""),
            Message.assistant("", [ToolCall("c1", "echo", {"text": "a"}), ToolCall("c2", "x")]),
            Message.tool('{"echo": "a"}', "c1", "echo"),
            Message.tool('{"error": "unknown tool"}', "c2", "x", is_error=True),
        ]
    )
    messages = client.requests[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool"]
    assert messages[2]["content"] == '{"echo": "a"}'
    assert messages[3]["content"] == 'ERROR: {"error": "unknown tool"}'


# -- Reporting ---------------------------------------------------------------


def test_summarize_transcript_renders_design_agent_events(tmp_path: Path) -> None:
    provider = MockProvider([ChatResponse(content="This design is verified.")])
    result = run_design_agent(
        "Design a heater", provider, workspace=CaseWorkspace(tmp_path), max_steps=1
    )
    md = summarize_transcript(result)
    assert "1. _unverified_text (not accepted)_: `This design is verified.`" in md
    assert f"**Answer:** {result.answer}" in md

    mixed = DesignAgentResult(
        "No design.",
        transcript=[
            {"event": "unverified_text", "content": "x" * 30, "accepted": False},
            {"tool": "list_cases", "arguments": {"limit": 1}, "result": {"cases": []}},
            {"event": "max_tokens", "content": "", "accepted": False},
        ],
    )
    md = summarize_transcript(mixed, max_chars=20)
    assert "1. _unverified_text (not accepted)_: `" + "x" * 20 + "...`" in md
    assert "2. **list_cases**(limit=1) -> `{'cases': []}`" in md
    assert "3. _max_tokens (not accepted)_: ``" in md
    assert md.endswith("**Answer:** No design.")
