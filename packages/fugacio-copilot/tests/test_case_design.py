from __future__ import annotations

import json

import pytest

from fugacio.copilot import (
    ChatResponse,
    DesignSession,
    MockProvider,
    ToolCall,
    default_registry,
    run_design_agent,
)
from fugacio.sim.cases import CaseRunner, CaseWorkspace
from fugacio.sim.cases.examples import example_case


def test_default_registry_exposes_portable_case_tools():
    registry = default_registry()
    assert {"case_format", "validate_case", "solve_case"} <= registry.keys()
    schema = registry["case_format"].run()
    assert schema["schema_version"] == 1
    assert "delta_K" in schema["units"]
    assert len(schema["unit_types"]) == 12


def test_unsupported_provider_text_cannot_become_design_answer(tmp_path):
    provider = MockProvider(
        [ChatResponse(content="This design is verified and costs exactly $123.")]
    )
    result = run_design_agent(
        "Design a heater", provider, workspace=CaseWorkspace(tmp_path), max_steps=2
    )
    assert result.stop_reason == "budget"
    assert result.design is None
    assert "$123" not in result.answer
    assert result.transcript[0]["event"] == "unverified_text"
    assert not result.transcript[0]["accepted"]


def test_scripted_agent_creates_runs_and_submits_recorded_values(tmp_path):
    def script(messages):
        results = [json.loads(m.content) for m in messages if m.role == "tool"]
        if not results:
            return ChatResponse(
                tool_calls=[ToolCall("create", "create_case", {"case": example_case().to_dict()})]
            )
        if len(results) == 1:
            return ChatResponse(
                tool_calls=[ToolCall("run", "run_case", {"case_id": results[0]["case_id"]})]
            )
        return ChatResponse(
            content="An invented value: 999 MW",
            tool_calls=[
                ToolCall(
                    "submit",
                    "submit_design",
                    {"run_id": results[-1]["artifact_id"], "metrics": ["duty", "annual_cost"]},
                )
            ],
        )

    result = run_design_agent(
        "Build and assess a heater", MockProvider(script), workspace=CaseWorkspace(tmp_path)
    )
    assert result.stop_reason == "submitted"
    assert result.design["metrics"]["duty"]["value"] == pytest.approx(2.0412998021092956)
    assert "999 MW" not in result.answer
    assert result.design["run_id"] in result.answer
    assert "not_evaluated" in result.answer


def test_saved_external_run_must_be_recomputed(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    case = example_case()
    run = CaseRunner(case).run(check=True)
    workspace.save_run(run)
    session = DesignSession(workspace)
    session.load_case(case.case_id)
    assert not session.inspect(run.run_id)["computed_in_session"]
    with pytest.raises(ValueError, match="computed in this session"):
        session.submit_design(run.run_id, ["duty"])
    computed = session.run_case(case.case_id)
    assert session.submit_design(computed["artifact_id"], ["duty"])["submitted"]


def test_stale_revision_unknown_metrics_and_failed_run_rejected(tmp_path):
    session = DesignSession(CaseWorkspace(tmp_path))
    created = session.create_case(example_case().to_dict())
    case_id = created["case_id"]
    run = session.run_case(case_id)
    with pytest.raises(ValueError, match="unique metric"):
        session.submit_design(run["artifact_id"], ["invented"])
    changed = session.set_parameters(case_id, {"temperature": {"value": 360, "unit": "K"}})
    with pytest.raises(ValueError, match="stale"):
        session.submit_design(run["artifact_id"], ["duty"])
    with pytest.raises(ValueError, match="stale"):
        session.update_case(case_id, example_case().to_dict())
    assert session.current_case_id == changed["case_id"]
    bad = example_case().to_dict()
    bad["metrics"]["undefined"] = {"expression": {"op": "divide", "args": [1, 0]}, "unit": "1"}
    failed_id = session.create_case(bad)["case_id"]
    failed = session.run_case(failed_id)
    assert not failed["accepted"]
    with pytest.raises(RuntimeError, match="failed"):
        session.submit_design(failed["artifact_id"], ["duty"])


def test_update_in_same_batch_invalidates_prior_submission(tmp_path):
    def script(messages):
        results = [json.loads(m.content) for m in messages if m.role == "tool"]
        if not results:
            return ChatResponse(
                tool_calls=[ToolCall("create", "create_case", {"case": example_case().to_dict()})]
            )
        if len(results) == 1:
            return ChatResponse(
                tool_calls=[ToolCall("run", "run_case", {"case_id": results[0]["case_id"]})]
            )
        return ChatResponse(
            tool_calls=[
                ToolCall(
                    "submit",
                    "submit_design",
                    {"run_id": results[1]["artifact_id"], "metrics": ["duty"]},
                ),
                ToolCall(
                    "change",
                    "set_case_parameters",
                    {
                        "case_id": results[0]["case_id"],
                        "overrides": {"temperature": {"value": 360, "unit": "K"}},
                    },
                ),
            ]
        )

    result = run_design_agent(
        "Assess a case", MockProvider(script), workspace=CaseWorkspace(tmp_path), max_steps=3
    )
    assert result.stop_reason == "budget"
    assert result.design is None


def test_literal_metrics_arent_submitted_as_computed_performance(tmp_path):
    session = DesignSession(CaseWorkspace(tmp_path))
    case = example_case().to_dict()
    case["metrics"]["claimed_cost"] = {
        "expression": {"value": 1, "unit": "USD/yr"},
        "unit": "USD/yr",
    }
    identity = session.create_case(case)["case_id"]
    run = session.run_case(identity)
    assert run["metrics"]["claimed_cost"]["source"] == "declared_input"
    with pytest.raises(ValueError, match="literal inputs"):
        session.submit_design(run["artifact_id"], ["claimed_cost"])
