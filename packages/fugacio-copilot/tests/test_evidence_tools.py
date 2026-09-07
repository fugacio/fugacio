import json

from fugacio.copilot.agent import _safe_call
from fugacio.copilot.tools import call_tool, default_registry


def test_missing_pair_is_reported_and_activity_tool_cannot_silently_idealize():
    inspected = call_tool("inspect_thermodynamic_evidence", {"components": ["water", "n-heptane"]})
    assert not inspected["applicability"]["parameters_available"]
    assert inspected["parameter_evidence"]["pairs"][0]["kind"] == "missing"
    result = _safe_call(
        "activity_coefficients",
        {
            "components": ["water", "n-heptane"],
            "method": "nrtl",
            "temperature": 330.0,
            "x": [0.5, 0.5],
        },
        default_registry(),
    )
    assert "error" in result


def test_measured_corpus_tool_is_offline_and_doesnt_claim_qualification():
    result = call_tool("measured_corpus", {})
    assert result["observations"] == 447
    assert len(result["systems"]) == 20
    json.dumps(result, allow_nan=False)


def test_existing_flash_returns_acceptance_and_assumptions():
    result = call_tool(
        "flash_drum",
        {
            "components": ["methane", "n-butane"],
            "z": [0.5, 0.5],
            "flow": 10.0,
            "temperature": 250.0,
            "pressure": 1e6,
        },
    )
    assert result["physical_acceptance"]["accepted"]
    assert result["parameter_evidence"]["pairs"][0]["kind"] == "assumed_zero"
    assert result["empirical_qualification"] == "not_checked"


def test_rejected_physical_state_returns_structured_report():
    result = _safe_call(
        "checked_flash",
        {
            "components": ["methane", "n-butane"],
            "z": [0.8, 0.8],
            "temperature": 250.0,
            "pressure": 1e6,
        },
        default_registry(),
    )
    assert "error" in result
    assert result["report"]["accepted"] is False
    assert result["report"]["input_valid"] is False


def test_predictive_method_reports_missing_group_assignments():
    result = call_tool(
        "inspect_thermodynamic_evidence", {"components": ["water", "hydrogen"], "method": "unifac"}
    )
    assert not result["applicability"]["parameters_available"]
    assert result["parameter_evidence"]["missing_components"] == ("hydrogen",)
