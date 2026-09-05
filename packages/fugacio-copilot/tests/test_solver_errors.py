"""Numerical failures remain structured and strict JSON at the agent boundary."""

import json

import jax.numpy as jnp
import pytest

from fugacio.copilot.agent import _safe_call
from fugacio.copilot.tools import ToolSpec, call_tool
from fugacio.thermo.diagnostics import ConvergenceError, residual_report


def test_agent_retains_solver_context_and_report():
    def fail():
        raise ConvergenceError(residual_report(jnp.array([jnp.nan])), "column stage 4")

    spec = ToolSpec("test", "Failure test", {"type": "object", "properties": {}}, fail)
    result = _safe_call("test", {}, {"test": spec})
    assert result["context"] == "column stage 4"
    assert result["report"]["status"] == "nonfinite"
    assert result["report"]["residual_norm"] is None
    json.dumps(result, allow_nan=False)


def test_tool_rejects_nonfinite_arguments_and_results():
    spec = ToolSpec(
        "test",
        "Nonfinite test",
        {"type": "object", "properties": {}},
        lambda **kwargs: {"value": float("nan")},
    )
    with pytest.raises(ValueError):
        call_tool("test", {}, {"test": spec})
    result = _safe_call("test", {"value": float("inf")}, {"test": spec})
    assert "error" in result
    json.dumps(result, allow_nan=False)
