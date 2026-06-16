"""Copilot tools for the dynamics & control layer (FOPDT id, PID tuning, loops)."""

import numpy as np
import pytest

from fugacio.copilot import call_tool, summarize_pid_tuning, tool_schemas


def test_dynamics_tools_are_registered() -> None:
    names = {s["name"] for s in tool_schemas()}
    assert names >= {"identify_fopdt", "tune_pid", "closed_loop_response", "recommend_pid_tuning"}


def test_identify_fopdt_recovers_synthetic_model() -> None:
    gain, tau, dead = 2.0, 5.0, 1.5
    t = np.linspace(0.0, 40.0, 160)
    y = np.where(t >= dead, gain * (1.0 - np.exp(-(t - dead) / tau)), 0.0)
    out = call_tool("identify_fopdt", {"time": t.tolist(), "response": y.tolist()})
    assert out["gain"] == pytest.approx(gain, rel=1e-2)
    assert out["tau_s"] == pytest.approx(tau, rel=5e-2)
    assert out["dead_time_s"] == pytest.approx(dead, rel=1e-1)
    assert out["r_squared"] > 0.999


def test_tune_pid_returns_gains_and_metrics() -> None:
    out = call_tool(
        "tune_pid", {"gain": 2.0, "tau": 5.0, "dead_time": 1.5, "rule": "imc", "controller": "PI"}
    )
    assert out["kc"] > 0.0
    assert out["tau_i_s"] == pytest.approx(5.0, rel=1e-6)  # IMC PI: tau_i = tau
    cl = out["closed_loop"]
    assert cl["iae"] > 0.0
    assert 0.0 <= cl["overshoot_fraction"] < 1.0
    assert cl["settling_time_s"] > 0.0


def test_tune_pid_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError, match="unknown tuning rule"):
        call_tool("tune_pid", {"gain": 1.0, "tau": 1.0, "dead_time": 0.1, "rule": "magic"})


def test_closed_loop_response_reaches_setpoint() -> None:
    out = call_tool(
        "closed_loop_response",
        {
            "gain": 2.0,
            "tau": 5.0,
            "dead_time": 1.0,
            "kc": 0.9,
            "tau_i": 5.0,
            "points": 21,
        },
    )
    assert len(out["time_s"]) == 21
    assert len(out["response"]) == 21
    assert out["response"][-1] == pytest.approx(out["setpoint"], abs=2e-2)
    assert out["metrics"]["steady_state_error"] == pytest.approx(0.0, abs=2e-2)


def test_recommend_pid_tuning_picks_lowest_iae() -> None:
    out = call_tool(
        "recommend_pid_tuning", {"gain": 2.0, "tau": 5.0, "dead_time": 1.5, "controller": "PI"}
    )
    candidates = out["candidates"]
    assert len(candidates) >= 3
    best = min(candidates, key=lambda c: c["iae"])
    assert out["recommended_rule"] == best["rule"]
    # The Markdown report renders the comparison and flags the recommendation.
    md = summarize_pid_tuning(out)
    assert "PID tuning comparison" in md
    assert out["recommended_rule"] in md


def test_saturation_limits_are_respected() -> None:
    out = call_tool(
        "closed_loop_response",
        {
            "gain": 2.0,
            "tau": 5.0,
            "dead_time": 1.0,
            "kc": 50.0,
            "tau_i": 2.0,
            "output_min": 0.0,
            "output_max": 1.0,
            "points": 31,
        },
    )
    assert max(out["control"]) <= 1.0 + 1e-9
    assert min(out["control"]) >= 0.0 - 1e-9
