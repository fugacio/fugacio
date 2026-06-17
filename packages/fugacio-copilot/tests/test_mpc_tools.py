"""Copilot tools for advanced process control (LQR/Kalman design, MPC, tuning)."""

import pytest

from fugacio.copilot import (
    call_tool,
    summarize_lqr_design,
    summarize_mpc_simulation,
    tool_schemas,
)

# A stable first-order SISO plant: x+ = 0.9 x + 0.1 u, y = x.
A = [[0.9]]
B = [[0.1]]
C = [[1.0]]


def test_mpc_tools_are_registered() -> None:
    names = {s["name"] for s in tool_schemas()}
    assert names >= {"lqr_design", "kalman_design", "simulate_mpc", "tune_mpc_weights"}


def test_lqr_design_is_stabilizing() -> None:
    out = call_tool("lqr_design", {"a": A, "b": B, "q": 1.0, "r": 0.1})
    assert out["stable"] is True
    assert len(out["gain"]) == 1 and len(out["gain"][0]) == 1
    # Discrete closed loop: all poles strictly inside the unit circle.
    assert max(out["pole_magnitudes"]) < 1.0
    md = summarize_lqr_design(out)
    assert "LQR design" in md and "u = -K x" in md


def test_lqr_design_continuous() -> None:
    # Continuous integrator xdot = u, penalize state and input.
    out = call_tool(
        "lqr_design", {"a": [[0.0]], "b": [[1.0]], "q": 1.0, "r": 1.0, "continuous": True}
    )
    assert out["continuous"] is True
    assert out["stable"] is True
    # x'=u with u=-Kx, K=1 -> pole at -1.
    assert out["pole_real_parts"][0] == pytest.approx(-1.0, rel=1e-6)


def test_kalman_design_is_stable() -> None:
    out = call_tool(
        "kalman_design",
        {"a": A, "c": C, "process_noise": 0.01, "measurement_noise": 0.1},
    )
    assert out["stable"] is True
    assert len(out["gain"]) == 1
    assert out["error_covariance"][0][0] > 0.0


def test_simulate_mpc_reaches_setpoint() -> None:
    out = call_tool(
        "simulate_mpc",
        {"a": A, "b": B, "c": C, "q": 1.0, "r": 0.01, "setpoint": [1.0], "n_steps": 80},
    )
    assert out["final_output"][0] == pytest.approx(1.0, abs=1e-2)
    assert out["metrics"][0]["steady_state_error"] == pytest.approx(0.0, abs=1e-2)
    md = summarize_mpc_simulation(out)
    assert "MPC closed-loop response" in md


def test_simulate_mpc_is_offset_free_under_disturbance() -> None:
    # A constant unmeasured output disturbance must be rejected to zero offset.
    out = call_tool(
        "simulate_mpc",
        {
            "a": A,
            "b": B,
            "c": C,
            "q": 1.0,
            "r": 0.01,
            "setpoint": [1.0],
            "disturbance": [0.3],
            "n_steps": 120,
        },
    )
    assert out["final_output"][0] == pytest.approx(1.0, abs=2e-2)


def test_simulate_mpc_respects_input_limits() -> None:
    out = call_tool(
        "simulate_mpc",
        {
            "a": A,
            "b": B,
            "c": C,
            "q": 1.0,
            "r": 1e-4,
            "setpoint": [1.0],
            "u_max": 1.5,
            "u_min": -1.5,
            "n_steps": 60,
        },
    )
    inputs = out["inputs"][0]
    assert max(inputs) <= 1.5 + 1e-6
    assert min(inputs) >= -1.5 - 1e-6


def test_tune_mpc_weights_does_not_worsen_cost() -> None:
    out = call_tool(
        "tune_mpc_weights",
        {
            "a": A,
            "b": B,
            "c": C,
            "setpoint": [1.0],
            "q0": 1.0,
            "r0": 1.0,
            "horizon": 8,
            "n_steps": 25,
            "max_iter": 6,
        },
    )
    assert out["tuned"]["cost"] <= out["initial"]["cost"] + 1e-6
    assert out["improved"] is True
    assert out["tuned"]["q"] > 0.0 and out["tuned"]["r"] > 0.0
