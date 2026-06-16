"""Copilot tools for the time-domain dynamics & control layer.

These expose :mod:`fugacio.sim.control` and :mod:`fugacio.sim.dynamics` to the LLM
design agent as deterministic, JSON-in/JSON-out calculations: identify a first-
order-plus-dead-time (FOPDT) model from step data, turn it into PID gains by a
named tuning rule, simulate the resulting closed loop, and compare tuning rules on
their closed-loop performance. They are kept in their own module (rather than the
already-large :mod:`fugacio.copilot.tools`) and folded into the registry there.

The closed-loop simulator integrates the FOPDT plant with an explicit fixed step
and represents the transport delay as an integer-sample shift buffer carried
through a :func:`jax.lax.scan`, so dead time is handled honestly rather than by a
Pade approximation. The controller is the library :class:`~fugacio.sim.control.PID`
marched with the same step, so the numbers match what the engine would produce.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from fugacio.sim import (
    PID,
    FOPDTModel,
    PIDState,
    amigo,
    cohen_coon,
    fit_fopdt,
    imc_tuning,
    step_info,
    ziegler_nichols,
)

JsonDict = dict[str, Any]

_RULES = {
    "imc": imc_tuning,
    "lambda": imc_tuning,
    "ziegler_nichols": ziegler_nichols,
    "cohen_coon": cohen_coon,
    "amigo": amigo,
}


def _build_pid(
    model: FOPDTModel,
    rule: str,
    controller: str,
    tau_c: float | None,
    output_min: float | None,
    output_max: float | None,
) -> PID:
    """Construct a :class:`PID` from an FOPDT model by a named tuning rule."""
    key = rule.lower()
    if key not in _RULES:
        raise ValueError(f"unknown tuning rule {rule!r}; use one of {sorted(_RULES)}")
    limits: JsonDict = {}
    if output_min is not None:
        limits["u_min"] = float(output_min)
    if output_max is not None:
        limits["u_max"] = float(output_max)
    if key in ("imc", "lambda"):
        return imc_tuning(model, tau_c=tau_c, controller=controller, **limits)
    return _RULES[key](model, controller=controller, **limits)


def _default_horizon(model: FOPDTModel) -> float:
    """A settling-capturing simulation horizon for an FOPDT loop."""
    tau = float(model.tau)
    lag = float(model.dead_time)
    return max(12.0 * (tau + lag), 10.0 * tau, 1.0)


def _simulate_closed_loop(
    model: FOPDTModel,
    pid: PID,
    setpoint: float,
    t_final: float,
    n_steps: int = 800,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Servo (setpoint-step) response of an FOPDT plant under ``pid``.

    Returns ``(t, y, u)`` sampled on a uniform grid of ``n_steps`` points. The
    plant starts at rest and the setpoint steps to ``setpoint`` at ``t = 0``; the
    transport delay is an integer-sample shift of the manipulated variable.
    """
    t = jnp.linspace(0.0, t_final, n_steps)
    dt = float(t_final) / float(n_steps - 1)
    gain = jnp.asarray(model.gain)
    tau = jnp.asarray(model.tau)
    ndelay = round(float(model.dead_time) / dt) if dt > 0.0 else 0
    buf_len = max(ndelay, 1)

    sp = jnp.asarray(float(setpoint))
    cstate0 = pid.init_state(0.0, 0.0)
    buf0 = jnp.zeros((buf_len,))

    def step(carry: tuple[jnp.ndarray, PIDState, jnp.ndarray], _: Any) -> tuple[Any, Any]:
        y, cs, buf = carry
        u = pid.output(cs, sp, y)
        u_applied = buf[0] if ndelay > 0 else u
        dy = (-y + gain * u_applied) / tau
        y_next = y + dt * dy
        deriv = pid.derivative(cs, sp, y)
        cs_next = PIDState(i=cs.i + dt * deriv.i, x_d=cs.x_d + dt * deriv.x_d)
        buf_next = jnp.concatenate([buf[1:], u[None]]) if ndelay > 0 else buf
        return (y_next, cs_next, buf_next), (y, u)

    _, (ys, us) = jax.lax.scan(step, (jnp.asarray(0.0), cstate0, buf0), None, length=n_steps)
    return t, ys, us


def _metrics(t: jnp.ndarray, y: jnp.ndarray, setpoint: float) -> JsonDict:
    info = step_info(t, y, setpoint)
    return {
        "overshoot_fraction": float(info.overshoot),
        "rise_time_s": float(info.rise_time),
        "settling_time_s": float(info.settling_time),
        "peak_time_s": float(info.peak_time),
        "steady_state_error": float(info.steady_state_error),
        "iae": float(info.iae),
    }


def _thin(values: jnp.ndarray, points: int) -> list[float]:
    n = values.shape[0]
    idx = jnp.linspace(0, n - 1, min(int(points), n)).round().astype(int)
    return [float(values[i]) for i in idx]


def _identify_fopdt(
    time: list[float],
    response: list[float],
    input_step: float = 1.0,
) -> JsonDict:
    """Fit an FOPDT model ``K e^{-Ls}/(tau s + 1)`` to a measured step response."""
    t = jnp.asarray(time, dtype=float)
    y = jnp.asarray(response, dtype=float)
    model = fit_fopdt(t, y, float(input_step))
    pred = y[0] + _fopdt_curve(t, model, float(input_step))
    ss_res = float(jnp.sum((y - pred) ** 2))
    ss_tot = float(jnp.sum((y - jnp.mean(y)) ** 2))
    return {
        "gain": float(model.gain),
        "tau_s": float(model.tau),
        "dead_time_s": float(model.dead_time),
        "input_step": float(input_step),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0,
    }


def _fopdt_curve(t: jnp.ndarray, model: FOPDTModel, u: float) -> jnp.ndarray:
    from fugacio.sim import fopdt_step

    return fopdt_step(t, model.gain, model.tau, model.dead_time, u=u)


def _tune_pid(
    gain: float,
    tau: float,
    dead_time: float,
    rule: str = "imc",
    controller: str = "PI",
    tau_c: float | None = None,
    setpoint: float = 1.0,
    output_min: float | None = None,
    output_max: float | None = None,
) -> JsonDict:
    """Tune a PID for an FOPDT model by a named rule and report closed-loop quality."""
    model = FOPDTModel(
        gain=jnp.asarray(float(gain)),
        tau=jnp.asarray(float(tau)),
        dead_time=jnp.asarray(float(dead_time)),
    )
    pid = _build_pid(model, rule, controller, tau_c, output_min, output_max)
    t_final = _default_horizon(model)
    t, y, _u = _simulate_closed_loop(model, pid, float(setpoint), t_final)
    return {
        "rule": rule,
        "controller": controller,
        "kc": float(jnp.asarray(pid.kc)),
        "tau_i_s": float(jnp.asarray(pid.tau_i)),
        "tau_d_s": float(jnp.asarray(pid.tau_d)),
        "closed_loop": _metrics(t, y, float(setpoint)),
    }


def _closed_loop_response(
    gain: float,
    tau: float,
    dead_time: float,
    kc: float,
    tau_i: float | None = None,
    tau_d: float = 0.0,
    setpoint: float = 1.0,
    t_final: float | None = None,
    points: int = 41,
    output_min: float | None = None,
    output_max: float | None = None,
) -> JsonDict:
    """Simulate the FOPDT + PID servo response for explicit gains; return the trajectory."""
    model = FOPDTModel(
        gain=jnp.asarray(float(gain)),
        tau=jnp.asarray(float(tau)),
        dead_time=jnp.asarray(float(dead_time)),
    )
    limits: JsonDict = {}
    if output_min is not None:
        limits["u_min"] = float(output_min)
    if output_max is not None:
        limits["u_max"] = float(output_max)
    pid = PID(
        kc=jnp.asarray(float(kc)),
        tau_i=jnp.inf if tau_i is None else jnp.asarray(float(tau_i)),
        tau_d=jnp.asarray(float(tau_d)),
        **limits,
    )
    horizon = _default_horizon(model) if t_final is None else float(t_final)
    t, y, u = _simulate_closed_loop(model, pid, float(setpoint), horizon)
    return {
        "setpoint": float(setpoint),
        "time_s": _thin(t, points),
        "response": _thin(y, points),
        "control": _thin(u, points),
        "metrics": _metrics(t, y, float(setpoint)),
    }


def _recommend_pid_tuning(
    gain: float,
    tau: float,
    dead_time: float,
    controller: str = "PI",
    setpoint: float = 1.0,
    output_min: float | None = None,
    output_max: float | None = None,
) -> JsonDict:
    """Compare PID tuning rules on the same FOPDT loop and recommend the lowest-IAE one."""
    model = FOPDTModel(
        gain=jnp.asarray(float(gain)),
        tau=jnp.asarray(float(tau)),
        dead_time=jnp.asarray(float(dead_time)),
    )
    t_final = _default_horizon(model)
    candidate_rules = ["imc", "ziegler_nichols", "cohen_coon", "amigo"]
    candidates: list[JsonDict] = []
    for rule in candidate_rules:
        try:
            pid = _build_pid(model, rule, controller, None, output_min, output_max)
        except ValueError:
            continue
        t, y, _u = _simulate_closed_loop(model, pid, float(setpoint), t_final)
        metrics = _metrics(t, y, float(setpoint))
        candidates.append(
            {
                "rule": rule,
                "kc": float(jnp.asarray(pid.kc)),
                "tau_i_s": float(jnp.asarray(pid.tau_i)),
                "tau_d_s": float(jnp.asarray(pid.tau_d)),
                "iae": metrics["iae"],
                "overshoot_fraction": metrics["overshoot_fraction"],
                "settling_time_s": metrics["settling_time_s"],
            }
        )
    finite = [c for c in candidates if jnp.isfinite(c["iae"])]
    best = min(finite or candidates, key=lambda c: c["iae"]) if candidates else None
    return {
        "controller": controller,
        "candidates": candidates,
        "recommended_rule": best["rule"] if best else None,
    }


def dynamics_tool_specs() -> list[Any]:
    """ToolSpecs for the dynamics & control layer (folded into ``default_registry``)."""
    from fugacio.copilot.tools import ToolSpec

    rule_enum = ["imc", "ziegler_nichols", "cohen_coon", "amigo"]
    return [
        ToolSpec(
            name="identify_fopdt",
            description=(
                "Identify a first-order-plus-dead-time (FOPDT) model "
                "K*exp(-L*s)/(tau*s+1) from a measured open-loop step response "
                "by differentiable least squares; returns gain, tau, dead time "
                "and the fit R^2."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "time": {"type": "array", "items": {"type": "number"}},
                    "response": {"type": "array", "items": {"type": "number"}},
                    "input_step": {
                        "type": "number",
                        "description": "Magnitude of the input step (default 1)",
                    },
                },
                "required": ["time", "response"],
            },
            run=_identify_fopdt,
        ),
        ToolSpec(
            name="tune_pid",
            description=(
                "Tune a PID/PI controller for an FOPDT process by a named rule "
                "(IMC/lambda, Ziegler-Nichols, Cohen-Coon, AMIGO), then simulate "
                "the closed loop and report overshoot, rise/settling time and IAE."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gain": {"type": "number", "description": "FOPDT steady-state gain K"},
                    "tau": {"type": "number", "description": "FOPDT time constant (s)"},
                    "dead_time": {"type": "number", "description": "FOPDT dead time L (s)"},
                    "rule": {"type": "string", "enum": rule_enum},
                    "controller": {"type": "string", "enum": ["PI", "PID", "P"]},
                    "tau_c": {
                        "type": "number",
                        "description": "Desired closed-loop time constant (IMC only)",
                    },
                    "setpoint": {"type": "number"},
                    "output_min": {"type": "number", "description": "Lower output saturation"},
                    "output_max": {"type": "number", "description": "Upper output saturation"},
                },
                "required": ["gain", "tau", "dead_time"],
            },
            run=_tune_pid,
        ),
        ToolSpec(
            name="closed_loop_response",
            description=(
                "Simulate the setpoint-step (servo) response of an FOPDT process "
                "under a PID controller with explicit gains (kc, tau_i, tau_d); "
                "returns the time/response/control trajectories and step metrics."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gain": {"type": "number"},
                    "tau": {"type": "number"},
                    "dead_time": {"type": "number"},
                    "kc": {"type": "number", "description": "Proportional gain"},
                    "tau_i": {
                        "type": "number",
                        "description": "Integral time (s); omit for P-only",
                    },
                    "tau_d": {"type": "number", "description": "Derivative time (s)"},
                    "setpoint": {"type": "number"},
                    "t_final": {"type": "number", "description": "Simulation horizon (s)"},
                    "points": {"type": "integer", "description": "Output samples to return"},
                    "output_min": {"type": "number"},
                    "output_max": {"type": "number"},
                },
                "required": ["gain", "tau", "dead_time", "kc"],
            },
            run=_closed_loop_response,
        ),
        ToolSpec(
            name="recommend_pid_tuning",
            description=(
                "Compare the standard PID tuning rules on the same FOPDT loop and "
                "recommend the one with the lowest closed-loop IAE; returns each "
                "candidate's gains and performance."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gain": {"type": "number"},
                    "tau": {"type": "number"},
                    "dead_time": {"type": "number"},
                    "controller": {"type": "string", "enum": ["PI", "PID"]},
                    "setpoint": {"type": "number"},
                    "output_min": {"type": "number"},
                    "output_max": {"type": "number"},
                },
                "required": ["gain", "tau", "dead_time"],
            },
            run=_recommend_pid_tuning,
        ),
    ]


__all__ = ["dynamics_tool_specs"]
