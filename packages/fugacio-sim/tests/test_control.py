"""The control toolkit: PID, linear blocks, metrics, linearization, and tuning.

The blocks are checked against their closed-form responses, the PID against its
defining algebra (proportional/integral/derivative terms, bumpless start, and
back-calculation anti-windup), the metrics against hand-computed values, the
linearization against the analytic poles and DC gain of a known plant, and the
tuning rules against an FOPDT model identified back from a synthetic step. A
dedicated regression test guards the ``tau_i`` gradient, which a naive
``jnp.where`` in the anti-windup tracking time once poisoned with a NaN.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    iae,
    ise,
    itae,
    odeint,
    overshoot,
    rise_time,
    settling_time,
    step_info,
)
from fugacio.sim.control import (
    PID,
    dc_gain,
    dead_band,
    first_order_ss,
    first_order_step,
    fit_fopdt,
    fopdt_step,
    is_controllable,
    is_observable,
    is_stable,
    lead_lag,
    linearize,
    p_only,
    pi,
    poles,
    rate_limit,
    saturate,
    second_order_ss,
    second_order_step,
)
from fugacio.sim.control.tuning import FOPDTModel, amigo, cohen_coon, imc_tuning, ziegler_nichols


# --------------------------------------------------------------------------- #
# Analytic linear blocks
# --------------------------------------------------------------------------- #
def test_first_order_step_matches_closed_form() -> None:
    t = jnp.linspace(0.0, 40.0, 81)
    y = first_order_step(t, gain=2.0, tau=4.0)
    assert jnp.allclose(y, 2.0 * (1.0 - jnp.exp(-t / 4.0)), atol=1e-12)
    assert float(y[-1]) == pytest.approx(2.0, rel=1e-3)  # settles to K*u after ~10 tau


def test_fopdt_step_is_flat_before_dead_time() -> None:
    t = jnp.linspace(0.0, 10.0, 101)
    y = fopdt_step(t, gain=1.5, tau=2.0, dead_time=3.0)
    assert jnp.all(y[t < 3.0] == 0.0)
    # After the delay it is a shifted first-order rise.
    late = t[t >= 3.0]
    assert jnp.allclose(
        fopdt_step(late, 1.5, 2.0, 3.0), 1.5 * (1.0 - jnp.exp(-(late - 3.0) / 2.0)), atol=1e-12
    )


def test_second_order_step_overshoots_when_underdamped() -> None:
    t = jnp.linspace(0.0, 30.0, 600)
    y = second_order_step(t, gain=1.0, wn=1.0, zeta=0.2)
    assert float(jnp.max(y)) > 1.3  # ~52% overshoot for zeta=0.2
    assert float(y[-1]) == pytest.approx(1.0, rel=1e-2)


def test_second_order_step_is_smooth_through_critical_damping() -> None:
    t = jnp.linspace(0.0, 12.0, 200)
    near = second_order_step(t, 1.0, 1.0, 0.999)
    crit = second_order_step(t, 1.0, 1.0, 1.0)
    over = second_order_step(t, 1.0, 1.0, 1.001)
    assert jnp.allclose(near, crit, atol=2e-3)
    assert jnp.allclose(over, crit, atol=2e-3)


def test_lead_lag_initial_and_final_values() -> None:
    t = jnp.linspace(0.0, 50.0, 200)
    y = lead_lag(t, gain=1.0, tau_lead=5.0, tau_lag=2.0)
    assert float(y[0]) == pytest.approx(5.0 / 2.0, rel=1e-6)  # instantaneous lead/lag ratio
    assert float(y[-1]) == pytest.approx(1.0, rel=1e-3)  # steady gain


def test_static_nonlinearities() -> None:
    assert float(saturate(5.0, 0.0, 1.0)) == 1.0
    assert float(saturate(-2.0, 0.0, 1.0)) == 0.0
    assert float(dead_band(0.3, 0.5)) == 0.0
    assert float(dead_band(1.0, 0.5)) == pytest.approx(0.5)
    assert float(rate_limit(10.0, 2.0)) == 2.0
    assert float(rate_limit(-10.0, 2.0)) == -2.0


# --------------------------------------------------------------------------- #
# PID controller algebra
# --------------------------------------------------------------------------- #
def test_p_only_output_is_proportional() -> None:
    c = p_only(kc=2.0, u_bias=0.5)
    state = c.init_state(0.0)
    # reverse acting: u = bias + kc (sp - pv)
    assert float(c.output(state, setpoint=3.0, pv=1.0)) == pytest.approx(0.5 + 2.0 * 2.0)


def test_direct_action_flips_sign() -> None:
    rev = PID(kc=1.0, tau_i=jnp.inf, direction="reverse")
    dir_ = PID(kc=1.0, tau_i=jnp.inf, direction="direct")
    s = rev.init_state(0.0)
    assert float(rev.output(s, 2.0, 0.0)) == pytest.approx(2.0)
    assert float(dir_.output(s, 2.0, 0.0)) == pytest.approx(-2.0)


def test_init_state_is_bumpless() -> None:
    c = pi(kc=1.0, tau_i=5.0, u_bias=0.0)
    state = c.init_state(pv0=2.0, u0=0.7)
    # At sp = pv = pv0 the initial output equals the requested u0.
    assert float(c.output(state, setpoint=2.0, pv=2.0)) == pytest.approx(0.7)


def test_pi_eliminates_steady_state_offset_on_first_order_plant() -> None:
    c = pi(kc=1.0, tau_i=3.0)
    kp, taup, sp = 2.0, 5.0, 1.0

    def rhs(t: jnp.ndarray, st: dict, th: None) -> dict:
        y, ic = st["y"], st["c"]
        u = c.output(ic, sp, y)
        return {"y": (-y + kp * u) / taup, "c": c.derivative(ic, sp, y)}

    ts = jnp.linspace(0.0, 80.0, 401)
    st0 = {"y": jnp.asarray(0.0), "c": c.init_state(0.0)}
    traj = odeint(rhs, st0, ts, method="rk4", substeps=4)
    assert float(traj["y"][-1]) == pytest.approx(sp, abs=1e-3)  # zero offset


def test_back_calculation_limits_windup_when_saturated() -> None:
    c = pi(kc=1.0, tau_i=2.0, u_min=0.0, u_max=1.0)
    state = c.init_state(0.0)
    # Huge error drives the unsaturated output well past u_max -> output saturates.
    assert float(c.output(state, setpoint=100.0, pv=0.0)) == pytest.approx(1.0)
    di = c.derivative(state, setpoint=100.0, pv=0.0).i
    ki_e = c._ki() * (100.0 - 0.0)
    # Anti-windup subtracts a positive tracking term, so di is below ki*e.
    assert float(di) < float(ki_e)


def test_derivative_acts_on_measurement_not_setpoint() -> None:
    # gamma=0: a setpoint change must not change the derivative term (no kick).
    c = PID(kc=1.0, tau_i=jnp.inf, tau_d=2.0, beta=1.0, gamma=0.0)
    state = c.init_state(0.0)
    u_lo = c._unsaturated(state, setpoint=1.0, pv=0.5)
    u_hi = c._unsaturated(state, setpoint=2.0, pv=0.5)
    # The whole difference is the proportional setpoint term kc*beta*dsp = 1.0.
    assert float(u_hi - u_lo) == pytest.approx(1.0, rel=1e-9)


def test_pid_gains_are_differentiable_pytree() -> None:
    def loss(kc: jnp.ndarray) -> jnp.ndarray:
        c = pi(kc=kc, tau_i=4.0)
        return c.output(c.init_state(0.0), 1.0, 0.0)

    assert float(jax.grad(loss)(jnp.asarray(2.0))) == pytest.approx(1.0, rel=1e-9)


def test_pi_tau_i_gradient_is_finite_regression() -> None:
    """Regression: the anti-windup tracking time must not poison the tau_i gradient.

    A bare ``sqrt(tau_i * tau_d)`` in the auto tracking-time made ``d/dtau_i`` NaN
    for a PI controller (tau_d = 0) via ``jnp.where``'s unselected branch.
    """
    ts = jnp.linspace(0.0, 40.0, 201)
    kp, taup, sp = 2.0, 5.0, 1.0

    def closed_loop_iae(gains: dict) -> jnp.ndarray:
        c = pi(kc=gains["kc"], tau_i=gains["tau_i"], u_min=-10.0, u_max=10.0)

        def rhs(t: jnp.ndarray, st: dict, th: None) -> dict:
            y, ic = st["y"], st["c"]
            u = c.output(ic, sp, y)
            return {"y": (-y + kp * u) / taup, "c": c.derivative(ic, sp, y)}

        st0 = {"y": jnp.asarray(0.0), "c": c.init_state(0.0)}
        traj = odeint(rhs, st0, ts, method="rk4", substeps=3)
        return iae(ts, traj["y"], sp)

    grads = jax.grad(closed_loop_iae)({"kc": jnp.asarray(0.5), "tau_i": jnp.asarray(8.0)})
    assert jnp.isfinite(grads["kc"])
    assert jnp.isfinite(grads["tau_i"])


# --------------------------------------------------------------------------- #
# Time-domain metrics
# --------------------------------------------------------------------------- #
def test_error_integrals_on_constant_offset() -> None:
    t = jnp.linspace(0.0, 10.0, 1001)
    y = jnp.zeros_like(t)  # never reaches sp = 1
    assert float(iae(t, y, 1.0)) == pytest.approx(10.0, rel=1e-6)
    assert float(ise(t, y, 1.0)) == pytest.approx(10.0, rel=1e-6)
    assert float(itae(t, y, 1.0)) == pytest.approx(50.0, rel=1e-6)  # integral t dt = 50


def test_overshoot_and_settling_on_second_order() -> None:
    t = jnp.linspace(0.0, 40.0, 2000)
    y = second_order_step(t, gain=1.0, wn=1.0, zeta=0.3)
    os = float(overshoot(y, 1.0))
    analytic = float(jnp.exp(-0.3 * jnp.pi / jnp.sqrt(1 - 0.3**2)))
    assert os == pytest.approx(analytic, rel=0.05)
    assert float(settling_time(t, y, 1.0, tol=0.02)) > 0.0
    assert float(rise_time(t, y, 1.0)) > 0.0


def test_step_info_bundle() -> None:
    t = jnp.linspace(0.0, 40.0, 2000)
    y = second_order_step(t, 1.0, 1.0, 0.5)
    info = step_info(t, y, 1.0)
    assert float(info.overshoot) > 0.0
    assert float(info.steady_state_error) == pytest.approx(0.0, abs=1e-2)
    assert float(info.iae) > 0.0


# --------------------------------------------------------------------------- #
# Linearization and state-space analysis
# --------------------------------------------------------------------------- #
def test_linearize_recovers_second_order_poles_and_gain() -> None:
    wn, zeta, gain = 2.0, 0.4, 3.0
    a, b, c, _d = second_order_ss(gain, wn, zeta)

    def rhs(x: jnp.ndarray, u: jnp.ndarray, th: None) -> jnp.ndarray:
        return a @ x + b @ jnp.atleast_1d(u)

    ss = linearize(rhs, jnp.zeros(2), jnp.zeros(1), output=lambda x, u, th: c @ x)
    pol = poles(ss)
    expected = jnp.array(
        [-zeta * wn + 1j * wn * jnp.sqrt(1 - zeta**2), -zeta * wn - 1j * wn * jnp.sqrt(1 - zeta**2)]
    )
    assert jnp.allclose(jnp.sort_complex(pol), jnp.sort_complex(expected), atol=1e-8)
    assert float(dc_gain(ss)[0, 0]) == pytest.approx(gain, rel=1e-8)
    assert bool(is_stable(ss))


def test_second_order_block_is_controllable_and_observable() -> None:
    a, b, c, _d = second_order_ss(1.0, 1.0, 0.5)

    def rhs(x: jnp.ndarray, u: jnp.ndarray, th: None) -> jnp.ndarray:
        return a @ x + b @ jnp.atleast_1d(u)

    ss = linearize(rhs, jnp.zeros(2), jnp.zeros(1), output=lambda x, u, th: c @ x)
    assert bool(is_controllable(ss))
    assert bool(is_observable(ss))


def test_first_order_ss_dc_gain() -> None:
    a, b, c, _d = first_order_ss(gain=4.0, tau=2.0)

    def rhs(x: jnp.ndarray, u: jnp.ndarray, th: None) -> jnp.ndarray:
        return a @ x + b @ jnp.atleast_1d(u)

    ss = linearize(rhs, jnp.zeros(1), jnp.zeros(1), output=lambda x, u, th: c @ x)
    assert float(dc_gain(ss)[0, 0]) == pytest.approx(4.0, rel=1e-8)


# --------------------------------------------------------------------------- #
# FOPDT identification and tuning rules
# --------------------------------------------------------------------------- #
def test_fit_fopdt_recovers_known_model() -> None:
    t = jnp.linspace(0.0, 60.0, 300)
    y = fopdt_step(t, gain=2.5, tau=8.0, dead_time=3.0)
    model = fit_fopdt(t, y)
    assert float(model.gain) == pytest.approx(2.5, rel=1e-3)
    assert float(model.tau) == pytest.approx(8.0, rel=2e-2)
    assert float(model.dead_time) == pytest.approx(3.0, rel=5e-2)


def test_tuning_rules_give_positive_sane_gains() -> None:
    model = FOPDTModel(gain=jnp.asarray(2.0), tau=jnp.asarray(10.0), dead_time=jnp.asarray(1.0))
    for rule in (ziegler_nichols, cohen_coon, amigo):
        c = rule(model, controller="PI")
        assert float(c.kc) > 0.0
        assert float(c.tau_i) > 0.0
    imc = imc_tuning(model, controller="PI")
    # IMC integral time equals the process time constant for a PI controller.
    assert float(imc.tau_i) == pytest.approx(10.0, rel=1e-6)


def test_imc_smaller_tau_c_is_more_aggressive() -> None:
    model = FOPDTModel(gain=jnp.asarray(1.0), tau=jnp.asarray(5.0), dead_time=jnp.asarray(1.0))
    fast = imc_tuning(model, tau_c=0.5, controller="PI")
    slow = imc_tuning(model, tau_c=10.0, controller="PI")
    assert float(fast.kc) > float(slow.kc)
