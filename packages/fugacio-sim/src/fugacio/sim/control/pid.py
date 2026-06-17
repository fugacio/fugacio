"""A differentiable PID controller in realizable, anti-windup form.

The controller is written so it can live *inside* a dynamic flowsheet ODE: its
memory (the integral action and the filtered derivative) are carried as ODE
states, exactly like a vessel holdup, so a closed loop is just a larger
differentiable ODE. Two things make it production-grade rather than a toy:

* **A realizable, filtered derivative.** Pure derivative action is not causal and
  amplifies noise; instead the derivative acts on the measurement through a
  first-order filter with time constant ``tau_d / n_filter``, carried as a state
  ``x_d``. This also removes "derivative kick" on setpoint changes.
* **Back-calculation anti-windup.** When the output saturates against its limits
  the integrator is unwound at tracking rate ``1/tau_t`` toward the achievable
  output, so the loop recovers cleanly from saturation instead of winding up.

The `PID` parameters are a registered JAX pytree, so the *gains themselves*
are differentiable: you can take a gradient of a closed-loop performance index
(IAE, overshoot, settling time) with respect to ``kc``, ``tau_i``, ``tau_d`` and
let an optimizer tune them (see `fugacio.sim.dynamics.tune_pid`).

Sign convention: the controller computes the error ``e = setpoint - measurement``
and a *positive* ``kc`` increases the output when the measurement is below
setpoint (a "reverse-acting" loop in process terms, ``direction="reverse"``). Use
``direction="direct"`` (or a negative ``kc``) when raising the output should lower
the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float


class PIDState(NamedTuple):
    """Internal controller state carried through the dynamic simulation.

    Attributes:
        i: Integral term contribution to the output (already scaled by the
            integral gain), in output units.
        x_d: First-order-filtered measurement used to form the derivative term.
    """

    i: Array
    x_d: Array


@dataclass(frozen=True)
class PID:
    """A filtered, anti-windup PID controller (a differentiable pytree).

    Attributes:
        kc: Proportional gain (output units per measurement unit).
        tau_i: Integral (reset) time, s. Use ``inf`` to disable integral action.
        tau_d: Derivative (rate) time, s. ``0`` disables derivative action.
        beta: Setpoint weight on the proportional term (``0..1``; ``1`` is classic).
        gamma: Setpoint weight on the derivative term (usually ``0``, derivative
            on measurement only, to avoid derivative kick).
        u_bias: Output bias / nominal output at zero error (manual reset).
        u_min: Lower output limit (saturation).
        u_max: Upper output limit (saturation).
        tau_t: Anti-windup tracking time, s. ``<= 0`` selects an automatic value
            (``sqrt(tau_i * tau_d)`` when derivative is active, else ``tau_i``).
        n_filter: Derivative-filter ratio (filter time is ``tau_d / n_filter``).
        direction: ``"reverse"`` (default) or ``"direct"`` acting; ``"direct"``
            flips the error sign.
    """

    kc: ArrayLike
    tau_i: ArrayLike = jnp.inf
    tau_d: ArrayLike = 0.0
    beta: ArrayLike = 1.0
    gamma: ArrayLike = 0.0
    u_bias: ArrayLike = 0.0
    u_min: ArrayLike = -jnp.inf
    u_max: ArrayLike = jnp.inf
    tau_t: ArrayLike = 0.0
    n_filter: float = field(default=10.0)
    direction: str = field(default="reverse")

    def _sign(self) -> float:
        return 1.0 if self.direction == "reverse" else -1.0

    def _ki(self) -> Array:
        """Integral gain ``kc / tau_i`` (0 when integral action is disabled)."""
        tau_i = jnp.asarray(self.tau_i)
        finite = jnp.isfinite(tau_i) & (tau_i > 0.0)
        return jnp.where(finite, jnp.asarray(self.kc) / jnp.where(finite, tau_i, 1.0), 0.0)

    def _tau_f(self) -> Array:
        """Derivative-filter time constant ``tau_d / n_filter`` (>= small floor)."""
        tau_d = jnp.asarray(self.tau_d)
        return jnp.where(tau_d > 0.0, tau_d / self.n_filter, 1.0)

    def _tau_t(self) -> Array:
        tau_t = jnp.asarray(self.tau_t)
        tau_i = jnp.asarray(self.tau_i)
        tau_d = jnp.asarray(self.tau_d)
        tau_i_safe = jnp.where(jnp.isfinite(tau_i) & (tau_i > 0.0), tau_i, 1.0)
        has_d = tau_d > 0.0
        # Guard the sqrt argument to stay strictly positive in *both* branches: a
        # ``jnp.where`` differentiates through its unselected branch, so a bare
        # ``sqrt(tau_i * tau_d)`` would feed ``sqrt(0)`` (infinite slope) into the
        # gradient w.r.t. ``tau_i`` whenever derivative action is off (``tau_d=0``).
        prod = jnp.where(has_d, tau_i_safe * tau_d, 1.0)
        auto = jnp.where(has_d, jnp.sqrt(prod), tau_i_safe)
        return jnp.where(tau_t > 0.0, tau_t, jnp.maximum(auto, 1e-6))

    def init_state(self, pv0: ArrayLike, u0: ArrayLike | None = None) -> PIDState:
        """Initial controller state for bumpless start at output ``u0`` (default bias).

        The integral term is preloaded so the controller's initial output equals
        ``u0`` (or `u_bias` if ``u0`` is ``None``) when the measurement is at
        ``pv0`` and the setpoint equals ``pv0``; the derivative filter starts at the
        measurement so the initial derivative action is zero.
        """
        u_start = jnp.asarray(self.u_bias if u0 is None else u0)
        return PIDState(i=u_start - jnp.asarray(self.u_bias), x_d=jnp.asarray(pv0))

    def _unsaturated(self, state: PIDState, setpoint: ArrayLike, pv: ArrayLike) -> Array:
        sp = jnp.asarray(setpoint)
        y = jnp.asarray(pv)
        s = self._sign()
        kc = jnp.asarray(self.kc)
        tau_d = jnp.asarray(self.tau_d)
        p_term = s * kc * (jnp.asarray(self.beta) * sp - y)
        d_term = jnp.where(
            tau_d > 0.0,
            -s * kc * tau_d * (y - jnp.asarray(self.gamma) * sp - state.x_d) / self._tau_f(),
            0.0,
        )
        return jnp.asarray(self.u_bias) + p_term + state.i + d_term

    def output(self, state: PIDState, setpoint: ArrayLike, pv: ArrayLike) -> Array:
        """Saturated controller output for the current state, setpoint and measurement."""
        u = self._unsaturated(state, setpoint, pv)
        return jnp.clip(u, jnp.asarray(self.u_min), jnp.asarray(self.u_max))

    def derivative(self, state: PIDState, setpoint: ArrayLike, pv: ArrayLike) -> PIDState:
        """Time derivative of the controller state (for embedding in a flowsheet ODE).

        Returns ``(di/dt, dx_d/dt)`` as a `PIDState`. The integral derivative
        includes the back-calculation anti-windup term so it stops winding up while
        the output is saturated.
        """
        sp = jnp.asarray(setpoint)
        y = jnp.asarray(pv)
        s = self._sign()
        e = s * (sp - y)
        u_unsat = self._unsaturated(state, setpoint, pv)
        u_sat = jnp.clip(u_unsat, jnp.asarray(self.u_min), jnp.asarray(self.u_max))
        di = self._ki() * e + (u_sat - u_unsat) / self._tau_t()
        tau_d = jnp.asarray(self.tau_d)
        dxd = jnp.where(tau_d > 0.0, (y - state.x_d) / self._tau_f(), 0.0)
        return PIDState(i=di, x_d=dxd)

    def step(
        self, state: PIDState, setpoint: ArrayLike, pv: ArrayLike, dt: ArrayLike
    ) -> tuple[Array, PIDState]:
        """Discrete one-step update by explicit Euler; returns ``(output, new_state)``.

        For standalone digital-controller loops. Inside a continuous dynamic
        simulation prefer `derivative` so the controller integrates with the
        same solver as the plant.
        """
        d = self.derivative(state, setpoint, pv)
        new_state = PIDState(
            i=state.i + jnp.asarray(dt) * d.i, x_d=state.x_d + jnp.asarray(dt) * d.x_d
        )
        return self.output(state, setpoint, pv), new_state


jax.tree_util.register_dataclass(
    PID,
    data_fields=["kc", "tau_i", "tau_d", "beta", "gamma", "u_bias", "u_min", "u_max", "tau_t"],
    meta_fields=["n_filter", "direction"],
)


def pi(kc: ArrayLike, tau_i: ArrayLike, **kwargs: ArrayLike) -> PID:
    """Convenience constructor for a PI controller (no derivative action)."""
    return PID(kc=kc, tau_i=tau_i, tau_d=0.0, **kwargs)  # type: ignore[arg-type]


def p_only(kc: ArrayLike, **kwargs: ArrayLike) -> PID:
    """Convenience constructor for a proportional-only controller."""
    return PID(kc=kc, tau_i=jnp.inf, tau_d=0.0, **kwargs)  # type: ignore[arg-type]


__all__ = ["PID", "PIDState", "p_only", "pi"]
