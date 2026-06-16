"""PID tuning rules and first-order-plus-dead-time model identification.

Classical tuning is a two-step recipe: reduce the process to a first-order-plus-
dead-time (FOPDT) model ``K e^{-L s} / (tau s + 1)``, then apply a correlation
that maps ``(K, tau, L)`` to PID gains. This module provides both halves:

* :func:`fit_fopdt` identifies ``(K, tau, L)`` from a measured step response by
  differentiable least squares (reusing :func:`fugacio.sim.least_squares`); and
* the tuning rules -- :func:`ziegler_nichols`, :func:`cohen_coon`,
  :func:`imc_tuning` (lambda tuning), and :func:`amigo` -- turn an FOPDT model
  into a ready :class:`~fugacio.sim.control.pid.PID`.

For a closed-loop, performance-index-optimal tune that exploits the fully
differentiable plant, see :func:`fugacio.sim.dynamics.tune_pid`, which descends an
IAE/ISE objective on the gains directly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from fugacio.sim.control.blocks import fopdt_step
from fugacio.sim.control.pid import PID
from fugacio.sim.optimize import least_squares

ArrayLike = Array | float


class FOPDTModel(NamedTuple):
    """A first-order-plus-dead-time process model ``K e^{-L s} / (tau s + 1)``.

    Attributes:
        gain: Steady-state gain ``K`` (output units per input unit).
        tau: Time constant ``tau`` (s).
        dead_time: Dead time / transport delay ``L`` (s).
    """

    gain: Array
    tau: Array
    dead_time: Array


def fit_fopdt(
    t: Array,
    y: Array,
    u_step: ArrayLike = 1.0,
    *,
    guess: FOPDTModel | None = None,
) -> FOPDTModel:
    """Identify an FOPDT model from a step response ``(t, y)`` to an input step ``u_step``.

    Fits ``(K, tau, L)`` by Levenberg-Marquardt least squares against the analytic
    FOPDT step response. A data-driven initial guess is used when ``guess`` is
    omitted (gain from the final value; ``tau`` and ``L`` from the 28.3% / 63.2%
    response times). The fit is differentiable in the data.
    """
    t = jnp.asarray(t, dtype=float)
    y = jnp.asarray(y, dtype=float)
    u = jnp.asarray(u_step)
    if guess is None:
        k0 = (y[-1] - y[0]) / u
        span = y[-1] - y[0]
        frac = (y - y[0]) / (span + 1e-30)
        t632 = t[jnp.argmax(frac >= 0.632)]
        t283 = t[jnp.argmax(frac >= 0.283)]
        tau0 = jnp.clip(1.5 * (t632 - t283), 1e-3, None)
        l0 = jnp.clip(t632 - tau0, 0.0, None)
        guess = FOPDTModel(gain=k0, tau=tau0, dead_time=l0)

    x0 = jnp.array([guess.gain, jnp.log(guess.tau), jnp.log(guess.dead_time + 1e-3)])

    def residual(x: Array, _: None) -> Array:
        gain = x[0]
        tau = jnp.exp(x[1])
        dead = jnp.exp(x[2]) - 1e-3
        return fopdt_step(t, gain, tau, jnp.clip(dead, 0.0, None), u=u) - (y - y[0])

    sol = least_squares(residual, x0, None, max_iter=200)
    x = sol.x
    return FOPDTModel(
        gain=x[0], tau=jnp.exp(x[1]), dead_time=jnp.clip(jnp.exp(x[2]) - 1e-3, 0.0, None)
    )


def _l(model: FOPDTModel) -> Array:
    # Floor the dead time so tuning rules that divide by it stay finite for a
    # delay-free process (a tiny effective delay ~ one integration step).
    return jnp.maximum(jnp.asarray(model.dead_time), 1e-3 * jnp.asarray(model.tau) + 1e-6)


def ziegler_nichols(model: FOPDTModel, *, controller: str = "PID", **kwargs: ArrayLike) -> PID:
    """Ziegler-Nichols open-loop (reaction-curve) tuning from an FOPDT model.

    ``controller`` is ``"P"``, ``"PI"`` or ``"PID"``. Aggressive (quarter-amplitude
    decay) tuning -- a classic baseline, often detuned in practice.
    """
    k, tau, lag = jnp.asarray(model.gain), jnp.asarray(model.tau), _l(model)
    r = tau / (k * lag)
    if controller == "P":
        return PID(kc=r, tau_i=jnp.inf, tau_d=0.0, **kwargs)  # type: ignore[arg-type]
    if controller == "PI":
        return PID(kc=0.9 * r, tau_i=3.33 * lag, tau_d=0.0, **kwargs)  # type: ignore[arg-type]
    if controller == "PID":
        return PID(kc=1.2 * r, tau_i=2.0 * lag, tau_d=0.5 * lag, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"controller must be 'P', 'PI' or 'PID', got {controller!r}")


def cohen_coon(model: FOPDTModel, *, controller: str = "PID", **kwargs: ArrayLike) -> PID:
    """Cohen-Coon tuning from an FOPDT model (better than ZN for larger ``L/tau``)."""
    k, tau, lag = jnp.asarray(model.gain), jnp.asarray(model.tau), _l(model)
    ratio = lag / tau
    if controller == "PI":
        kc = (1.0 / k) * (tau / lag) * (0.9 + ratio / 12.0)
        tau_i = lag * (30.0 + 3.0 * ratio) / (9.0 + 20.0 * ratio)
        return PID(kc=kc, tau_i=tau_i, tau_d=0.0, **kwargs)  # type: ignore[arg-type]
    if controller == "PID":
        kc = (1.0 / k) * (tau / lag) * (4.0 / 3.0 + ratio / 4.0)
        tau_i = lag * (32.0 + 6.0 * ratio) / (13.0 + 8.0 * ratio)
        tau_d = lag * 4.0 / (11.0 + 2.0 * ratio)
        return PID(kc=kc, tau_i=tau_i, tau_d=tau_d, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"controller must be 'PI' or 'PID', got {controller!r}")


def imc_tuning(
    model: FOPDTModel,
    *,
    tau_c: ArrayLike | None = None,
    controller: str = "PI",
    **kwargs: ArrayLike,
) -> PID:
    """Internal-model-control (lambda) tuning from an FOPDT model.

    ``tau_c`` is the desired closed-loop time constant; it defaults to
    ``max(0.1 tau, 0.8 L)``, a robust middle-of-the-road choice. Larger ``tau_c``
    gives a slower but more robust loop.
    """
    k, tau, lag = jnp.asarray(model.gain), jnp.asarray(model.tau), _l(model)
    tc = jnp.maximum(0.1 * tau, 0.8 * lag) if tau_c is None else jnp.asarray(tau_c)
    if controller == "PI":
        return PID(kc=tau / (k * (tc + lag)), tau_i=tau, tau_d=0.0, **kwargs)  # type: ignore[arg-type]
    if controller == "PID":
        kc = (tau + 0.5 * lag) / (k * (tc + 0.5 * lag))
        tau_i = tau + 0.5 * lag
        tau_d = tau * lag / (2.0 * tau + lag)
        return PID(kc=kc, tau_i=tau_i, tau_d=tau_d, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"controller must be 'PI' or 'PID', got {controller!r}")


def amigo(model: FOPDTModel, *, controller: str = "PI", **kwargs: ArrayLike) -> PID:
    """AMIGO (Astrom-Hagglund) robust tuning from an FOPDT model."""
    k, tau, lag = jnp.asarray(model.gain), jnp.asarray(model.tau), _l(model)
    if controller == "PI":
        kc = (1.0 / k) * (0.15 + (0.35 - lag * tau / (lag + tau) ** 2) * (tau / lag))
        tau_i = 0.35 * lag + (13.0 * lag * tau**2) / (tau**2 + 12.0 * lag * tau + 7.0 * lag**2)
        return PID(kc=kc, tau_i=tau_i, tau_d=0.0, **kwargs)  # type: ignore[arg-type]
    if controller == "PID":
        kc = (1.0 / k) * (0.2 + 0.45 * tau / lag)
        tau_i = (0.4 * lag + 0.8 * tau) / (lag + 0.1 * tau) * lag
        tau_d = 0.5 * lag * tau / (0.3 * lag + tau)
        return PID(kc=kc, tau_i=tau_i, tau_d=tau_d, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"controller must be 'PI' or 'PID', got {controller!r}")


__all__ = [
    "FOPDTModel",
    "amigo",
    "cohen_coon",
    "fit_fopdt",
    "imc_tuning",
    "ziegler_nichols",
]
