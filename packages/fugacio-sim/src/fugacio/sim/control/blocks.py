"""Linear dynamic blocks and analytic step responses for control studies.

These are the small building blocks of classical process control: first- and
second-order lags, the first-order-plus-dead-time (FOPDT) model that almost every
tuning rule is built on, lead-lag compensators, and the static nonlinearities
(saturation, dead band, rate limit) that real actuators impose. Each linear block
is offered two ways:

* an **analytic step response** (closed form), handy for plotting, for fitting an
  FOPDT model, and for checking the numerical integrators; and
* a **state-space realization** (``A, B, C, D`` arrays), so the block can be
  dropped into a dynamic simulation and integrated with everything else.

Everything is `jax.numpy`, hence differentiable in the block parameters
(gain, time constants, damping), the basis for gradient-based identification.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float


# --------------------------------------------------------------------------- #
# Analytic step responses
# --------------------------------------------------------------------------- #
def first_order_step(t: ArrayLike, gain: ArrayLike, tau: ArrayLike, *, u: ArrayLike = 1.0) -> Array:
    """Response of ``K/(tau s + 1)`` to a step of size ``u`` at ``t = 0``.

    ``y(t) = K u (1 - exp(-t/tau))`` for ``t >= 0``.
    """
    t = jnp.asarray(t)
    resp = (
        jnp.asarray(gain)
        * jnp.asarray(u)
        * (1.0 - jnp.exp(-jnp.clip(t, 0.0, None) / jnp.asarray(tau)))
    )
    return jnp.where(t < 0.0, 0.0, resp)


def fopdt_step(
    t: ArrayLike, gain: ArrayLike, tau: ArrayLike, dead_time: ArrayLike, *, u: ArrayLike = 1.0
) -> Array:
    """Response of a first-order-plus-dead-time process to a step of size ``u``.

    ``y(t) = K u (1 - exp(-(t - L)/tau))`` for ``t >= L`` (the dead time ``L``),
    else ``0``. The workhorse model for PID tuning.
    """
    t = jnp.asarray(t)
    shifted = t - jnp.asarray(dead_time)
    resp = (
        jnp.asarray(gain)
        * jnp.asarray(u)
        * (1.0 - jnp.exp(-jnp.clip(shifted, 0.0, None) / jnp.asarray(tau)))
    )
    return jnp.where(shifted < 0.0, 0.0, resp)


def second_order_step(
    t: ArrayLike, gain: ArrayLike, wn: ArrayLike, zeta: ArrayLike, *, u: ArrayLike = 1.0
) -> Array:
    """Step response of ``K wn^2 / (s^2 + 2 zeta wn s + wn^2)`` (size ``u``).

    Handles the under-, critically-, and over-damped regimes with a single smooth
    expression (the under/over branches are selected by `jax.numpy.where`, so
    the result is differentiable in ``zeta`` through ``zeta = 1`` as well).
    """
    t = jnp.clip(jnp.asarray(t), 0.0, None)
    k = jnp.asarray(gain) * jnp.asarray(u)
    wn = jnp.asarray(wn)
    zeta = jnp.asarray(zeta)
    # Underdamped (zeta < 1).
    wd = wn * jnp.sqrt(jnp.abs(1.0 - zeta**2) + 1e-30)
    phi = jnp.arctan2(jnp.sqrt(jnp.abs(1.0 - zeta**2) + 1e-30), zeta)
    under = 1.0 - jnp.exp(-zeta * wn * t) * jnp.sin(wd * t + phi) / jnp.sqrt(
        jnp.abs(1.0 - zeta**2) + 1e-30
    )
    # Overdamped (zeta > 1): two real poles.
    r = wn * jnp.sqrt(jnp.abs(zeta**2 - 1.0) + 1e-30)
    s1 = -zeta * wn + r
    s2 = -zeta * wn - r
    over = 1.0 + (s2 * jnp.exp(s1 * t) - s1 * jnp.exp(s2 * t)) / (s1 - s2 + 1e-30)
    # Critically damped (zeta ~ 1).
    crit = 1.0 - jnp.exp(-wn * t) * (1.0 + wn * t)
    shape = jnp.where(zeta < 0.999, under, jnp.where(zeta > 1.001, over, crit))
    return k * shape


# --------------------------------------------------------------------------- #
# State-space realizations (for embedding in a dynamic simulation)
# --------------------------------------------------------------------------- #
def first_order_ss(gain: ArrayLike, tau: ArrayLike) -> tuple[Array, Array, Array, Array]:
    """State-space ``(A, B, C, D)`` of ``K/(tau s + 1)`` (one state)."""
    tau = jnp.asarray(tau)
    a = jnp.array([[-1.0]]) / tau
    b = jnp.array([[1.0]]) / tau
    c = jnp.asarray(gain).reshape(1, 1)
    d = jnp.zeros((1, 1))
    return a, b, c, d


def second_order_ss(
    gain: ArrayLike, wn: ArrayLike, zeta: ArrayLike
) -> tuple[Array, Array, Array, Array]:
    """State-space ``(A, B, C, D)`` of the standard second-order block (two states)."""
    wn = jnp.asarray(wn)
    zeta = jnp.asarray(zeta)
    a = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    a = a.at[1, 0].set(-(wn**2)).at[1, 1].set(-2.0 * zeta * wn)
    b = jnp.array([[0.0], [1.0]]) * (wn**2)
    c = jnp.zeros((1, 2)).at[0, 0].set(jnp.asarray(gain))
    d = jnp.zeros((1, 1))
    return a, b, c, d


def lead_lag(
    t: ArrayLike, gain: ArrayLike, tau_lead: ArrayLike, tau_lag: ArrayLike, *, u: ArrayLike = 1.0
) -> Array:
    """Step response of a lead-lag compensator ``K (tau_lead s + 1)/(tau_lag s + 1)``."""
    t = jnp.clip(jnp.asarray(t), 0.0, None)
    k = jnp.asarray(gain) * jnp.asarray(u)
    tl = jnp.asarray(tau_lead)
    tg = jnp.asarray(tau_lag)
    return k * (1.0 - (1.0 - tl / tg) * jnp.exp(-t / tg))


# --------------------------------------------------------------------------- #
# Static actuator nonlinearities (differentiable a.e.)
# --------------------------------------------------------------------------- #
def saturate(u: ArrayLike, u_min: ArrayLike, u_max: ArrayLike) -> Array:
    """Clamp ``u`` to ``[u_min, u_max]``."""
    return jnp.clip(jnp.asarray(u), jnp.asarray(u_min), jnp.asarray(u_max))


def dead_band(e: ArrayLike, width: ArrayLike) -> Array:
    """Symmetric dead band of total width ``2*width`` centred at zero."""
    e = jnp.asarray(e)
    w = jnp.asarray(width)
    return jnp.sign(e) * jnp.clip(jnp.abs(e) - w, 0.0, None)


def rate_limit(du_desired: ArrayLike, max_rate: ArrayLike) -> Array:
    """Limit a desired rate of change to ``+/- max_rate``."""
    return jnp.clip(jnp.asarray(du_desired), -jnp.asarray(max_rate), jnp.asarray(max_rate))


__all__ = [
    "dead_band",
    "first_order_ss",
    "first_order_step",
    "fopdt_step",
    "lead_lag",
    "rate_limit",
    "saturate",
    "second_order_ss",
    "second_order_step",
]
