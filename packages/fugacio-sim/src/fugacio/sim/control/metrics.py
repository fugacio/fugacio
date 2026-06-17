"""Time-domain performance metrics for a response trajectory.

Given a sampled response ``(t, y)`` toward a setpoint, these compute the figures
of merit an engineer actually tunes for: overshoot, rise and settling time, and
the error integrals (IAE / ISE / ITAE). The error integrals are smooth functionals
of the trajectory, so they are the natural (and differentiable) objectives for
gradient-based controller tuning (`fugacio.sim.dynamics.tune_pid`); the
event-style metrics (overshoot, settling time) are primarily for reporting.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float


def _trapz(y: Array, t: Array) -> Array:
    return jnp.trapezoid(y, t)


def iae(t: Array, y: Array, setpoint: ArrayLike) -> Array:
    """Integral of the absolute error, ``integral |sp - y| dt``."""
    return _trapz(jnp.abs(jnp.asarray(setpoint) - y), t)


def ise(t: Array, y: Array, setpoint: ArrayLike) -> Array:
    """Integral of the squared error, ``integral (sp - y)^2 dt``."""
    return _trapz((jnp.asarray(setpoint) - y) ** 2, t)


def itae(t: Array, y: Array, setpoint: ArrayLike) -> Array:
    """Integral of time-weighted absolute error, ``integral t |sp - y| dt``."""
    return _trapz(t * jnp.abs(jnp.asarray(setpoint) - y), t)


def overshoot(y: Array, setpoint: ArrayLike, *, y0: ArrayLike | None = None) -> Array:
    """Fractional overshoot of the peak beyond the setpoint (0 if none).

    ``(y_peak - sp) / (sp - y0)`` for a positive step; ``y0`` defaults to the first
    sample. Returns 0 when the response does not exceed the setpoint.
    """
    sp = jnp.asarray(setpoint)
    start = y[0] if y0 is None else jnp.asarray(y0)
    span = sp - start
    direction = jnp.sign(span)
    peak = jnp.where(direction >= 0, jnp.max(y), jnp.min(y))
    os = (peak - sp) * direction / (jnp.abs(span) + 1e-30)
    return jnp.clip(os, 0.0, None)


def peak_time(t: Array, y: Array, setpoint: ArrayLike) -> Array:
    """Time of the response extremum in the step direction."""
    sp = jnp.asarray(setpoint)
    direction = jnp.sign(sp - y[0])
    idx = jnp.where(direction >= 0, jnp.argmax(y), jnp.argmin(y))
    return t[idx]


def rise_time(
    t: Array, y: Array, setpoint: ArrayLike, *, lo: float = 0.1, hi: float = 0.9
) -> Array:
    """Time to rise from ``lo`` to ``hi`` fraction of the step (``nan`` if never reached)."""
    sp = jnp.asarray(setpoint)
    start = y[0]
    span = sp - start
    frac = (y - start) / (span + 1e-30)
    t_lo = _first_crossing_time(t, frac, lo)
    t_hi = _first_crossing_time(t, frac, hi)
    return t_hi - t_lo


def settling_time(t: Array, y: Array, setpoint: ArrayLike, *, tol: float = 0.02) -> Array:
    """Last time the response leaves the ``+/- tol`` band around the setpoint.

    Returns the time after which ``|y - sp| <= tol * |sp - y0|`` holds for the rest
    of the record (``t[-1]`` if it never settles within the record).
    """
    sp = jnp.asarray(setpoint)
    span = jnp.abs(sp - y[0]) + 1e-30
    outside = jnp.abs(y - sp) > tol * span
    # Index of the last sample outside the band; settling time is the next sample.
    any_outside = jnp.any(outside)
    last_out = jnp.max(jnp.where(outside, jnp.arange(t.shape[0]), 0))
    idx = jnp.minimum(last_out + 1, t.shape[0] - 1)
    return jnp.where(any_outside, t[idx], t[0])


def steady_state_error(y: Array, setpoint: ArrayLike) -> Array:
    """Offset of the final value from the setpoint, ``sp - y[-1]``."""
    return jnp.asarray(setpoint) - y[-1]


def _first_crossing_time(t: Array, frac: Array, level: float) -> Array:
    """Linear-interpolated time at which ``frac`` first reaches ``level``."""
    reached = frac >= level
    idx = jnp.argmax(reached)
    idx = jnp.clip(idx, 1, t.shape[0] - 1)
    f0, f1 = frac[idx - 1], frac[idx]
    t0, t1 = t[idx - 1], t[idx]
    w = jnp.clip((level - f0) / (f1 - f0 + 1e-30), 0.0, 1.0)
    interp = t0 + w * (t1 - t0)
    return jnp.where(jnp.any(reached), interp, jnp.nan)


class StepInfo(NamedTuple):
    """A bundle of step-response metrics.

    Attributes:
        overshoot: Fractional overshoot beyond the setpoint.
        peak_time: Time of the response extremum.
        rise_time: 10-90% rise time.
        settling_time: 2% settling time.
        steady_state_error: Final offset ``sp - y[-1]``.
        iae: Integral of absolute error.
    """

    overshoot: Array
    peak_time: Array
    rise_time: Array
    settling_time: Array
    steady_state_error: Array
    iae: Array


def step_info(t: Array, y: Array, setpoint: ArrayLike, *, settle_tol: float = 0.02) -> StepInfo:
    """Compute the standard step-response metrics for a trajectory ``(t, y)``."""
    return StepInfo(
        overshoot=overshoot(y, setpoint),
        peak_time=peak_time(t, y, setpoint),
        rise_time=rise_time(t, y, setpoint),
        settling_time=settling_time(t, y, setpoint, tol=settle_tol),
        steady_state_error=steady_state_error(y, setpoint),
        iae=iae(t, y, setpoint),
    )


__all__ = [
    "StepInfo",
    "iae",
    "ise",
    "itae",
    "overshoot",
    "peak_time",
    "rise_time",
    "settling_time",
    "steady_state_error",
    "step_info",
]
