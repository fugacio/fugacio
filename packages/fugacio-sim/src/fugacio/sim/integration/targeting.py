"""Energy targeting: the problem table algorithm, composite curves, and the pinch.

This is the differentiable heart of heat integration. Given a set of hot and cold
`HeatStream` objects and a minimum
approach temperature ``dt_min``, the **problem table algorithm** computes the
thermodynamic minimum hot- and cold-utility duties and locates the **pinch**, the
temperature that divides the process into a heat-deficit region (above) and a
heat-surplus region (below) and sets the ceiling on heat recovery.

The construction is the textbook one (Linnhoff & Flower):

1. *Shift* temperatures into a common interval scale (hot streams down by
   ``dt_min / 2`` and cold streams up by ``dt_min / 2``) so that a hot and a
   cold stream exactly ``dt_min`` apart in real temperature coincide, and any
   overlap in shifted temperature is feasible heat exchange.
2. Form temperature intervals at the shifted supply/target temperatures and, in
   each, net the hot against the cold heat-capacity flowrates to get the interval
   heat surplus/deficit.
3. *Cascade* heat down the intervals. The most negative point of the cascade
   started with zero input is the minimum hot utility; adding it makes the
   cascade non-negative everywhere and the point where it touches zero is the
   pinch. The heat leaving the bottom is the minimum cold utility.

Every quantity is a smooth (a.e.) function of the stream temperatures, the
heat-capacity flowrates, and ``dt_min``, so ``jax.grad`` flows through the whole
target, the basis for gradient-based heat-recovery optimisation.

Composite curves (`composite_curves`) and the grand composite curve
(`grand_composite_curve`) return the canonical temperature-enthalpy data
for plotting and area targeting.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from fugacio.sim.integration.streams import HeatStream, stack

ArrayLike = Array | float

#: Duty (W) below which a utility target is treated as zero (threshold problem).
_PINCH_TOL = 1e-6


class HeatCascade(NamedTuple):
    """The problem-table heat cascade and the targets read off it.

    Attributes:
        dt_min: Minimum approach temperature used (K).
        shifted_temperatures: Interval-boundary temperatures on the shifted
            scale, descending (K), shape ``(m,)``.
        interval_cp: Net ``sum(CP_hot) - sum(CP_cold)`` in each interval (W/K),
            shape ``(m - 1,)``.
        interval_dh: Net heat surplus of each interval (W), shape ``(m - 1,)``.
        cascade: Feasible (non-negative) cascaded heat flow at each boundary (W),
            shape ``(m,)``; ``cascade[0]`` is the hot utility, ``cascade[-1]`` the
            cold utility, and it touches zero at the pinch.
        hot_utility: Minimum hot-utility duty ``Q_h,min`` (W).
        cold_utility: Minimum cold-utility duty ``Q_c,min`` (W).
        pinch_shifted_temperature: Shifted temperature of the pinch (K).
        has_pinch: Whether a genuine pinch exists (both utilities non-zero).
    """

    dt_min: Array
    shifted_temperatures: Array
    interval_cp: Array
    interval_dh: Array
    cascade: Array
    hot_utility: Array
    cold_utility: Array
    pinch_shifted_temperature: Array
    has_pinch: Array


def heat_cascade(streams: list[HeatStream], dt_min: ArrayLike) -> HeatCascade:
    """Run the problem table algorithm and return the full `HeatCascade`.

    Args:
        streams: The hot and cold process streams (classified by their own
            supply/target temperatures).
        dt_min: Minimum approach temperature ``dt_min`` (K).

    Returns:
        A `HeatCascade` carrying the minimum utilities, the pinch, and the
        interval data.
    """
    t_supply, t_target, cp, _ = stack(streams)
    dt = jnp.asarray(dt_min, dtype=float)
    half = 0.5 * dt

    is_hot = t_supply > t_target
    # Shift each stream onto the common (interval) temperature scale.
    shift = jnp.where(is_hot, -half, half)
    sh_supply = t_supply + shift
    sh_target = t_target + shift
    hi = jnp.maximum(sh_supply, sh_target)
    lo = jnp.minimum(sh_supply, sh_target)
    # Signed heat-capacity flowrate: hot streams release (+), cold absorb (-).
    signed_cp = jnp.where(is_hot, cp, -cp)

    # Interval boundaries: every shifted supply/target temperature, descending.
    edges = jnp.sort(jnp.concatenate([hi, lo]))[::-1]
    top = edges[:-1]
    bottom = edges[1:]
    mid = 0.5 * (top + bottom)

    # A stream is present in an interval when the interval midpoint lies within
    # its shifted temperature span. (n_intervals, n_streams)
    present = (lo[None, :] <= mid[:, None]) & (mid[:, None] <= hi[None, :])
    interval_cp = jnp.sum(present * signed_cp[None, :], axis=1)
    interval_dh = interval_cp * (top - bottom)

    # Infeasible cascade (zero heat input at the top), then shift up to feasible.
    infeasible = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(interval_dh)])
    hot_utility = jnp.maximum(0.0, -jnp.min(infeasible))
    cascade = infeasible + hot_utility
    cold_utility = cascade[-1]
    pinch_index = jnp.argmin(infeasible)
    pinch_shifted = edges[pinch_index]
    has_pinch = (hot_utility > _PINCH_TOL) & (cold_utility > _PINCH_TOL)

    return HeatCascade(
        dt_min=dt,
        shifted_temperatures=edges,
        interval_cp=interval_cp,
        interval_dh=interval_dh,
        cascade=cascade,
        hot_utility=hot_utility,
        cold_utility=cold_utility,
        pinch_shifted_temperature=pinch_shifted,
        has_pinch=has_pinch,
    )


class PinchResult(NamedTuple):
    """Headline energy targets for a heat-integration problem.

    Attributes:
        dt_min: Minimum approach temperature (K).
        hot_utility: Minimum hot-utility duty ``Q_h,min`` (W).
        cold_utility: Minimum cold-utility duty ``Q_c,min`` (W).
        heat_recovery: Process-to-process heat recovered at the target (W).
        pinch_temperature: Mean (shifted) pinch temperature (K).
        hot_pinch_temperature: Pinch temperature on the hot-stream scale (K).
        cold_pinch_temperature: Pinch temperature on the cold-stream scale (K).
        has_pinch: Whether a genuine pinch exists (``False`` for a threshold
            problem needing only one utility).
    """

    dt_min: Array
    hot_utility: Array
    cold_utility: Array
    heat_recovery: Array
    pinch_temperature: Array
    hot_pinch_temperature: Array
    cold_pinch_temperature: Array
    has_pinch: Array


def pinch_analysis(streams: list[HeatStream], dt_min: ArrayLike) -> PinchResult:
    """Compute the minimum utilities, heat recovery, and pinch temperatures.

    Args:
        streams: Hot and cold process streams.
        dt_min: Minimum approach temperature (K).

    Returns:
        A `PinchResult`.
    """
    casc = heat_cascade(streams, dt_min)
    half = 0.5 * casc.dt_min
    _, _, cp, _ = stack(streams)
    t_supply, t_target, _, _ = stack(streams)
    is_hot = t_supply > t_target
    total_hot_duty = jnp.sum(jnp.where(is_hot, cp * jnp.abs(t_supply - t_target), 0.0))
    # Heat recovered = hot duty served by the process rather than cold utility.
    heat_recovery = total_hot_duty - casc.cold_utility
    return PinchResult(
        dt_min=casc.dt_min,
        hot_utility=casc.hot_utility,
        cold_utility=casc.cold_utility,
        heat_recovery=heat_recovery,
        pinch_temperature=casc.pinch_shifted_temperature,
        hot_pinch_temperature=casc.pinch_shifted_temperature + half,
        cold_pinch_temperature=casc.pinch_shifted_temperature - half,
        has_pinch=casc.has_pinch,
    )


def minimum_utilities(streams: list[HeatStream], dt_min: ArrayLike) -> tuple[Array, Array]:
    """Return just ``(hot_utility, cold_utility)`` minimum duties (W), a convenience."""
    casc = heat_cascade(streams, dt_min)
    return casc.hot_utility, casc.cold_utility


class CompositeSegments(NamedTuple):
    """A composite curve as contiguous temperature segments (one side, hot or cold).

    Attributes:
        t_lo: Lower temperature of each segment (K), shape ``(k,)``.
        t_hi: Upper temperature of each segment (K), shape ``(k,)``.
        h_lo: Cumulative enthalpy at ``t_lo`` (W), shape ``(k,)``.
        h_hi: Cumulative enthalpy at ``t_hi`` (W), shape ``(k,)``.
        cp: Total heat-capacity flowrate of the segment (W/K), shape ``(k,)``.
        inv_h: Enthalpy-weighted mean film resistance ``sum(CP/h)/sum(CP)``
            (m^2*K/W), shape ``(k,)``, the area-target weight of the segment.
    """

    t_lo: Array
    t_hi: Array
    h_lo: Array
    h_hi: Array
    cp: Array
    inv_h: Array


def _side_segments(streams: list[HeatStream], *, hot: bool) -> CompositeSegments:
    """Build the composite segments for one side from its member streams.

    The hot/cold classification is structural, so it is read from the concrete
    temperature leaves via ``float`` (which works even inside a trace, where a
    staged ``>`` comparison would not); the segment *values* (enthalpies,
    ``CP``) remain differentiable.
    """
    side = [s for s in streams if (float(s.t_supply) > float(s.t_target)) == hot]
    if not side:
        zero = jnp.zeros((0,))
        return CompositeSegments(zero, zero, zero, zero, zero, zero)
    t_lo_s = jnp.stack([s.t_cold for s in side])
    t_hi_s = jnp.stack([s.t_hot for s in side])
    cp_s = jnp.stack([jnp.asarray(s.cp, dtype=float) for s in side])
    h_s = jnp.stack([jnp.asarray(s.h, dtype=float) for s in side])

    # Sorted (fixed-size) breakpoints; duplicate temperatures simply yield
    # zero-width segments, so the shapes stay static and jit-/grad-traceable.
    temps = jnp.sort(jnp.concatenate([t_lo_s, t_hi_s]))  # ascending
    seg_lo = temps[:-1]
    seg_hi = temps[1:]
    seg_mid = 0.5 * (seg_lo + seg_hi)
    present = (t_lo_s[None, :] <= seg_mid[:, None]) & (seg_mid[:, None] <= t_hi_s[None, :])
    seg_cp = jnp.sum(present * cp_s[None, :], axis=1)
    seg_cp_over_h = jnp.sum(present * (cp_s / h_s)[None, :], axis=1)
    inv_h = seg_cp_over_h / jnp.where(seg_cp > 0.0, seg_cp, 1.0)

    dh = seg_cp * (seg_hi - seg_lo)
    h_boundaries = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(dh)])
    return CompositeSegments(
        t_lo=seg_lo,
        t_hi=seg_hi,
        h_lo=h_boundaries[:-1],
        h_hi=h_boundaries[1:],
        cp=seg_cp,
        inv_h=inv_h,
    )


class CompositeCurves(NamedTuple):
    """Hot and cold composite curves positioned for the given ``dt_min``.

    The cold composite is offset in enthalpy by the cold-utility target so the
    curves overlap exactly over the recoverable heat and approach to ``dt_min``
    at the pinch.

    Attributes:
        hot_t, hot_h: Hot composite temperature (K) and enthalpy (W) breakpoints.
        cold_t, cold_h: Cold composite temperature (K) and enthalpy (W).
        min_approach: Minimum vertical temperature gap between the curves (K);
            equals ``dt_min`` at a pinched problem.
    """

    hot_t: Array
    hot_h: Array
    cold_t: Array
    cold_h: Array
    min_approach: Array


def composite_curves(streams: list[HeatStream], dt_min: ArrayLike) -> CompositeCurves:
    """Hot and cold composite curves (temperature-enthalpy), positioned by ``dt_min``.

    Args:
        streams: Hot and cold process streams.
        dt_min: Minimum approach temperature (K).

    Returns:
        A `CompositeCurves` with the two curves and the achieved minimum
        approach (a consistency check: it equals ``dt_min`` for a pinched
        problem).
    """
    casc = heat_cascade(streams, dt_min)
    hot = _side_segments(streams, hot=True)
    cold = _side_segments(streams, hot=False)

    hot_t = jnp.concatenate([hot.t_lo[:1], hot.t_hi])
    hot_h = jnp.concatenate([hot.h_lo[:1], hot.h_hi])
    cold_t = jnp.concatenate([cold.t_lo[:1], cold.t_hi])
    cold_h = jnp.concatenate([cold.h_lo[:1], cold.h_hi]) + casc.cold_utility

    # Minimum vertical approach over the overlapping enthalpy range. The gap
    # between two piecewise-linear curves is extremised at a breakpoint of
    # either, so sampling the union of breakpoints (clipped to the overlap) is
    # exact.
    overlap_lo = jnp.maximum(hot_h[0], cold_h[0])
    overlap_hi = jnp.minimum(hot_h[-1], cold_h[-1])
    samples = jnp.clip(jnp.concatenate([hot_h, cold_h]), overlap_lo, overlap_hi)
    hot_temp = jnp.interp(samples, hot_h, hot_t)
    cold_temp = jnp.interp(samples, cold_h, cold_t)
    min_approach = jnp.min(hot_temp - cold_temp)

    return CompositeCurves(
        hot_t=hot_t,
        hot_h=hot_h,
        cold_t=cold_t,
        cold_h=cold_h,
        min_approach=min_approach,
    )


class GrandComposite(NamedTuple):
    """The grand composite curve: net heat flow vs shifted temperature.

    Attributes:
        shifted_temperature: Interval-boundary shifted temperatures (K), shape
            ``(m,)``, descending.
        net_heat_flow: Feasible cascaded heat flow at each boundary (W); the
            curve touches zero at the pinch, equals the hot utility at the top and
            the cold utility at the bottom.
    """

    shifted_temperature: Array
    net_heat_flow: Array


def grand_composite_curve(streams: list[HeatStream], dt_min: ArrayLike) -> GrandComposite:
    """Grand composite curve (shifted temperature vs net heat flow).

    The GCC is the master diagram for utility selection: its shape shows where
    multiple utility levels can be placed and where heat pockets recover
    internally.
    """
    casc = heat_cascade(streams, dt_min)
    return GrandComposite(
        shifted_temperature=casc.shifted_temperatures,
        net_heat_flow=casc.cascade,
    )
