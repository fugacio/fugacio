"""Area, unit-count, and capital targeting, and the capital-energy trade-off.

Energy targets (`fugacio.sim.integration.targeting`) fix the *operating*
cost floor; to trade it against *capital* you need the network's area and unit
count before designing it. This module supplies the classic targets:

* `area_target` -- the **Bath formula** minimum heat-transfer area from the
  balanced composite curves (vertical heat exchange), accounting for individual
  stream film coefficients;
* `units_target` -- the minimum number of heat-exchange units from Euler's
  graph relation, respecting the pinch division (the MER unit count);
* `capital_cost_target` -- an installed-capital estimate from the area and
  unit targets via a smooth exchanger cost law;
* `supertarget` and `total_annual_cost_target` -- the total annual
  cost (annualised capital plus utilities) at a given ``dt_min``;
* `optimal_dt_min` -- the ``dt_min`` that minimises total annual cost,
  found by gradient-based optimisation straight through the differentiable
  targets (the "supertargeting" curve), the showcase of an end-to-end
  differentiable heat-integration model.

Everything is differentiable in the stream data and ``dt_min``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.economics import (
    HOURS_PER_YEAR,
    annualized_capital,
    lmtd,
    utility_cost,
)
from fugacio.sim.integration.streams import HeatStream, stack
from fugacio.sim.integration.targeting import (
    _side_segments,
    heat_cascade,
)

ArrayLike = Array | float

#: Reciprocal golden ratio, for the golden-section search in `optimal_dt_min`.
_INV_PHI = 0.6180339887498949

#: Default smooth exchanger installed-cost law ``cost = a + b * A**c`` ($, A in m^2).
DEFAULT_AREA_COST = (0.0, 1200.0, 0.6)


def _process_duties(streams: list[HeatStream]) -> tuple[Array, Array]:
    """Total hot and cold process duties ``(Q_hot, Q_cold)`` (W)."""
    t_supply, t_target, cp, _ = stack(streams)
    is_hot = t_supply > t_target
    duty = cp * jnp.abs(t_supply - t_target)
    q_hot = jnp.sum(jnp.where(is_hot, duty, 0.0))
    q_cold = jnp.sum(jnp.where(is_hot, 0.0, duty))
    return q_hot, q_cold


def _temperature_extents(streams: list[HeatStream]) -> tuple[Array, Array]:
    """Hottest and coldest process temperatures across all streams (K)."""
    t_supply, t_target, _, _ = stack(streams)
    t_max = jnp.maximum(jnp.max(t_supply), jnp.max(t_target))
    t_min = jnp.minimum(jnp.min(t_supply), jnp.min(t_target))
    return t_max, t_min


def _side_index(boundaries_hi: Array, point: Array) -> Array:
    """Index of the interval (with ascending upper bounds) that contains ``point``."""
    n = boundaries_hi.shape[0]
    return jnp.clip(jnp.searchsorted(boundaries_hi, point, side="left"), 0, n - 1)


def _temperature_at(
    enthalpy: Array, h_lo: Array, h_hi: Array, t_lo: Array, t_hi: Array, idx: Array
) -> Array:
    """Linear-interpolated temperature at ``enthalpy`` within interval ``idx``."""
    lo = h_lo[idx]
    hi = h_hi[idx]
    span = hi - lo
    frac = jnp.where(span > 1e-12, (enthalpy - lo) / jnp.where(span > 1e-12, span, 1.0), 0.0)
    return t_lo[idx] + (t_hi[idx] - t_lo[idx]) * frac


def area_target(
    streams: list[HeatStream],
    dt_min: ArrayLike,
    *,
    hot_utility_t: ArrayLike | None = None,
    cold_utility_t: ArrayLike | None = None,
    hot_utility_h: ArrayLike = 5000.0,
    cold_utility_h: ArrayLike = 5000.0,
) -> Array:
    """Minimum heat-transfer area target (m^2) from the balanced composite curves.

    Implements the Bath formula: the balanced composite curves (process streams
    plus the utility duties that close the enthalpy balance) are split into
    enthalpy intervals, and within each the area for vertical heat exchange is

        ``A_k = (dH_k / dT_lm,k) * sum_i (CP_i / h_i)``

    summed over the hot and cold streams present, with ``dT_lm`` the log-mean of
    the hot-cold temperature gaps at the interval ends. Summing over intervals
    gives a target no real (vertically-matched) network can beat. Differentiable
    in the stream data, film coefficients, and ``dt_min``.

    Args:
        streams: Hot and cold process streams (each carrying a film coefficient
            ``h``).
        dt_min: Minimum approach temperature (K).
        hot_utility_t: Hot-utility temperature (K); defaults to just above the
            hottest process temperature.
        cold_utility_t: Cold-utility temperature (K); defaults to just below the
            coldest process temperature.
        hot_utility_h: Hot-utility film coefficient (W/m^2/K).
        cold_utility_h: Cold-utility film coefficient (W/m^2/K).

    Returns:
        The area target (m^2).
    """
    casc = heat_cascade(streams, dt_min)
    q_h = casc.hot_utility
    q_c = casc.cold_utility
    q_hot, _q_cold = _process_duties(streams)
    h_total = q_hot + q_h

    t_max, t_min = _temperature_extents(streams)
    dt = jnp.asarray(dt_min, dtype=float)
    t_hu = t_max + dt if hot_utility_t is None else jnp.asarray(hot_utility_t, dtype=float)
    t_cu = t_min - dt if cold_utility_t is None else jnp.asarray(cold_utility_t, dtype=float)

    hot = _side_segments(streams, hot=True)
    cold = _side_segments(streams, hot=False)

    # Hot side: process composite [0, q_hot] then the (isothermal) hot utility.
    hot_h_lo = jnp.concatenate([hot.h_lo, jnp.atleast_1d(q_hot)])
    hot_h_hi = jnp.concatenate([hot.h_hi, jnp.atleast_1d(h_total)])
    hot_t_lo = jnp.concatenate([hot.t_lo, jnp.atleast_1d(t_hu)])
    hot_t_hi = jnp.concatenate([hot.t_hi, jnp.atleast_1d(t_hu)])
    hot_inv = jnp.concatenate([hot.inv_h, jnp.atleast_1d(1.0 / jnp.asarray(hot_utility_h))])

    # Cold side: the (isothermal) cold utility [0, q_c] then the process composite.
    cold_h_lo = jnp.concatenate([jnp.zeros((1,)), cold.h_lo + q_c])
    cold_h_hi = jnp.concatenate([jnp.atleast_1d(q_c), cold.h_hi + q_c])
    cold_t_lo = jnp.concatenate([jnp.atleast_1d(t_cu), cold.t_lo])
    cold_t_hi = jnp.concatenate([jnp.atleast_1d(t_cu), cold.t_hi])
    cold_inv = jnp.concatenate([jnp.atleast_1d(1.0 / jnp.asarray(cold_utility_h)), cold.inv_h])

    # Common enthalpy grid = union of every interval boundary on both sides.
    boundaries = jnp.sort(jnp.concatenate([hot_h_lo, hot_h_hi, cold_h_lo, cold_h_hi]))
    a = boundaries[:-1]
    b = boundaries[1:]
    dh = b - a
    mid = 0.5 * (a + b)

    hot_idx = _side_index(hot_h_hi, mid)
    cold_idx = _side_index(cold_h_hi, mid)
    hot_t_a = _temperature_at(a, hot_h_lo, hot_h_hi, hot_t_lo, hot_t_hi, hot_idx)
    hot_t_b = _temperature_at(b, hot_h_lo, hot_h_hi, hot_t_lo, hot_t_hi, hot_idx)
    cold_t_a = _temperature_at(a, cold_h_lo, cold_h_hi, cold_t_lo, cold_t_hi, cold_idx)
    cold_t_b = _temperature_at(b, cold_h_lo, cold_h_hi, cold_t_lo, cold_t_hi, cold_idx)

    dt_a = jnp.clip(hot_t_a - cold_t_a, 1e-6, None)
    dt_b = jnp.clip(hot_t_b - cold_t_b, 1e-6, None)
    lm = lmtd(dt_a, dt_b)
    inv_h_sum = hot_inv[hot_idx] + cold_inv[cold_idx]
    contributions = jnp.where(dh > 1e-12, dh / lm * inv_h_sum, 0.0)
    return jnp.sum(contributions)


class UnitsTarget(NamedTuple):
    """Minimum heat-exchange unit count target.

    Attributes:
        units: Minimum number of units (MER count when a pinch exists, else the
            single-region ``N - 1``).
        above_pinch: Stream + utility count above the pinch.
        below_pinch: Stream + utility count below the pinch.
    """

    units: Array
    above_pinch: Array
    below_pinch: Array


def units_target(streams: list[HeatStream], dt_min: ArrayLike) -> UnitsTarget:
    """Minimum number of heat-exchange units (Euler's relation, pinch-respecting).

    For a network of ``S`` streams (including utilities) the minimum unit count
    is ``S - 1``; designing for minimum energy splits the problem at the pinch, so
    the MER target sums ``(S - 1)`` over the above- and below-pinch regions.
    """
    casc = heat_cascade(streams, dt_min)
    t_supply, t_target, _, _ = stack(streams)
    half = 0.5 * casc.dt_min
    is_hot = t_supply > t_target
    shift = jnp.where(is_hot, -half, half)
    sh_hi = jnp.maximum(t_supply, t_target) + shift
    sh_lo = jnp.minimum(t_supply, t_target) + shift
    pinch = casc.pinch_shifted_temperature
    tol = 1e-6

    has_above = sh_hi > pinch + tol
    has_below = sh_lo < pinch - tol
    n_above = jnp.sum(has_above) + (casc.hot_utility > tol).astype(float)
    n_below = jnp.sum(has_below) + (casc.cold_utility > tol).astype(float)

    n_total = (
        float(len(streams))
        + (casc.hot_utility > tol).astype(float)
        + (casc.cold_utility > tol).astype(float)
    )
    mer_units = jnp.maximum(n_above - 1.0, 0.0) + jnp.maximum(n_below - 1.0, 0.0)
    units = jnp.where(casc.has_pinch, mer_units, jnp.maximum(n_total - 1.0, 0.0))
    return UnitsTarget(units=units, above_pinch=n_above, below_pinch=n_below)


def capital_cost_target(
    streams: list[HeatStream],
    dt_min: ArrayLike,
    *,
    area_cost: tuple[float, float, float] = DEFAULT_AREA_COST,
    area_kwargs: dict | None = None,
) -> Array:
    """Installed-capital target ($) from the area and unit targets.

    Distributes the area target equally over the unit target and costs each
    exchanger by the smooth law ``a + b * (A / N)**c``; differentiable in the
    stream data and ``dt_min``.
    """
    area = area_target(streams, dt_min, **(area_kwargs or {}))
    units = jnp.maximum(units_target(streams, dt_min).units, 1.0)
    a, b, c = area_cost
    area_each = area / units
    return units * (a + b * area_each**c)


class SuperTargetResult(NamedTuple):
    """Total-annual-cost target and its breakdown at a given ``dt_min``.

    Attributes:
        dt_min: Minimum approach temperature (K).
        hot_utility: Minimum hot-utility duty (W).
        cold_utility: Minimum cold-utility duty (W).
        area: Area target (m^2).
        units: Unit-count target.
        capital: Installed-capital target ($).
        annualized_capital: Annualised capital charge ($/yr).
        utility_cost: Annual utility (operating) cost ($/yr).
        total_annual_cost: Annualised capital plus utilities ($/yr).
    """

    dt_min: Array
    hot_utility: Array
    cold_utility: Array
    area: Array
    units: Array
    capital: Array
    annualized_capital: Array
    utility_cost: Array
    total_annual_cost: Array


def total_annual_cost_target(
    streams: list[HeatStream],
    dt_min: ArrayLike,
    *,
    hot_utility: str = "hp_steam",
    cold_utility: str = "cooling_water",
    hours_per_year: ArrayLike = HOURS_PER_YEAR,
    interest_rate: ArrayLike = 0.1,
    years: ArrayLike = 10.0,
    area_cost: tuple[float, float, float] = DEFAULT_AREA_COST,
    area_kwargs: dict | None = None,
) -> SuperTargetResult:
    """Total annual cost (annualised capital + utilities) at a given ``dt_min``.

    Args:
        streams: Hot and cold process streams.
        dt_min: Minimum approach temperature (K).
        hot_utility: Hot-utility key priced via `fugacio.sim.economics.UTILITIES`.
        cold_utility: Cold-utility key priced via `fugacio.sim.economics.UTILITIES`.
        hours_per_year: Operating hours per year.
        interest_rate: Annual interest rate for the capital-recovery factor.
        years: Project life (years) for the capital-recovery factor.
        area_cost: ``(a, b, c)`` exchanger cost-law coefficients.
        area_kwargs: Extra keyword arguments forwarded to `area_target`.

    Returns:
        A `SuperTargetResult`.
    """
    casc = heat_cascade(streams, dt_min)
    area = area_target(streams, dt_min, **(area_kwargs or {}))
    units = units_target(streams, dt_min).units
    capital = capital_cost_target(streams, dt_min, area_cost=area_cost, area_kwargs=area_kwargs)
    ann_cap = annualized_capital(capital, rate=interest_rate, years=years)
    op_cost = utility_cost(
        casc.hot_utility, hot_utility, hours_per_year=hours_per_year
    ) + utility_cost(casc.cold_utility, cold_utility, hours_per_year=hours_per_year)
    return SuperTargetResult(
        dt_min=casc.dt_min,
        hot_utility=casc.hot_utility,
        cold_utility=casc.cold_utility,
        area=area,
        units=units,
        capital=capital,
        annualized_capital=ann_cap,
        utility_cost=op_cost,
        total_annual_cost=ann_cap + op_cost,
    )


def supertarget(
    streams: list[HeatStream],
    dt_min_grid: ArrayLike,
    **kwargs: object,
) -> SuperTargetResult:
    """Vectorised `total_annual_cost_target` over a grid of ``dt_min`` values.

    Returns a `SuperTargetResult` whose fields are arrays aligned with
    ``dt_min_grid`` -- the data behind the supertargeting (cost-vs-``dt_min``)
    plot.
    """
    grid = jnp.asarray(dt_min_grid, dtype=float)
    results = [total_annual_cost_target(streams, float(dt), **kwargs) for dt in grid]  # type: ignore[arg-type]
    stacked = {
        field: jnp.stack([jnp.asarray(getattr(r, field)) for r in results])
        for field in SuperTargetResult._fields
    }
    return SuperTargetResult(**stacked)


class OptimalDtMin(NamedTuple):
    """Result of the capital-energy trade-off optimisation.

    Attributes:
        dt_min: Optimal minimum approach temperature (K).
        total_annual_cost: Total annual cost at the optimum ($/yr).
        target: The full `SuperTargetResult` at the optimum.
        converged: Whether the optimiser converged.
    """

    dt_min: Array
    total_annual_cost: Array
    target: SuperTargetResult
    converged: Array


def optimal_dt_min(
    streams: list[HeatStream],
    *,
    bounds: tuple[float, float] = (1.0, 60.0),
    grid: int = 121,
    refine_iters: int = 40,
    hot_utility: str = "hp_steam",
    cold_utility: str = "cooling_water",
    hours_per_year: ArrayLike = HOURS_PER_YEAR,
    interest_rate: ArrayLike = 0.1,
    years: ArrayLike = 10.0,
    area_cost: tuple[float, float, float] = DEFAULT_AREA_COST,
    area_kwargs: dict | None = None,
) -> OptimalDtMin:
    """Find the ``dt_min`` that minimises total annual cost (supertargeting).

    The capital-energy trade-off curve is broadly U-shaped -- the area target
    diverges as ``dt_min`` -> 0 while utilities rise with ``dt_min`` -- but it is
    only piecewise-smooth: the integer unit-count target steps at the
    threshold/pinch transitions, so the curve has genuine kinks and small jumps.
    A smooth gradient step would stall on those, so the optimum is found by a
    vectorised **grid scan** (which locates the global basin despite the jumps)
    followed by a **golden-section polish** inside the smooth neighbouring bracket.
    The total-annual-cost target itself is fully differentiable between kinks (see
    `total_annual_cost_target`).

    Args:
        streams: Hot and cold process streams.
        bounds: ``(lower, upper)`` search interval for ``dt_min`` (K).
        grid: Number of grid points for the global scan.
        refine_iters: Golden-section iterations for the local polish.
        hot_utility: Hot-utility key, forwarded to `total_annual_cost_target`.
        cold_utility: Cold-utility key, forwarded to `total_annual_cost_target`.
        hours_per_year: Operating hours per year, forwarded to `total_annual_cost_target`.
        interest_rate: Annual interest rate, forwarded to `total_annual_cost_target`.
        years: Project life (years), forwarded to `total_annual_cost_target`.
        area_cost: Exchanger cost-law coefficients, forwarded to `total_annual_cost_target`.
        area_kwargs: Extra `area_target` keyword arguments, forwarded to the target.

    Returns:
        An `OptimalDtMin` with the optimal approach temperature and the
        full target breakdown there.
    """

    def tac(dt: ArrayLike) -> Array:
        return total_annual_cost_target(
            streams,
            dt,
            hot_utility=hot_utility,
            cold_utility=cold_utility,
            hours_per_year=hours_per_year,
            interest_rate=interest_rate,
            years=years,
            area_cost=area_cost,
            area_kwargs=area_kwargs,
        ).total_annual_cost

    lo, hi = float(bounds[0]), float(bounds[1])
    xs = jnp.linspace(lo, hi, grid)
    fs = jax.vmap(tac)(xs)
    i = int(jnp.argmin(fs))
    # Golden-section polish within the bracket around the best grid point. The
    # scalar objective is jitted so the refinement loop does not recompile.
    tac = jax.jit(tac)
    a = float(xs[max(i - 1, 0)])
    b = float(xs[min(i + 1, grid - 1)])
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc, fd = float(tac(c)), float(tac(d))
    for _ in range(refine_iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = float(tac(c))
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = float(tac(d))
    dt_star = jnp.asarray(0.5 * (a + b))
    target = total_annual_cost_target(
        streams,
        dt_star,
        hot_utility=hot_utility,
        cold_utility=cold_utility,
        hours_per_year=hours_per_year,
        interest_rate=interest_rate,
        years=years,
        area_cost=area_cost,
        area_kwargs=area_kwargs,
    )
    return OptimalDtMin(
        dt_min=dt_star,
        total_annual_cost=target.total_annual_cost,
        target=target,
        converged=jnp.asarray(True),
    )
