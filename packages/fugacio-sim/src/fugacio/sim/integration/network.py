"""Heat-exchanger network synthesis by the pinch design method.

Targeting says *how much* heat to recover and *how much* area it needs; this
module proposes an actual network of exchangers that meets the minimum-utility
(MER) target. It follows Linnhoff & Hindmarsh's **pinch design method**:

1. Divide the problem at the pinch -- no heat crosses it, so the above- and
   below-pinch regions are designed independently.
2. Start matches *at* the pinch, where driving forces are tightest, honouring the
   CP-feasibility criterion (``CP_hot <= CP_cold`` above the pinch,
   ``CP_hot >= CP_cold`` below) so every exchanger keeps ``dt_min``.
3. **Tick off** each match at the larger feasible duty, eliminating one stream at
   a time, and top up the unmet duties with hot utility above the pinch and cold
   utility below.

The synthesiser is a screening-level heuristic (it does not perform automatic
multi-branch stream splitting), so its result is always passed through a rigorous
:func:`verify_network`: per-exchanger terminal approaches, per-stream energy
balances, and whether the design hits the MER utility targets. Areas use the
film coefficients carried by each :class:`~fugacio.sim.integration.streams.HeatStream`.

Unlike the targeting layer this is a discrete design step, so it works in plain
Python floats rather than as a differentiable computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fugacio.sim.economics import lmtd
from fugacio.sim.integration.streams import DEFAULT_FILM_COEFFICIENT, HeatStream
from fugacio.sim.integration.targeting import heat_cascade, pinch_analysis

_TOL = 1e-6


@dataclass
class Exchanger:
    """One heat-exchange unit in a synthesised network.

    Attributes:
        kind: ``"process"`` (stream-stream), ``"heater"`` (hot utility), or
            ``"cooler"`` (cold utility).
        hot: Label of the hot side (a stream name, or the hot-utility label).
        cold: Label of the cold side (a stream name, or the cold-utility label).
        duty: Exchanger duty (W).
        t_hot_in, t_hot_out: Hot-side inlet/outlet temperatures (K).
        t_cold_in, t_cold_out: Cold-side inlet/outlet temperatures (K).
        dt_a, dt_b: Terminal temperature approaches at the two ends (K).
        area: Heat-transfer area from ``Q / (U * LMTD)`` with series film
            resistances (m^2).
    """

    kind: str
    hot: str
    cold: str
    duty: float
    t_hot_in: float
    t_hot_out: float
    t_cold_in: float
    t_cold_out: float
    dt_a: float
    dt_b: float
    area: float


@dataclass
class HeatExchangerNetwork:
    """A synthesised heat-exchanger network and its verification.

    Attributes:
        exchangers: The units (process exchangers, heaters, coolers).
        hot_utility: Total hot-utility duty used (W).
        cold_utility: Total cold-utility duty used (W).
        dt_min: Minimum approach temperature the design targeted (K).
        n_units: Number of units.
        total_area: Sum of exchanger areas (m^2).
        min_approach: Smallest terminal approach over all exchangers (K).
        feasible: Whether every exchanger respects ``dt_min`` and every stream's
            energy balance closes.
        achieves_mer: Whether the utilities match the minimum (MER) targets.
    """

    exchangers: list[Exchanger]
    hot_utility: float
    cold_utility: float
    dt_min: float
    n_units: int = 0
    total_area: float = 0.0
    min_approach: float = 0.0
    feasible: bool = False
    achieves_mer: bool = False
    _targets: tuple[float, float] = field(default=(0.0, 0.0), repr=False)


@dataclass
class _Side:
    """Mutable per-stream bookkeeping during a region design."""

    name: str
    cp: float
    h: float
    duty: float
    cursor: float  # temperature at the processed (pinch-side) end
    target: float  # the stream's target temperature (for utility top-up)


def _exchanger_area(duty: float, h_hot: float, h_cold: float, dt_a: float, dt_b: float) -> float:
    """Area of one exchanger from series film resistances and the LMTD."""
    u = 1.0 / (1.0 / h_hot + 1.0 / h_cold)
    lm = float(lmtd(max(dt_a, _TOL), max(dt_b, _TOL)))
    return abs(duty) / (u * lm)


def _region_design(
    hots: list[_Side], colds: list[_Side], *, above: bool, dt_min: float, enforce_cp: bool = True
) -> list[Exchanger]:
    """Design one side of the pinch by tick-off, honouring the CP criterion.

    Above the pinch the pinch is the cold end (temperatures rise away from it);
    below, it is the hot end. ``_Side.cursor`` tracks the processed-end
    temperature of each stream, advancing away from the pinch as matches stack.
    ``enforce_cp`` applies the pinch CP-feasibility rule (off for threshold
    problems, which have no pinch and are anchored at the closed end instead).
    """
    sign = 1.0 if above else -1.0
    matches: list[Exchanger] = []

    def at_pinch(s: _Side, pinch_t: float) -> bool:
        return abs(s.cursor - pinch_t) < _TOL

    hot_pinch = hots[0].cursor if hots else 0.0
    cold_pinch = colds[0].cursor if colds else 0.0

    def cp_feasible(h: _Side, c: _Side) -> bool:
        # Only the genuine pinch match (both streams still at the pinch) is
        # constrained; away from the pinch the approach has already opened up.
        if not enforce_cp or not (at_pinch(h, hot_pinch) and at_pinch(c, cold_pinch)):
            return True
        return (h.cp <= c.cp + _TOL) if above else (h.cp >= c.cp - _TOL)

    def make_match(h: _Side, c: _Side, q: float) -> Exchanger:
        dt_h = q / h.cp
        dt_c = q / c.cp
        if above:
            hot_out, hot_in = h.cursor, h.cursor + dt_h
            cold_in, cold_out = c.cursor, c.cursor + dt_c
        else:
            hot_in, hot_out = h.cursor, h.cursor - dt_h
            cold_out, cold_in = c.cursor, c.cursor - dt_c
        dt_pinch_end = (hot_out - cold_in) if above else (hot_in - cold_out)
        dt_far_end = (hot_in - cold_out) if above else (hot_out - cold_in)
        h.cursor += sign * dt_h
        c.cursor += sign * dt_c
        h.duty -= q
        c.duty -= q
        return Exchanger(
            kind="process",
            hot=h.name,
            cold=c.name,
            duty=q,
            t_hot_in=hot_in,
            t_hot_out=hot_out,
            t_cold_in=cold_in,
            t_cold_out=cold_out,
            dt_a=dt_pinch_end,
            dt_b=dt_far_end,
            area=_exchanger_area(q, h.h, c.h, dt_pinch_end, dt_far_end),
        )

    # Phase 1: pinch matches -- match streams still at the pinch, CP-feasible,
    # processing the larger-CP hot streams first (the binding ones).
    if enforce_cp:
        for h in sorted(hots, key=lambda s: -s.cp):
            if h.duty <= _TOL or not at_pinch(h, hot_pinch):
                continue
            candidates = [
                c for c in colds if c.duty > _TOL and at_pinch(c, cold_pinch) and cp_feasible(h, c)
            ]
            if not candidates:
                continue
            # Tightest feasible partner keeps spare cold capacity for later matches.
            c = (
                min(candidates, key=lambda s: s.cp)
                if above
                else max(candidates, key=lambda s: s.cp)
            )
            matches.append(make_match(h, c, min(h.duty, c.duty)))

    # Phase 2: away-from-pinch matches, largest remaining duty first.
    while True:
        active_h = [h for h in hots if h.duty > _TOL]
        active_c = [c for c in colds if c.duty > _TOL]
        pairs = [(h, c) for h in active_h for c in active_c if cp_feasible(h, c)]
        if not pairs:
            break
        h, c = max(pairs, key=lambda p: min(p[0].duty, p[1].duty))
        matches.append(make_match(h, c, min(h.duty, c.duty)))

    return matches


def synthesize_network(
    streams: list[HeatStream],
    dt_min: float,
    *,
    hot_utility_t: float | None = None,
    cold_utility_t: float | None = None,
    hot_utility_h: float = DEFAULT_FILM_COEFFICIENT,
    cold_utility_h: float = DEFAULT_FILM_COEFFICIENT,
) -> HeatExchangerNetwork:
    """Synthesise an MER heat-exchanger network by the pinch design method.

    Args:
        streams: Hot and cold process streams.
        dt_min: Minimum approach temperature (K).
        hot_utility_t: Hot-utility temperature (K); defaults to above the hottest
            process temperature.
        cold_utility_t: Cold-utility temperature (K); defaults to below the
            coldest process temperature.
        hot_utility_h, cold_utility_h: Utility film coefficients (W/m^2/K).

    Returns:
        A verified :class:`HeatExchangerNetwork`. Inspect ``feasible`` and
        ``achieves_mer`` to confirm the design.
    """
    dt = float(dt_min)
    res = pinch_analysis(streams, dt)
    casc = heat_cascade(streams, dt)
    hot_pinch = float(res.hot_pinch_temperature)
    cold_pinch = float(res.cold_pinch_temperature)
    q_h_target = float(casc.hot_utility)
    q_c_target = float(casc.cold_utility)

    temps = [float(s.t_supply) for s in streams] + [float(s.t_target) for s in streams]
    t_hu = (max(temps) + dt) if hot_utility_t is None else float(hot_utility_t)
    t_cu = (min(temps) - dt) if cold_utility_t is None else float(cold_utility_t)

    exchangers: list[Exchanger] = []

    if bool(res.has_pinch):
        exchangers += _design_pinched(streams, hot_pinch, cold_pinch, dt)
    else:
        exchangers += _design_threshold(streams, dt)

    # Utility exchangers close the remaining duty on each unfinished stream.
    hu_used, cu_used = _place_utilities(
        streams, exchangers, dt, t_hu, t_cu, hot_utility_h, cold_utility_h
    )

    net = HeatExchangerNetwork(
        exchangers=exchangers,
        hot_utility=hu_used,
        cold_utility=cu_used,
        dt_min=dt,
        _targets=(q_h_target, q_c_target),
    )
    return verify_network(net, streams)


def _stream_bands(
    streams: list[HeatStream], pinch_t_hot: float, pinch_t_cold: float
) -> tuple[list[_Side], list[_Side], list[_Side], list[_Side]]:
    """Split process streams into above/below-pinch hot and cold ``_Side`` records."""
    hot_above: list[_Side] = []
    hot_below: list[_Side] = []
    cold_above: list[_Side] = []
    cold_below: list[_Side] = []
    for s in streams:
        ts, tt = float(s.t_supply), float(s.t_target)
        cp, h, name = float(s.cp), float(s.h), s.name
        if ts > tt:  # hot
            if ts > pinch_t_hot + _TOL:
                top = ts
                bot = max(tt, pinch_t_hot)
                hot_above.append(_Side(name, cp, h, cp * (top - bot), pinch_t_hot, top))
            if tt < pinch_t_hot - _TOL:
                top = min(ts, pinch_t_hot)
                bot = tt
                hot_below.append(_Side(name, cp, h, cp * (top - bot), pinch_t_hot, bot))
        else:  # cold
            if tt > pinch_t_cold + _TOL:
                top = tt
                bot = max(ts, pinch_t_cold)
                cold_above.append(_Side(name, cp, h, cp * (top - bot), pinch_t_cold, top))
            if ts < pinch_t_cold - _TOL:
                top = min(tt, pinch_t_cold)
                bot = ts
                cold_below.append(_Side(name, cp, h, cp * (top - bot), pinch_t_cold, bot))
    return hot_above, hot_below, cold_above, cold_below


def _design_pinched(
    streams: list[HeatStream], hot_pinch: float, cold_pinch: float, dt: float
) -> list[Exchanger]:
    hot_above, hot_below, cold_above, cold_below = _stream_bands(streams, hot_pinch, cold_pinch)
    above = _region_design(hot_above, cold_above, above=True, dt_min=dt)
    below = _region_design(hot_below, cold_below, above=False, dt_min=dt)
    return above + below


def _design_threshold(streams: list[HeatStream], dt: float) -> list[Exchanger]:
    """Threshold problem (no pinch): one region anchored at the closed end.

    With no pinch the design starts from the end that needs no utility -- the hot
    end when only cold utility is required, the cold end when only hot utility is
    -- so driving forces only open up away from it. The CP rule (a pinch concept)
    does not apply; feasibility is confirmed afterwards by :func:`verify_network`.
    """
    casc = heat_cascade(streams, dt)
    anchor_hot_end = float(casc.hot_utility) <= float(casc.cold_utility)
    hots: list[_Side] = []
    colds: list[_Side] = []
    for s in streams:
        ts, tt = float(s.t_supply), float(s.t_target)
        cp, h, name = float(s.cp), float(s.h), s.name
        if ts > tt:  # hot
            cursor = ts if anchor_hot_end else tt
            hots.append(_Side(name, cp, h, cp * (ts - tt), cursor, tt))
        else:  # cold
            cursor = tt if anchor_hot_end else ts
            colds.append(_Side(name, cp, h, cp * (tt - ts), cursor, tt))
    return _region_design(hots, colds, above=not anchor_hot_end, dt_min=dt, enforce_cp=False)


def _place_utilities(
    streams: list[HeatStream],
    exchangers: list[Exchanger],
    dt: float,
    t_hu: float,
    t_cu: float,
    h_hu: float,
    h_cu: float,
) -> tuple[float, float]:
    """Add heaters/coolers for unmet stream duty; return ``(hot, cold)`` utility used."""
    recovered_cold: dict[str, float] = {}
    recovered_hot: dict[str, float] = {}
    for e in exchangers:
        recovered_cold[e.cold] = recovered_cold.get(e.cold, 0.0) + e.duty
        recovered_hot[e.hot] = recovered_hot.get(e.hot, 0.0) + e.duty

    hu_used = cu_used = 0.0
    for s in streams:
        ts, tt = float(s.t_supply), float(s.t_target)
        cp, h, name = float(s.cp), float(s.h), s.name
        total = cp * abs(ts - tt)
        if tt > ts:  # cold stream: heater tops up to target
            unmet = total - recovered_cold.get(name, 0.0)
            if unmet > _TOL:
                cold_in = tt - unmet / cp
                dt_a = t_hu - tt
                dt_b = t_hu - cold_in
                exchangers.append(
                    Exchanger(
                        "heater",
                        "HU",
                        name,
                        unmet,
                        t_hu,
                        t_hu,
                        cold_in,
                        tt,
                        dt_a,
                        dt_b,
                        _exchanger_area(unmet, h_hu, h, dt_a, dt_b),
                    )
                )
                hu_used += unmet
        else:  # hot stream: cooler removes the rest
            unmet = total - recovered_hot.get(name, 0.0)
            if unmet > _TOL:
                hot_in = tt + unmet / cp
                dt_a = hot_in - t_cu
                dt_b = tt - t_cu
                exchangers.append(
                    Exchanger(
                        "cooler",
                        name,
                        "CU",
                        unmet,
                        hot_in,
                        tt,
                        t_cu,
                        t_cu,
                        dt_a,
                        dt_b,
                        _exchanger_area(unmet, h, h_cu, dt_a, dt_b),
                    )
                )
                cu_used += unmet
    return hu_used, cu_used


def verify_network(net: HeatExchangerNetwork, streams: list[HeatStream]) -> HeatExchangerNetwork:
    """Check a network's approaches, energy balances, and MER attainment.

    Fills in ``n_units``, ``total_area``, ``min_approach``, ``feasible`` and
    ``achieves_mer`` and returns the same (mutated) network.
    """
    min_approach = min((min(e.dt_a, e.dt_b) for e in net.exchangers), default=float("inf"))
    total_area = sum(e.area for e in net.exchangers)

    # Per-stream energy balance: duty served must match the stream requirement.
    served: dict[str, float] = {}
    for e in net.exchangers:
        if e.kind in ("process", "cooler"):
            served[e.hot] = served.get(e.hot, 0.0) + e.duty
        if e.kind in ("process", "heater"):
            served[e.cold] = served.get(e.cold, 0.0) + e.duty
    balanced = True
    for s in streams:
        required = float(s.cp) * abs(float(s.t_supply) - float(s.t_target))
        if abs(served.get(s.name, 0.0) - required) > 1e-3 * max(required, 1.0):
            balanced = False

    q_h_target, q_c_target = net._targets
    feasible = bool(balanced and min_approach >= net.dt_min - 1e-3)
    achieves_mer = bool(
        abs(net.hot_utility - q_h_target) <= 1e-3 * max(q_h_target, 1.0)
        and abs(net.cold_utility - q_c_target) <= 1e-3 * max(q_c_target, 1.0)
    )

    net.n_units = len(net.exchangers)
    net.total_area = total_area
    net.min_approach = min_approach
    net.feasible = feasible
    net.achieves_mer = achieves_mer
    return net
