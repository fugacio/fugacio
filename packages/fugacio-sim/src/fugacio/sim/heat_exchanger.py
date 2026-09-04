"""Two-sided heat exchanger: a hot and a cold stream coupled by one duty.

The `fugacio.sim.units.heater` block moves heat between a stream and an
unspecified utility. A real exchanger couples *two* process streams, and the
plant-level questions (how much area, what approach temperatures, which side
pinches, what happens to the cold outlet when the hot inlet changes) need both
sides in one model. `heat_exchanger` provides that: an energy balance shared by
the two streams, outlet states from each side's own property package (so a
steam-heated reboiler can run IAPWS-95 on one side and Peng-Robinson on the
other), and a rating/design relation ``Q = UA * LMTD`` that handles phase change
by integrating the duty in zones.

One specification closes the model. It can be the duty, either outlet
temperature, a minimum approach temperature (the pinch specification of heat
integration), or the exchanger size ``UA`` (or ``area`` with a coefficient
``u``), in which case the duty is the root of ``UA_required(Q) - UA = 0``. That
root is found by the bracketed solver of `fugacio.thermo.implicit`, so the whole
exchanger is differentiable with respect to both inlets, the size, and the
pressure drops.

Counter-current and parallel (co-current) arrangements are supported. With
``zones > 1`` the temperature-duty curves of both streams are sampled at
intermediate duties (each through the package's isenthalpic flash, so condensing
and boiling plateaus are captured) and the log-mean driving force is applied
zone by zone, the standard treatment for exchangers with a phase change on
either side.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from fugacio.sim.economics import lmtd
from fugacio.sim.properties import Model, resolve_package
from fugacio.sim.stream import Stream
from fugacio.thermo import PR, CubicEOS, PropertyPackage
from fugacio.thermo.implicit import bracketed_root

ArrayLike = Array | float


class HeatExchangerResult(NamedTuple):
    """Converged two-sided exchanger.

    Attributes:
        hot_out: Hot-side outlet `Stream`.
        cold_out: Cold-side outlet `Stream`.
        duty: Heat transferred from the hot to the cold stream (W, positive).
        ua: Overall ``U A`` the duty requires (W/K).
        lmtd: Effective log-mean temperature difference ``Q / UA`` (K).
        area: Heat-transfer area (m^2) when ``u`` was given, else ``nan``.
        approach_hot_end: Temperature approach at the hot-inlet end (K).
        approach_cold_end: Temperature approach at the hot-outlet end (K).
        min_approach: Smallest approach anywhere along the exchanger (K).
        hot_curve: Hot-side temperatures at the zone boundaries (K), inlet first.
        cold_curve: Cold-side temperatures at the same positions (K).
    """

    hot_out: Stream
    cold_out: Stream
    duty: Array
    ua: Array
    lmtd: Array
    area: Array
    approach_hot_end: Array
    approach_cold_end: Array
    min_approach: Array
    hot_curve: Array
    cold_curve: Array


_FLOW_FLOOR = 1.0e-12
"""Molar flow below which a side is treated as empty (mol/s)."""


def _nonempty(stream: Stream) -> Stream:
    """Give an empty stream a trace of uniform composition for property evaluation.

    A recycle seeded with zero flow has no composition; every activity or EOS
    model would return NaN for it. A trace at ``_FLOW_FLOOR`` exchanges no heat
    to any tolerance but keeps the curves finite, so a loop can start empty.
    """
    total = jnp.sum(stream.n)
    trace = jnp.full_like(stream.n, _FLOW_FLOOR / stream.n.shape[0])
    n = jnp.where(total > _FLOW_FLOOR, stream.n, trace)
    return Stream(n=n, t=stream.t, p=stream.p, components=stream.components)


def _outlet_temperature(
    pkg: PropertyPackage, stream: Stream, h_molar: Array, p_out: Array, t_init: float
) -> Array:
    return pkg.flash_ph(p_out, h_molar, stream.z, t_init=t_init).t


def _curves(
    q: Array,
    zones: int,
    flow: str,
    hot: Stream,
    cold: Stream,
    pkg_h: PropertyPackage,
    pkg_c: PropertyPackage,
    h_hot_in: Array,
    h_cold_in: Array,
    p_hot_out: Array,
    p_cold_out: Array,
    t_init: float,
) -> tuple[Array, Array]:
    """Hot and cold temperatures at ``zones + 1`` positions along the exchanger.

    Position 0 is the hot inlet. The hot stream has given up ``k/zones`` of the
    duty at position ``k``; counter-current, the cold stream there has absorbed
    ``(zones - k)/zones`` of it (it enters at the far end), parallel it has
    absorbed ``k/zones``.
    """
    frac = jnp.linspace(0.0, 1.0, zones + 1)
    cold_frac = (1.0 - frac) if flow == "counter" else frac
    # An empty side (a recycle seeded with zero flow) exchanges no heat; the
    # floor keeps its molar enthalpy finite instead of dividing by zero.
    n_hot = jnp.maximum(hot.total, _FLOW_FLOOR)
    n_cold = jnp.maximum(cold.total, _FLOW_FLOOR)

    def t_hot(fr: Array) -> Array:
        return _outlet_temperature(pkg_h, hot, h_hot_in - fr * q / n_hot, p_hot_out, t_init)

    def t_cold(fr: Array) -> Array:
        return _outlet_temperature(pkg_c, cold, h_cold_in + fr * q / n_cold, p_cold_out, t_init)

    # `lax.map` (a scan) compiles each side's isenthalpic flash once and runs it
    # sequentially over the positions; a Python loop would emit ``zones + 1``
    # copies of the nested solver and `vmap` would turn its phase-regime `switch`
    # into a `select` that evaluates non-existent phases.
    t_h = lax.map(t_hot, frac)
    t_c = lax.map(t_cold, cold_frac)
    return t_h, t_c


def _ua_required(q: Array, t_h: Array, t_c: Array) -> Array:
    """``sum_k (Q / zones) / LMTD_k`` over the zones between consecutive positions."""
    dt = t_h - t_c
    zones = t_h.shape[0] - 1
    safe = jnp.maximum(dt, 1e-12)
    lm = lmtd(safe[:-1], safe[1:])
    # A pinch (zero or crossed approach) needs infinite area.
    return jnp.where(jnp.min(dt) <= 0.0, jnp.inf, jnp.sum((q / zones) / lm))


@partial(jax.jit, static_argnames=("spec", "zones", "flow", "t_init", "tol"))
def _solve(
    pkg_h: PropertyPackage,
    pkg_c: PropertyPackage,
    hot: Stream,
    cold: Stream,
    value: Array,
    dp_hot: Array,
    dp_cold: Array,
    *,
    spec: str,
    zones: int,
    flow: str,
    t_init: float,
    tol: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Duty, both temperature curves, required ``UA``, and the outlet pressures."""
    p_hot_out = hot.p - dp_hot
    p_cold_out = cold.p - dp_cold
    h_hot_in = pkg_h.mixture_enthalpy(hot.t, hot.p, hot.z)
    h_cold_in = pkg_c.mixture_enthalpy(cold.t, cold.p, cold.z)

    # Second-law duty cap: hot cooled to the cold inlet, cold heated to the hot inlet.
    q_hot_max = hot.total * (h_hot_in - pkg_h.mixture_enthalpy(cold.t, p_hot_out, hot.z))
    q_cold_max = cold.total * (pkg_c.mixture_enthalpy(hot.t, p_cold_out, cold.z) - h_cold_in)
    q_max = jnp.maximum(jnp.minimum(q_hot_max, q_cold_max), 1e-9)

    def curves(q: Array, th: Any) -> tuple[Array, Array]:
        pkg_h_, pkg_c_, hot_, cold_, hh, hc, ph, pc = th
        return _curves(q, zones, flow, hot_, cold_, pkg_h_, pkg_c_, hh, hc, ph, pc, t_init)

    theta_curves: Any = (pkg_h, pkg_c, hot, cold, h_hot_in, h_cold_in, p_hot_out, p_cold_out)

    if spec == "duty":
        q = value
    elif spec == "t_hot_out":
        q = hot.total * (h_hot_in - pkg_h.mixture_enthalpy(value, p_hot_out, hot.z))
    elif spec == "t_cold_out":
        q = cold.total * (pkg_c.mixture_enthalpy(value, p_cold_out, cold.z) - h_cold_in)
    else:
        # Size or approach spec: a monotone scalar root in the duty fraction.
        def residual(frac: Array, th: Any) -> Array:
            th_curves, qmax, target = th
            q_ = frac * qmax
            t_h, t_c = curves(q_, th_curves)
            if spec == "min_approach":
                return jnp.min(t_h - t_c) - target
            return jnp.log(target) - jnp.log(_ua_required(q_, t_h, t_c))

        lo = jnp.asarray(1e-6)
        hi = jnp.asarray(1.0 - 1e-6)
        frac_star = bracketed_root(residual, (theta_curves, q_max, value), lo, hi, tol, 200)
        q = frac_star * q_max

    q = jnp.clip(q, 0.0, q_max)
    t_h, t_c = curves(q, theta_curves)
    return q, t_h, t_c, _ua_required(q, t_h, t_c), p_hot_out, p_cold_out


def heat_exchanger(
    hot: Stream,
    cold: Stream,
    *,
    duty: ArrayLike | None = None,
    t_hot_out: ArrayLike | None = None,
    t_cold_out: ArrayLike | None = None,
    min_approach: ArrayLike | None = None,
    ua: ArrayLike | None = None,
    area: ArrayLike | None = None,
    u: ArrayLike | None = None,
    dp_hot: ArrayLike = 0.0,
    dp_cold: ArrayLike = 0.0,
    flow: str = "counter",
    zones: int = 1,
    model: Model = None,
    model_hot: Model = None,
    model_cold: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    t_init: float = 300.0,
    tol: float = 1e-9,
) -> HeatExchangerResult:
    """Exchange heat between a hot and a cold stream under one specification.

    Exactly one of ``duty``, ``t_hot_out``, ``t_cold_out``, ``min_approach``,
    ``ua``, or ``area`` (with ``u``) must be given. The duty is always positive
    (heat flows from ``hot`` to ``cold``) and is capped by the second law: the
    cold outlet cannot exceed the hot inlet nor the hot outlet drop below the
    cold inlet.

    Args:
        hot: Hot inlet stream.
        cold: Cold inlet stream.
        duty: Heat transferred (W).
        t_hot_out: Hot outlet temperature (K).
        t_cold_out: Cold outlet temperature (K).
        min_approach: Minimum temperature approach (K) between the two curves;
            the duty is raised until the tighter end (or interior zone boundary)
            reaches it.
        ua: Exchanger size ``U A`` (W/K); the duty is solved from ``Q = UA * LMTD``.
        area: Heat-transfer area (m^2), used with ``u``.
        u: Overall heat-transfer coefficient (W/m^2/K); with ``area`` gives
            ``ua`` and also reports the ``area`` a duty spec requires.
        dp_hot: Hot-side pressure drop (Pa).
        dp_cold: Cold-side pressure drop (Pa).
        flow: ``"counter"`` (default) or ``"parallel"``.
        zones: Number of duty zones the driving force is integrated over.
        model: Property package for both sides (see
            `fugacio.sim.models.package_for`); defaults to Peng-Robinson.
        model_hot: Hot-side package overriding ``model``.
        model_cold: Cold-side package overriding ``model``.
        eos: Cubic EOS for the default package.
        kij: Binary interaction matrix for the default package.
        t_init: Seed for the isenthalpic outlet solves.
        tol: Tolerance of the duty root solve (relative to the duty cap).

    Returns:
        A `HeatExchangerResult`, differentiable with respect to both inlets,
        the specification, the pressure drops, and the packages.

    Raises:
        ValueError: if the specification is missing or ambiguous, or ``flow`` /
            ``zones`` are invalid.
    """
    if flow not in ("counter", "parallel"):
        raise ValueError(f"flow must be 'counter' or 'parallel', got {flow!r}")
    if zones < 1:
        raise ValueError("zones must be at least 1")
    given = [
        s
        for s, v in (
            ("duty", duty),
            ("t_hot_out", t_hot_out),
            ("t_cold_out", t_cold_out),
            ("min_approach", min_approach),
            ("ua", ua),
            ("area", area),
        )
        if v is not None
    ]
    if len(given) != 1:
        raise ValueError(
            "heat_exchanger needs exactly one of duty, t_hot_out, t_cold_out, min_approach, "
            f"ua, or area (with u); got {given or 'none'}"
        )
    if area is not None and u is None:
        raise ValueError("an area specification needs the coefficient u as well")

    pkg_h = resolve_package(
        hot.components, model_hot if model_hot is not None else model, eos=eos, kij=kij
    )
    pkg_c = resolve_package(
        cold.components, model_cold if model_cold is not None else model, eos=eos, kij=kij
    )
    if duty is not None:
        spec, value = "duty", duty
    elif t_hot_out is not None:
        spec, value = "t_hot_out", t_hot_out
    elif t_cold_out is not None:
        spec, value = "t_cold_out", t_cold_out
    elif min_approach is not None:
        spec, value = "min_approach", min_approach
    elif ua is not None:
        spec, value = "ua", ua
    else:
        spec, value = "ua", jnp.asarray(area, dtype=float) * jnp.asarray(u, dtype=float)

    q, t_h, t_c, ua_req, p_hot_out, p_cold_out = _solve(
        pkg_h,
        pkg_c,
        _nonempty(hot),
        _nonempty(cold),
        jnp.asarray(value, dtype=float),
        jnp.asarray(dp_hot, dtype=float),
        jnp.asarray(dp_cold, dtype=float),
        spec=spec,
        zones=zones,
        flow=flow,
        t_init=t_init,
        tol=tol,
    )
    hot_out = Stream(n=hot.n, t=t_h[-1], p=p_hot_out, components=hot.components)
    cold_out = Stream(
        n=cold.n,
        t=t_c[-1] if flow == "parallel" else t_c[0],
        p=p_cold_out,
        components=cold.components,
    )
    dt = t_h - t_c
    area_out = ua_req / jnp.asarray(u, dtype=float) if u is not None else jnp.asarray(jnp.nan)
    return HeatExchangerResult(
        hot_out=hot_out,
        cold_out=cold_out,
        duty=q,
        ua=ua_req,
        lmtd=q / ua_req,
        area=area_out,
        approach_hot_end=dt[0],
        approach_cold_end=dt[-1],
        min_approach=jnp.min(dt),
        hot_curve=t_h,
        cold_curve=t_c,
    )


__all__ = ["HeatExchangerResult", "heat_exchanger"]
