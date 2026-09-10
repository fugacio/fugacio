"""Differentiable unit operations with rigorous material *and* energy balances.

Every block here is built on a `fugacio.thermo.PropertyPackage`, the object that
owns both the phase equilibrium and the energy properties of the mixture, so
each unit is differentiable end-to-end with respect to its operating conditions,
its feed, and the thermodynamic parameters. That is the headline Fugacio claim
made concrete: you can take a gradient of a product purity, a duty, or a shaft
power with respect to a drum temperature, an outlet pressure, a split fraction, a
feed flow, or an NRTL parameter: the basis for gradient-based flowsheet
optimisation and parameter regression through a whole plant.

The library covers the staples of a process flowsheet:

* `flash_drum`: isothermal-isobaric vapour/liquid separator;
* `heater`: heater/cooler on either a temperature or a duty spec;
* `valve`: isenthalpic (Joule-Thomson) pressure letdown;
* `pump`: incompressible-liquid pump with an efficiency;
* `compressor` / `turbine`: isentropic machines with an efficiency;
* `mix`: adiabatic mixer (energy-balanced);
* `splitter`: flow splitter;
* `component_separator`: idealised component split.

Every energy-balanced unit takes a ``model`` argument selecting the property
package (see `fugacio.sim.models.package_for`); leaving it unset falls back to
Peng-Robinson on the stream's components, tunable through the legacy ``eos`` /
``kij`` arguments. Outlet temperatures of the energy-specified blocks are found
with the package's differentiable ``flash_ph`` / ``flash_ps`` solves, whose
temperatures carry implicit-function gradients.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.properties import (
    Model,
    _composition,
    molar_enthalpy,
    molar_entropy,
    resolve_package,
)
from fugacio.sim.stream import Stream
from fugacio.thermo import PR, CubicEOS
from fugacio.thermo.diagnostics import SolveReport, SolveStatus, require_converged, residual_report
from fugacio.thermo.package import _phase_classification

ArrayLike = Array | float


class HeaterResult(NamedTuple):
    """Outlet of a heater/cooler together with the heat duty.

    Attributes:
        outlet: Product `Stream`.
        duty: Heat duty (W). Positive means heat *added*; negative means cooling.
    """

    outlet: Stream
    duty: Array
    report: SolveReport


class PumpResult(NamedTuple):
    """Outlet of a pump together with the shaft work.

    Attributes:
        outlet: Product `Stream`.
        work: Shaft power delivered to the fluid (W).
    """

    outlet: Stream
    work: Array
    report: SolveReport


class WorkResult(NamedTuple):
    """Outlet of a compressor/turbine with actual and ideal shaft work.

    Attributes:
        outlet: Product `Stream`.
        work: Actual shaft power into the fluid (W); negative for a turbine
            (the fluid does work on the surroundings).
        ideal_work: Reversible (isentropic) shaft power into the fluid (W).
    """

    outlet: Stream
    work: Array
    ideal_work: Array
    report: SolveReport


def _efficiency(value: ArrayLike, context: str) -> Array:
    """Validate a machine efficiency in eager and compiled calculations."""
    efficiency = jnp.asarray(value, dtype=float)
    report = residual_report(
        jnp.atleast_1d(jnp.where((efficiency > 0) & (efficiency <= 1), 0.0, 1.0)),
        failure=SolveStatus.INVALID_INPUT,
    )
    require_converged(report, context, ("efficiency must be in (0, 1]",))
    return efficiency * jnp.where(report.converged, 1.0, jnp.nan)


def _energy_outlet(
    outlet: Stream, target: Array, pkg: Any, context: str, *, allow_extrapolation: bool = False
) -> tuple[Stream, SolveReport]:
    """Independently verify energy closure before handing a state downstream."""
    from fugacio.thermo.acceptance import AcceptancePolicy, assess_flash
    from fugacio.thermo.equilibrium import FlashResult

    error = (molar_enthalpy(outlet, model=pkg) - target) / jnp.maximum(jnp.abs(target), 1e4)
    vapor_n = jnp.asarray(outlet.vapor_n)
    beta = jnp.sum(vapor_n) / jnp.maximum(outlet.total, 1e-300)
    x, y = _composition(outlet.n - vapor_n), _composition(vapor_n)
    state = FlashResult(beta, x, y, y / jnp.maximum(x, 1e-300))
    physical = assess_flash(
        pkg,
        outlet.t,
        outlet.p,
        _composition(outlet.n),
        state,
        policy=AcceptancePolicy(check_stability=False),
    )
    errors = jnp.array(
        [
            error,
            physical.equilibrium_error,
            physical.phase_error,
            jnp.where(
                physical.applicability.parameters_available
                if allow_extrapolation
                else physical.applicability.accepted,
                0.0,
                1.0,
            ),
        ]
    )
    report = residual_report(jnp.where(outlet.total > 0, errors, 0.0), tol=1e-7)
    report = report._replace(
        status=jnp.where(outlet.report.converged, report.status, SolveStatus.INVALID_INPUT)
    )
    require_converged(report, context, ("molar enthalpy",))
    checked = jax.tree_util.tree_map(lambda x: jnp.where(report.converged, x, jnp.nan), outlet)
    return checked, report


def flash_drum(
    feed: Stream,
    t: ArrayLike,
    p: ArrayLike,
    *,
    model: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
) -> tuple[Stream, Stream]:
    """Flash a feed stream at temperature ``t`` and pressure ``p``.

    Args:
        feed: Inlet `Stream`.
        t: Drum temperature (K).
        p: Drum pressure (Pa).
        model: Property package (or equilibrium model) to flash with; defaults
            to a cubic EOS built from ``eos`` / ``kij``.
        eos: Cubic equation of state for the default package (Peng-Robinson).
        kij: Optional binary interaction matrix for the default package.

    Returns:
        ``(vapor, liquid)`` product streams. Their flows are differentiable with
        respect to ``t``, ``p``, the feed, and the package parameters.
    """
    pkg = resolve_package(feed.components, model, eos=eos, kij=kij)
    z = _composition(feed.n)
    classified = _phase_classification(pkg, t, p, z)
    total = feed.total

    def single_phase(_: None) -> tuple[Array, Array]:
        vapor_flow = jnp.where(classified.beta <= 0.0, jnp.zeros_like(feed.n), feed.n)
        return vapor_flow, feed.n - vapor_flow

    def two_phase(_: None) -> tuple[Array, Array]:
        result = pkg.flash_pt(t, p, z)
        return result.y * result.beta * total, result.x * (1.0 - result.beta) * total

    # Outside the two-phase region, phase flows follow the material balance
    # directly. Don't differentiate an absent phase's trial equilibrium state.
    vapor_flow, liquid_flow = jax.lax.cond(
        jnp.isfinite(classified.beta) & ((classified.beta <= 0.0) | (classified.beta >= 1.0)),
        single_phase,
        two_phase,
        None,
    )
    t_arr = jnp.asarray(t, dtype=float)
    p_arr = jnp.asarray(p, dtype=float)
    vapor = Stream(
        n=vapor_flow,
        vapor_n=vapor_flow,
        t=t_arr,
        p=p_arr,
        components=feed.components,
    )
    liquid = Stream(
        n=liquid_flow,
        vapor_n=jnp.zeros_like(feed.n),
        t=t_arr,
        p=p_arr,
        components=feed.components,
    )
    return vapor, liquid


def heater(
    feed: Stream,
    *,
    t_out: ArrayLike | None = None,
    duty: ArrayLike | None = None,
    dp: ArrayLike = 0.0,
    model: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    t_init: float = 300.0,
    allow_extrapolation: bool = False,
) -> HeaterResult:
    """Heat or cool a stream on either a temperature or a duty specification.

    Provide exactly one of ``t_out`` (target outlet temperature) or ``duty``
    (signed heat added, W). With ``t_out`` the duty is computed from the enthalpy
    change; with ``duty`` the outlet temperature is found by an isenthalpic solve,
    so the block correctly handles partial vaporisation/condensation. ``dp`` is the
    pressure drop across the block.

    Raises:
        ValueError: if not exactly one of ``t_out`` / ``duty`` is given.
    """
    if (t_out is None) == (duty is None):
        raise ValueError("heater requires exactly one of t_out or duty")
    pkg = resolve_package(feed.components, model, eos=eos, kij=kij)
    p_out = feed.p - jnp.asarray(dp)
    h_in = molar_enthalpy(feed, model=pkg)
    if t_out is not None:
        outlet = Stream(
            n=feed.n, t=jnp.asarray(t_out, dtype=float), p=p_out, components=feed.components
        )
        duty_w = feed.total * (molar_enthalpy(outlet, model=pkg) - h_in)
        report = residual_report(jnp.atleast_1d(jnp.where(jnp.isfinite(duty_w), 0.0, jnp.nan)))
        report = report._replace(
            status=jnp.where(outlet.report.converged, report.status, SolveStatus.INVALID_INPUT)
        )
        require_converged(report, "heater")
        outlet = jax.tree_util.tree_map(lambda x: jnp.where(report.converged, x, jnp.nan), outlet)
        return HeaterResult(outlet=outlet, duty=duty_w, report=report)
    h_spec = h_in + jnp.asarray(duty) / jnp.where(feed.total > 0, feed.total, 1.0)
    r = pkg.flash_ph(p_out, h_spec, _composition(feed.n), t_init=t_init)
    outlet = Stream(
        n=feed.n, t=r.t, p=p_out, components=feed.components, vapor_n=r.beta * feed.total * r.y
    )
    outlet, report = _energy_outlet(
        outlet, h_spec, pkg, "heater", allow_extrapolation=allow_extrapolation
    )
    feasible = (feed.total > 0) | (jnp.asarray(duty) == 0)
    report = report._replace(status=jnp.where(feasible, report.status, SolveStatus.INFEASIBLE))
    require_converged(report, "heater: duty on an empty stream")
    outlet = jax.tree_util.tree_map(lambda x: jnp.where(report.converged, x, jnp.nan), outlet)
    return HeaterResult(outlet=outlet, duty=jnp.asarray(duty, dtype=float), report=report)


def valve(
    feed: Stream,
    p_out: ArrayLike,
    *,
    model: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    t_init: float = 300.0,
    allow_extrapolation: bool = False,
) -> Stream:
    """Isenthalpic (Joule-Thomson) pressure letdown to ``p_out``.

    Enthalpy is conserved, so the outlet temperature (and any flashing that
    results from the pressure drop) follows from an isenthalpic solve. The
    outlet may be two-phase; it is returned as a single bulk stream at the solved
    temperature.
    """
    pkg = resolve_package(feed.components, model, eos=eos, kij=kij)
    h_spec = molar_enthalpy(feed, model=pkg)
    r = pkg.flash_ph(p_out, h_spec, _composition(feed.n), t_init=t_init)
    outlet = Stream(
        n=feed.n,
        t=r.t,
        p=jnp.asarray(p_out, dtype=float),
        components=feed.components,
        vapor_n=r.beta * feed.total * r.y,
    )
    return _energy_outlet(outlet, h_spec, pkg, "valve", allow_extrapolation=allow_extrapolation)[0]


def pump(
    feed: Stream,
    p_out: ArrayLike,
    *,
    efficiency: ArrayLike = 0.75,
    model: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    t_init: float = 300.0,
) -> PumpResult:
    """Pump an (incompressible) liquid from ``feed.p`` to ``p_out``.

    The reversible work is ``v_L (p_out - p_in)`` per mole using the liquid molar
    volume; the actual work is divided by ``efficiency`` and the inefficiency is
    deposited as heat, so the outlet temperature is found from an isenthalpic
    balance on ``H_out = H_in + W_actual``.
    """
    pkg = resolve_package(feed.components, model, eos=eos, kij=kij)
    v_l = pkg.volume(feed.t, feed.p, _composition(feed.n), phase="liquid")
    w_ideal = v_l * (jnp.asarray(p_out) - feed.p)
    w_actual = w_ideal / _efficiency(efficiency, "pump")
    h_out = molar_enthalpy(feed, model=pkg) + w_actual
    r = pkg.flash_ph(p_out, h_out, _composition(feed.n), t_init=t_init)
    outlet = Stream(
        n=feed.n,
        t=r.t,
        p=jnp.asarray(p_out, dtype=float),
        components=feed.components,
        vapor_n=r.beta * feed.total * r.y,
    )
    outlet, report = _energy_outlet(outlet, h_out, pkg, "pump")
    return PumpResult(outlet=outlet, work=w_actual * feed.total, report=report)


def _compress(
    feed: Stream,
    p_out: ArrayLike,
    efficiency: ArrayLike,
    model: Model,
    eos: CubicEOS,
    kij: Array | None,
    t_init: float,
    *,
    is_turbine: bool,
) -> WorkResult:
    """Shared isentropic-machine model for `compressor` and `turbine`."""
    pkg = resolve_package(feed.components, model, eos=eos, kij=kij)
    z = _composition(feed.n)
    h_in = molar_enthalpy(feed, model=pkg)
    s_in = molar_entropy(feed, model=pkg)
    iso = pkg.flash_ps(p_out, s_in, z, t_init=t_init)
    iso_stream = Stream(
        feed.n, iso.t, jnp.asarray(p_out), feed.components, iso.beta * feed.total * iso.y
    )
    entropy_error = (molar_entropy(iso_stream, model=pkg) - s_in) / jnp.maximum(jnp.abs(s_in), 10.0)
    entropy_report = residual_report(jnp.atleast_1d(entropy_error), tol=1e-7)
    require_converged(entropy_report, "isentropic machine", ("entropy",))
    h_out_ideal = jnp.where(
        entropy_report.converged, molar_enthalpy(iso_stream, model=pkg), jnp.nan
    )
    w_ideal = h_out_ideal - h_in
    eff = _efficiency(efficiency, "turbine" if is_turbine else "compressor")
    w_actual = eff * w_ideal if is_turbine else w_ideal / eff
    h_out = h_in + w_actual
    r = pkg.flash_ph(p_out, h_out, z, t_init=t_init)
    outlet = Stream(
        n=feed.n,
        t=r.t,
        p=jnp.asarray(p_out, dtype=float),
        components=feed.components,
        vapor_n=r.beta * feed.total * r.y,
    )
    outlet, report = _energy_outlet(outlet, h_out, pkg, "turbine" if is_turbine else "compressor")
    return WorkResult(
        outlet=outlet, work=w_actual * feed.total, ideal_work=w_ideal * feed.total, report=report
    )


def compressor(
    feed: Stream,
    p_out: ArrayLike,
    *,
    efficiency: ArrayLike = 0.75,
    model: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    t_init: float = 300.0,
) -> WorkResult:
    """Compress a stream to ``p_out`` with an isentropic ``efficiency`` (< 1).

    The reversible outlet is the isentropic state at ``p_out``; the actual work is
    ``W_ideal / efficiency`` and the extra enthalpy sets the (higher) real outlet
    temperature via an isenthalpic solve.
    """
    return _compress(feed, p_out, efficiency, model, eos, kij, t_init, is_turbine=False)


def turbine(
    feed: Stream,
    p_out: ArrayLike,
    *,
    efficiency: ArrayLike = 0.85,
    model: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    t_init: float = 300.0,
) -> WorkResult:
    """Expand a stream to ``p_out`` with an isentropic ``efficiency`` (< 1).

    The fluid recovers ``efficiency`` of the reversible work; the returned
    ``work`` is negative (power delivered to the shaft) and the real outlet is
    warmer than the isentropic outlet because of the lost work.
    """
    return _compress(feed, p_out, efficiency, model, eos, kij, t_init, is_turbine=True)


def splitter(feed: Stream, fractions: ArrayLike) -> tuple[Stream, ...]:
    """Split ``feed`` into multiple outlets that share its composition and state.

    ``fractions`` is a sequence of split fractions (one per outlet); each outlet
    carries that fraction of every component flow. The fractions should sum to one
    for a conservative split. Temperature and pressure are passed through.
    """
    fr = jnp.asarray(fractions)
    k = fr.shape[0]
    return tuple(feed.scaled(fr[i]) for i in range(k))


def component_separator(
    feed: Stream,
    split_to_top: ArrayLike,
    *,
    top_t: ArrayLike | None = None,
    top_p: ArrayLike | None = None,
    bottom_t: ArrayLike | None = None,
    bottom_p: ArrayLike | None = None,
) -> tuple[Stream, Stream]:
    """Idealised separator with a per-component recovery to the top product.

    ``split_to_top`` is a per-component fraction (aligned with ``feed.components``)
    sent to the top outlet; the remainder leaves in the bottom. This is the
    workhorse "spec" separator for conceptual flowsheets, a stand-in for a column
    or absorber whose recoveries are known. Product temperatures and pressures
    default to the feed's.
    """
    frac = jnp.asarray(split_to_top)
    top = Stream(
        n=feed.n * frac,
        t=feed.t if top_t is None else jnp.asarray(top_t),
        p=feed.p if top_p is None else jnp.asarray(top_p),
        components=feed.components,
    )
    bottom = Stream(
        n=feed.n * (1.0 - frac),
        t=feed.t if bottom_t is None else jnp.asarray(bottom_t),
        p=feed.p if bottom_p is None else jnp.asarray(bottom_p),
        components=feed.components,
    )
    return top, bottom


def mix(
    streams: list[Stream],
    *,
    t: ArrayLike | None = None,
    p: ArrayLike | None = None,
    model: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    t_init: float = 300.0,
) -> Stream:
    """Combine streams with an exact material balance and an adiabatic energy balance.

    Flows add component-by-component. By default the outlet temperature is found
    from an *adiabatic* energy balance (total enthalpy in equals total enthalpy
    out) via an isenthalpic solve, so heat of mixing and any phase change are
    accounted for; pass ``t`` to fix the outlet temperature instead. Outlet
    pressure defaults to the lowest inlet pressure.

    Raises:
        ValueError: if the streams are empty or do not share a component list.
    """
    if not streams:
        raise ValueError("mix requires at least one stream")
    components = streams[0].components
    for s in streams[1:]:
        if s.components != components:
            raise ValueError("all streams must share the same component list to mix")
    n_total = jnp.sum(jnp.stack([s.n for s in streams]), axis=0)
    p_out = jnp.min(jnp.stack([s.p for s in streams])) if p is None else jnp.asarray(p, dtype=float)
    if t is not None:
        return Stream(n=n_total, t=jnp.asarray(t, dtype=float), p=p_out, components=components)
    pkg = resolve_package(components, model, eos=eos, kij=kij)
    total = jnp.sum(n_total)
    z = n_total / jnp.where(total > 0.0, total, 1.0)
    z = jnp.where(total > 0.0, z, jnp.ones_like(z) / z.size)
    # An empty inlet (a zero-flow recycle guess on the first tear iteration)
    # contributes no enthalpy; guard it so its undefined composition cannot
    # poison the balance.
    h_in = jnp.sum(
        jnp.stack(
            [jnp.where(s.total > 0.0, s.total * molar_enthalpy(s, model=pkg), 0.0) for s in streams]
        )
    )
    target = jnp.where(
        total > 0, h_in / jnp.where(total > 0, total, 1.0), molar_enthalpy(streams[0], model=pkg)
    )
    r = pkg.flash_ph(p_out, target, z, t_init=t_init)
    outlet = Stream(n=n_total, t=r.t, p=p_out, components=components, vapor_n=r.beta * total * r.y)
    return _energy_outlet(outlet, target, pkg, "mixer")[0]
