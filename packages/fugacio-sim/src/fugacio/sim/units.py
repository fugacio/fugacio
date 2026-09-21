"""Differentiable unit operations with rigorous material *and* energy balances.

Every block here is built on a `fugacio.thermo.PropertyPackage`, the object that
owns both the phase equilibrium and the energy properties of the mixture, so
each unit is differentiable end-to-end with respect to its operating conditions,
its feed, and the thermodynamic parameters: you can take a gradient of a product
purity, a duty, or a shaft power with respect to a drum temperature, an outlet
pressure, a split fraction, a feed flow, or an NRTL parameter.

The library covers the staples of a process flowsheet:

* `flash_drum` / `flash_drum_with_info`: isothermal-isobaric separator;
* `adiabatic_flash`: pressure-and-duty separator (a PH flash), the right
  specification for a letdown into a drum or a pure fluid on its saturation line;
* `heater`: heater/cooler on a temperature, duty, or outlet vapor-fraction spec;
* `valve`: isenthalpic (Joule-Thomson) pressure letdown;
* `pump`: incompressible-liquid pump with an efficiency;
* `compressor` / `turbine`: isentropic machines with an efficiency;
* `mix`: adiabatic mixer (energy-balanced);
* `splitter`: flow splitter;
* `component_separator`: idealised component split.

**Contract.** Every unit is one compiled kernel. The property package and every
specification are dynamic arguments, so a unit compiles once per package
structure and specification kind, then evaluates new operating points without
recompiling. Each kernel returns a `fugacio.thermo.diagnostics.SolveReport`
alongside its outlets, and a failed kernel's outlets are NaN. An eager call
raises: `fugacio.thermo.diagnostics.ConvergenceError` for a failed solve and
`ValueError` for a violated operating limit (`fugacio.sim.unit_limits`). A
traced call (inside ``jax.jit``, a flowsheet, or an optimizer) can't raise and
instead returns NaN outlets with nonfinite derivatives.

The flashes here solve one liquid and one vapor. Whether a feed splits into two
liquids is a separate, more expensive question: check it with `flash_drum_checked`
(`fugacio.sim.acceptance`), a flowsheet's post-solve audit, or the package's
``stability`` method.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim import unit_limits as limits
from fugacio.sim.properties import (
    Model,
    _composition,
    molar_enthalpy,
    molar_entropy,
    resolve_package,
)
from fugacio.sim.stream import Stream
from fugacio.thermo.acceptance import PhysicalAcceptanceWarning
from fugacio.thermo.diagnostics import (
    SolveReport,
    SolveStatus,
    nan_unless_converged,
    require_converged,
    residual_report,
    with_status,
)
from fugacio.thermo.package import _is_pytree, _phase_classification

ArrayLike = Array | float


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class FlashDrumResult(NamedTuple):
    """Vapor and liquid products of a separator, its heat duty, and its report.

    Attributes:
        vapor: Vapor product `Stream`.
        liquid: Liquid product `Stream`.
        duty: Heat added to hold the drum at its specification (W).
        report: Solve and specification report.
    """

    vapor: Stream
    liquid: Stream
    duty: Array
    report: SolveReport

    @property
    def outlets(self) -> tuple[Stream, Stream]:
        """``(vapor, liquid)``, the order a flowsheet assigns outputs."""
        return self.vapor, self.liquid

    @property
    def heat(self) -> Array:
        """Heat into the fluid (W), the flowsheet's energy-accounting name."""
        return self.duty


class HeaterResult(NamedTuple):
    """Outlet of a heater/cooler together with the heat duty.

    Attributes:
        outlet: Product `Stream`.
        duty: Heat duty (W). Positive means heat *added*; negative means cooling.
        report: Solve and specification report.
    """

    outlet: Stream
    duty: Array
    report: SolveReport

    @property
    def outlets(self) -> tuple[Stream]:
        """The single outlet, as a flowsheet output tuple."""
        return (self.outlet,)

    @property
    def heat(self) -> Array:
        """Heat into the fluid (W)."""
        return self.duty


class PumpResult(NamedTuple):
    """Outlet of a pump together with the shaft work.

    Attributes:
        outlet: Product `Stream`.
        work: Shaft power delivered to the fluid (W).
        report: Solve and specification report.
    """

    outlet: Stream
    work: Array
    report: SolveReport

    @property
    def outlets(self) -> tuple[Stream]:
        """The single outlet, as a flowsheet output tuple."""
        return (self.outlet,)


class WorkResult(NamedTuple):
    """Outlet of a compressor/turbine with actual and ideal shaft work.

    Attributes:
        outlet: Product `Stream`.
        work: Actual shaft power into the fluid (W); negative for a turbine
            (the fluid does work on the surroundings).
        ideal_work: Reversible (isentropic) shaft power into the fluid (W).
        report: Solve and specification report.
    """

    outlet: Stream
    work: Array
    ideal_work: Array
    report: SolveReport

    @property
    def outlets(self) -> tuple[Stream]:
        """The single outlet, as a flowsheet output tuple."""
        return (self.outlet,)


# --------------------------------------------------------------------------- #
# Kernel plumbing
# --------------------------------------------------------------------------- #


def _f(value: ArrayLike) -> Array:
    """A strongly typed float array, so Python and JAX scalars share one compilation."""
    return jnp.asarray(value, dtype=float)


def _typed(tree: Any) -> Any:
    """Remove weak types from every array leaf (a weak leaf would force a retrace)."""
    return jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=jnp.asarray(v).dtype), tree)


def _run(kernel: Any, impl: Any, pkg: Any, *args: Any, **static: Any) -> Any:
    """Evaluate a unit through its compiled kernel, or eagerly for an opaque package.

    Registered pytree packages (every built-in package) compile once per
    structure; an arbitrary Python object can't cross a compilation boundary, so
    it runs the same implementation eagerly.
    """
    if _is_pytree(pkg):
        return kernel(*_typed(args), _typed(pkg), **static)
    return impl(*args, pkg, **static)


def _raise(report: SolveReport, context: str, labels: tuple[str, ...] = ()) -> None:
    """Raise for a concrete failed report (a no-op while tracing)."""
    require_converged(report, context, labels)


def _checked(ok: Array, report: SolveReport) -> SolveReport:
    """Mark a report ``INVALID_INPUT`` where an operating limit is violated."""
    return with_status(report, ~ok, SolveStatus.INVALID_INPUT)


def _converged() -> SolveReport:
    return residual_report(jnp.zeros(1))


def _state_report(streams: tuple[Stream, ...], report: SolveReport) -> SolveReport:
    """Combine a unit's report with the physical validity of its outlets."""
    valid = jnp.all(jnp.array([s.report.converged for s in streams]))
    return with_status(report, report.converged & ~valid, SolveStatus.NONFINITE)


def _energy_state(
    pkg: Any,
    solved: Any,
    n: Array,
    p: Array,
    components: tuple[str, ...],
    allow_extrapolation: bool,
) -> tuple[Stream, SolveReport]:
    """Outlet of an energy-specified flash and its verified report.

    The flash report already verifies energy, material, and equilibrium
    closure independently of the temperature iteration. Parameter
    applicability (the package evidence's observed bounds) is added here.
    """
    from fugacio.thermo.provenance import PackageEvidence, assess_applicability

    state = solved.value
    total = jnp.sum(n)
    outlet = Stream(
        n=n, t=state.t, p=p, components=components, vapor_n=state.beta * total * state.y
    )
    applicability = assess_applicability(
        getattr(pkg, "evidence", PackageEvidence()), state.t, p, _composition(n)
    )
    applicable = (
        applicability.parameters_available if allow_extrapolation else applicability.accepted
    )
    report = with_status(solved.report, ~applicable, SolveStatus.OUT_OF_DOMAIN)
    # An empty stream has no composition to verify; its state is trivially valid.
    report = jax.tree_util.tree_map(lambda a, b: jnp.where(total > 0, a, b), report, _converged())
    return outlet, report


# --------------------------------------------------------------------------- #
# Separators
# --------------------------------------------------------------------------- #


def _flash_drum_impl(feed: Stream, t: Array, p: Array, pkg: Any, *, duty: bool) -> FlashDrumResult:
    z = _composition(feed.n)
    total = feed.total
    detached = _detached_flash(pkg, t, p, z)
    beta = detached.value.beta

    def single_phase(_: None) -> tuple[Array, Array]:
        vapor_flow = jnp.where(beta <= 0.0, jnp.zeros_like(feed.n), feed.n)
        return vapor_flow, feed.n - vapor_flow

    def two_phase(_: None) -> tuple[Array, Array]:
        # Differentiate the equilibrium split only when both phases exist.
        result = pkg.flash_pt(t, p, z)
        return result.y * result.beta * total, result.x * (1.0 - result.beta) * total

    vapor_flow, liquid_flow = jax.lax.cond(
        jnp.isfinite(beta) & ((beta <= 0.0) | (beta >= 1.0)), single_phase, two_phase, None
    )
    report = detached.report
    if pkg.n_components == 1:
        # A pure fluid at its own resolved saturation state (a valve outlet, a
        # PH stream) keeps its inventory: a PT specification on the saturation
        # line can't determine the phase amounts, but the feed already has.
        same = (
            feed.phase_known
            & (jnp.abs(t - feed.t) <= 1e-9 * jnp.abs(t))
            & (jnp.abs(p - feed.p) <= 1e-9 * jnp.abs(p))
        )
        retained = jnp.asarray(feed.vapor_n)
        vapor_flow = jnp.where(same, retained, vapor_flow)
        liquid_flow = jnp.where(same, feed.n - retained, liquid_flow)
        report = jax.tree_util.tree_map(lambda a, b: jnp.where(same, a, b), _converged(), report)
    vapor = Stream(n=vapor_flow, vapor_n=vapor_flow, t=t, p=p, components=feed.components)
    liquid = Stream(
        n=liquid_flow, vapor_n=jnp.zeros_like(feed.n), t=t, p=p, components=feed.components
    )
    heat = jnp.asarray(0.0)
    if duty:
        heat = (
            vapor.total * molar_enthalpy(vapor, model=pkg)
            + liquid.total * molar_enthalpy(liquid, model=pkg)
            - total * molar_enthalpy(feed, model=pkg)
        )
    report = _state_report((vapor, liquid), report)
    vapor, liquid, heat = nan_unless_converged((vapor, liquid, heat), report)
    return FlashDrumResult(vapor, liquid, heat, report)


def _detached_flash(pkg: Any, t: Array, p: Array, z: Array) -> Any:
    """Locate the phase regime and report without tracing an unused flash derivative.

    Inputs are detached before the flash (stopping only its output is too late:
    its implicit rule would already have been linearized). Opaque packages pass
    through unchanged.
    """
    package, temperature, pressure, composition = jax.tree_util.tree_map(
        lambda v: jax.lax.stop_gradient(v) if isinstance(v, Array | jax.core.Tracer) else v,
        (pkg, t, p, z),
    )
    return jax.lax.stop_gradient(package.flash_pt_with_info(temperature, pressure, composition))


_flash_drum_kernel = jax.jit(_flash_drum_impl, static_argnames=("duty",))

#: Tangent-plane distance below which an outlet is reported as unstable.
_STABILITY_TOLERANCE = 1e-7

_UNSTABLE_OUTLET = (
    "a flash-drum outlet is unstable: a trial phase lowers its Gibbs energy, so the "
    "feed splits into more phases than a vapor-liquid flash represents (for example, "
    "two liquids). Use decanter or three_phase_flash with an activity-coefficient "
    "package, or flash_drum_checked to make this an error."
)


def _outlet_stability_impl(vapor: Stream, liquid: Stream, pkg: Any) -> Array:
    """Most negative tangent-plane distance over the non-empty outlets."""

    def distance(stream: Stream) -> Array:
        found = pkg.stability(stream.t, stream.p, _composition(stream.n))
        return jnp.where(stream.total > 0, found.tpd, jnp.inf)

    return jnp.minimum(distance(vapor), distance(liquid))


_outlet_stability_kernel = jax.jit(_outlet_stability_impl)


def _warn_if_unstable(pkg: Any, vapor: Stream, liquid: Stream) -> None:
    """Warn, for a concrete result, when an outlet phase isn't stable."""
    if isinstance(vapor.t, jax.core.Tracer):
        return
    tpd = _run(_outlet_stability_kernel, _outlet_stability_impl, pkg, vapor, liquid)
    if float(tpd) < -_STABILITY_TOLERANCE:
        warnings.warn(PhysicalAcceptanceWarning(_UNSTABLE_OUTLET), stacklevel=3)


def flash_drum_with_info(
    feed: Stream, t: ArrayLike, p: ArrayLike, *, model: Model = None
) -> FlashDrumResult:
    """Flash a feed at temperature ``t`` and pressure ``p``, with duty and report.

    Args:
        feed: Inlet `Stream`.
        t: Drum temperature (K).
        p: Drum pressure (Pa).
        model: Property package; defaults to Peng-Robinson over the feed.

    Returns:
        The vapor and liquid products, the duty to hold the drum at ``t``, and
        the report. A pure fluid specified at its own resolved saturation state
        keeps its phase inventory.

    Raises:
        ConvergenceError: If an eager flash fails.
        ValueError: If ``p`` exceeds the feed pressure (a drum can't raise it).
    """
    pkg = resolve_package(feed.components, model)
    limits.require(limits.pressure_not_raised(feed.p, p), "flash drum: " + limits.NO_PRESSURE_RISE)
    result = _run(_flash_drum_kernel, _flash_drum_impl, pkg, feed, _f(t), _f(p), duty=True)
    _raise(result.report, "flash drum", ("PT flash",))
    _warn_if_unstable(pkg, result.vapor, result.liquid)
    return result


def flash_drum(
    feed: Stream, t: ArrayLike, p: ArrayLike, *, model: Model = None
) -> tuple[Stream, Stream]:
    """Flash a feed stream at temperature ``t`` and pressure ``p``.

    Args:
        feed: Inlet `Stream`.
        t: Drum temperature (K).
        p: Drum pressure (Pa).
        model: Property package; defaults to Peng-Robinson over the feed.

    Returns:
        ``(vapor, liquid)`` product streams. Their flows are differentiable with
        respect to ``t``, ``p``, the feed, and the package parameters.

    Raises:
        ConvergenceError: If an eager flash fails (NaN products when traced).
        ValueError: If ``p`` exceeds the feed pressure.

    Warns:
        PhysicalAcceptanceWarning: If an eager call's outlet phase is unstable
            (the feed forms more phases than a vapor-liquid flash represents).
    """
    pkg = resolve_package(feed.components, model)
    limits.require(limits.pressure_not_raised(feed.p, p), "flash drum: " + limits.NO_PRESSURE_RISE)
    result = _run(_flash_drum_kernel, _flash_drum_impl, pkg, feed, _f(t), _f(p), duty=False)
    _raise(result.report, "flash drum", ("PT flash",))
    _warn_if_unstable(pkg, result.vapor, result.liquid)
    return result.vapor, result.liquid


def _adiabatic_flash_impl(
    feed: Stream, p: Array, duty: Array, t_init: Array, pkg: Any, *, allow_extrapolation: bool
) -> FlashDrumResult:
    total = feed.total
    h_spec = molar_enthalpy(feed, model=pkg) + duty / jnp.where(total > 0, total, 1.0)
    z = _composition(feed.n)
    solved = pkg.flash_ph_with_info(p, h_spec, z, t_init=t_init)
    state = solved.value
    _, report = _energy_state(pkg, solved, feed.n, p, feed.components, allow_extrapolation)
    vapor_n = state.beta * total * state.y
    vapor = Stream(n=vapor_n, vapor_n=vapor_n, t=state.t, p=p, components=feed.components)
    liquid = Stream(
        n=feed.n - vapor_n,
        vapor_n=jnp.zeros_like(feed.n),
        t=state.t,
        p=p,
        components=feed.components,
    )
    report = _state_report((vapor, liquid), report)
    vapor, liquid, heat = nan_unless_converged((vapor, liquid, duty), report)
    return FlashDrumResult(vapor, liquid, heat, report)


_adiabatic_flash_kernel = jax.jit(_adiabatic_flash_impl, static_argnames=("allow_extrapolation",))


def adiabatic_flash(
    feed: Stream,
    p: ArrayLike,
    *,
    duty: ArrayLike = 0.0,
    model: Model = None,
    t_init: ArrayLike = 300.0,
    allow_extrapolation: bool = False,
) -> FlashDrumResult:
    """Separate a feed at pressure ``p`` with heat ``duty`` added (a PH flash drum).

    The drum temperature follows from the energy balance, so this is the
    separator for a letdown into a drum, a partial condenser on a duty, or a
    pure fluid on its saturation line (where a temperature can't fix the phase
    amounts).

    Args:
        feed: Inlet `Stream`.
        p: Drum pressure (Pa).
        duty: Heat added (W); zero for an adiabatic drum.
        model: Property package; defaults to Peng-Robinson over the feed.
        t_init: Initial temperature guess (K) for the energy solve.
        allow_extrapolation: Accept states outside the parameters' observed bounds.

    Returns:
        The vapor and liquid products, the duty, and the report.

    Raises:
        ConvergenceError: If an eager solve fails.
        ValueError: If ``p`` exceeds the feed pressure.
    """
    pkg = resolve_package(feed.components, model)
    limits.require(limits.pressure_not_raised(feed.p, p), "flash drum: " + limits.NO_PRESSURE_RISE)
    result = _run(
        _adiabatic_flash_kernel,
        _adiabatic_flash_impl,
        pkg,
        feed,
        _f(p),
        _f(duty),
        _f(t_init),
        allow_extrapolation=allow_extrapolation,
    )
    _raise(result.report, "adiabatic flash", ("energy and equilibrium",))
    return result


# --------------------------------------------------------------------------- #
# Heater / cooler
# --------------------------------------------------------------------------- #


def _saturation_outlet(
    feed: Stream, p_out: Array, fraction: Array, pkg: Any
) -> tuple[Stream, SolveReport]:
    """Outlet at a specified vapor fraction.

    A pure fluid takes any quality; a mixture is saturated liquid (0) or vapor (1).
    """
    z = _composition(feed.n)
    if pkg.n_components == 1:
        solved = pkg.bubble_temperature_with_info(p_out, z)
        report = solved.report
        t_out = solved.value.value
        vapor_n = fraction * feed.n
        ok = (fraction >= 0.0) & (fraction <= 1.0)
    else:
        bubble = pkg.bubble_temperature_with_info(p_out, z)
        dew = pkg.dew_temperature_with_info(p_out, z)
        at_dew = fraction >= 0.5
        report = jax.tree_util.tree_map(
            lambda a, b: jnp.where(at_dew, a, b), dew.report, bubble.report
        )
        t_out = jnp.where(at_dew, dew.value.value, bubble.value.value)
        vapor_n = jnp.where(at_dew, feed.n, jnp.zeros_like(feed.n))
        ok = (fraction == 0.0) | (fraction == 1.0)
    outlet = Stream(n=feed.n, t=t_out, p=p_out, components=feed.components, vapor_n=vapor_n)
    return outlet, _checked(ok, report)


def _heater_impl(
    feed: Stream,
    value: Array,
    dp: Array,
    t_init: Array,
    pkg: Any,
    *,
    spec: str,
    allow_extrapolation: bool,
) -> HeaterResult:
    p_out = feed.p - dp
    total = feed.total
    h_in = molar_enthalpy(feed, model=pkg)
    if spec == "duty":
        h_spec = h_in + value / jnp.where(total > 0, total, 1.0)
        solved = pkg.flash_ph_with_info(p_out, h_spec, _composition(feed.n), t_init=t_init)
        outlet, report = _energy_state(
            pkg, solved, feed.n, p_out, feed.components, allow_extrapolation
        )
        report = with_status(report, (total <= 0) & (value != 0), SolveStatus.INFEASIBLE)
        duty = value
    else:
        if spec == "t_out":
            outlet = Stream(n=feed.n, t=value, p=p_out, components=feed.components)
            report = _converged()
            if pkg.n_components == 1:
                # A pure fluid exactly at its saturation temperature has an
                # undetermined quality: that's a vapor-fraction specification.
                saturation = pkg.bubble_temperature_with_info(p_out, jnp.ones(1))
                on_line = saturation.report.converged & (
                    jnp.abs(value - saturation.value.value) <= 1e-6
                )
                report = _checked(~on_line, report)
        else:
            outlet, report = _saturation_outlet(feed, p_out, value, pkg)
        duty = total * (molar_enthalpy(outlet, model=pkg) - h_in)
        report = with_status(report, report.converged & ~jnp.isfinite(duty), SolveStatus.NONFINITE)
    report = _checked(dp >= 0.0, report)
    report = _state_report((outlet,), report)
    outlet, duty = nan_unless_converged((outlet, duty), report)
    return HeaterResult(outlet=outlet, duty=duty, report=report)


_heater_kernel = jax.jit(_heater_impl, static_argnames=("spec", "allow_extrapolation"))


def heater(
    feed: Stream,
    *,
    t_out: ArrayLike | None = None,
    duty: ArrayLike | None = None,
    vapor_fraction: ArrayLike | None = None,
    dp: ArrayLike = 0.0,
    model: Model = None,
    t_init: ArrayLike = 300.0,
    allow_extrapolation: bool = False,
) -> HeaterResult:
    """Heat or cool a stream on a temperature, duty, or vapor-fraction specification.

    Provide exactly one of:

    * ``t_out``: outlet temperature (K); the duty follows from the enthalpy change;
    * ``duty``: signed heat added (W); the outlet temperature follows from an
      isenthalpic solve, so partial vaporization/condensation is handled;
    * ``vapor_fraction``: outlet molar vapor fraction at its saturation
      temperature, any value in ``[0, 1]`` for a pure fluid (a condenser to
      saturated liquid is ``0``) and ``0`` (bubble point) or ``1`` (dew point)
      for a mixture.

    ``dp`` is the (non-negative) pressure drop across the block. A pure fluid
    specified by ``t_out`` exactly at its saturation temperature is rejected:
    its quality is undetermined, so use ``vapor_fraction`` or ``duty``.

    Raises:
        ValueError: If not exactly one specification is given, or ``dp < 0``.
        ConvergenceError: If an eager solve fails.
    """
    specs = {"t_out": t_out, "duty": duty, "vapor_fraction": vapor_fraction}
    given = [k for k, v in specs.items() if v is not None]
    if len(given) != 1:
        raise ValueError("heater requires exactly one of t_out, duty, or vapor_fraction")
    limits.require(jnp.asarray(dp) >= 0.0, "heater: the pressure drop can't be negative")
    pkg = resolve_package(feed.components, model)
    spec = given[0]
    value = specs[spec]
    assert value is not None
    result = _run(
        _heater_kernel,
        _heater_impl,
        pkg,
        feed,
        _f(value),
        _f(dp),
        _f(t_init),
        spec=spec,
        allow_extrapolation=allow_extrapolation,
    )
    status = result.report.status
    if (
        spec == "t_out"
        and pkg.n_components == 1
        and not isinstance(status, jax.core.Tracer)
        and int(status) == int(SolveStatus.INVALID_INPUT)
    ):
        raise ValueError(
            "heater: t_out is the pure fluid's saturation temperature at the outlet "
            "pressure, where the vapor fraction is undetermined; specify vapor_fraction "
            "(0 for saturated liquid, 1 for saturated vapor) or duty instead"
        )
    _raise(result.report, "heater", ("outlet state",))
    return result


# --------------------------------------------------------------------------- #
# Pressure-changing equipment
# --------------------------------------------------------------------------- #


def _valve_impl(
    feed: Stream, p_out: Array, t_init: Array, pkg: Any, *, allow_extrapolation: bool
) -> tuple[Stream, SolveReport]:
    h_spec = molar_enthalpy(feed, model=pkg)
    solved = pkg.flash_ph_with_info(p_out, h_spec, _composition(feed.n), t_init=t_init)
    outlet, report = _energy_state(pkg, solved, feed.n, p_out, feed.components, allow_extrapolation)
    report = _checked(limits.pressure_not_raised(feed.p, p_out), report)
    report = _state_report((outlet,), report)
    return nan_unless_converged(outlet, report), report


_valve_kernel = jax.jit(_valve_impl, static_argnames=("allow_extrapolation",))


def valve(
    feed: Stream,
    p_out: ArrayLike,
    *,
    model: Model = None,
    t_init: ArrayLike = 300.0,
    allow_extrapolation: bool = False,
) -> Stream:
    """Isenthalpic (Joule-Thomson) pressure letdown to ``p_out``.

    Enthalpy is conserved, so the outlet temperature (and any flashing that
    results from the pressure drop) follows from an isenthalpic solve. The
    outlet may be two-phase; its phase inventory is retained on the stream.

    Raises:
        ValueError: If ``p_out`` exceeds the inlet pressure.
        ConvergenceError: If an eager solve fails.
    """
    limits.require(limits.pressure_not_raised(feed.p, p_out), "valve: " + limits.NO_PRESSURE_RISE)
    pkg = resolve_package(feed.components, model)
    outlet, report = _run(
        _valve_kernel,
        _valve_impl,
        pkg,
        feed,
        _f(p_out),
        _f(t_init),
        allow_extrapolation=allow_extrapolation,
    )
    _raise(report, "valve", ("outlet state",))
    return outlet


def _efficiency_ok(efficiency: Array) -> Array:
    return (efficiency > 0) & (efficiency <= 1)


def _pump_impl(
    feed: Stream, p_out: Array, efficiency: Array, t_init: Array, pkg: Any
) -> PumpResult:
    z = _composition(feed.n)
    vapor_fraction = jnp.where(
        feed.phase_known,
        jnp.sum(jnp.asarray(feed.vapor_n)) / jnp.maximum(feed.total, 1e-300),
        _phase_classification(pkg, feed.t, feed.p, z).beta,
    )
    v_l = pkg.volume(feed.t, feed.p, z, phase="liquid")
    w_ideal = v_l * (p_out - feed.p)
    w_actual = w_ideal / efficiency
    h_out = molar_enthalpy(feed, model=pkg) + w_actual
    solved = pkg.flash_ph_with_info(p_out, h_out, z, t_init=t_init)
    outlet, report = _energy_state(pkg, solved, feed.n, p_out, feed.components, False)
    ok = (
        _efficiency_ok(efficiency)
        & limits.pressure_not_lowered(feed.p, p_out)
        & (limits.liquid_inlet(vapor_fraction) | (feed.total <= 0))
    )
    report = _state_report((outlet,), _checked(ok, report))
    outlet, work = nan_unless_converged((outlet, w_actual * feed.total), report)
    return PumpResult(outlet=outlet, work=work, report=report)


_pump_kernel = jax.jit(_pump_impl)


def pump(
    feed: Stream,
    p_out: ArrayLike,
    *,
    efficiency: ArrayLike = 0.75,
    model: Model = None,
    t_init: ArrayLike = 300.0,
) -> PumpResult:
    """Pump an (incompressible) liquid from ``feed.p`` to ``p_out``.

    The reversible work is ``v_L (p_out - p_in)`` per mole using the liquid molar
    volume; the actual work is divided by ``efficiency`` and the inefficiency is
    deposited as heat, so the outlet temperature is found from an isenthalpic
    balance on ``H_out = H_in + W_actual``.

    Raises:
        ValueError: For an efficiency outside ``(0, 1]``, a pressure decrease, or
            a feed that isn't liquid (use `compressor` for a vapor).
        ConvergenceError: If an eager solve fails.
    """
    limits.require(_efficiency_ok(jnp.asarray(efficiency)), "pump: efficiency must be in (0, 1]")
    limits.require(limits.pressure_not_lowered(feed.p, p_out), "pump: " + limits.NO_PRESSURE_DROP)
    pkg = resolve_package(feed.components, model)
    result = _run(_pump_kernel, _pump_impl, pkg, feed, _f(p_out), _f(efficiency), _f(t_init))
    if not isinstance(result.report.status, jax.core.Tracer) and int(result.report.status) == int(
        SolveStatus.INVALID_INPUT
    ):
        raise ValueError("pump: " + limits.LIQUID_INLET)
    _raise(result.report, "pump", ("outlet state",))
    return result


def _compress_impl(
    feed: Stream,
    p_out: Array,
    efficiency: Array,
    t_init: Array,
    pkg: Any,
    *,
    is_turbine: bool,
) -> WorkResult:
    z = _composition(feed.n)
    h_in = molar_enthalpy(feed, model=pkg)
    s_in = molar_entropy(feed, model=pkg)
    iso = pkg.flash_ps_with_info(p_out, s_in, z, t_init=t_init)
    iso_stream = Stream(
        feed.n, iso.value.t, p_out, feed.components, iso.value.beta * feed.total * iso.value.y
    )
    h_out_ideal = molar_enthalpy(iso_stream, model=pkg)
    w_ideal = h_out_ideal - h_in
    w_actual = efficiency * w_ideal if is_turbine else w_ideal / efficiency
    solved = pkg.flash_ph_with_info(p_out, h_in + w_actual, z, t_init=t_init)
    outlet, report = _energy_state(pkg, solved, feed.n, p_out, feed.components, False)
    report = with_status(report, report.converged & ~iso.report.converged, iso.report.status)
    direction = (
        limits.pressure_not_raised(feed.p, p_out)
        if is_turbine
        else limits.pressure_not_lowered(feed.p, p_out)
    )
    report = _state_report((outlet,), _checked(_efficiency_ok(efficiency) & direction, report))
    outlet, work, ideal = nan_unless_converged(
        (outlet, w_actual * feed.total, w_ideal * feed.total), report
    )
    return WorkResult(outlet=outlet, work=work, ideal_work=ideal, report=report)


_compress_kernel = jax.jit(_compress_impl, static_argnames=("is_turbine",))


def _machine(
    feed: Stream,
    p_out: ArrayLike,
    efficiency: ArrayLike,
    model: Model,
    t_init: ArrayLike,
    *,
    is_turbine: bool,
) -> WorkResult:
    name = "turbine" if is_turbine else "compressor"
    limits.require(_efficiency_ok(jnp.asarray(efficiency)), f"{name}: efficiency must be in (0, 1]")
    if is_turbine:
        limits.require(
            limits.pressure_not_raised(feed.p, p_out), f"{name}: {limits.EXPANSION_ONLY}"
        )
    else:
        limits.require(
            limits.pressure_not_lowered(feed.p, p_out), f"{name}: {limits.NO_PRESSURE_DROP}"
        )
    pkg = resolve_package(feed.components, model)
    result = _run(
        _compress_kernel,
        _compress_impl,
        pkg,
        feed,
        _f(p_out),
        _f(efficiency),
        _f(t_init),
        is_turbine=is_turbine,
    )
    _raise(result.report, name, ("outlet state",))
    return result


def compressor(
    feed: Stream,
    p_out: ArrayLike,
    *,
    efficiency: ArrayLike = 0.75,
    model: Model = None,
    t_init: ArrayLike = 300.0,
) -> WorkResult:
    """Compress a stream to ``p_out`` with an isentropic ``efficiency`` (< 1).

    The reversible outlet is the isentropic state at ``p_out``; the actual work is
    ``W_ideal / efficiency`` and the extra enthalpy sets the (higher) real outlet
    temperature via an isenthalpic solve.

    Raises:
        ValueError: For an efficiency outside ``(0, 1]`` or ``p_out`` below the inlet.
        ConvergenceError: If an eager solve fails.
    """
    return _machine(feed, p_out, efficiency, model, t_init, is_turbine=False)


def turbine(
    feed: Stream,
    p_out: ArrayLike,
    *,
    efficiency: ArrayLike = 0.85,
    model: Model = None,
    t_init: ArrayLike = 300.0,
) -> WorkResult:
    """Expand a stream to ``p_out`` with an isentropic ``efficiency`` (< 1).

    The fluid recovers ``efficiency`` of the reversible work; the returned
    ``work`` is negative (power delivered to the shaft) and the real outlet is
    warmer than the isentropic outlet because of the lost work.

    Raises:
        ValueError: For an efficiency outside ``(0, 1]`` or ``p_out`` above the inlet.
        ConvergenceError: If an eager solve fails.
    """
    return _machine(feed, p_out, efficiency, model, t_init, is_turbine=True)


# --------------------------------------------------------------------------- #
# Splitters and the mixer
# --------------------------------------------------------------------------- #


def splitter(feed: Stream, fractions: ArrayLike) -> tuple[Stream, ...]:
    """Split ``feed`` into outlets that share its composition and state.

    ``fractions`` holds one split fraction per outlet; each outlet carries that
    fraction of every component flow (and of the vapor inventory). The fractions
    must each lie in ``[0, 1]`` and sum to one, so the split conserves matter.

    Raises:
        ValueError: For invalid fractions (NaN outlets when traced).
    """
    fr = jnp.asarray(fractions, dtype=float)
    ok = limits.require(limits.split_fractions_valid(fr), "splitter: " + limits.SPLIT_FRACTIONS)
    gate = jnp.where(ok, 1.0, jnp.nan)
    return tuple(feed.scaled(fr[i] * gate) for i in range(fr.shape[0]))


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

    ``split_to_top`` is a per-component fraction in ``[0, 1]`` (aligned with
    ``feed.components``) sent to the top outlet; the remainder leaves in the
    bottom. This is the workhorse "spec" separator for conceptual flowsheets, a
    stand-in for a column or absorber whose recoveries are known. Product
    temperatures and pressures default to the feed's; product pressures can't
    exceed it.

    Raises:
        ValueError: For recoveries outside ``[0, 1]`` or a raised pressure.
    """
    frac = jnp.asarray(split_to_top, dtype=float)
    top_pressure = feed.p if top_p is None else jnp.asarray(top_p)
    bottom_pressure = feed.p if bottom_p is None else jnp.asarray(bottom_p)
    ok = limits.require(
        limits.recoveries_valid(frac), "component separator: " + limits.RECOVERIES
    ) & limits.require(
        limits.pressure_not_raised(feed.p, jnp.maximum(top_pressure, bottom_pressure)),
        "component separator: " + limits.NO_PRESSURE_RISE,
    )
    gate = jnp.where(ok, 1.0, jnp.nan)
    top = Stream(
        n=feed.n * frac * gate,
        t=feed.t if top_t is None else jnp.asarray(top_t),
        p=top_pressure,
        components=feed.components,
    )
    bottom = Stream(
        n=feed.n * (1.0 - frac) * gate,
        t=feed.t if bottom_t is None else jnp.asarray(bottom_t),
        p=bottom_pressure,
        components=feed.components,
    )
    return top, bottom


def _mix_impl(
    streams: tuple[Stream, ...], p_out: Array, t_init: Array, pkg: Any
) -> tuple[Stream, SolveReport]:
    components = streams[0].components
    n_total = jnp.sum(jnp.stack([s.n for s in streams]), axis=0)
    total = jnp.sum(n_total)
    z = _composition(n_total)
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
    solved = pkg.flash_ph_with_info(p_out, target, z, t_init=t_init)
    outlet, report = _energy_state(pkg, solved, n_total, p_out, components, False)
    lowest = jnp.min(jnp.stack([s.p for s in streams]))
    report = _state_report((outlet,), _checked(limits.pressure_not_raised(lowest, p_out), report))
    return nan_unless_converged(outlet, report), report


_mix_kernel = jax.jit(_mix_impl)


def mix(
    streams: list[Stream],
    *,
    t: ArrayLike | None = None,
    p: ArrayLike | None = None,
    model: Model = None,
    t_init: ArrayLike = 300.0,
) -> Stream:
    """Combine streams with an exact material balance and an adiabatic energy balance.

    Flows add component-by-component. By default the outlet temperature is found
    from an *adiabatic* energy balance (total enthalpy in equals total enthalpy
    out) via an isenthalpic solve, so heat of mixing and any phase change are
    accounted for; pass ``t`` to fix the outlet temperature instead (its heat
    is then the enthalpy change). Outlet pressure defaults to the lowest inlet
    pressure and can't exceed it.

    Raises:
        ValueError: if the streams are empty, don't share a component list, or
            ``p`` exceeds the lowest inlet pressure.
        ConvergenceError: If an eager energy solve fails.
    """
    if not streams:
        raise ValueError("mix requires at least one stream")
    components = streams[0].components
    for s in streams[1:]:
        if s.components != components:
            raise ValueError("all streams must share the same component list to mix")
    lowest = jnp.min(jnp.stack([jnp.asarray(s.p) for s in streams]))
    p_out = lowest if p is None else jnp.asarray(p, dtype=float)
    limits.require(limits.pressure_not_raised(lowest, p_out), "mixer: " + limits.NO_PRESSURE_RISE)
    n_total = jnp.sum(jnp.stack([s.n for s in streams]), axis=0)
    if t is not None:
        return Stream(n=n_total, t=jnp.asarray(t, dtype=float), p=p_out, components=components)
    pkg = resolve_package(components, model)
    outlet, report = _run(_mix_kernel, _mix_impl, pkg, tuple(streams), _f(p_out), _f(t_init))
    _raise(report, "mixer", ("outlet state",))
    return outlet


__all__ = [
    "FlashDrumResult",
    "HeaterResult",
    "PumpResult",
    "WorkResult",
    "adiabatic_flash",
    "component_separator",
    "compressor",
    "flash_drum",
    "flash_drum_with_info",
    "heater",
    "mix",
    "pump",
    "splitter",
    "turbine",
    "valve",
]
