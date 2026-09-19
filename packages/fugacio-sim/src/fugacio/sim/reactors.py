"""Reactor unit operations on a common property package.

These blocks turn the reaction thermochemistry and kinetics in `fugacio.thermo`
into flowsheet units that consume and produce `Stream` objects:

* `equilibrium_reactor`, `cstr`, and `pfr`: homogeneous reactors solved by the
  checked implementation in `fugacio.sim.reaction_units.reaction_reactor`, with
  phase-specific package properties, formation-consistent energy balances, and
  retained reports and profiles;
* `stoichiometric_reactor`: a specified extent or key-reactant conversion on
  the package's formation-consistent energy basis;
* `batch_reactor`: a constant-volume, ideal-gas batch reactor integrated in time.

Every reactor supports an energy specification: an outlet temperature (the heat
to hold it is returned, carrying the heat of reaction) or a heat duty (zero for
adiabatic operation, the outlet temperature is then solved). Results are
differentiable in the feed, the operating conditions, *and* the reaction and
kinetic parameters. Eager calls raise on failure, and a traced failure has NaN
products (outlet, duty, and extents) while its report, balances, and profiles
are kept for diagnosis. Pass ``check=False`` to get the best iterate instead,
with a failed report and nonfinite derivatives.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.properties import Model, enthalpy_flow, resolve_package
from fugacio.sim.reaction_units import ReactionResult, reaction_reactor
from fugacio.sim.stream import Stream
from fugacio.thermo.constants import R
from fugacio.thermo.diagnostics import (
    SolveReport,
    SolveStatus,
    nan_unless_converged,
    require_converged,
    residual_report,
    with_status,
)
from fugacio.thermo.ideal import cp_ig, enthalpy_ig
from fugacio.thermo.package import HelmholtzPackage
from fugacio.thermo.reaction_system import ReactionSet
from fugacio.thermo.reactions import Reaction, delta_h_rxn, reaction_arrays

ArrayLike = Array | float


def _as_reactions(reactions: Reaction | Sequence[Reaction]) -> list[Reaction]:
    return [reactions] if isinstance(reactions, Reaction) else list(reactions)


def _stack_nu(reactions: Sequence[Reaction], components: tuple[str, ...]) -> Array:
    """Stack reaction stoichiometries into an ``(n_reactions, n_components)`` matrix."""
    rows = []
    for r in reactions:
        if tuple(r.components) != tuple(components):
            raise ValueError(
                "each reaction must be defined over the reactor feed's components in the same order"
            )
        rows.append(jnp.asarray(r.nu))
    return jnp.stack(rows)


def _system(
    reactions: Reaction | Sequence[Reaction] | ReactionSet,
    rate_laws: Any,
    phase: str,
    rate_basis: str,
) -> ReactionSet:
    if isinstance(reactions, ReactionSet):
        if rate_laws is not None:
            raise ValueError("a ReactionSet already supplies its rate laws")
        return reactions
    return ReactionSet.from_reactions(
        reactions, () if rate_laws is None else rate_laws, phase=phase, rate_basis=rate_basis
    )


def _nan_on_failure(result: ReactionResult, check: bool) -> ReactionResult:
    """With ``check``, NaN products for a failed reactor (report and profiles remain)."""
    if not check:
        return result
    outlet, duty, extent, generation = nan_unless_converged(
        (result.outlet, result.duty, result.extent, result.generation), result.report
    )
    return result._replace(outlet=outlet, duty=duty, extent=extent, generation=generation)


def equilibrium_reactor(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction] | ReactionSet,
    *,
    model: Model = None,
    t_out: ArrayLike | None = None,
    duty: ArrayLike | None = None,
    dp: ArrayLike = 0.0,
    phase: str = "vapor",
    tol: float = 1e-10,
    max_iter: int = 100,
    check: bool = True,
) -> ReactionResult:
    """Reactor whose outlet is the homogeneous chemical-equilibrium composition.

    Args:
        feed: Inlet stream; reactions must be defined over ``feed.components``.
        reactions: One reaction, several sharing the feed's component order, or a
            validated `ReactionSet`.
        model: Property package; defaults to Peng-Robinson over the feed.
        t_out: Isothermal outlet temperature (K); the default is the feed's.
        duty: Heat added (W) instead of ``t_out``; zero is adiabatic.
        dp: Non-negative pressure drop (Pa).
        phase: Reacting phase for a newly constructed reaction set.
        tol: Scaled nonlinear residual tolerance.
        max_iter: Newton iteration cap.
        check: Raise on a rejected concrete result.

    Returns:
        A `fugacio.sim.reaction_units.ReactionResult` with the outlet, duty,
        extents, balances, and report.
    """
    return _nan_on_failure(
        reaction_reactor(
            feed,
            _system(reactions, None, phase, "concentration"),
            kind="equilibrium",
            model=model,
            t_out=t_out,
            duty=duty,
            dp=dp,
            tol=tol,
            max_iter=max_iter,
            check=check,
        ),
        check,
    )


def cstr(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction] | ReactionSet,
    volume: ArrayLike,
    rate_laws: Any = None,
    *,
    model: Model = None,
    t_out: ArrayLike | None = None,
    duty: ArrayLike | None = None,
    dp: ArrayLike = 0.0,
    phase: str = "vapor",
    rate_basis: str = "concentration",
    tol: float = 1e-10,
    max_iter: int = 100,
    check: bool = True,
) -> ReactionResult:
    """Continuous stirred-tank reactor (perfectly mixed) at steady state.

    Solves ``F_out = F_in + V (r . Nu)`` with outlet-condition rates on the
    package's phase concentrations, jointly with the energy balance when a
    ``duty`` is specified.

    Args:
        feed: Inlet stream (``feed.n`` are molar flows, mol/s).
        reactions: Reaction(s) over ``feed.components``, or a `ReactionSet`.
        volume: Reacting-phase volume (m^3).
        rate_laws: One rate law per reaction; omit for a ReactionSet.
        model: Property package; defaults to Peng-Robinson over the feed.
        t_out: Isothermal temperature (K); the default is the feed's.
        duty: Heat added (W) instead of ``t_out``; zero is adiabatic.
        dp: Non-negative pressure drop (Pa).
        phase: Reacting phase for a newly constructed reaction set.
        rate_basis: Kinetic input convention for a newly constructed set.
        tol: Scaled nonlinear residual tolerance.
        max_iter: Newton iteration cap.
        check: Raise on a rejected concrete result.

    Returns:
        A `fugacio.sim.reaction_units.ReactionResult`.
    """
    return _nan_on_failure(
        reaction_reactor(
            feed,
            _system(reactions, rate_laws, phase, rate_basis),
            kind="cstr",
            model=model,
            volume=volume,
            t_out=t_out,
            duty=duty,
            dp=dp,
            tol=tol,
            max_iter=max_iter,
            check=check,
        ),
        check,
    )


def pfr(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction] | ReactionSet,
    volume: ArrayLike,
    rate_laws: Any = None,
    *,
    model: Model = None,
    t_out: ArrayLike | None = None,
    duty: ArrayLike | None = None,
    dp: ArrayLike = 0.0,
    phase: str = "vapor",
    rate_basis: str = "concentration",
    steps: int = 64,
    tol: float = 1e-10,
    max_iter: int = 100,
    check: bool = True,
) -> ReactionResult:
    """Plug-flow reactor integrated along its volume with step-doubling error control.

    Args:
        feed: Inlet stream (``feed.n`` are molar flows, mol/s).
        reactions: Reaction(s) over ``feed.components``, or a `ReactionSet`.
        volume: Reacting-phase volume (m^3).
        rate_laws: One rate law per reaction; omit for a ReactionSet.
        model: Property package; defaults to Peng-Robinson over the feed.
        t_out: Isothermal temperature (K); the default is the feed's.
        duty: Heat added (W) instead of ``t_out``, distributed uniformly; zero
            is adiabatic.
        dp: Non-negative pressure drop (Pa), linear along the reactor.
        phase: Reacting phase for a newly constructed reaction set.
        rate_basis: Kinetic input convention for a newly constructed set.
        steps: Coarse mesh size; a refined mesh estimates the integration error.
        tol: Scaled nonlinear residual tolerance.
        max_iter: Newton iteration cap for implicit steps.
        check: Raise on a rejected concrete result.

    Returns:
        A `fugacio.sim.reaction_units.ReactionResult` with axial profiles.
    """
    return _nan_on_failure(
        reaction_reactor(
            feed,
            _system(reactions, rate_laws, phase, rate_basis),
            kind="pfr",
            model=model,
            volume=volume,
            t_out=t_out,
            duty=duty,
            dp=dp,
            steps=steps,
            tol=tol,
            max_iter=max_iter,
            check=check,
        ),
        check,
    )


class StoichiometricResult(NamedTuple):
    """Products of a specified-extent reactor.

    Attributes:
        outlet: Product `Stream`.
        duty: Heat added (W), positive into the fluid.
        extent: Extent of each reaction (mol/s).
        generation: Net component generation ``extent @ nu`` (mol/s).
        report: Specification and energy-solve report.
    """

    outlet: Stream
    duty: Array
    extent: Array
    generation: Array
    report: SolveReport

    @property
    def outlets(self) -> tuple[Stream]:
        """The single outlet, as a flowsheet output tuple."""
        return (self.outlet,)

    @property
    def heat(self) -> Array:
        """Heat into the fluid (W)."""
        return self.duty


def stoichiometric_reactor(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction],
    *,
    extent: ArrayLike | None = None,
    conversion: ArrayLike | None = None,
    t_out: ArrayLike | None = None,
    duty: ArrayLike | None = None,
    dp: ArrayLike = 0.0,
    model: Model = None,
) -> StoichiometricResult:
    """Reactor with a *specified* extent or key-reactant conversion (no equilibrium).

    Provide exactly one of ``extent`` (per reaction, mol/s) or ``conversion`` (a
    single reaction's fractional conversion of its limiting reactant, in
    ``[0, 1]``). The outlet is ``n = n_feed + extent @ nu`` and every outlet flow
    must stay non-negative. Energy uses the package's sensible and residual
    enthalpies plus ideal-gas formation enthalpies: with ``t_out`` (default: the
    feed temperature) the duty follows; with ``duty`` (zero is adiabatic) the
    outlet temperature follows from a PH flash.

    Raises:
        ValueError: For conflicting or missing specifications, a conversion
            outside ``[0, 1]``, a negative outlet flow, or a reference-fluid package.
        ConvergenceError: If an eager adiabatic energy solve fails.
    """
    if (extent is None) == (conversion is None):
        raise ValueError("provide exactly one of 'extent' or 'conversion'")
    if t_out is not None and duty is not None:
        raise ValueError("provide at most one of 't_out' or 'duty'")
    comps = feed.components
    nu = _stack_nu(_as_reactions(reactions), comps)
    pkg = resolve_package(comps, model)
    if isinstance(pkg, HelmholtzPackage):
        raise ValueError("reaction energy requires the ideal-gas reference of a mixture package")
    hf, _gf, _coeffs = reaction_arrays(list(comps))
    n_feed = feed.n
    if conversion is not None:
        if nu.shape[0] != 1:
            raise ValueError("'conversion' is only defined for a single reaction")
        fraction = jnp.asarray(conversion, dtype=float)
        valid = (fraction >= 0.0) & (fraction <= 1.0)
        if not isinstance(valid, jax.core.Tracer) and not bool(valid):
            raise ValueError("conversion must lie in [0, 1]")
        row = nu[0]
        reactant = row < 0.0
        extent_max = jnp.min(jnp.where(reactant, n_feed / jnp.where(reactant, -row, 1.0), jnp.inf))
        extent_arr = jnp.reshape(fraction * extent_max, (1,))
    else:
        extent_arr = jnp.atleast_1d(jnp.asarray(extent, dtype=float))
        valid = jnp.asarray(True)
    generation = extent_arr @ nu
    n_out = n_feed + generation
    inventory = jnp.all(n_out >= -1e-12 * jnp.maximum(jnp.sum(n_feed), 1.0))
    if not isinstance(inventory, jax.core.Tracer) and not bool(inventory):
        raise ValueError("the specified extent consumes more of a reactant than the feed holds")
    p = jnp.asarray(feed.p - jnp.asarray(dp), dtype=float)
    h_in = enthalpy_flow(feed, model=pkg) + n_feed @ hf
    report = residual_report(jnp.zeros(1))
    if duty is None:
        t = jnp.asarray(feed.t if t_out is None else t_out, dtype=float)
        outlet = Stream(n_out, t, p, comps)
        heat = enthalpy_flow(outlet, model=pkg) + n_out @ hf - h_in
    else:
        heat = jnp.asarray(duty, dtype=float)
        flow = jnp.sum(n_out)
        target = (h_in + heat - n_out @ hf) / flow
        solved = pkg.flash_ph_with_info(p, target, n_out / flow, t_init=feed.t)
        state = solved.value
        report = solved.report
        outlet = Stream(n_out, state.t, p, comps, state.beta * flow * state.y)
    report = with_status(report, ~(valid & inventory), SolveStatus.INVALID_INPUT)
    report = with_status(report, report.converged & ~outlet.report.converged, SolveStatus.NONFINITE)
    require_converged(report, "stoichiometric reactor")
    outlet, heat, extent_arr, generation = nan_unless_converged(
        (outlet, heat, extent_arr, generation), report
    )
    return StoichiometricResult(outlet, heat, extent_arr, generation, report)


class BatchResult(NamedTuple):
    """Contents of a closed, constant-volume batch reactor after a reaction time.

    Attributes:
        contents: Final contents as a `Stream` of moles (mol), at the final
            temperature and ideal-gas pressure.
        heat: Cumulative heat added (J); ``Delta U`` for an isothermal run, zero
            when adiabatic.
        extent: Extent of each reaction (mol).
    """

    contents: Stream
    heat: Array
    extent: Array


def batch_reactor(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction],
    rate_laws: Any,
    volume: ArrayLike,
    time: ArrayLike,
    *,
    t_out: ArrayLike | None = None,
    adiabatic: bool = False,
    steps: int = 200,
) -> BatchResult:
    """Closed, constant-volume, ideal-gas batch reactor integrated over time.

    Here ``feed.n`` are the *initial moles* (mol). Marches ``dN_i/dt = V (r . Nu)_i``
    with concentrations ``c_i = N_i / V`` by explicit RK4. The vessel is closed
    and rigid, so the first law conserves internal energy, not enthalpy: an
    adiabatic run integrates ``(sum_i N_i Cv_i) dT/dt = -V sum_j r_j Delta U_j``
    with ``Cv = Cp - R`` and ``Delta U_j = Delta H_j - R T sum_i nu_ji``, and an
    isothermal run returns the heat ``Q = Delta U``. The final pressure is the
    ideal-gas ``N R T / V``.

    Raises:
        ValueError: If the number of rate laws doesn't match the reactions.
    """
    comps = feed.components
    nu = _stack_nu(_as_reactions(reactions), comps)
    laws = list(rate_laws) if isinstance(rate_laws, list | tuple) else [rate_laws]
    if len(laws) != nu.shape[0]:
        raise ValueError(f"expected {nu.shape[0]} rate law(s), got {len(laws)}")
    hf, _gf, coeffs = reaction_arrays(list(comps))
    a, b, c, d, e = coeffs
    n0 = feed.n
    vol = jnp.asarray(volume, dtype=float)
    t0 = jnp.asarray(feed.t if t_out is None else t_out, dtype=float)
    dt_step = jnp.asarray(time, dtype=float) / steps
    dnu = jnp.sum(nu, axis=1)

    def internal_energy(n: Array, t: Array) -> Array:
        return jnp.sum(n * (hf + enthalpy_ig(t, a, b, c, d, e))) - R * t * jnp.sum(n)

    def deriv(n: Array, t: Array) -> tuple[Array, Array]:
        conc = jnp.clip(n, 0.0, None) / vol
        rates = jnp.stack([law.rate(t, conc) for law in laws])
        dn = vol * (rates @ nu)
        if not adiabatic:
            return dn, jnp.asarray(0.0)
        du = (
            jnp.stack([delta_h_rxn(nu[j], t, hf, a, b, c, d, e) for j in range(nu.shape[0])])
            - R * t * dnu
        )
        cv = jnp.sum(n * (cp_ig(t, a, b, c, d, e) - R))
        return dn, -vol * jnp.sum(rates * du) / cv

    def rk4(state: tuple[Array, Array], _: None) -> tuple[tuple[Array, Array], None]:
        x, t = state
        dx1, dt1 = deriv(x, t)
        dx2, dt2 = deriv(x + 0.5 * dt_step * dx1, t + 0.5 * dt_step * dt1)
        dx3, dt3 = deriv(x + 0.5 * dt_step * dx2, t + 0.5 * dt_step * dt2)
        dx4, dt4 = deriv(x + dt_step * dx3, t + dt_step * dt3)
        x = x + (dt_step / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        t = t + (dt_step / 6.0) * (dt1 + 2.0 * dt2 + 2.0 * dt3 + dt4)
        return (jnp.clip(x, 0.0, None), t), None

    (n_final, t_final), _ = jax.lax.scan(rk4, (n0, t0), None, length=steps)
    extent, *_ = jnp.linalg.lstsq(nu.T, n_final - n0, rcond=None)
    pressure = jnp.sum(n_final) * R * t_final / vol
    heat = (
        jnp.asarray(0.0)
        if adiabatic
        else internal_energy(n_final, t_final) - internal_energy(n0, jnp.asarray(feed.t))
    )
    return BatchResult(Stream(n_final, t_final, pressure, comps), heat, extent)


def conversion(feed: Stream, outlet: Stream, component_index: int) -> Array:
    """Fractional conversion of a feed component, ``(n_in - n_out) / n_in``."""
    n0 = feed.n[component_index]
    return (n0 - outlet.n[component_index]) / n0


__all__ = [
    "BatchResult",
    "StoichiometricResult",
    "batch_reactor",
    "conversion",
    "cstr",
    "equilibrium_reactor",
    "pfr",
    "stoichiometric_reactor",
]
