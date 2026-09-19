"""Reactive separations: simultaneous chemical reaction and phase equilibrium.

Real separation equipment often runs *with* a reaction happening inside it: the
whole point of reactive distillation is to push a reaction past its equilibrium
limit by continuously pulling products into a different phase. This module
provides the common-package units:

* `reactive_flash`: an isothermal flash in which the liquid simultaneously
  reaches **chemical** equilibrium (one or more reactions) and **phase**
  equilibrium (vapour-liquid). The extents of reaction and the V/L split are
  solved together, reusing the validated gamma-phi flash and the ideal-gas
  reaction thermochemistry. Works for any net mole change.

* `reactive_column`: energy-balanced MESH with a common property package and
  volumetric rates on a declared liquid or vapor phase.

The reaction equilibrium constant ``K(T)`` comes from the ideal-gas formation data
in `fugacio.thermo.reactions`; at vapour-liquid equilibrium the component
fugacities are equal across phases, so the ideal-gas-referenced equilibrium is
written consistently in terms of the liquid activities
``a_i = x_i gamma_i f_i^{0,L}/P_ref``. Every result is a differentiable
`Stream` (or profile of them): conversions, product
purities, and stage profiles carry gradients with respect to the feed, operating
conditions, the activity-model parameters, *and* the kinetic/thermochemical
parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.properties import Model, molar_enthalpy, resolve_package
from fugacio.sim.reaction_units import reaction_parameter_validity
from fugacio.sim.stream import Stream
from fugacio.thermo.constants import P_REF
from fugacio.thermo.diagnostics import (
    SolveReport,
    SolveStatus,
    nan_unless_converged,
    require_converged,
)
from fugacio.thermo.implicit import (
    bracketed_root_with_info,
    gate_derivative,
    newton_system_with_info,
)
from fugacio.thermo.reaction_system import ReactionSet
from fugacio.thermo.reactions import Reaction

if TYPE_CHECKING:
    from fugacio.sim.distillation import ColumnFeed, RigorousColumnResult

ArrayLike = Array | float


def _as_reactions(reactions: Reaction | Sequence[Reaction]) -> list[Reaction]:
    return [reactions] if isinstance(reactions, Reaction) else list(reactions)


def _stack_nu(reactions: Sequence[Reaction], components: tuple[str, ...]) -> Array:
    rows = []
    for r in reactions:
        if tuple(r.components) != tuple(components):
            raise ValueError(
                "each reaction must be defined over the feed's components in the same order"
            )
        rows.append(jnp.asarray(r.nu))
    return jnp.stack(rows)


class ReactiveFlashResult(NamedTuple):
    """Outcome of a simultaneous reaction + vapour-liquid flash.

    Attributes:
        vapor: Vapour product `Stream`.
        liquid: Liquid product `Stream`.
        beta: Vapour fraction (mol vapour / mol after reaction).
        extent: Equilibrium extent of each reaction (mol/s), shape ``(n_reactions,)``.
        duty: External heat input, including formation energy exactly once (W).
        generation: Component generation (mol/s).
        report: Chemical solve and final phase-solve acceptance.
        phase_report: Actual final PT flash iteration report.
    """

    vapor: Stream
    liquid: Stream
    beta: Array
    extent: Array
    duty: Array
    generation: Array
    report: SolveReport
    phase_report: SolveReport

    @property
    def outlets(self) -> tuple[Stream, Stream]:
        """``(vapor, liquid)``, as a flowsheet output tuple."""
        return self.vapor, self.liquid

    @property
    def heat(self) -> Array:
        """Heat into the fluid (W), including formation energy once."""
        return self.duty


def reactive_flash(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction] | ReactionSet,
    t: ArrayLike,
    p: ArrayLike,
    model: Model,
    *,
    tol: float = 1e-9,
    max_iter: int = 100,
    check: bool = True,
) -> ReactiveFlashResult:
    """Solve isothermal chemical equilibrium coupled to a package PT flash.

    All property and thermochemical parameters pass explicitly through the
    implicit solve, including under JIT, JVP, and VJP. The reaction quotient
    uses the liquid fugacity when liquid is present, otherwise the vapor
    fugacity. Absent phases never determine chemical equilibrium. Formation
    enthalpy is included once in the reported heat input (W).

    Args:
        feed: Inlet material and thermal state.
        reactions: One reaction, a sequence, or a validated ReactionSet.
        t: Drum temperature (K).
        p: Drum pressure (Pa).
        model: Common property package or compatible equilibrium model.
        tol: Chemical-equilibrium residual tolerance.
        max_iter: Root iteration cap.
        check: Raise for failed concrete solves; compiled callers inspect report.

    Returns:
        Products preserving phase inventories, reaction extents, heat input,
        generation, and the actual chemical solve report. Process-case audits
        independently assess product equilibrium, stability, and balances.
    """
    if not isinstance(reactions, ReactionSet):
        _stack_nu(_as_reactions(reactions), feed.components)
    system = (
        reactions if isinstance(reactions, ReactionSet) else ReactionSet.from_reactions(reactions)
    )
    pkg = resolve_package(feed.components, model)
    system.check_package(pkg)
    if system.components != feed.components:
        raise ValueError("reaction and feed component order differs")
    nu = system.nu
    nr = nu.shape[0]
    tt, pp = jnp.asarray(t, dtype=float), jnp.asarray(p, dtype=float)
    theta = {"feed": feed.n, "t": tt, "p": pp, "package": pkg, "system": system}

    def residual(xi: Array, th: Any) -> Array:
        rx, package = th["system"], th["package"]
        n = th["feed"] + xi @ rx.nu
        z = n / jnp.sum(n)
        phases = package.flash_pt(th["t"], th["p"], z)

        def log_activity(phase: str, composition: Array) -> Array:
            return (
                jnp.log(jnp.maximum(composition, 1e-300))
                + package.ln_phi(th["t"], th["p"], composition, phase=phase)
                + jnp.log(th["p"] / P_REF)
            )

        ln_a = jax.lax.cond(
            phases.beta < 1.0,
            lambda _: log_activity("liquid", phases.x),
            lambda _: log_activity("vapor", phases.y),
            None,
        )
        return rx.nu @ ln_a - rx.ln_equilibrium_constants(th["t"])

    if nr == 1:
        row = nu[0]
        hi = jnp.min(jnp.where(row < 0, feed.n / jnp.where(row < 0, -row, 1), jnp.inf))
        lo = -jnp.min(jnp.where(row > 0, feed.n / jnp.where(row > 0, row, 1), jnp.inf))
        epsilon = 1e-10 * (hi - lo)
        solved = bracketed_root_with_info(
            lambda x, th: residual(jnp.atleast_1d(x), th)[0],
            theta,
            lo + epsilon,
            hi - epsilon,
            tol * 1e-3,
            max_iter,
            residual_tol=tol,
        )
        extent = jnp.atleast_1d(solved.value)
    else:
        # Log component flows keep trial compositions positive; extents enforce
        # stoichiometry and the final equation residual verifies both systems.
        c = len(feed.components)
        scale = jnp.maximum(feed.total, 1.0)
        theta = {**theta, "scale": scale}

        def joint(u: Array, th: Any) -> Array:
            n = jnp.exp(u[:c]) * th["scale"]
            xi = u[c:] * th["scale"]
            # Evaluate chemical equations on the positive global mole coordinates.
            chemical = residual(jnp.zeros(nr), {**th, "feed": n})
            balance = (n - th["feed"] - xi @ th["system"].nu) / th["scale"]
            return jnp.concatenate([balance, chemical])

        seed = jnp.concatenate([jnp.log(jnp.maximum(feed.n / scale, 1e-3)), jnp.zeros(nr)])
        solved = newton_system_with_info(
            joint,
            seed,
            theta,
            tol,
            max_iter,
            lower=jnp.concatenate([jnp.full(c, -70.0), jnp.full(nr, -jnp.inf)]),
            upper=jnp.concatenate([jnp.full(c, 20.0), jnp.full(nr, jnp.inf)]),
        )
        extent = solved.value[c:] * scale
    n = feed.n + extent @ nu
    valid = (
        reaction_parameter_validity(system)
        & jnp.all(n >= 0)
        & (jnp.sum(n) > 0)
        & (tt > 0)
        & (pp > 0)
        & feed.report.converged
    )
    report = solved.report._replace(
        status=jnp.where(valid, solved.report.status, SolveStatus.INVALID_INPUT)
    )
    package_d, t_d, p_d, z_d = jax.lax.stop_gradient((pkg, tt, pp, n / jnp.sum(n)))
    phase_report = package_d.flash_pt_with_info(t_d, p_d, z_d).report
    report = report._replace(
        status=jnp.where(report.converged, phase_report.status, report.status),
        residual_norm=jnp.maximum(report.residual_norm, phase_report.residual_norm),
    )
    if check:
        require_converged(report, "reactive flash")
    extent = gate_derivative(extent, report.converged)
    n = feed.n + extent @ nu
    phases = pkg.flash_pt(tt, pp, n / jnp.sum(n))
    nv, nl = phases.y * phases.beta * jnp.sum(n), phases.x * (1 - phases.beta) * jnp.sum(n)
    vapor = Stream(nv, tt, pp, feed.components, nv)
    liquid = Stream(nl, tt, pp, feed.components, jnp.zeros_like(nl))
    # Single-phase primitives avoid asking an empty product for a composition.
    hout = jax.lax.cond(
        phases.beta > 0,
        lambda _: jnp.sum(nv) * pkg.enthalpy(tt, pp, phases.y, phase="vapor"),
        lambda _: jnp.asarray(0.0),
        None,
    )
    hout += jax.lax.cond(
        phases.beta < 1,
        lambda _: jnp.sum(nl) * pkg.enthalpy(tt, pp, phases.x, phase="liquid"),
        lambda _: jnp.asarray(0.0),
        None,
    )

    hin = feed.total * molar_enthalpy(feed, model=pkg)
    duty = hout - hin + (extent @ nu) @ system.formation_enthalpy
    finite = jnp.isfinite(duty) & jnp.all(jnp.isfinite(nv)) & jnp.all(jnp.isfinite(nl))
    report = report._replace(status=jnp.where(finite, report.status, SolveStatus.NONFINITE))
    if check:
        require_converged(report, "reactive flash")
    vapor, liquid, beta, extent, duty = jax.tree_util.tree_map(
        lambda value: gate_derivative(value, report.converged),
        (vapor, liquid, phases.beta, extent, duty),
    )
    if check:
        # A traced failure can't raise: its products are NaN, as for every unit.
        vapor, liquid, beta, extent, duty = nan_unless_converged(
            (vapor, liquid, beta, extent, duty), report
        )
    return ReactiveFlashResult(vapor, liquid, beta, extent, duty, extent @ nu, report, phase_report)


def reactive_column(
    feeds: Sequence[ColumnFeed],
    n_stages: int,
    reactions: ReactionSet,
    reaction_volumes: ArrayLike,
    **kwargs: Any,
) -> RigorousColumnResult:
    """Solve reactive MESH distillation with formation-consistent stage energy.

    Arguments follow rigorous_column. Volumes are reacting-phase m^3; a scalar
    selects interior stages and a vector explicitly selects each stage. This
    rigorous entry point closes stage energy balances for any reaction
    stoichiometry.
    """
    from fugacio.sim.distillation import rigorous_column

    return rigorous_column(
        feeds, n_stages, reactions=reactions, reaction_volumes=reaction_volumes, **kwargs
    )


__all__ = [
    "ReactiveFlashResult",
    "reactive_column",
    "reactive_flash",
]
