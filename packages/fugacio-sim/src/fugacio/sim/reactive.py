"""Reactive separations: simultaneous chemical reaction and phase equilibrium.

Real separation equipment often runs *with* a reaction happening inside it: the
whole point of reactive distillation is to push a reaction past its equilibrium
limit by continuously pulling products into a different phase. This module adds
common-package units alongside the legacy gamma-phi approximation:

* `reactive_flash`: an isothermal flash in which the liquid simultaneously
  reaches **chemical** equilibrium (one or more reactions) and **phase**
  equilibrium (vapour-liquid). The extents of reaction and the V/L split are
  solved together, reusing the validated gamma-phi flash and the ideal-gas
  reaction thermochemistry. Works for any net mole change.

* `reactive_column`: energy-balanced MESH with a common property package and
  volumetric rates on a declared liquid or vapor phase.

* `reactive_distillation`: a legacy multistage approximation (Wang-Henke
  bubble-point, constant molar overflow) with a **rate-based** reaction source on
  each reactive stage: ``S_{j,i} = H_j * sum_r nu_{r,i} * rate_r(T_j, a_j)`` with
  the liquid-phase activities ``a_i = x_i gamma_i`` and a per-stage molar holdup
  ``H_j``. For an equimolar reaction (``sum_i nu_i = 0``, the dominant reactive
  distillation class: esterification, transesterification, metathesis,
  isomerisation) the source conserves total moles. Constant molar overflow remains
  a material-balance approximation; it does not establish energy closure.

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
from fugacio.thermo.acceptance import accepted_value
from fugacio.thermo.constants import P_REF, R
from fugacio.thermo.diagnostics import SolveReport, SolveStatus, require_converged
from fugacio.thermo.gammaphi import gamma_phi_k_values
from fugacio.thermo.implicit import (
    bracketed_root_with_info,
    newton_system_with_info,
)
from fugacio.thermo.package import flash_pt_with_info
from fugacio.thermo.phase import GammaPhiModel
from fugacio.thermo.reaction_system import ReactionSet
from fugacio.thermo.reactions import Reaction, delta_g_rxn
from fugacio.thermo.reference import liquid_reference_fugacity

if TYPE_CHECKING:
    from fugacio.sim.distillation import ColumnFeed, RigorousColumnResult

ArrayLike = Array | float

_TINY = 1e-300


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


def _ln_k(nu: Array, t: ArrayLike, hf: Array, gf: Array, coeffs: Any) -> Array:
    """Row vector of ``ln K_r(T)`` for each reaction (ideal-gas reference)."""
    a, b, c, d, e = coeffs
    t = jnp.asarray(t)
    return jnp.stack(
        [-delta_g_rxn(nu[j], t, hf, gf, a, b, c, d, e) / (R * t) for j in range(nu.shape[0])]
    )


def _ln_activity_liquid(model: GammaPhiModel, t: ArrayLike, p: ArrayLike, x: Array) -> Array:
    """Log liquid-phase activities ``ln a_i = ln(x_i gamma_i f_i^{0,L}/P_ref)``.

    This is the ideal-gas-referenced activity used by the reaction equilibrium:
    ``a_i = f_i^L / P_ref`` with the gamma-phi liquid fugacity
    ``f_i^L = x_i gamma_i f_i^{0,L}``, so it pairs consistently with ``K(T)`` from
    the ideal-gas formation data.
    """
    f_ref, _ = liquid_reference_fugacity(
        model.eos,
        t,
        p,
        model.tc,
        model.pc,
        model.omega,
        poynting=model.poynting,
        phi_saturation=model.phi_saturation,
    )
    ln_gamma = model.activity.ln_gamma(x, t)
    return jnp.log(jnp.clip(x, _TINY, None)) + ln_gamma + jnp.log(f_ref) - jnp.log(P_REF)


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
    detached = jax.lax.stop_gradient((pkg, tt, pp, n / jnp.sum(n)))
    phase_report = flash_pt_with_info(*detached).report
    report = report._replace(
        status=jnp.where(report.converged, phase_report.status, report.status),
        residual_norm=jnp.maximum(report.residual_norm, phase_report.residual_norm),
    )
    if check:
        require_converged(report, "reactive flash")
    extent = accepted_value(extent, report.converged)
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
        lambda value: accepted_value(value, report.converged),
        (vapor, liquid, phases.beta, extent, duty),
    )
    return ReactiveFlashResult(vapor, liquid, beta, extent, duty, extent @ nu, report, phase_report)


class ReactiveColumnResult(NamedTuple):
    """Converged profile and products of a reactive distillation column.

    Attributes:
        t: Stage temperatures (K), top stage first, shape ``(n_stages,)``.
        x: Liquid mole fractions, shape ``(n_stages, n_components)``.
        y: Vapour mole fractions, shape ``(n_stages, n_components)``.
        distillate: Distillate product `Stream`.
        bottoms: Bottoms product `Stream`.
        reflux: Reflux ratio used.
        generation: Net mole generation by reaction on each stage (mol/s),
            shape ``(n_stages, n_components)``.
    """

    t: Array
    x: Array
    y: Array
    distillate: Stream
    bottoms: Stream
    reflux: Array
    generation: Array


def reactive_distillation(
    feed: Stream,
    model: GammaPhiModel,
    reactions: Reaction | Sequence[Reaction],
    rate_laws: Any,
    holdup: ArrayLike,
    n_stages: int,
    feed_stage: int,
    reflux: ArrayLike,
    distillate_rate: ArrayLike,
    *,
    reactive_stages: tuple[int, int] | None = None,
    q: ArrayLike = 1.0,
    t_top: ArrayLike | None = None,
    t_bottom: ArrayLike | None = None,
    t_min: float = 200.0,
    t_max: float = 700.0,
    tol: float = 1e-11,
    max_iter: int = 600,
) -> ReactiveColumnResult:
    """Rate-based reactive distillation by the gamma-phi Wang-Henke method (CMO).

    A total condenser sits above stage 1 and a partial reboiler is stage
    ``n_stages``; one feed of quality ``q`` enters at ``feed_stage`` (1-indexed).
    Each stage equilibrates by the gamma-phi bubble-point method, and on every
    *reactive* stage a rate-based source ``H * sum_r nu_r rate_r(T, a)`` (liquid
    activities ``a_i = x_i gamma_i``, molar holdup ``H``) is added to the component
    balance. The whole profile is converged by the Wegstein tear solver, so the
    products and profiles are differentiable with respect to ``reflux``,
    ``distillate_rate``, ``holdup``, the feed, and the model/kinetic parameters.

    This legacy screening model has no stage energy equations. Equimolar
    stoichiometry does not establish energy closure. Use reactive_column for
    energy-balanced design and non-equimolar reactions.

    Args:
        feed: Feed stream.
        model: Gamma-phi property model for the (non-ideal) liquid.
        reactions: One reaction or several over ``feed.components``.
        rate_laws: One rate law per reaction (``rate(T, a)``; activities passed as
            the concentration argument for a pseudo-homogeneous, activity-based rate).
        holdup: Liquid molar holdup ``H`` on each reactive stage (mol).
        n_stages: Number of equilibrium stages including the reboiler.
        feed_stage: 1-indexed feed stage.
        reflux: Reflux ratio ``L/D``.
        distillate_rate: Distillate molar flow (mol/s).
        reactive_stages: Inclusive 1-indexed ``(first, last)`` reactive stage range;
            defaults to all interior stages ``(2, n_stages - 1)``.
        q: Feed thermal quality (1 = saturated liquid).
        t_top: Optional initial top-stage temperature (K).
        t_bottom: Optional initial bottom-stage temperature (K).
        t_min: Lower per-stage temperature clamp (K).
        t_max: Upper per-stage temperature clamp (K).
        tol: Convergence tolerance for the outer fixed point.
        max_iter: Maximum number of outer sweeps.

    Returns:
        A `ReactiveColumnResult`.
    """
    from fugacio.sim.flowsheet import tear_solve

    comps = feed.components
    rxns = _as_reactions(reactions)
    nu = _stack_nu(rxns, comps)
    laws = list(rate_laws) if isinstance(rate_laws, (list, tuple)) else [rate_laws]
    if len(laws) != nu.shape[0]:
        raise ValueError(f"expected {nu.shape[0]} rate law(s), got {len(laws)}")

    n = n_stages
    n_c = len(comps)
    f_idx = feed_stage - 1
    p = jnp.asarray(feed.p)
    z = feed.z
    big_f = feed.total
    q_arr = jnp.asarray(q, dtype=float)
    idx = jnp.arange(n)
    feed_comp = jnp.zeros((n, n_c)).at[f_idx].set(big_f * z)

    lo, hi = (2, n - 1) if reactive_stages is None else reactive_stages
    react_mask = (idx + 1 >= lo) & (idx + 1 <= hi)
    h_stage = jnp.where(react_mask, jnp.asarray(holdup, dtype=float), 0.0)

    def stage_k(t_j: Array, x_j: Array, y_j: Array) -> Array:
        return gamma_phi_k_values(
            model.activity,
            t_j,
            p,
            x_j,
            y_j,
            model.tc,
            model.pc,
            model.omega,
            eos=model.eos,
            kij=model.kij,
            vapor=model.vapor,
            poynting=model.poynting,
            phi_saturation=model.phi_saturation,
        )

    def stage_source(t_j: Array, x_j: Array, h_j: Array) -> Array:
        a_j = x_j * jnp.exp(model.activity.ln_gamma(x_j, t_j))
        rates = jnp.stack([law.rate(t_j, a_j) for law in laws])
        return h_j * (rates @ nu)

    def cmo_flows(r: Array, d: Array) -> tuple[Array, Array]:
        b = big_f - d
        v_rect = (r + 1.0) * d
        v_strip = (r + 1.0) * d - (1.0 - q_arr) * big_f
        l_rect = r * d
        l_strip = r * d + q_arr * big_f
        v = jnp.where(idx + 1 <= feed_stage, v_rect, v_strip)
        liq = jnp.where(idx + 1 < feed_stage, l_rect, jnp.where(idx + 1 < n, l_strip, b))
        return v, liq

    def tridiag_component(k_col: Array, f_col: Array, v: Array, liq: Array, r: Array) -> Array:
        diag = -(1.0 + v * k_col / liq)
        diag = diag.at[0].set(-1.0 - k_col[0] / r)
        sub = jnp.ones(n - 1)
        sup = v[1:] * k_col[1:] / liq[1:]
        mat = jnp.diag(diag) + jnp.diag(sub, -1) + jnp.diag(sup, 1)
        return jnp.linalg.solve(mat, -f_col)

    def sweep(state: tuple[Array, Array, Array], theta: dict[str, Array]) -> tuple[Array, ...]:
        t, x, y = state
        r, d = theta["R"], theta["D"]
        v, liq_flows = cmo_flows(r, d)
        k = jax.vmap(stage_k)(t, x, y)
        source = jax.vmap(stage_source)(t, x, h_stage)
        rhs = feed_comp + source
        liq = jax.vmap(tridiag_component, in_axes=(1, 1, None, None, None), out_axes=1)(
            k, rhs, v, liq_flows, r
        )
        liq = jnp.maximum(liq, 1e-12)
        x_new = liq / jnp.sum(liq, axis=1, keepdims=True)

        def bubble_residual(t_j: Array, x_j: Array, y_j: Array) -> Array:
            return jnp.sum(stage_k(t_j, x_j, y_j) * x_j) - 1.0

        r_bp = jax.vmap(bubble_residual)(t, x_new, y)
        dr_bp = jax.vmap(jax.grad(bubble_residual))(t, x_new, y)
        step = jnp.clip(r_bp / dr_bp, -25.0, 25.0)
        t_new = jnp.clip(t - step, t_min, t_max)
        k_new = jax.vmap(stage_k)(t_new, x_new, y)
        y_unnorm = k_new * x_new
        y_new = y_unnorm / jnp.sum(y_unnorm, axis=1, keepdims=True)
        return t_new, x_new, y_new

    t_hi = feed.t + 25.0 if t_bottom is None else jnp.asarray(t_bottom)
    t_lo = feed.t - 5.0 if t_top is None else jnp.asarray(t_top)
    t0 = jnp.linspace(t_lo, t_hi, n)
    x0 = jnp.broadcast_to(z, (n, n_c))
    theta = {"R": jnp.asarray(reflux, dtype=float), "D": jnp.asarray(distillate_rate, dtype=float)}
    t_star, x_star, y_star = tear_solve(
        sweep, (t0, x0, x0), theta, q_min=-5.0, q_max=0.0, tol=tol, max_iter=max_iter
    )

    big_d = jnp.asarray(distillate_rate, dtype=float)
    big_b = big_f - big_d
    distillate = Stream(big_d * y_star[0], t_star[0], p, comps)
    bottoms = Stream(big_b * x_star[-1], t_star[-1], p, comps)
    generation = jax.vmap(stage_source)(t_star, x_star, h_stage)
    return ReactiveColumnResult(
        t=t_star,
        x=x_star,
        y=y_star,
        distillate=distillate,
        bottoms=bottoms,
        reflux=theta["R"],
        generation=generation,
    )


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
    rigorous entry point replaces CMO assumptions for process design. The older
    reactive_distillation function retains its historical molar-holdup screening
    convention and should only be used for comparisons with those old examples.
    """
    from fugacio.sim.distillation import rigorous_column

    return rigorous_column(
        feeds, n_stages, reactions=reactions, reaction_volumes=reaction_volumes, **kwargs
    )


__all__ = [
    "ReactiveColumnResult",
    "ReactiveFlashResult",
    "reactive_column",
    "reactive_distillation",
    "reactive_flash",
]
