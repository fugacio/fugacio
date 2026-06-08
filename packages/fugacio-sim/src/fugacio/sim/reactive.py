"""Reactive separations: simultaneous chemical reaction and phase equilibrium.

Real separation equipment often runs *with* a reaction happening inside it -- the
whole point of reactive distillation is to push a reaction past its equilibrium
limit by continuously pulling products into a different phase. This module adds
two such units on top of the gamma-phi property system:

* :func:`reactive_flash` -- an isothermal flash in which the liquid simultaneously
  reaches **chemical** equilibrium (one or more reactions) and **phase**
  equilibrium (vapour-liquid). The extents of reaction and the V/L split are
  solved together, reusing the validated gamma-phi flash and the ideal-gas
  reaction thermochemistry. Works for any net mole change.

* :func:`reactive_distillation` -- a rigorous multistage column (Wang-Henke
  bubble-point, constant molar overflow) with a **rate-based** reaction source on
  each reactive stage: ``S_{j,i} = H_j * sum_r nu_{r,i} * rate_r(T_j, a_j)`` with
  the liquid-phase activities ``a_i = x_i gamma_i`` and a per-stage molar holdup
  ``H_j``. For an equimolar reaction (``sum_i nu_i = 0`` -- the dominant reactive
  distillation class: esterification, transesterification, metathesis,
  isomerisation) the source conserves total moles, so constant molar overflow is
  exact and the model is rigorous.

The reaction equilibrium constant ``K(T)`` comes from the ideal-gas formation data
in :mod:`fugacio.thermo.reactions`; at vapour-liquid equilibrium the component
fugacities are equal across phases, so the ideal-gas-referenced equilibrium is
written consistently in terms of the liquid activities
``a_i = x_i gamma_i f_i^{0,L}/P_ref``. Every result is a differentiable
:class:`~fugacio.sim.stream.Stream` (or profile of them): conversions, product
purities, and stage profiles carry gradients with respect to the feed, operating
conditions, the activity-model parameters, *and* the kinetic/thermochemical
parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.stream import Stream
from fugacio.thermo.constants import P_REF, R
from fugacio.thermo.gammaphi import gamma_phi_k_values
from fugacio.thermo.implicit import bracketed_root, newton_system
from fugacio.thermo.phase import GammaPhiModel
from fugacio.thermo.reactions import Reaction, delta_g_rxn, reaction_arrays
from fugacio.thermo.reference import liquid_reference_fugacity

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
        vapor: Vapour product :class:`~fugacio.sim.stream.Stream`.
        liquid: Liquid product :class:`~fugacio.sim.stream.Stream`.
        beta: Vapour fraction (mol vapour / mol after reaction).
        extent: Equilibrium extent of each reaction (mol/s), shape ``(n_reactions,)``.
    """

    vapor: Stream
    liquid: Stream
    beta: Array
    extent: Array


def reactive_flash(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction],
    t: ArrayLike,
    p: ArrayLike,
    model: GammaPhiModel,
    *,
    tol: float = 1e-11,
    max_iter: int = 80,
) -> ReactiveFlashResult:
    """Isothermal flash with simultaneous chemical and phase equilibrium.

    Solves for the reaction extents that satisfy chemical equilibrium *while* the
    mixture is split by a gamma-phi vapour-liquid flash at ``(T, P)``. The reaction
    equilibrium is imposed on the liquid activities (equivalently the equal vapour
    fugacities), so it is consistent across the whole vapour-fraction range -- it
    even pins the bubble/dew composition when the flash is single-phase.

    Args:
        feed: Inlet stream; reactions must be defined over ``feed.components``.
        reactions: One reaction or several over the feed's component ordering.
        t, p: Temperature (K) and pressure (Pa).
        model: A :class:`~fugacio.thermo.GammaPhiModel` (activity liquid + EOS/ideal
            vapour) -- the right tool for the non-ideal mixtures reactive flashes
            target.
        tol, max_iter: Solver controls.

    Returns:
        A :class:`ReactiveFlashResult`. Differentiable in the feed, ``(T, P)``, the
        activity-model parameters, and the reaction thermochemistry.
    """
    comps = feed.components
    rxns = _as_reactions(reactions)
    nu = _stack_nu(rxns, comps)
    n_rxn = nu.shape[0]
    hf, gf, coeffs = reaction_arrays(list(comps))
    n_feed = feed.n
    t_arr = jnp.asarray(t, dtype=float)
    p_arr = jnp.asarray(p, dtype=float)

    reactant = nu < 0.0
    product = nu > 0.0
    cap_r = jnp.where(reactant, n_feed[None, :] / jnp.where(reactant, -nu, 1.0), jnp.inf)
    cap_p = jnp.where(product, n_feed[None, :] / jnp.where(product, nu, 1.0), jnp.inf)
    xi_hi = jnp.min(cap_r, axis=1)
    xi_lo = -jnp.min(cap_p, axis=1)

    def reaction_residual(xi: Array, theta: tuple[Array, Array, Array]) -> Array:
        nf, tt, pp = theta
        n = nf + xi @ nu
        z = n / jnp.sum(n)
        res = model.flash_pt(tt, pp, z)
        ln_a = _ln_activity_liquid(model, tt, pp, res.x)
        return nu @ ln_a - _ln_k(nu, tt, hf, gf, coeffs)

    theta = (n_feed, t_arr, p_arr)
    if n_rxn == 1:
        span = xi_hi[0] - xi_lo[0]

        def scalar(xi_s: Array, th: tuple[Array, Array, Array]) -> Array:
            return reaction_residual(jnp.reshape(xi_s, (1,)), th)[0]

        xi_star = bracketed_root(
            scalar, theta, xi_lo[0] + 1e-6 * span, xi_hi[0] - 1e-6 * span, tol, max_iter
        )
        extent = jnp.reshape(xi_star, (1,))
    else:
        extent = newton_system(reaction_residual, jnp.zeros(n_rxn), theta, tol, max_iter)

    n = n_feed + extent @ nu
    total = jnp.sum(n)
    z = n / total
    res = model.flash_pt(t_arr, p_arr, z)
    vapor = Stream(res.y * res.beta * total, t_arr, p_arr, comps)
    liquid = Stream(res.x * (1.0 - res.beta) * total, t_arr, p_arr, comps)
    return ReactiveFlashResult(vapor=vapor, liquid=liquid, beta=res.beta, extent=extent)


class ReactiveColumnResult(NamedTuple):
    """Converged profile and products of a reactive distillation column.

    Attributes:
        t: Stage temperatures (K), top stage first, shape ``(n_stages,)``.
        x: Liquid mole fractions, shape ``(n_stages, n_components)``.
        y: Vapour mole fractions, shape ``(n_stages, n_components)``.
        distillate: Distillate product :class:`~fugacio.sim.stream.Stream`.
        bottoms: Bottoms product :class:`~fugacio.sim.stream.Stream`.
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

    For an equimolar reaction the source conserves total moles and constant molar
    overflow is exact. (Non-equimolar reactions also run, but the constant-overflow
    traffic then neglects the reaction's volume change.)

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
        t_top, t_bottom: Optional initial top/bottom temperatures.
        t_min, t_max: Per-stage temperature clamp.
        tol, max_iter: Outer fixed-point controls.

    Returns:
        A :class:`ReactiveColumnResult`.
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


__all__ = [
    "ReactiveColumnResult",
    "ReactiveFlashResult",
    "reactive_distillation",
    "reactive_flash",
]
