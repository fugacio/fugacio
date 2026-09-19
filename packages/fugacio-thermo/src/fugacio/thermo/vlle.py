"""Three-phase vapour-liquid-liquid equilibrium (VLLE).

When a vapour coexists with two partially miscible liquids (the heterogeneous
azeotropes that make water/organic distillation and decantation work), neither a
two-phase VLE flash nor an LLE flash alone suffices. VLLE couples them: a vapour
``V`` and two liquids ``I`` and ``II`` all at equal temperature, pressure, and
component fugacity.

Following Michelsen, both non-reference phases are referred to liquid ``I`` through
two sets of K-values,

    K_i^V = y_i / x_i^I = gamma_i^I f_i^{0,L} / (phi_i^V P),
    K_i^L = x_i^II / x_i^I = gamma_i^I / gamma_i^II,

so the material balance gives ``x_i^I = z_i / D_i`` with
``D_i = 1 + beta_V (K_i^V - 1) + beta_II (K_i^L - 1)``. The two phase fractions
``(beta_V, beta_II)`` solve the pair of Rachford-Rice equations
``sum_i z_i (K_i^V - 1)/D_i = 0`` and ``sum_i z_i (K_i^L - 1)/D_i = 0`` (a small
2x2 Newton with an analytic Jacobian), and the K-values are updated from the
fugacity equalities until consistent.

The flash is seeded from a VLE flash followed by a stability-driven split of its
liquid. When no three-phase solution exists (the Rachford-Rice root leaves the
phase-fraction simplex) the flash reports ``INFEASIBLE`` rather than clipping
phase fractions away from their compositions: use the two-phase VLE or LLE
flash there. A converged three-phase state is differentiable through the fixed
point.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.activity.models import ActivityModel
from fugacio.thermo.diagnostics import (
    SolveReport,
    SolveStatus,
    nan_unless_converged,
    residual_report,
    with_status,
)
from fugacio.thermo.eos import PR, CubicEOS
from fugacio.thermo.equilibrium import input_report
from fugacio.thermo.gammaphi import (
    flash_pt_gamma_with_info,
    gamma_phi_k_values,
    reference_domain,
)
from fugacio.thermo.implicit import fixed_point_with_info, gate_tree, scanned_root_with_info
from fugacio.thermo.lle import flash_lle_with_info
from fugacio.thermo.reference import saturation_pressures_with_info

ArrayLike = Array | float


class HeterogeneousAzeotrope(NamedTuple):
    """A binary heterogeneous (two-liquid) azeotrope at fixed pressure.

    Attributes:
        t: Azeotrope temperature (K).
        x_i: Composition of liquid ``I``.
        x_ii: Composition of liquid ``II``.
        y: Common vapour composition in equilibrium with *both* liquids.
    """

    t: Array
    x_i: Array
    x_ii: Array
    y: Array


class VLLEResult(NamedTuple):
    """Result of a three-phase vapour-liquid-liquid flash.

    Attributes:
        beta_v: Vapour mole fraction of the feed.
        beta_l1: Mole fraction in liquid ``I`` (the reference liquid).
        beta_l2: Mole fraction in liquid ``II``.
        y: Vapour composition.
        x_i: Liquid ``I`` composition.
        x_ii: Liquid ``II`` composition.
        three_phase: ``True`` when all three phase fractions are strictly positive.
    """

    beta_v: Array
    beta_l1: Array
    beta_l2: Array
    y: Array
    x_i: Array
    x_ii: Array
    three_phase: Array


class VLLESolveResult(NamedTuple):
    """Three-phase flash and its report."""

    value: VLLEResult
    report: SolveReport


def _two_phase_rr(z: Array, kv: Array, kl: Array, iters: int = 80) -> Array:
    """Solve the 2x2 Rachford-Rice system for ``(beta_V, beta_II)`` robustly.

    The pair ``(f1, f2)`` is the gradient of the convex Michelsen objective
    ``Q = -sum_i z_i ln D_i`` on the polytope ``{D_i > 0}``, so a Newton step in the
    *descent* direction, capped to keep every ``D_i`` strictly positive, converges
    globally without the overshoot that plagues an unguarded Newton. Starts from
    the feasible origin ``(0, 0)`` where ``D_i = 1``.
    """
    cv = kv - 1.0
    cl = kl - 1.0

    def body(_: int, beta: Array) -> Array:
        d = 1.0 + beta[0] * cv + beta[1] * cl
        d = jnp.maximum(d, 1e-300)
        f1 = jnp.sum(z * cv / d)
        f2 = jnp.sum(z * cl / d)
        h11 = jnp.sum(z * cv * cv / d**2) + 1e-12
        h12 = jnp.sum(z * cv * cl / d**2)
        h22 = jnp.sum(z * cl * cl / d**2) + 1e-12
        det = h11 * h22 - h12 * h12
        det = jnp.where(jnp.abs(det) < 1e-30, 1e-30, det)
        # Newton step on the convex objective: H delta = (f1, f2).
        dv = (h22 * f1 - h12 * f2) / det
        d2 = (-h12 * f1 + h11 * f2) / det
        # Cap the step so no D_i crosses zero (feasibility line search, closed form).
        c_dir = dv * cv + d2 * cl
        ratio = jnp.where(c_dir < 0.0, -d / c_dir, jnp.inf)
        lam = jnp.minimum(1.0, 0.99 * jnp.min(ratio))
        return beta + lam * jnp.array([dv, d2])

    return jax.lax.fori_loop(0, iters, body, jnp.zeros(2))


def flash_vlle_with_info(
    model: ActivityModel,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    tol: float = 1e-11,
    max_iter: int = 300,
) -> VLLESolveResult:
    """Isothermal-isobaric three-phase (V-L-L) flash of feed ``z`` at ``(T, P)``.

    Seeds a vapour/liquid split from the gamma-phi VLE flash and a
    liquid/liquid split of its liquid from `fugacio.thermo.lle.flash_lle_with_info`,
    then drives both K-value sets to consistency with a 2x2 Rachford-Rice inner
    solve. Phase fractions and compositions come from the same root, so the
    returned state always closes the material balance.

    Returns:
        The state and its report. The report is ``INFEASIBLE`` (and the value
        must not be used) when the converged root has a phase fraction outside
        ``[0, 1]``, i.e. the feed isn't three-phase at ``(T, P)``.
    """
    z = jnp.asarray(z)
    options: dict[str, Any] = {
        "eos": eos,
        "kij": kij,
        "vapor": vapor,
        "poynting": poynting,
        "phi_saturation": phi_saturation,
    }
    vle = flash_pt_gamma_with_info(model, t, p, z, tc, pc, omega, **options).value
    x_bulk = jnp.where(vle.beta < 1.0, vle.x, z)
    split = flash_lle_with_info(model, t, x_bulk).value
    kv0 = jax.lax.stop_gradient(vle.y / jnp.clip(split.x_i, 1e-12, None))
    kl0 = jax.lax.stop_gradient(split.k)
    theta = (model, jnp.asarray(t, dtype=float), jnp.asarray(p, dtype=float), z, tc, pc, omega)

    def phases(z_: Array, kv: Array, kl: Array) -> tuple[Array, Array, Array, Array, Array]:
        beta = _two_phase_rr(z_, kv, kl)
        d = 1.0 + beta[0] * (kv - 1.0) + beta[1] * (kl - 1.0)
        return beta[0], beta[1], z_ / d, kl * z_ / d, kv * z_ / d

    def g(state: Array, theta: Any) -> Array:
        model_, t_, p_, z_, tc_, pc_, omega_ = theta
        _, _, x_i, x_ii, y = phases(z_, jnp.exp(state[0]), jnp.exp(state[1]))
        x_i, x_ii, y = x_i / jnp.sum(x_i), x_ii / jnp.sum(x_ii), y / jnp.sum(y)
        kv_new = gamma_phi_k_values(model_, t_, p_, x_i, y, tc_, pc_, omega_, **options)
        kl_new = jnp.exp(model_.ln_gamma(x_i, t_) - model_.ln_gamma(x_ii, t_))
        return jnp.stack([jnp.log(kv_new), jnp.log(kl_new)])

    state0 = jnp.stack([jnp.log(jnp.clip(kv0, 1e-10, 1e10)), jnp.log(jnp.clip(kl0, 1e-10, 1e10))])
    solved = fixed_point_with_info(g, jnp.nan_to_num(state0), theta, tol, max_iter)
    bv, b2, x_i, x_ii, y = phases(z, jnp.exp(solved.value[0]), jnp.exp(solved.value[1]))
    b1 = 1.0 - bv - b2
    # The unclipped root: every phase fraction must be a fraction.
    three_phase = (bv > 1e-9) & (b1 > 1e-9) & (b2 > 1e-9)
    # Material balance and normalization of every phase (the inner 2x2
    # Rachford-Rice solve must have converged for the phases to be fractions).
    closure = residual_report(
        jnp.concatenate(
            (
                b1 * x_i + b2 * x_ii + bv * y - z,
                jnp.stack([jnp.sum(x_i) - 1.0, jnp.sum(x_ii) - 1.0, jnp.sum(y) - 1.0]),
            )
        ),
        1e-9,
    )
    report = with_status(solved.report, ~closure.converged, SolveStatus.MAX_ITERATIONS)
    report = with_status(report, ~three_phase, SolveStatus.INFEASIBLE)
    domain = reference_domain(t, z, tc, eos, pc, omega)
    report = with_status(report, ~domain, SolveStatus.OUT_OF_DOMAIN)
    report = with_status(report, ~input_report(t, p, z), SolveStatus.INVALID_INPUT)
    value = VLLEResult(
        beta_v=bv, beta_l1=b1, beta_l2=b2, y=y, x_i=x_i, x_ii=x_ii, three_phase=three_phase
    )
    return VLLESolveResult(gate_tree(value, report.converged), report)


def flash_vlle(
    model: ActivityModel,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    **options: Any,
) -> VLLEResult:
    """Three-phase flash value; NaN when `flash_vlle_with_info` reports failure."""
    solved = flash_vlle_with_info(model, t, p, z, tc, pc, omega, **options)
    return nan_unless_converged(solved.value, solved.report)


def heterogeneous_azeotrope(
    model: ActivityModel,
    p: ArrayLike,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    feed: ArrayLike = 0.5,
    t_min: float = 250.0,
    t_max: float = 500.0,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> HeterogeneousAzeotrope:
    """Binary heterogeneous azeotrope: the boiling temperature of two conjugate liquids.

    Inside a miscibility gap the two conjugate liquids have equal component
    activities (``a_i = x_i gamma_i`` matches across the LLE tie-line), so they
    necessarily boil to the *same* vapour ``y_i = a_i Psat_i / P``. The
    heterogeneous azeotrope is the temperature at which that shared vapour's total
    pressure ``sum_i a_i Psat_i`` reaches the system pressure ``P``, a single,
    well-posed scalar root, located on a scanned bracket and
    differentiable in ``P`` and the model parameters.

    Returns:
        A `HeterogeneousAzeotrope` (the temperature, both liquid
        compositions, and the common vapour), or NaN when no two-liquid boiling
        point exists on ``[t_min, t_max]``. Use a ``feed`` mole fraction (of
        component 1) that lies inside the miscibility gap.
    """
    z0 = jnp.asarray([feed, 1.0 - feed])

    def activities(t: Array) -> tuple[Array, Array, Array]:
        split = flash_lle_with_info(model, t, z0)
        # Only a genuine two-liquid split defines a heterogeneous azeotrope.
        ok = split.report.converged & (split.value.psi > 0) & (split.value.psi < 1)
        a = split.value.x_i * jnp.exp(model.ln_gamma(split.value.x_i, t))
        return jnp.where(ok, a, jnp.nan), split.value.x_i, split.value.x_ii

    def residual(t: Array, params: Any) -> Array:
        (p_,) = params
        a, _, _ = activities(t)
        psat, _ = saturation_pressures_with_info(eos, t, tc, pc, omega)
        valid = reference_domain(t, z0, tc, eos, pc, omega)
        return jnp.where(valid, jnp.log(jnp.sum(a * psat) / p_), jnp.nan)

    solved = scanned_root_with_info(
        residual,
        (jnp.asarray(p, dtype=float),),
        jnp.asarray(t_min),
        jnp.asarray(t_max),
        tol,
        max_iter,
    )
    t_star = solved.value
    a, x_i, x_ii = activities(t_star)
    psat, _ = saturation_pressures_with_info(eos, t_star, tc, pc, omega)
    y = a * psat / jnp.asarray(p)
    y = y / jnp.sum(y)
    result = HeterogeneousAzeotrope(t=t_star, x_i=x_i, x_ii=x_ii, y=y)
    return nan_unless_converged(result, solved.report)
