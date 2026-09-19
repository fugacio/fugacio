"""Phase equilibrium on a cubic equation of state: flash and saturation.

This module turns the cubic equation of state (`fugacio.thermo.eos`) into
the equilibrium calculations a process simulator calls:

* `wilson_k`: the classic K-value initial guess;
* `rachford_rice`: the material-balance root for the vapour fraction;
* `flash_pt_with_info`: an isothermal-isobaric vapour-liquid flash;
* `psat_eos_with_info`: pure-component saturation pressure by equifugacity;
* `bubble_pressure_eos_with_info` / `dew_pressure_eos_with_info`: mixture
  saturation pressures and incipient-phase compositions.

Every calculation has a checked ``_with_info`` form that returns its best state
together with a `fugacio.thermo.diagnostics.SolveReport`, and a value-only form
that returns NaN whenever that report fails. A failed solve therefore never
returns a finite number. Converged results carry implicit-function-theorem
derivatives with respect to temperature, pressure, composition, and the model
constants; failed results have nonfinite derivatives.

A two-phase flash can't detect a second liquid. Whether a feed splits at all is
answered by the tangent-plane search in `fugacio.thermo.stability`, which every
property package exposes as ``stability``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.diagnostics import (
    SolveReport,
    SolveResult,
    SolveStatus,
    nan_unless_converged,
    residual_report,
    with_status,
)
from fugacio.thermo.eos import CubicEOS, ln_phi_mixture, ln_phi_pure
from fugacio.thermo.implicit import fixed_point_with_info, gate_tree, implicit_solution

ArrayLike = Array | float


class FlashResult(NamedTuple):
    """Result of an isothermal-isobaric flash.

    Attributes:
        beta: Vapour molar fraction (mol vapour / mol feed).
        x: Liquid-phase mole fractions (the incipient liquid when ``beta == 1``).
        y: Vapour-phase mole fractions (the incipient vapour when ``beta == 0``).
        k: Equilibrium ratios ``K_i = y_i / x_i`` at the solution.
    """

    beta: Array
    x: Array
    y: Array
    k: Array


class FlashSolveResult(NamedTuple):
    """PT flash and the report of its fixed-point iteration."""

    value: FlashResult
    report: SolveReport


class SaturationResult(NamedTuple):
    """A saturation point and the composition of the incipient phase.

    Attributes:
        value: Saturation pressure (Pa) or temperature (K).
        composition: Incipient-phase mole fractions: the vapour of a bubble
            point or the liquid of a dew point.
    """

    value: Array
    composition: Array


class SaturationSolveResult(NamedTuple):
    """A saturation point and its solve report."""

    value: SaturationResult
    report: SolveReport


def wilson_k(t: ArrayLike, p: ArrayLike, tc: Array, pc: Array, omega: Array) -> Array:
    """Wilson correlation for initial K-values ``K_i = y_i / x_i``.

    ``K_i = (Pc_i / P) * exp[5.373 (1 + omega_i)(1 - Tc_i / T)]``.
    """
    t = jnp.asarray(t)
    p = jnp.asarray(p)
    return (pc / p) * jnp.exp(5.373 * (1.0 + omega) * (1.0 - tc / t))


def wilson_psat(t: ArrayLike, tc: ArrayLike, pc: ArrayLike, omega: ArrayLike) -> Array:
    """Wilson's vapour-pressure estimate (Pa), an initialization only."""
    t = jnp.asarray(t, dtype=float)
    return jnp.asarray(pc) * jnp.exp(5.373 * (1.0 + jnp.asarray(omega)) * (1.0 - tc / t))


def _rr_residual(beta: Array, z: Array, k: Array) -> Array:
    return jnp.sum(z * (k - 1.0) / (1.0 + beta * (k - 1.0)))


@jax.custom_jvp
def rachford_rice(z: Array, k: Array) -> Array:
    """Solve the Rachford-Rice equation for the vapour fraction ``beta``.

    Returns ``beta`` clamped to ``[0, 1]``: ``0`` for a subcooled liquid, ``1``
    for a superheated vapour, and the interior root otherwise. The residual
    ``sum_i z_i (K_i - 1) / (1 + beta (K_i - 1))`` is monotonically decreasing in
    ``beta`` on ``(0, 1)``, so a bisection is used for the interior root.
    """
    z = jnp.asarray(z)
    k = jnp.asarray(k)
    f0 = _rr_residual(jnp.asarray(0.0), z, k)
    f1 = _rr_residual(jnp.asarray(1.0), z, k)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        lo, hi, i = carry
        mid = 0.5 * (lo + hi)
        f_mid = _rr_residual(mid, z, k)
        lo_new = jnp.where(f_mid > 0.0, mid, lo)
        hi_new = jnp.where(f_mid > 0.0, hi, mid)
        return lo_new, hi_new, i + 1

    lo, hi, _ = jax.lax.while_loop(
        lambda c: c[2] < 80,
        body,
        (jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(0)),
    )
    beta_interior = 0.5 * (lo + hi)
    return jnp.where(f0 <= 0.0, 0.0, jnp.where(f1 >= 0.0, 1.0, beta_interior))


@rachford_rice.defjvp
def _rachford_rice_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    z, k = primals
    z_dot, k_dot = tangents
    beta = rachford_rice(z, k)
    f0 = _rr_residual(jnp.asarray(0.0), z, k)
    f1 = _rr_residual(jnp.asarray(1.0), z, k)
    interior = (f0 > 0.0) & (f1 < 0.0)
    f_beta = jax.grad(_rr_residual, argnums=0)(beta, z, k)
    f_z = jax.grad(_rr_residual, argnums=1)(beta, z, k)
    f_k = jax.grad(_rr_residual, argnums=2)(beta, z, k)
    beta_dot_interior = -(jnp.vdot(f_z, z_dot) + jnp.vdot(f_k, k_dot)) / f_beta
    beta_dot = jnp.where(interior, beta_dot_interior, 0.0)
    return beta, beta_dot


#: ``max |ln K|`` below which a converged K-iteration is the trivial solution.
_TRIVIAL_LN_K = 1.0e-6


def classify_trivial(z: Array, k: Array, beta: Array, k_wilson: Array, z_factor: Array) -> Array:
    """Vapour fraction of a single-phase feed when the K-iteration went trivial.

    Successive substitution on a feed that is single phase collapses to the
    trivial solution ``K = 1`` (identical trial phases), and the Rachford-Rice
    residual at ``K = 1`` is zero on both ends, so ``beta`` alone cannot tell a
    superheated vapour from a subcooled liquid. When ``max |ln K|`` is below
    `_TRIVIAL_LN_K`, the phase is decided from the Wilson K-values instead: the
    feed is a vapour if it lies below its Wilson dew pressure
    (``sum z_i (K_i - 1) / K_i >= 0``) and a liquid if above its Wilson bubble
    pressure (``sum z_i (K_i - 1) <= 0``). If Wilson claims two phases where
    the rigorous iteration found one, the compressibility factor of the single
    root decides (``Z > 0.5`` is vapour-like). Away from the trivial solution
    ``beta`` is returned unchanged.
    """
    trivial = jnp.max(jnp.abs(jnp.log(k))) < _TRIVIAL_LN_K
    f0 = jnp.sum(z * (k_wilson - 1.0))
    f1 = jnp.sum(z * (k_wilson - 1.0) / k_wilson)
    single = jnp.where(
        f1 >= 0.0, 1.0, jnp.where(f0 <= 0.0, 0.0, jnp.where(z_factor > 0.5, 1.0, 0.0))
    )
    return jnp.where(trivial, single, beta)


def phase_compositions(z: Array, k: Array, beta: Array) -> tuple[Array, Array]:
    """Liquid and vapour compositions from the material balance at ``(K, beta)``.

    A present phase's composition follows from Rachford-Rice. An absent phase
    is the incipient trial phase, normalized so both are mole fractions.
    """
    denom = 1.0 + beta * (k - 1.0)
    x = z / denom
    y = k * x
    x = jnp.where(beta >= 1.0, x / jnp.sum(x), x)
    y = jnp.where(beta <= 0.0, y / jnp.sum(y), y)
    return x, y


def input_report(t: ArrayLike, p: ArrayLike, z: Array) -> Array:
    """Whether ``(T, P, z)`` is a physical state specification."""
    t = jnp.asarray(t)
    p = jnp.asarray(p)
    z = jnp.asarray(z)
    return (
        jnp.isfinite(t)
        & jnp.isfinite(p)
        & (t > 0)
        & (p > 0)
        & jnp.all(jnp.isfinite(z))
        & jnp.all(z >= 0)
        & (jnp.sum(z) > 0)
    )


def flash_pt(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    kij: Array | None = None,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> FlashResult:
    """Isothermal flash value; NaN when `flash_pt_with_info` reports failure."""
    solved = flash_pt_with_info(eos, t, p, z, tc, pc, omega, kij=kij, tol=tol, max_iter=max_iter)
    return nan_unless_converged(solved.value, solved.report)


def flash_pt_with_info(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    kij: Array | None = None,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> FlashSolveResult:
    """Isothermal-isobaric vapour-liquid flash by successive substitution.

    Solves the equal-fugacity conditions ``phi_i^L x_i = phi_i^V y_i`` together
    with the Rachford-Rice material balance, starting from Wilson K-values. A
    single-phase feed collapses onto the trivial solution ``K = 1``, whose phase
    is decided by `classify_trivial`. The converged state is differentiable with
    respect to ``(T, P, z, ...)`` via implicit differentiation of the fixed point.
    This flash considers one liquid and one vapour; test the result's stability
    when a second liquid can form.
    """
    z = jnp.asarray(z)
    tc = jnp.asarray(tc)
    pc = jnp.asarray(pc)
    omega = jnp.asarray(omega)
    n = z.shape[0]
    kij_arr = jnp.zeros((n, n)) if kij is None else jnp.asarray(kij)
    k0 = wilson_k(t, p, tc, pc, omega)
    theta = (jnp.asarray(t), jnp.asarray(p), z, tc, pc, omega, kij_arr)

    def g(ln_k: Array, theta: Any) -> Array:
        t_, p_, z_, tc_, pc_, omega_, kij_ = theta
        k = jnp.exp(ln_k)
        beta = rachford_rice(z_, k)
        # Normalized trial phases: a no-op at an interior Rachford-Rice root, and
        # the incipient phase of a single-phase feed (where the unnormalized
        # ``z / K`` would make the equation of state see a non-composition).
        x, y = phase_compositions(z_, k, beta)
        x, y = x / jnp.sum(x), y / jnp.sum(y)
        ln_phi_l, _ = ln_phi_mixture(eos, t_, p_, x, tc_, pc_, omega_, phase="liquid", kij=kij_)
        ln_phi_v, _ = ln_phi_mixture(eos, t_, p_, y, tc_, pc_, omega_, phase="vapor", kij=kij_)
        return ln_phi_l - ln_phi_v

    solved = fixed_point_with_info(g, jnp.log(k0), theta, tol, max_iter)
    k = jnp.exp(solved.value)
    beta = rachford_rice(z, k)
    _, z_single = ln_phi_mixture(eos, t, p, z, tc, pc, omega, phase="vapor", kij=kij_arr)
    beta = classify_trivial(z, k, beta, k0, z_single)
    x, y = phase_compositions(z, k, beta)
    report = with_status(solved.report, ~input_report(t, p, z), SolveStatus.INVALID_INPUT)
    return FlashSolveResult(FlashResult(beta=beta, x=x, y=y, k=k), report)


# --------------------------------------------------------------------------- #
# Pure-component saturation
# --------------------------------------------------------------------------- #

#: Compressibility separating vapour-like from liquid-like single cubic roots.
_Z_SPLIT = 0.3


def _psat_terms(eos: CubicEOS, ln_p: Array, theta: Any) -> tuple[Array, tuple[Array, Array]]:
    """Equifugacity residual in ``ln P``, whether the roots coincide, and ``Z^V``."""
    t, tc, pc, omega = theta
    p = jnp.exp(ln_p)
    ln_phi_l, z_l = ln_phi_pure(eos, t, p, tc, pc, omega, phase="liquid")
    ln_phi_v, z_v = ln_phi_pure(eos, t, p, tc, pc, omega, phase="vapor")
    single = jnp.abs(z_v - z_l) <= 1e-9 * jnp.maximum(jnp.abs(z_v), 1e-12)
    return ln_phi_l - ln_phi_v, (single, z_v)


def psat_eos_with_info(
    eos: CubicEOS,
    t: ArrayLike,
    tc: ArrayLike,
    pc: ArrayLike,
    omega: ArrayLike,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> SolveResult:
    """Pure-component saturation pressure (Pa) from the EOS by equifugacity.

    Solves ``ln phi^L(T, P) = ln phi^V(T, P)`` for ``ln P`` by Newton's method
    from the Wilson estimate. A trial pressure outside the three-root region has
    coincident liquid and vapour roots, where the residual vanishes trivially;
    such a step is redirected toward the coexistence region instead of being
    accepted. The loop stops on its own recorded convergence flag, so scalar and
    vectorized evaluations follow identical arithmetic.

    At or above the critical temperature no saturation pressure exists: the
    report is ``OUT_OF_DOMAIN`` and the value is the (differentiable) Wilson
    extrapolation, a usable initialization but not a property. Keeping that
    value finite lets a mixture model carry an absent supercritical component
    without poisoning derivatives. A converged value is differentiable in ``T``
    and the critical constants (Clapeyron's implicit derivative); failures
    inside the domain have nonfinite derivatives.

    Args:
        eos: Cubic equation of state.
        t: Temperature (K).
        tc: Critical temperature (K).
        pc: Critical pressure (Pa).
        omega: Acentric factor.
        tol: Residual tolerance on ``ln phi^L - ln phi^V``.
        max_iter: Newton iteration cap.

    Returns:
        The saturation pressure and its report.
    """
    t = jnp.asarray(t, dtype=float)
    tc = jnp.asarray(tc, dtype=float)
    pc = jnp.asarray(pc, dtype=float)
    omega = jnp.asarray(omega, dtype=float)
    in_domain = (t > 0.0) & (t < tc) & jnp.isfinite(t)
    # Every lane (including vectorized supercritical ones) solves inside the
    # domain, so a discarded solve never contributes a nonfinite derivative.
    params = (jnp.where(in_domain, t, 0.9 * tc), tc, pc, omega)
    theta = jax.lax.stop_gradient(params)
    ln_p0 = jnp.log(jax.lax.stop_gradient(wilson_psat(*params)))

    def terms(ln_p: Array) -> tuple[Array, Array, Array]:
        r, (single, z_v) = _psat_terms(eos, ln_p, theta)
        return r, single, z_v

    def cond(carry: tuple) -> Array:
        _, _, _, i, done = carry
        return ~done & (i < max_iter)

    def body(carry: tuple) -> tuple:
        ln_p, _, _, i, _ = carry
        (r, (single, z_v)), slope = jax.value_and_grad(
            lambda x: _psat_terms(eos, x, theta), has_aux=True
        )(ln_p)
        newton = jnp.clip(-r / jnp.where(jnp.abs(slope) > 1e-300, slope, -1.0), -3.0, 3.0)
        # One real root: a vapour-like root means P is below the coexistence
        # window, a liquid-like root that it's above it.
        escape = jnp.where(z_v > _Z_SPLIT, 0.7, -0.7)
        ln_new = ln_p + jnp.where(single | ~jnp.isfinite(newton), escape, newton)
        r_new, single_new, _ = terms(ln_new)
        done = ((jnp.abs(r_new) <= tol) & ~single_new) | ~jnp.isfinite(r_new)
        return ln_new, r_new, single_new, i + 1, done

    r0, single0, _ = terms(ln_p0)
    done0 = ((jnp.abs(r0) <= tol) & ~single0) | ~jnp.isfinite(r0)
    ln_p, r, single, iterations, _ = jax.lax.while_loop(
        cond, body, (ln_p0, r0, single0, jnp.asarray(0), done0)
    )
    solve = residual_report(jnp.atleast_1d(r), tol, iterations=iterations)
    solve = with_status(solve, single, SolveStatus.TRIVIAL)
    ln_psat = implicit_solution(
        lambda x, th: jnp.atleast_1d(_psat_terms(eos, x[0], th)[0]),
        jnp.atleast_1d(ln_p),
        params,
        solve.converged,
    )[0]
    value = jnp.where(in_domain, jnp.exp(ln_psat), wilson_psat(t, tc, pc, omega))
    report = with_status(solve, ~in_domain, SolveStatus.OUT_OF_DOMAIN)
    return SolveResult(value, report)


def psat_eos(
    eos: CubicEOS,
    t: ArrayLike,
    tc: ArrayLike,
    pc: ArrayLike,
    omega: ArrayLike,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> Array:
    """Saturation pressure (Pa); NaN when `psat_eos_with_info` reports failure."""
    solved = psat_eos_with_info(eos, t, tc, pc, omega, tol=tol, max_iter=max_iter)
    return nan_unless_converged(solved.value, solved.report)


def psat_seeds(eos: CubicEOS, t: ArrayLike, tc: Array, pc: Array, omega: Array) -> Array:
    """Finite, detached pure-component pressures for initializing mixture solves.

    Converged EOS saturation pressures where they exist, and the Wilson
    extrapolation elsewhere (for example, supercritical components).
    """

    def one(a: Array, b: Array, c: Array) -> Array:
        solved = psat_eos_with_info(eos, t, a, b, c)
        return jnp.where(solved.report.converged, solved.value, wilson_psat(t, a, b, c))

    return jax.lax.stop_gradient(
        jax.vmap(one)(jnp.asarray(tc), jnp.asarray(pc), jnp.asarray(omega))
    )


# --------------------------------------------------------------------------- #
# Mixture saturation pressures
# --------------------------------------------------------------------------- #


def _trivial_saturation(
    eos: CubicEOS,
    t: ArrayLike,
    p: Array,
    liquid: Array,
    vapor: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    kij: Array,
) -> Array:
    """Whether a converged saturation point is the trivial one-root solution.

    At a genuine bubble or dew point the phases differ in density even when
    their compositions coincide (an azeotrope). A trivial solution has equal
    compositions on a single EOS root.
    """
    _, z_l = ln_phi_mixture(eos, t, p, liquid, tc, pc, omega, phase="liquid", kij=kij)
    _, z_v = ln_phi_mixture(eos, t, p, vapor, tc, pc, omega, phase="vapor", kij=kij)
    same_density = jnp.abs(z_v - z_l) <= 1e-6 * jnp.maximum(jnp.abs(z_v), 1e-12)
    return jax.lax.stop_gradient(same_density & (jnp.max(jnp.abs(vapor - liquid)) <= 1e-6))


def bubble_pressure_eos_with_info(
    eos: CubicEOS,
    t: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    kij: Array | None = None,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> SaturationSolveResult:
    """Bubble-point pressure and incipient vapour composition at fixed ``T``, ``x``.

    Solved as a coupled fixed point in ``(ln P, y)`` from Raoult's-law seeds. A
    trivial solution (above the cricondenbar, where the iteration collapses onto
    ``y = x`` on one root) is reported as ``TRIVIAL``.
    """
    x = jnp.asarray(x)
    tc = jnp.asarray(tc)
    pc = jnp.asarray(pc)
    omega = jnp.asarray(omega)
    n = x.shape[0]
    kij_arr = jnp.zeros((n, n)) if kij is None else jnp.asarray(kij)
    psats = psat_seeds(eos, t, tc, pc, omega)
    p0 = jnp.sum(x * psats)
    y0 = x * psats / p0
    state0 = jnp.concatenate([jnp.log(p0)[None], y0])
    theta = (jnp.asarray(t), x, tc, pc, omega, kij_arr)

    def g(state: Array, theta: Any) -> Array:
        t_, x_, tc_, pc_, omega_, kij_ = theta
        p = jnp.exp(state[0])
        y = state[1:]
        ln_phi_l, _ = ln_phi_mixture(eos, t_, p, x_, tc_, pc_, omega_, phase="liquid", kij=kij_)
        ln_phi_v, _ = ln_phi_mixture(eos, t_, p, y, tc_, pc_, omega_, phase="vapor", kij=kij_)
        k = jnp.exp(ln_phi_l - ln_phi_v)
        y_unnorm = k * x_
        s = jnp.sum(y_unnorm)
        return jnp.concatenate([(state[0] + jnp.log(s))[None], y_unnorm / s])

    solved = fixed_point_with_info(g, state0, theta, tol, max_iter)
    p, y = jnp.exp(solved.value[0]), solved.value[1:]
    trivial = _trivial_saturation(eos, t, p, x, y, tc, pc, omega, kij_arr)
    report = with_status(solved.report, trivial, SolveStatus.TRIVIAL)
    report = with_status(report, ~input_report(t, 1.0, x), SolveStatus.INVALID_INPUT)
    value = gate_tree(SaturationResult(p, y), report.converged)
    return SaturationSolveResult(value, report)


def bubble_pressure_eos(
    eos: CubicEOS,
    t: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    kij: Array | None = None,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> SaturationResult:
    """Bubble pressure and incipient vapour ``(P, y)``; NaN on failure."""
    solved = bubble_pressure_eos_with_info(
        eos, t, x, tc, pc, omega, kij=kij, tol=tol, max_iter=max_iter
    )
    return nan_unless_converged(solved.value, solved.report)


def dew_pressure_eos_with_info(
    eos: CubicEOS,
    t: ArrayLike,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    kij: Array | None = None,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> SaturationSolveResult:
    """Dew-point pressure and incipient liquid composition at fixed ``T``, ``y``.

    Coupled fixed point in ``(ln P, x)``. A trivial one-root solution is
    reported as ``TRIVIAL``.
    """
    y = jnp.asarray(y)
    tc = jnp.asarray(tc)
    pc = jnp.asarray(pc)
    omega = jnp.asarray(omega)
    n = y.shape[0]
    kij_arr = jnp.zeros((n, n)) if kij is None else jnp.asarray(kij)
    psats = psat_seeds(eos, t, tc, pc, omega)
    p0 = 1.0 / jnp.sum(y / psats)
    x0 = y * p0 / psats
    state0 = jnp.concatenate([jnp.log(p0)[None], x0])
    theta = (jnp.asarray(t), y, tc, pc, omega, kij_arr)

    def g(state: Array, theta: Any) -> Array:
        t_, y_, tc_, pc_, omega_, kij_ = theta
        p = jnp.exp(state[0])
        x = state[1:]
        ln_phi_l, _ = ln_phi_mixture(eos, t_, p, x, tc_, pc_, omega_, phase="liquid", kij=kij_)
        ln_phi_v, _ = ln_phi_mixture(eos, t_, p, y_, tc_, pc_, omega_, phase="vapor", kij=kij_)
        k = jnp.exp(ln_phi_l - ln_phi_v)
        x_unnorm = y_ / k
        s = jnp.sum(x_unnorm)
        return jnp.concatenate([(state[0] - jnp.log(s))[None], x_unnorm / s])

    solved = fixed_point_with_info(g, state0, theta, tol, max_iter)
    p, x = jnp.exp(solved.value[0]), solved.value[1:]
    trivial = _trivial_saturation(eos, t, p, x, y, tc, pc, omega, kij_arr)
    report = with_status(solved.report, trivial, SolveStatus.TRIVIAL)
    report = with_status(report, ~input_report(t, 1.0, y), SolveStatus.INVALID_INPUT)
    value = gate_tree(SaturationResult(p, x), report.converged)
    return SaturationSolveResult(value, report)


def dew_pressure_eos(
    eos: CubicEOS,
    t: ArrayLike,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    kij: Array | None = None,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> SaturationResult:
    """Dew pressure and incipient liquid ``(P, x)``; NaN on failure."""
    solved = dew_pressure_eos_with_info(
        eos, t, y, tc, pc, omega, kij=kij, tol=tol, max_iter=max_iter
    )
    return nan_unless_converged(solved.value, solved.report)
