"""Phase equilibrium: K-values, Rachford-Rice, flash, saturation, and stability.

This module turns the cubic equation of state (`fugacio.thermo.eos`) into
the equilibrium calculations a process simulator actually calls:

* `wilson_k` -- the classic K-value initial guess;
* `rachford_rice` -- the material-balance root for the vapour fraction;
* `flash_pt` -- an isothermal-isobaric two-phase flash;
* `psat_eos` -- pure-component saturation pressure by equifugacity;
* `bubble_pressure_eos` / `dew_pressure_eos` -- phase envelopes;
* `stability_analysis` -- Michelsen's tangent-plane-distance test.

Every iterative result is differentiable end-to-end: the scalar solves carry
hand-written implicit-function-theorem rules (`jax.custom_jvp`) and the
flash/saturation loops reuse `fugacio.thermo.implicit.fixed_point`. You can
therefore take a gradient of *any* equilibrium output with respect to ``T``,
``P``, composition, or model parameters -- the property that makes Fugacio a
differentiable core rather than just another flash package.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.eos import CubicEOS, ln_phi_mixture, ln_phi_pure
from fugacio.thermo.implicit import fixed_point

ArrayLike = Array | float


class FlashResult(NamedTuple):
    """Result of an isothermal-isobaric flash.

    Attributes:
        beta: Vapour molar fraction (mol vapour / mol feed).
        x: Liquid-phase mole fractions.
        y: Vapour-phase mole fractions.
        k: Equilibrium ratios ``K_i = y_i / x_i`` at the solution.
    """

    beta: Array
    x: Array
    y: Array
    k: Array


class StabilityResult(NamedTuple):
    """Result of a tangent-plane stability analysis.

    Attributes:
        stable: ``True`` if the feed is single-phase stable.
        tpd: The smallest (most negative) modified tangent-plane distance found.
    """

    stable: Array
    tpd: Array


def wilson_k(t: ArrayLike, p: ArrayLike, tc: Array, pc: Array, omega: Array) -> Array:
    """Wilson correlation for initial K-values ``K_i = y_i / x_i``.

    ``K_i = (Pc_i / P) * exp[5.373 (1 + omega_i)(1 - Tc_i / T)]``.
    """
    t = jnp.asarray(t)
    p = jnp.asarray(p)
    return (pc / p) * jnp.exp(5.373 * (1.0 + omega) * (1.0 - tc / t))


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
    """Isothermal-isobaric two-phase flash by accelerated successive substitution.

    Solves the equal-fugacity conditions ``phi_i^L x_i = phi_i^V y_i`` together
    with the Rachford-Rice material balance, starting from Wilson K-values. The
    converged solution -- and therefore ``beta``, ``x``, ``y`` -- is
    differentiable with respect to ``(T, P, z, ...)`` via implicit
    differentiation of the fixed point.
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
        denom = 1.0 + beta * (k - 1.0)
        x = z_ / denom
        y = k * x
        ln_phi_l, _ = ln_phi_mixture(eos, t_, p_, x, tc_, pc_, omega_, phase="liquid", kij=kij_)
        ln_phi_v, _ = ln_phi_mixture(eos, t_, p_, y, tc_, pc_, omega_, phase="vapor", kij=kij_)
        return ln_phi_l - ln_phi_v

    ln_k_star = fixed_point(g, jnp.log(k0), theta, tol, max_iter)
    k = jnp.exp(ln_k_star)
    beta = rachford_rice(z, k)
    denom = 1.0 + beta * (k - 1.0)
    x = z / denom
    y = k * x
    return FlashResult(beta=beta, x=x, y=y, k=k)


def _psat_residual(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    tc: ArrayLike,
    pc: ArrayLike,
    omega: ArrayLike,
) -> Array:
    ln_phi_l, _ = ln_phi_pure(eos, t, p, tc, pc, omega, phase="liquid")
    ln_phi_v, _ = ln_phi_pure(eos, t, p, tc, pc, omega, phase="vapor")
    return ln_phi_l - ln_phi_v


@partial(jax.custom_jvp, nondiff_argnums=(0, 5, 6))
def psat_eos(
    eos: CubicEOS,
    t: ArrayLike,
    tc: ArrayLike,
    pc: ArrayLike,
    omega: ArrayLike,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> Array:
    """Pure-component saturation pressure (Pa) from the EOS by equifugacity.

    Solves ``ln phi^L(T, P) = ln phi^V(T, P)`` for ``P`` with a Newton iteration
    in ``ln P`` (which keeps the pressure positive), initialised from the Wilson
    vapour-pressure estimate. Differentiable in ``T`` (and the critical
    constants) via the Clapeyron-like implicit derivative ``dP/dT``.
    """
    t = jnp.asarray(t)
    p0 = pc * jnp.exp(5.373 * (1.0 + omega) * (1.0 - tc / t))

    def cond(carry: tuple[Array, Array]) -> Array:
        p, i = carry
        return (jnp.abs(_psat_residual(eos, t, p, tc, pc, omega)) > tol) & (i < max_iter)

    def body(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        p, i = carry
        r = _psat_residual(eos, t, p, tc, pc, omega)
        dr_dp = jax.grad(lambda pp: _psat_residual(eos, t, pp, tc, pc, omega))(p)
        ln_p_new = jnp.log(p) - r / (p * dr_dp)
        return jnp.exp(ln_p_new), i + 1

    p_star, _ = jax.lax.while_loop(cond, body, (p0, jnp.asarray(0)))
    return p_star


@psat_eos.defjvp
def _psat_eos_jvp(
    eos: CubicEOS,
    tol: float,
    max_iter: int,
    primals: tuple[Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    t, tc, pc, omega = primals
    t_dot, tc_dot, pc_dot, omega_dot = tangents
    p = psat_eos(eos, t, tc, pc, omega, tol, max_iter)
    r_p = jax.grad(lambda pp: _psat_residual(eos, t, pp, tc, pc, omega))(p)
    r_t = jax.grad(lambda tt: _psat_residual(eos, tt, p, tc, pc, omega))(t)
    r_tc = jax.grad(lambda v: _psat_residual(eos, t, p, v, pc, omega))(tc)
    r_pc = jax.grad(lambda v: _psat_residual(eos, t, p, tc, v, omega))(pc)
    r_om = jax.grad(lambda v: _psat_residual(eos, t, p, tc, pc, v))(omega)
    p_dot = -(r_t * t_dot + r_tc * tc_dot + r_pc * pc_dot + r_om * omega_dot) / r_p
    return p, p_dot


def _all_psat(eos: CubicEOS, t: ArrayLike, tc: Array, pc: Array, omega: Array) -> Array:
    return jax.vmap(lambda a, b, c: psat_eos(eos, t, a, b, c))(tc, pc, omega)


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
) -> tuple[Array, Array]:
    """Bubble-point pressure and incipient vapour composition at fixed ``T``, ``x``.

    Returns ``(P, y)``. Solved as a coupled fixed point in ``(ln P, y)`` so the
    result is differentiable in temperature and composition.
    """
    x = jnp.asarray(x)
    tc = jnp.asarray(tc)
    pc = jnp.asarray(pc)
    omega = jnp.asarray(omega)
    n = x.shape[0]
    kij_arr = jnp.zeros((n, n)) if kij is None else jnp.asarray(kij)
    psats = _all_psat(eos, t, tc, pc, omega)
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

    state = fixed_point(g, state0, theta, tol, max_iter)
    return jnp.exp(state[0]), state[1:]


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
) -> tuple[Array, Array]:
    """Dew-point pressure and incipient liquid composition at fixed ``T``, ``y``.

    Returns ``(P, x)``, differentiable in temperature and composition.
    """
    y = jnp.asarray(y)
    tc = jnp.asarray(tc)
    pc = jnp.asarray(pc)
    omega = jnp.asarray(omega)
    n = y.shape[0]
    kij_arr = jnp.zeros((n, n)) if kij is None else jnp.asarray(kij)
    psats = _all_psat(eos, t, tc, pc, omega)
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

    state = fixed_point(g, state0, theta, tol, max_iter)
    return jnp.exp(state[0]), state[1:]


def stability_analysis(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    kij: Array | None = None,
    iters: int = 40,
) -> StabilityResult:
    """Michelsen tangent-plane stability test for a feed ``z`` at ``(T, P)``.

    Performs two trial-phase searches (vapour-like and liquid-like). If the
    modified tangent-plane distance ``tm`` dips below zero for either trial, a
    second phase can lower the Gibbs energy and the feed is *unstable* (it will
    split). Returns the worst (smallest) ``tm`` found and a stability flag.
    """
    z = jnp.asarray(z)
    tc = jnp.asarray(tc)
    pc = jnp.asarray(pc)
    omega = jnp.asarray(omega)
    n = z.shape[0]
    kij_arr = jnp.zeros((n, n)) if kij is None else jnp.asarray(kij)

    ln_phi_zl, _ = ln_phi_mixture(eos, t, p, z, tc, pc, omega, phase="liquid", kij=kij_arr)
    ln_phi_zv, _ = ln_phi_mixture(eos, t, p, z, tc, pc, omega, phase="vapor", kij=kij_arr)
    g_l = jnp.sum(z * (jnp.log(z) + ln_phi_zl))
    g_v = jnp.sum(z * (jnp.log(z) + ln_phi_zv))
    ln_phi_z = jnp.where(g_l < g_v, ln_phi_zl, ln_phi_zv)
    d = jnp.log(z) + ln_phi_z

    k_wilson = wilson_k(t, p, tc, pc, omega)

    def run_trial(w0: Array, phase: str) -> Array:
        def body(_: Array, w: Array) -> Array:
            wn = w / jnp.sum(w)
            ln_phi_w, _ = ln_phi_mixture(eos, t, p, wn, tc, pc, omega, phase=phase, kij=kij_arr)
            return jnp.exp(d - ln_phi_w)

        w = jax.lax.fori_loop(0, iters, body, w0)
        wn = w / jnp.sum(w)
        ln_phi_w, _ = ln_phi_mixture(eos, t, p, wn, tc, pc, omega, phase=phase, kij=kij_arr)
        return 1.0 + jnp.sum(w * (jnp.log(w) + ln_phi_w - d - 1.0))

    tm_vapor = run_trial(z * k_wilson, "vapor")
    tm_liquid = run_trial(z / k_wilson, "liquid")
    tpd = jnp.minimum(tm_vapor, tm_liquid)
    return StabilityResult(stable=tpd >= -1e-8, tpd=tpd)
