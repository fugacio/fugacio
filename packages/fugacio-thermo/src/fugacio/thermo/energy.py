"""Energy-balance phase equilibrium: two-phase enthalpy/entropy and PH/PS flash.

The isothermal flash in :mod:`fugacio.thermo.equilibrium` answers "what splits?"
at a *given* temperature. Process units instead fix an *energy* specification --
a heat duty, an adiabatic mix, an isentropic compression -- and the temperature
is unknown. This module supplies:

* :func:`mixture_enthalpy` / :func:`mixture_entropy` -- the molar enthalpy and
  entropy of an equilibrium feed at ``(T, P)``, correctly blending the vapour and
  liquid products of the flash (so the latent heat is included automatically);
* :func:`flash_ph` -- the isenthalpic (adiabatic) flash: solve for the
  temperature at which the mixture enthalpy meets a target, then return the split;
* :func:`flash_ps` -- the isentropic flash, the backbone of compressor and
  turbine models.

Both flashes solve a scalar, monotone energy residual (enthalpy or entropy minus
a specification) for the temperature with a *safeguarded* Newton iteration: the
forward pass brackets the root in ``[t_min, t_max]`` and only ever evaluates the
residual's value (never its gradient), falling back to bisection whenever a
Newton step would leave the bracket. This is robust even when a trial temperature
crosses a phase boundary, where the underlying flash gradient is ill-defined. The
converged temperature -- and everything derived from it -- is differentiable with
respect to the energy specification, pressure, feed, and model parameters by the
implicit function theorem (a hand-written ``custom_jvp`` rule), with no
differentiation through the iteration itself.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import P_REF, T_REF
from fugacio.thermo.eos import CubicEOS
from fugacio.thermo.equilibrium import flash_pt
from fugacio.thermo.properties import CpCoeffs, molar_enthalpy, molar_entropy

ArrayLike = Array | float


class EnergyFlashResult(NamedTuple):
    """Result of an energy-specified flash (PH or PS).

    Attributes:
        t: Solved temperature (K).
        beta: Vapour molar fraction.
        x: Liquid-phase mole fractions.
        y: Vapour-phase mole fractions.
        k: Equilibrium ratios at the solution.
    """

    t: Array
    beta: Array
    x: Array
    y: Array
    k: Array


def mixture_enthalpy(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    *,
    kij: Array | None = None,
    t_ref: float = T_REF,
) -> Array:
    """Molar enthalpy of an equilibrium feed ``z`` at ``(T, P)`` (J/mol of feed).

    Runs the isothermal flash and blends the phase enthalpies by vapour fraction,
    ``H = (1 - beta) H^L(x) + beta H^V(y)``; in the single-phase region ``beta``
    is 0 or 1 and this reduces to the single-phase enthalpy.
    """
    r = flash_pt(eos, t, p, z, tc, pc, omega, kij=kij)
    h_l = molar_enthalpy(
        t, p, r.x, tc, pc, omega, cp, eos=eos, phase="liquid", kij=kij, t_ref=t_ref
    )
    h_v = molar_enthalpy(t, p, r.y, tc, pc, omega, cp, eos=eos, phase="vapor", kij=kij, t_ref=t_ref)
    return (1.0 - r.beta) * h_l + r.beta * h_v


def mixture_entropy(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    *,
    kij: Array | None = None,
    t_ref: float = T_REF,
    p_ref: float = P_REF,
) -> Array:
    """Molar entropy of an equilibrium feed ``z`` at ``(T, P)`` (J/mol/K of feed)."""
    r = flash_pt(eos, t, p, z, tc, pc, omega, kij=kij)
    s_l = molar_entropy(
        t, p, r.x, tc, pc, omega, cp, eos=eos, phase="liquid", kij=kij, t_ref=t_ref, p_ref=p_ref
    )
    s_v = molar_entropy(
        t, p, r.y, tc, pc, omega, cp, eos=eos, phase="vapor", kij=kij, t_ref=t_ref, p_ref=p_ref
    )
    return (1.0 - r.beta) * s_l + r.beta * s_v


@partial(jax.custom_jvp, nondiff_argnums=(0, 2, 3, 4, 5, 6))
def _implicit_temperature(
    residual: Callable[[Array, Any], Array],
    params: Any,
    t_init: float,
    t_min: float,
    t_max: float,
    tol: float,
    max_iter: int,
) -> Array:
    """Solve ``residual(T, params) = 0`` for the temperature ``T`` in ``[t_min, t_max]``.

    ``residual`` is a smooth, monotonically *increasing* energy residual (enthalpy
    or entropy minus a specification, since both rise with temperature); ``params``
    is the differentiable pytree it depends on.

    The forward pass is a safeguarded Newton iteration. It maintains a bracket
    ``[lo, hi]`` (initialised to ``[t_min, t_max]``) that always contains the root,
    using only residual *values* -- the slope is estimated by a one-sided finite
    difference rather than ``jax.grad``, because a Newton trial can cross a phase
    boundary where the flash gradient is undefined (``NaN``). A Newton step is
    accepted only if it stays inside the bracket and the slope is usable; otherwise
    the step bisects. This converges from any starting point without ever
    propagating a ``NaN``.

    The converged temperature is differentiated by the implicit function theorem
    in the ``custom_jvp`` rule below: ``dT* = -(dr/dparams . dparams) / (dr/dT)``,
    using only *first-order* sensitivities of the residual at the solution (so the
    flash's own ``custom_vjp`` handles them and nothing differentiates through the
    iteration itself).
    """

    def cond(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
        _, _, _, i, err = carry
        return (err > tol) & (i < max_iter)

    def body(
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        lo, hi, t, i, _ = carry
        r = residual(t, params)
        # The residual increases with T, so the sign of r tells us which side of
        # the root we are on; tighten the bracket accordingly.
        lo = jnp.where(r <= 0.0, t, lo)
        hi = jnp.where(r > 0.0, t, hi)
        # One-sided finite-difference slope, probing *inside* the bracket so we
        # never evaluate the residual outside [t_min, t_max] (where it may be NaN).
        h = jnp.maximum(1e-4 * jnp.abs(t), 1e-4)
        t_probe = jnp.maximum(t - h, lo)
        dr = (r - residual(t_probe, params)) / jnp.maximum(t - t_probe, 1e-12)
        t_newton = t - r / dr
        usable = jnp.isfinite(t_newton) & (t_newton > lo) & (t_newton < hi) & (jnp.abs(dr) > 1e-12)
        t_next = jnp.where(usable, t_newton, 0.5 * (lo + hi))
        return lo, hi, t_next, i + 1, jnp.abs(t_next - t)

    lo0 = jnp.asarray(t_min, dtype=float)
    hi0 = jnp.asarray(t_max, dtype=float)
    t0 = jnp.clip(jnp.asarray(t_init, dtype=float), lo0, hi0)
    init = (lo0, hi0, t0, jnp.asarray(0), jnp.asarray(jnp.inf))
    _, _, t_star, _, _ = jax.lax.while_loop(cond, body, init)
    return t_star


@_implicit_temperature.defjvp
def _implicit_temperature_jvp(
    residual: Callable[[Array, Any], Array],
    t_init: float,
    t_min: float,
    t_max: float,
    tol: float,
    max_iter: int,
    primals: tuple[Any],
    tangents: tuple[Any],
) -> tuple[Array, Array]:
    (params,) = primals
    (params_dot,) = tangents
    t_star = _implicit_temperature(residual, params, t_init, t_min, t_max, tol, max_iter)
    r_t = jax.grad(lambda tt: residual(tt, params))(t_star)
    grad_params = jax.grad(lambda pp: residual(t_star, pp))(params)
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda g, d: jnp.vdot(g, d), grad_params, params_dot)
    )
    r_dot = sum(leaves, jnp.asarray(0.0))
    t_dot = -r_dot / r_t
    return t_star, t_dot


def flash_ph(
    eos: CubicEOS,
    p: ArrayLike,
    h_spec: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    *,
    kij: Array | None = None,
    t_init: ArrayLike = 300.0,
    t_min: float = 50.0,
    t_max: float = 1500.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> EnergyFlashResult:
    """Isenthalpic (adiabatic) flash: find ``T`` so the feed enthalpy equals ``h_spec``.

    Returns the solved temperature together with the equilibrium split, all
    differentiable with respect to ``(p, h_spec, z, ...)``. ``dT/d h_spec`` is the
    reciprocal of the two-phase heat capacity, including latent effects. The
    temperature is bracketed to ``[t_min, t_max]`` (raise ``t_max`` for very hot
    streams, but keep it where the EOS evaluates cleanly).
    """
    z = jnp.asarray(z)
    n = z.shape[0]
    kij_arr = jnp.zeros((n, n)) if kij is None else jnp.asarray(kij)
    params = (jnp.asarray(p, dtype=float), jnp.asarray(h_spec, dtype=float), z, tc, pc, omega)

    def residual(t: Array, params: Any) -> Array:
        p_, h_, z_, tc_, pc_, omega_ = params
        return mixture_enthalpy(eos, t, p_, z_, tc_, pc_, omega_, cp, kij=kij_arr) - h_

    t_star = _implicit_temperature(residual, params, float(t_init), t_min, t_max, tol, max_iter)
    r = flash_pt(eos, t_star, p, z, tc, pc, omega, kij=kij_arr)
    return EnergyFlashResult(t=t_star, beta=r.beta, x=r.x, y=r.y, k=r.k)


def flash_ps(
    eos: CubicEOS,
    p: ArrayLike,
    s_spec: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    *,
    kij: Array | None = None,
    t_init: ArrayLike = 300.0,
    t_min: float = 50.0,
    t_max: float = 1500.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> EnergyFlashResult:
    """Isentropic flash: find ``T`` so the feed entropy equals ``s_spec``.

    The backbone of isentropic compressor and turbine models: given an inlet
    entropy and an outlet pressure, ``flash_ps`` returns the ideal outlet state.
    The temperature is bracketed to ``[t_min, t_max]``.
    """
    z = jnp.asarray(z)
    n = z.shape[0]
    kij_arr = jnp.zeros((n, n)) if kij is None else jnp.asarray(kij)
    params = (jnp.asarray(p, dtype=float), jnp.asarray(s_spec, dtype=float), z, tc, pc, omega)

    def residual(t: Array, params: Any) -> Array:
        p_, s_, z_, tc_, pc_, omega_ = params
        return mixture_entropy(eos, t, p_, z_, tc_, pc_, omega_, cp, kij=kij_arr) - s_

    t_star = _implicit_temperature(residual, params, float(t_init), t_min, t_max, tol, max_iter)
    r = flash_pt(eos, t_star, p, z, tc, pc, omega, kij=kij_arr)
    return EnergyFlashResult(t=t_star, beta=r.beta, x=r.x, y=r.y, k=r.k)
