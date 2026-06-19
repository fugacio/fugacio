"""PC-SAFT properties at ``(rho, T, x)`` and ``(T, P, x)`` from ``alpha_res``.

Every function here is an algebraic combination of the autodiff partials of the
reduced residual Helmholtz energy `fugacio.thermo.saft.pcsaft.alpha_residual`,
exactly mirroring the reference-fluid property layer
(`fugacio.thermo.helmholtz.props`). The compressibility factor follows from
the density derivative,

    Z = 1 + rho (d alpha_res / d rho)_{T,x},

the mixture fugacity coefficients from the mole-number gradient of the total
residual Helmholtz energy at fixed ``(T, V)``,

    ln phi_i = (d (n alpha_res) / d n_i)_{T,V} - ln Z,

and the residual (departure) enthalpy/entropy/heat capacity from the temperature
derivatives. The ``(T, P)`` entry points first solve ``P(rho, T, x) = P`` for the
molar density on the requested phase branch (seeded from the ideal gas for the
vapour and from a dense packing fraction for the liquid) through the
implicit-differentiation helpers of `fugacio.thermo.implicit`, so the returned
density, and everything built on it, carries exact gradients in ``T``, ``P``,
composition, and the PC-SAFT parameters.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import N_A, R
from fugacio.thermo.implicit import bracketed_root, newton_root
from fugacio.thermo.saft.parameters import SaftParameters, segment_diameter
from fugacio.thermo.saft.pcsaft import alpha_residual

ArrayLike = Array | float

#: Packing-fraction ceiling used to bracket the density solve (face-centred close packing).
_ETA_MAX = 0.7405


def compressibility_factor(params: SaftParameters, rho: ArrayLike, t: ArrayLike, x: Array) -> Array:
    """Compressibility factor ``Z = 1 + rho (d alpha_res / d rho)`` at ``(rho, T, x)``."""
    rho = jnp.asarray(rho, dtype=float)
    dalpha = jax.grad(lambda r: alpha_residual(params, r, t, x))(rho)
    return 1.0 + rho * dalpha


def pressure(params: SaftParameters, rho: ArrayLike, t: ArrayLike, x: Array) -> Array:
    """Pressure (Pa) at molar density ``rho`` (mol/m^3), temperature ``t`` (K), comp. ``x``."""
    rho = jnp.asarray(rho, dtype=float)
    t = jnp.asarray(t, dtype=float)
    return compressibility_factor(params, rho, t, x) * rho * R * t


def _packing_density(params: SaftParameters, t: ArrayLike, x: Array, eta: float) -> Array:
    """Molar density (mol/m^3) at a target packing fraction ``eta``."""
    d = segment_diameter(params, t)
    x = jnp.asarray(x, dtype=float)
    denom = (jnp.pi / 6.0) * N_A * jnp.sum(x * params.m * d**3)
    return eta / denom


def _log_rho_residual(ln_rho: Array, params_t_p_x: tuple) -> Array:
    params, t, p, x = params_t_p_x
    return pressure(params, jnp.exp(ln_rho), t, x) / p - 1.0


@partial(jax.jit, static_argnames=("phase",))
def _molar_density(params: SaftParameters, t: Array, p: Array, x: Array, phase: str) -> Array:
    if phase == "vapor":
        rho0 = p / (R * t)
        ln_rho = newton_root(_log_rho_residual, (params, t, p, x), jnp.log(rho0), 1e-13, 100)
        return jnp.exp(ln_rho)
    if phase == "liquid":
        rho0 = _packing_density(params, t, x, 0.5)
        ln_rho = newton_root(_log_rho_residual, (params, t, p, x), jnp.log(rho0), 1e-13, 100)
        return jnp.exp(ln_rho)
    lo = jnp.log(_packing_density(params, t, x, 1e-10))
    hi = jnp.log(_packing_density(params, t, x, _ETA_MAX))
    ln_rho = bracketed_root(_log_rho_residual, (params, t, p, x), lo, hi, 1e-14, 300)
    return jnp.exp(ln_rho)


def molar_density(
    params: SaftParameters, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str = "liquid"
) -> Array:
    """Molar density ``rho(T, P, x)`` (mol/m^3) on the requested phase branch.

    Args:
        params: PC-SAFT parameter set.
        t: Temperature (K).
        p: Pressure (Pa).
        x: Mole fractions, shape ``(n,)``.
        phase: ``"liquid"`` or ``"vapor"`` seed the Newton solve from a dense
            packing fraction or the ideal gas; ``"stable"`` runs a bracketed
            bisection over the whole density range and returns the root with the
            lower molar Gibbs energy when more than one branch exists.

    Returns:
        The converged molar density, differentiable in ``T``, ``P``, ``x``, and
        the PC-SAFT parameters.
    """
    if phase not in ("liquid", "vapor", "stable"):
        raise ValueError(f"unknown phase {phase!r}; use 'liquid', 'vapor' or 'stable'")
    t = jnp.asarray(t, dtype=float)
    p = jnp.asarray(p, dtype=float)
    x = jnp.asarray(x, dtype=float)
    if phase == "stable":
        rho_l = _molar_density(params, t, p, x, "liquid")
        rho_v = _molar_density(params, t, p, x, "vapor")
        g_l = _branch_gibbs(params, rho_l, t, p, x)
        g_v = _branch_gibbs(params, rho_v, t, p, x)
        # Both branches seed a Newton solve that can miss (or diverge from) a
        # root that does not exist on this isotherm (e.g. no vapour root deep in
        # the compressed liquid). ``_branch_gibbs`` returns +inf for such a
        # spurious root, so the surviving real branch is always selected.
        return jnp.where(g_l <= g_v, rho_l, rho_v)
    return _molar_density(params, t, p, x, phase)


def _reduced_gibbs(params: SaftParameters, rho: Array, t: Array, p: Array, x: Array) -> Array:
    """Residual molar Gibbs energy over ``RT`` at a given density (for branch choice)."""
    z = p / (rho * R * t)
    a = alpha_residual(params, rho, t, x)
    return a + (z - 1.0) - jnp.log(z)


def _branch_gibbs(params: SaftParameters, rho: Array, t: Array, p: Array, x: Array) -> Array:
    """Residual molar Gibbs over ``RT`` for branch selection, +inf if ``rho`` is spurious.

    A density-root candidate is only valid if it is finite, positive, and
    actually satisfies ``P(rho, T, x) = P`` (a Newton solve seeded for a
    non-existent branch can return either a non-finite value or a point off the
    isotherm). Invalid candidates are scored ``+inf`` so they are never chosen.
    """
    valid = jnp.isfinite(rho) & (rho > 0.0)
    rho_safe = jnp.where(valid, rho, 1.0)  # keep the residual computation NaN-free
    residual_ok = jnp.abs(pressure(params, rho_safe, t, x) / p - 1.0) < 1e-6
    return jnp.where(valid & residual_ok, _reduced_gibbs(params, rho_safe, t, p, x), jnp.inf)


def ln_fugacity_coefficients(
    params: SaftParameters, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str = "liquid"
) -> Array:
    """Log fugacity coefficients ``ln phi_i`` of every component at ``(T, P, x)``.

    Args:
        params: PC-SAFT parameter set.
        t: Temperature (K).
        p: Pressure (Pa).
        x: Mole fractions, shape ``(n,)``.
        phase: Density branch (see `molar_density`).

    Returns:
        The vector of ``ln phi_i``, differentiable in ``(T, P, x)`` and the
        PC-SAFT parameters.
    """
    t = jnp.asarray(t, dtype=float)
    p = jnp.asarray(p, dtype=float)
    x = jnp.asarray(x, dtype=float)
    rho = molar_density(params, t, p, x, phase=phase)
    z = p / (rho * R * t)
    volume = 1.0 / rho  # volume holding exactly one mole of mixture

    def total_residual(n: Array) -> Array:
        n_tot = jnp.sum(n)
        return n_tot * alpha_residual(params, n_tot / volume, t, n / n_tot)

    mu_residual = jax.grad(total_residual)(x)
    return mu_residual - jnp.log(z)


class ResidualProperties(NamedTuple):
    """Residual (departure) molar properties at ``(T, P)`` relative to the ideal gas.

    Attributes:
        z: Compressibility factor.
        enthalpy: Residual molar enthalpy ``H - H^ig`` (J/mol).
        entropy: Residual molar entropy ``S - S^ig`` (J/mol/K).
        gibbs: Residual molar Gibbs energy ``G - G^ig`` (J/mol).
        cp: Residual molar isobaric heat capacity ``cp - cp^ig`` (J/mol/K).
    """

    z: Array
    enthalpy: Array
    entropy: Array
    gibbs: Array
    cp: Array


def residual_properties(
    params: SaftParameters, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str = "liquid"
) -> ResidualProperties:
    """All residual molar properties at ``(T, P, x)`` on a phase branch.

    Args:
        params: PC-SAFT parameter set.
        t: Temperature (K).
        p: Pressure (Pa).
        x: Mole fractions, shape ``(n,)``.
        phase: Density branch (see `molar_density`).

    Returns:
        A `ResidualProperties` of ``Z`` and the enthalpy/entropy/Gibbs/cp
        departures, each differentiable in ``(T, P, x)`` and the parameters.
    """
    t = jnp.asarray(t, dtype=float)
    p = jnp.asarray(p, dtype=float)
    x = jnp.asarray(x, dtype=float)
    rho = molar_density(params, t, p, x, phase=phase)
    z = p / (rho * R * t)
    ln_z = jnp.log(z)

    a = alpha_residual(params, rho, t, x)
    a_t = jax.grad(lambda tt: alpha_residual(params, rho, tt, x))(t)
    a_tt = jax.grad(jax.grad(lambda tt: alpha_residual(params, rho, tt, x)))(t)

    enthalpy = R * t * (-t * a_t + (z - 1.0))
    entropy = R * (-a - t * a_t + ln_z)
    gibbs = R * t * (a + (z - 1.0) - ln_z)

    cv_res = -R * (2.0 * t * a_t + t**2 * a_tt)
    dp_dt = jax.grad(lambda tt: pressure(params, rho, tt, x))(t)
    dp_drho = jax.grad(lambda rr: pressure(params, rr, t, x))(rho)
    cp_res = cv_res - R + t * dp_dt**2 / (rho**2 * dp_drho)

    return ResidualProperties(z=z, enthalpy=enthalpy, entropy=entropy, gibbs=gibbs, cp=cp_res)


__all__ = [
    "ResidualProperties",
    "compressibility_factor",
    "ln_fugacity_coefficients",
    "molar_density",
    "pressure",
    "residual_properties",
]
