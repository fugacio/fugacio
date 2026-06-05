"""Residual (departure) functions of the cubic equation of state.

A *residual* property is the difference between the real-fluid property and the
ideal-gas property at the **same temperature, pressure, and composition**::

    M_real(T, P, x) = M_ideal_gas(T, P, x) + M_residual(T, P, x)

so the residual functions here are exactly what must be added to the ideal-gas
integrals in :mod:`fugacio.thermo.ideal` to obtain real-fluid enthalpy, entropy,
and Gibbs energy. They are the keystone that turns the property engine into one
that can close *energy* balances (heat duties, adiabatic mixing, compression),
not just material balances.

The closed forms are written in terms of the same departure log-term ``g`` that
:func:`fugacio.thermo.eos.ln_phi_mixture` already uses, so they are *consistent
by construction* with the fugacity coefficients that are validated against
CoolProp. Writing ``A = aP/(RT)^2``, ``B = bP/(RT)`` and
``g = ln[(Z+sigma B)/(Z+epsilon B)] / (sigma - epsilon)``::

    G_res / (RT) = (Z - 1) - ln(Z - B) - (A / B) * g
    H_res        = R T (Z - 1) + ((T a' - a) / b) * g
    S_res        = R ln(Z - B) + (a' / b) * g

with ``a' = da/dT`` for the mixture (obtained by automatic differentiation of
the van der Waals one-fluid mixing rule, so binary interaction parameters and
any ``alpha(T)`` law are handled exactly). One can verify
``G_res = H_res - T S_res`` and ``H_res = G_res - T (dG_res/dT)_P`` directly;
:mod:`fugacio.thermo.tests` turns those identities into graded consistency
checks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import R
from fugacio.thermo.eos import CubicEOS, _ab_mixture, _departure_g, compress_factor

ArrayLike = Array | float


class ResidualProperties(NamedTuple):
    """Residual (departure) molar properties at a given ``T, P, x`` and phase.

    Attributes:
        gibbs: Residual molar Gibbs energy ``G - G_ig`` (J/mol).
        enthalpy: Residual molar enthalpy ``H - H_ig`` (J/mol).
        entropy: Residual molar entropy ``S - S_ig`` (J/mol/K).
        z: Compressibility factor of the selected phase root.
    """

    gibbs: Array
    enthalpy: Array
    entropy: Array
    z: Array


def _kij_matrix(n: int, kij: Array | None) -> Array:
    return jnp.zeros((n, n)) if kij is None else jnp.asarray(kij)


def _a_mixture(
    eos: CubicEOS,
    t: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    kij: Array,
) -> Array:
    """Scalar mixture attractive parameter ``a_mix(T)`` (J m^3 / mol^2)."""
    # Pressure is irrelevant to ``a_mix``; pass a dummy 1 bar so we can reuse the
    # validated ``_ab_mixture`` mixing rule and differentiate it in ``T``.
    _, _, _, a_mix, _, _ = _ab_mixture(eos, t, 1.0e5, x, tc, pc, omega, kij)
    return a_mix


def _da_dt(
    eos: CubicEOS,
    t: Array,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    kij: Array,
) -> Array:
    """Temperature derivative ``da_mix/dT`` by automatic differentiation."""
    return jax.grad(lambda tt: _a_mixture(eos, tt, x, tc, pc, omega, kij))(t)


def residual_properties(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    phase: str = "vapor",
    kij: Array | None = None,
) -> ResidualProperties:
    """All residual molar properties (G, H, S) at fixed ``T, P, x`` in one pass.

    Computing them together shares the compressibility root, the departure
    log-term, and ``da/dT``, which the individual accessors below simply select
    from. The Gibbs residual equals ``R T sum_i x_i ln(phi_i)`` -- the
    partial-molar identity tying this module to
    :func:`fugacio.thermo.eos.ln_phi_mixture`.
    """
    t = jnp.asarray(t, dtype=float)
    p = jnp.asarray(p, dtype=float)
    x = jnp.asarray(x)
    kij_arr = _kij_matrix(x.shape[0], kij)
    big_a, big_b, _, a_mix, b_mix, _ = _ab_mixture(eos, t, p, x, tc, pc, omega, kij_arr)
    z = compress_factor(big_a, big_b, eos.u, eos.w, phase == "vapor")
    g = _departure_g(z, big_b, eos)
    da_dt = _da_dt(eos, t, x, tc, pc, omega, kij_arr)
    gibbs = R * t * ((z - 1.0) - jnp.log(z - big_b) - (big_a / big_b) * g)
    enthalpy = R * t * (z - 1.0) + ((t * da_dt - a_mix) / b_mix) * g
    entropy = R * jnp.log(z - big_b) + (da_dt / b_mix) * g
    return ResidualProperties(gibbs=gibbs, enthalpy=enthalpy, entropy=entropy, z=z)


def residual_gibbs(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    phase: str = "vapor",
    kij: Array | None = None,
) -> Array:
    """Residual molar Gibbs energy ``G - G_ig`` at fixed ``T, P`` (J/mol)."""
    return residual_properties(eos, t, p, x, tc, pc, omega, phase=phase, kij=kij).gibbs


def residual_enthalpy(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    phase: str = "vapor",
    kij: Array | None = None,
) -> Array:
    """Residual molar enthalpy ``H - H_ig`` at fixed ``T, P`` (J/mol)."""
    return residual_properties(eos, t, p, x, tc, pc, omega, phase=phase, kij=kij).enthalpy


def residual_entropy(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    phase: str = "vapor",
    kij: Array | None = None,
) -> Array:
    """Residual molar entropy ``S - S_ig`` at fixed ``T, P`` (J/mol/K)."""
    return residual_properties(eos, t, p, x, tc, pc, omega, phase=phase, kij=kij).entropy


def residual_cp(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    phase: str = "vapor",
    kij: Array | None = None,
) -> Array:
    """Residual molar heat capacity ``Cp - Cp_ig`` at fixed ``T, P`` (J/mol/K).

    Obtained as ``(d H_res / dT)_P`` by automatic differentiation of
    :func:`residual_enthalpy`, so it needs no separately derived second-derivative
    formula and stays consistent with the enthalpy departure.
    """
    t = jnp.asarray(t, dtype=float)

    def h_of_t(tt: Array) -> Array:
        return residual_enthalpy(eos, tt, p, x, tc, pc, omega, phase=phase, kij=kij)

    return jax.grad(h_of_t)(t)
