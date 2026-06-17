"""Ideal-gas thermodynamic properties from heat-capacity correlations.

These functions integrate the four-parameter ``Cp/R = a + b*T + c*T**2 + d/T**2``
correlation (see `fugacio.thermo.components.CpIdeal`) into the ideal-gas
enthalpy, entropy, and Gibbs energy. They are the temperature-dependent backbone
that the equation-of-state *departure functions* are added to in order to obtain
real-fluid properties::

    H_real(T, P) = H_ideal(T) + H_departure(T, P)

Everything is written in `jax.numpy`, so the coefficients ``a, b, c, d`` may
themselves be differentiated through (useful when regressing Cp data).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.components import Component
from fugacio.thermo.constants import P_REF, T_REF, R

ArrayLike = Array | float


def cp_ig(
    t: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    e: ArrayLike = 0.0,
) -> Array:
    """Ideal-gas molar heat capacity ``Cp`` (J/mol/K).

    ``Cp = R * (a + b*T + c*T**2 + d/T**2 + e*T**3)``.
    """
    t = jnp.asarray(t)
    return R * (a + b * t + c * t**2 + d / t**2 + e * t**3)


def enthalpy_ig(
    t: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    e: ArrayLike = 0.0,
    t_ref: float = T_REF,
) -> Array:
    """Ideal-gas molar enthalpy relative to ``t_ref`` (J/mol).

    Analytic integral ``integral_{t_ref}^{t} Cp dT``.
    """
    t = jnp.asarray(t)
    return R * (
        a * (t - t_ref)
        + b / 2.0 * (t**2 - t_ref**2)
        + c / 3.0 * (t**3 - t_ref**3)
        - d * (1.0 / t - 1.0 / t_ref)
        + e / 4.0 * (t**4 - t_ref**4)
    )


def entropy_ig(
    t: ArrayLike,
    p: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    e: ArrayLike = 0.0,
    t_ref: float = T_REF,
    p_ref: float = P_REF,
) -> Array:
    """Ideal-gas molar entropy relative to ``(t_ref, p_ref)`` (J/mol/K).

    Analytic integral ``integral Cp/T dT`` minus the pressure term
    ``R * ln(P / p_ref)``.
    """
    t = jnp.asarray(t)
    p = jnp.asarray(p)
    s_temperature = R * (
        a * jnp.log(t / t_ref)
        + b * (t - t_ref)
        + c / 2.0 * (t**2 - t_ref**2)
        - d / 2.0 * (1.0 / t**2 - 1.0 / t_ref**2)
        + e / 3.0 * (t**3 - t_ref**3)
    )
    return s_temperature - R * jnp.log(p / p_ref)


def gibbs_ig(
    t: ArrayLike,
    p: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    e: ArrayLike = 0.0,
    t_ref: float = T_REF,
    p_ref: float = P_REF,
) -> Array:
    """Ideal-gas molar Gibbs energy ``G = H - T*S`` relative to the reference state."""
    h = enthalpy_ig(t, a, b, c, d, e, t_ref=t_ref)
    s = entropy_ig(t, p, a, b, c, d, e, t_ref=t_ref, p_ref=p_ref)
    return h - jnp.asarray(t) * s


def enthalpy_ig_mixture(
    t: ArrayLike,
    x: Array,
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    e: Array,
    t_ref: float = T_REF,
) -> Array:
    """Ideal-gas molar enthalpy of a mixture relative to ``t_ref`` (J/mol).

    Ideal-gas enthalpy is independent of pressure and of mixing, so the mixture
    value is just the mole-fraction average of the component enthalpies. ``a..e``
    are 1-D coefficient arrays aligned with ``x`` (see `ideal_gas_coeffs`).
    """
    x = jnp.asarray(x)
    return jnp.sum(x * enthalpy_ig(t, a, b, c, d, e, t_ref=t_ref))


def entropy_ig_mixture(
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    e: Array,
    t_ref: float = T_REF,
    p_ref: float = P_REF,
) -> Array:
    """Ideal-gas molar entropy of a mixture relative to ``(t_ref, p_ref)`` (J/mol/K).

    Includes the ideal entropy of mixing ``-R sum_i x_i ln x_i`` (equivalently,
    each component is evaluated at its partial pressure ``x_i P``).
    """
    x = jnp.asarray(x)
    pure = jnp.sum(x * entropy_ig(t, p, a, b, c, d, e, t_ref=t_ref, p_ref=p_ref))
    x_ln_x = jnp.where(x > 0.0, x * jnp.log(jnp.where(x > 0.0, x, 1.0)), 0.0)
    return pure - R * jnp.sum(x_ln_x)


def gibbs_ig_mixture(
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    e: Array,
    t_ref: float = T_REF,
    p_ref: float = P_REF,
) -> Array:
    """Ideal-gas molar Gibbs energy of a mixture ``G = H - T S`` (J/mol)."""
    h = enthalpy_ig_mixture(t, x, a, b, c, d, e, t_ref=t_ref)
    s = entropy_ig_mixture(t, p, x, a, b, c, d, e, t_ref=t_ref, p_ref=p_ref)
    return h - jnp.asarray(t) * s


def ideal_gas_coeffs(
    components: list[Component],
) -> tuple[Array, Array, Array, Array, Array]:
    """Stack the ``Cp`` coefficients of several components into ``(a, b, c, d, e)``.

    Raises:
        ValueError: if any component lacks ideal-gas heat-capacity data.
    """
    missing = [c.name for c in components if c.cp_ig is None]
    if missing:
        raise ValueError(f"missing ideal-gas Cp data for: {missing}")
    cps = [c.cp_ig for c in components if c.cp_ig is not None]
    a = jnp.asarray([cp.a for cp in cps])
    b = jnp.asarray([cp.b for cp in cps])
    cc = jnp.asarray([cp.c for cp in cps])
    d = jnp.asarray([cp.d for cp in cps])
    e = jnp.asarray([cp.e for cp in cps])
    return a, b, cc, d, e
