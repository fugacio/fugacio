"""Vapor-liquid equilibrium for binary mixtures (modified Raoult's law).

A deliberately small but real vertical slice that sits on top of
:mod:`fugacio.thermo`: it combines Antoine vapor pressures with Margules
activity coefficients to compute a bubble-point pressure. Everything is
differentiable end-to-end with respect to composition, temperature, and the
activity-model parameters.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo import margules_gamma

ArrayLike = Array | float
Antoine = tuple[float, float, float]


def antoine_psat(temperature: ArrayLike, a: ArrayLike, b: ArrayLike, c: ArrayLike) -> Array:
    """Saturation pressure from the base-10 Antoine equation.

    ``log10(Psat) = a - b / (temperature + c)``

    Pressure and temperature units follow whatever convention the Antoine
    constants were fit in; Fugacio does not impose one here.
    """
    return jnp.power(10.0, a - b / (jnp.asarray(temperature) + c))


def bubble_pressure(
    x1: ArrayLike,
    temperature: ArrayLike,
    antoine1: Antoine,
    antoine2: Antoine,
    a12: ArrayLike = 0.0,
    a21: ArrayLike = 0.0,
) -> tuple[Array, Array]:
    """Bubble-point pressure and vapor composition for a binary at fixed ``T``, ``x``.

    Uses modified Raoult's law ``p_i = x_i * gamma_i * Psat_i`` with two-parameter
    Margules activity coefficients. Setting ``a12 = a21 = 0`` recovers ideal
    (Raoult's law) behaviour.

    Returns:
        ``(pressure, y1)``, where ``y1`` is the vapor mole fraction of component 1.
    """
    x1 = jnp.asarray(x1)
    x2 = 1.0 - x1
    gamma1, gamma2 = margules_gamma(x1, a12, a21)
    psat1 = antoine_psat(temperature, *antoine1)
    psat2 = antoine_psat(temperature, *antoine2)
    p1 = x1 * gamma1 * psat1
    p2 = x2 * gamma2 * psat2
    pressure = p1 + p2
    y1 = p1 / pressure
    return pressure, y1
