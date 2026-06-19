"""Differentiable PC-SAFT parameter estimation.

Because every PC-SAFT property is differentiable with respect to the model
parameters (`fugacio.thermo.saft.parameters`), fitting them to data is plain
gradient-based least squares, reusing the `levenberg_marquardt` optimiser of
`fugacio.thermo.regression`. Two fitters are provided:

* `fit_saft_pure` regresses the three pure-component parameters
  (``m``, ``sigma``, ``epsilon``) to saturated-vapour-pressure and
  saturated-liquid-density data, the standard PC-SAFT pure fit; and
* `fit_saft_kij` regresses a single binary correction ``k_ij`` to isothermal
  bubble-pressure data.

Both differentiate straight through the saturation / density / bubble-point
solvers, so there are no finite-difference parameter sweeps anywhere.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.regression import levenberg_marquardt
from fugacio.thermo.saft.equilibrium import bubble_pressure_saft, psat_saft
from fugacio.thermo.saft.parameters import ANGSTROM, SaftParameters
from fugacio.thermo.saft.properties import molar_density

ArrayLike = Array | float


def fit_saft_pure(
    base: SaftParameters,
    t: Array,
    psat_exp: Array,
    rho_liquid_exp: Array,
    *,
    density_weight: float = 1.0,
    max_iter: int = 60,
) -> tuple[SaftParameters, Array]:
    """Fit pure ``(m, sigma, epsilon)`` to saturation pressure and liquid density.

    Args:
        base: Single-component PC-SAFT parameter set; its current values seed the
            fit and any association parameters are held fixed.
        t: Temperatures of the data points (K), shape ``(k,)``.
        psat_exp: Experimental saturation pressures (Pa), shape ``(k,)``.
        rho_liquid_exp: Experimental saturated-liquid molar densities (mol/m^3),
            shape ``(k,)``.
        density_weight: Relative weight of the density residuals against the
            pressure residuals.
        max_iter: Levenberg-Marquardt iteration cap.

    Returns:
        ``(params, cost)``: the fitted single-component `SaftParameters` and the
        final half-sum-of-squares cost.
    """
    t = jnp.asarray(t, dtype=float)
    psat_exp = jnp.asarray(psat_exp, dtype=float)
    rho_liquid_exp = jnp.asarray(rho_liquid_exp, dtype=float)
    x = jnp.ones(1)

    theta0 = {
        "m": base.m,
        "sigma_a": base.sigma / ANGSTROM,
        "epsilon": base.epsilon,
    }

    def make(theta: dict[str, Array]) -> SaftParameters:
        return replace(
            base, m=theta["m"], sigma=theta["sigma_a"] * ANGSTROM, epsilon=theta["epsilon"]
        )

    def residual(theta: dict[str, Array]) -> Array:
        params = make(theta)

        def one(ti: Array, pi: Array, ri: Array) -> Array:
            p_calc = psat_saft(params, ti, pi)
            rho_calc = molar_density(params, ti, p_calc, x, phase="liquid")
            return jnp.array([p_calc / pi - 1.0, density_weight * (rho_calc / ri - 1.0)])

        rows = jax.vmap(one)(t, psat_exp, rho_liquid_exp)
        return rows.reshape(-1)

    theta, cost = levenberg_marquardt(residual, theta0, max_iter=max_iter)
    return make(theta), cost


def fit_saft_kij(
    base: SaftParameters,
    t: ArrayLike,
    x: Array,
    p_exp: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    max_iter: int = 40,
) -> tuple[Array, Array]:
    """Fit a single binary correction ``k_ij`` to isothermal bubble-pressure data.

    Args:
        base: Binary PC-SAFT parameter set (its ``kij`` is overwritten by the fit).
        t: Temperature of the isotherm (K).
        x: Liquid mole fractions of component 0 at each point, shape ``(k,)``;
            the second component is ``1 - x``.
        p_exp: Measured bubble pressures (Pa), shape ``(k,)``.
        tc: Critical temperatures of the two components, K (for Wilson seeding).
        pc: Critical pressures of the two components, Pa (for Wilson seeding).
        omega: Acentric factors of the two components (for Wilson seeding).
        max_iter: Levenberg-Marquardt iteration cap.

    Returns:
        ``(kij, cost)``: the fitted scalar binary correction and the final cost.
    """
    t = jnp.asarray(t, dtype=float)
    x = jnp.asarray(x, dtype=float)
    p_exp = jnp.asarray(p_exp, dtype=float)
    tc = jnp.asarray(tc, dtype=float)
    pc = jnp.asarray(pc, dtype=float)
    omega = jnp.asarray(omega, dtype=float)

    def residual(theta: dict[str, Array]) -> Array:
        kij_value = theta["kij"]
        kij = jnp.array([[0.0, kij_value], [kij_value, 0.0]])
        params = replace(base, kij=kij)

        def one(xi: Array, pi: Array) -> Array:
            comp = jnp.array([xi, 1.0 - xi])
            p_calc, _ = bubble_pressure_saft(params, t, comp, tc, pc, omega)
            return p_calc / pi - 1.0

        return jax.vmap(one)(x, p_exp)

    theta, cost = levenberg_marquardt(residual, {"kij": jnp.asarray(0.0)}, max_iter=max_iter)
    return theta["kij"], cost


__all__ = ["fit_saft_kij", "fit_saft_pure"]
