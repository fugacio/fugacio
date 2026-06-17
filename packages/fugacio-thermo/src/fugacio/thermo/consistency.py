"""First-principles thermodynamic consistency laws (data-free oracles).

These checks need no external reference data: they encode laws that *any* correct
model must obey, and return a residual that should be zero (to numerical
precision). They are the backbone of the README's "executable acceptance
harness": thousands of small graded checks that anchor correctness as the
engine grows.

Implemented laws:

* `partial_molar_symmetry_residual`: the Gibbs-Duhem relation, expressed
  as the symmetry of the Hessian of ``n_T g^E`` (equivalently of the Jacobian of
  ``ln gamma_i``, or ``ln phi_i``, with respect to mole numbers). This holds
  at constant ``T, P`` for any model derived from a single Gibbs-energy surface.
* `equifugacity_residual`: equality of component fugacities between phases
  at equilibrium, ``x_i phi_i^L = y_i phi_i^V``.
* `fugacity_pressure_residual`: the pure-fluid identity
  ``(d ln phi / dP)_T = (Z - 1) / P``, a direct consequence of ``dG = V dP``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.eos import CubicEOS, ln_phi_mixture, ln_phi_pure

ArrayLike = Array | float
LogActivityFn = Callable[[Array], Array]


def partial_molar_symmetry_residual(ln_coeff_fn: LogActivityFn, x: Array) -> Array:
    """Gibbs-Duhem residual for a vector of log partial-molar coefficients.

    Given ``ln_coeff_fn`` mapping mole fractions to ``ln(gamma_i)`` (activity) or
    ``ln(phi_i)`` (fugacity, at fixed ``T, P``), this forms the Jacobian with
    respect to *mole numbers* and returns the max-norm of its antisymmetric part.
    A thermodynamically consistent model gives zero (to machine precision),
    because the coefficients are first derivatives of one scalar potential.
    """
    x = jnp.asarray(x, dtype=float)

    def in_mole_numbers(n: Array) -> Array:
        return ln_coeff_fn(n / jnp.sum(n))

    jac = jax.jacobian(in_mole_numbers)(x)
    return jnp.max(jnp.abs(jac - jac.T))


def gibbs_duhem_residual(ln_gamma_fn: LogActivityFn, x: Array) -> Array:
    """Gibbs-Duhem residual for an activity-coefficient model (see module docs)."""
    return partial_molar_symmetry_residual(ln_gamma_fn, x)


def equifugacity_residual(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    kij: Array | None = None,
) -> Array:
    """Max equilibrium fugacity mismatch ``|ln(x_i phi_i^L) - ln(y_i phi_i^V)|``."""
    ln_phi_l, _ = ln_phi_mixture(eos, t, p, x, tc, pc, omega, phase="liquid", kij=kij)
    ln_phi_v, _ = ln_phi_mixture(eos, t, p, y, tc, pc, omega, phase="vapor", kij=kij)
    return jnp.max(jnp.abs((jnp.log(x) + ln_phi_l) - (jnp.log(y) + ln_phi_v)))


def fugacity_pressure_residual(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    tc: ArrayLike,
    pc: ArrayLike,
    omega: ArrayLike,
    *,
    phase: str = "vapor",
) -> Array:
    """Residual of the pure-fluid identity ``(d ln phi / dP)_T = (Z - 1) / P``."""

    def ln_phi_of_p(pp: Array) -> Array:
        value, _ = ln_phi_pure(eos, t, pp, tc, pc, omega, phase=phase)
        return value

    d_ln_phi_dp = jax.grad(ln_phi_of_p)(jnp.asarray(p, dtype=float))
    _, z = ln_phi_pure(eos, t, p, tc, pc, omega, phase=phase)
    return jnp.abs(d_ln_phi_dp - (z - 1.0) / jnp.asarray(p))
