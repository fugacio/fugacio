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
* `gibbs_helmholtz_residual`: ``H = -T^2 (d(G/T)/dT)_P`` on one branch of a
  property package, which ties its enthalpy to its entropy.
* `fugacity_enthalpy_residual`: ``H^V - H^L = -R T^2 d/dT sum_i x_i (ln phi_i^V -
  ln phi_i^L)`` for a property package, which ties the fugacity coefficients
  that decide phase equilibrium to the enthalpies that close energy balances.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import R
from fugacio.thermo.eos import CubicEOS, ln_phi_mixture, ln_phi_pure

if TYPE_CHECKING:
    from fugacio.thermo.package import PropertyPackage

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


def gibbs_helmholtz_residual(
    pkg: PropertyPackage, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str
) -> Array:
    """Relative residual of ``H = -T^2 (d(G/T)/dT)_{P,x}`` on one branch of a package.

    ``G = H - T S`` is built from the package's own enthalpy and entropy, so a
    nonzero residual means the two disagree (``(dS/dT)_P != C_P / T``).
    """
    t = jnp.asarray(t, dtype=float)
    h = pkg.enthalpy(t, p, x, phase=phase)

    def g_over_t(tt: Array) -> Array:
        g = pkg.enthalpy(tt, p, x, phase=phase) - tt * pkg.entropy(tt, p, x, phase=phase)
        return g / tt

    slope = jax.grad(g_over_t)(t)
    return jnp.abs(-(t**2) * slope - h) / jnp.maximum(jnp.abs(h), 1.0)


def fugacity_enthalpy_residual(pkg: PropertyPackage, t: ArrayLike, p: ArrayLike, x: Array) -> Array:
    """Relative residual of ``H^V - H^L = -R T^2 d/dT sum_i x_i (ln phi_i^V - ln phi_i^L)``.

    Evaluate at a ``(T, P, x)`` where both branches exist. A reference fluid
    uses the gas constant of its own formulation.
    """
    t = jnp.asarray(t, dtype=float)
    x = jnp.asarray(x, dtype=float)
    gas_constant = getattr(getattr(pkg, "fluid", None), "gas_constant", R)
    dh = pkg.enthalpy(t, p, x, phase="vapor") - pkg.enthalpy(t, p, x, phase="liquid")

    def ln_phi_gap(tt: Array) -> Array:
        gap = pkg.ln_phi(tt, p, x, phase="vapor") - pkg.ln_phi(tt, p, x, phase="liquid")
        return jnp.sum(x * gap)

    dh_phi = -gas_constant * t**2 * jax.grad(ln_phi_gap)(t)
    return jnp.abs(dh - dh_phi) / jnp.abs(dh)
