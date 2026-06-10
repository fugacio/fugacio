"""Density solves ``rho(T, P)`` for a reference Helmholtz EOS.

A Helmholtz EOS is explicit in ``(rho, T)``, so going from the process-side
specification ``(T, P)`` requires a root solve of ``P(rho, T) = P``. Below the
critical temperature that equation has up to three roots (liquid, unstable
middle, vapor); the solver picks the physical branch by *initialization* --
the saturated-liquid ancillary seeds the liquid branch and the ideal gas seeds
the vapor branch -- exactly the strategy reference implementations use. Above
the critical temperature the isotherm is monotonic and a bracketed bisection
is unconditionally robust.

All solves run in ``ln(delta)`` (scale-free, positivity-safe) through the
implicit-differentiation helpers of :mod:`fugacio.thermo.implicit`, so the
returned density carries exact gradients with respect to ``T``, ``P``, and the
EOS coefficients regardless of iteration count. The public solve is
jit-compiled with the fluid as a pytree argument (eager re-tracing of solver
loops would dwarf the numerical work); the first call per fluid pays a
one-time compilation.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.helmholtz.fluids import HelmholtzFluid
from fugacio.thermo.helmholtz.props import pressure
from fugacio.thermo.helmholtz.saturation import rho_liquid_ancillary
from fugacio.thermo.implicit import bracketed_root, newton_root

ArrayLike = Array | float


def log_delta_residual(x: Array, params: tuple[HelmholtzFluid, Array, Array]) -> Array:
    """Scaled pressure residual ``P(exp(x) * rho_reducing, T)/P_target - 1``."""
    fluid, t, p = params
    rho = jnp.exp(x) * fluid.rho_reducing
    return pressure(fluid, rho, t) / p - 1.0


def _newton_density(
    fluid: HelmholtzFluid, t: ArrayLike, p: ArrayLike, rho_init: ArrayLike
) -> Array:
    t = jnp.asarray(t, dtype=float)
    p = jnp.asarray(p, dtype=float)
    x0 = jnp.log(jnp.asarray(rho_init, dtype=float) / fluid.rho_reducing)
    x_star = newton_root(log_delta_residual, (fluid, t, p), x0, 1e-13, 80)
    return jnp.exp(x_star) * fluid.rho_reducing


def _bracketed_density(fluid: HelmholtzFluid, t: ArrayLike, p: ArrayLike) -> Array:
    t = jnp.asarray(t, dtype=float)
    p = jnp.asarray(p, dtype=float)
    # P -> 0 as delta -> 0 and P(rho_max) exceeds any sane target, so the
    # residual always changes sign across the bracket. On supercritical
    # isotherms (the intended use) the root is unique.
    lo = jnp.log(jnp.asarray(1e-10))
    hi = jnp.log(jnp.asarray(fluid.rho_max / fluid.rho_reducing))
    x_star = bracketed_root(log_delta_residual, (fluid, t, p), lo, hi, 1e-14, 300)
    return jnp.exp(x_star) * fluid.rho_reducing


def vapor_density_guess(fluid: HelmholtzFluid, t: ArrayLike, p: ArrayLike) -> Array:
    """Ideal-gas molar density (mol/m^3), the vapor-branch seed."""
    return jnp.asarray(p, dtype=float) / (fluid.gas_constant * jnp.asarray(t, dtype=float))


def liquid_density_guess(fluid: HelmholtzFluid, t: ArrayLike) -> Array:
    """Saturated-liquid ancillary density (mol/m^3), the liquid-branch seed.

    Clipped to the subcritical range where the ancillary is defined; for
    ``t >= t_critical`` the isotherm has a single root anyway, so the seed only
    needs to be on the dense side.
    """
    t_anc = jnp.clip(jnp.asarray(t, dtype=float), fluid.t_triple, 0.9999 * fluid.t_critical)
    return rho_liquid_ancillary(fluid, t_anc)


@partial(jax.jit, static_argnames=("phase",))
def _molar_density(fluid: HelmholtzFluid, t: Array, p: Array, phase: str) -> Array:
    if phase == "vapor":
        return _newton_density(fluid, t, p, vapor_density_guess(fluid, t, p))
    if phase == "liquid":
        return _newton_density(fluid, t, p, liquid_density_guess(fluid, t))
    return _bracketed_density(fluid, t, p)


def molar_density(
    fluid: HelmholtzFluid, t: ArrayLike, p: ArrayLike, *, phase: str = "supercritical"
) -> Array:
    """Molar density ``rho(T, P)`` (mol/m^3) on the requested branch.

    Args:
        fluid: Reference EOS.
        t: Temperature (K).
        p: Pressure (Pa).
        phase: ``"liquid"`` or ``"vapor"`` select the subcritical branch by
            seeding Newton from the saturated-liquid ancillary or the ideal
            gas; ``"supercritical"`` runs a bracketed bisection over the whole
            density range (robust whenever the isotherm is monotonic, i.e.
            above ``t_critical``).

    Returns:
        The converged molar density; differentiable in ``t``, ``p``, and the
        EOS coefficients via the implicit function theorem. For states near
        the critical point (within ~0.1% of ``t_critical``) prefer the
        bracketed branch.
    """
    if phase not in ("liquid", "vapor", "supercritical"):
        raise ValueError(f"unknown phase {phase!r}; use 'liquid', 'vapor' or 'supercritical'")
    return _molar_density(fluid, jnp.asarray(t, dtype=float), jnp.asarray(p, dtype=float), phase)


__all__ = [
    "liquid_density_guess",
    "molar_density",
    "vapor_density_guess",
]
