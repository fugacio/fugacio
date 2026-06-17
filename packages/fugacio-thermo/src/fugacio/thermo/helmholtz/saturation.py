"""Vapor-liquid saturation of a reference Helmholtz EOS (Maxwell construction).

At saturation the two coexisting densities satisfy mechanical and chemical
equilibrium:

    P(rho_liquid, T) = P(rho_vapor, T)      and
    g(rho_liquid, T) = g(rho_vapor, T),

a 2x2 root problem in ``(ln delta_liquid, ln delta_vapor)`` solved by the
damped Newton of `fugacio.thermo.implicit` and seeded by the published
saturation ancillary equations. Because the solve is wrapped in an implicit
``custom_vjp``, the saturation line is *differentiable*: ``d(psat)/dT``
computed by `jax.grad` through this solve reproduces the
Clausius-Clapeyron relation ``dP/dT = h_vap / (T dv)`` to machine precision --
one of the consistency oracles in the test suite.

``saturation_state`` evaluates the full coexistence state (densities,
enthalpies, entropies); ``saturation_temperature`` inverts the line at a given
pressure with a three-unknown Newton seeded by an ancillary bisection.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.helmholtz.fluids import Ancillary, HelmholtzFluid
from fugacio.thermo.helmholtz.props import enthalpy, entropy, gibbs_energy, pressure
from fugacio.thermo.implicit import bracketed_root, newton_system

ArrayLike = Array | float

#: Fraction of ``t_critical`` beyond which the two-density Newton is considered
#: degenerate (the Jacobian is singular at the critical point itself).
T_SAT_MAX_FRACTION = 0.99999


def _evaluate_ancillary(anc: Ancillary, t: ArrayLike) -> Array:
    """Evaluate one saturation ancillary at ``t`` (K)."""
    t = jnp.asarray(t, dtype=float)
    theta = jnp.clip(1.0 - t / anc.t_reducing, 0.0, 1.0)
    # theta**t has an unbounded theta-gradient at the critical point for
    # fractional exponents < 1; the double where keeps it finite there.
    positive = theta > 0.0
    powered = jnp.where(positive, theta, 1.0) ** anc.t
    total = jnp.sum(anc.n * jnp.where(positive, powered, 0.0))
    if anc.noexp:
        return anc.reducing * (1.0 + total)
    factor = anc.t_reducing / t if anc.using_tau_r else jnp.asarray(1.0)
    return anc.reducing * jnp.exp(factor * total)


def psat_ancillary(fluid: HelmholtzFluid, t: ArrayLike) -> Array:
    """Ancillary (initial-guess) saturation pressure (Pa)."""
    return _evaluate_ancillary(fluid.anc_psat, t)


def rho_liquid_ancillary(fluid: HelmholtzFluid, t: ArrayLike) -> Array:
    """Ancillary saturated-liquid molar density (mol/m^3)."""
    return _evaluate_ancillary(fluid.anc_rho_liquid, t)


def rho_vapor_ancillary(fluid: HelmholtzFluid, t: ArrayLike) -> Array:
    """Ancillary saturated-vapor molar density (mol/m^3)."""
    return _evaluate_ancillary(fluid.anc_rho_vapor, t)


def _coexistence_residual(x: Array, params: tuple[HelmholtzFluid, Array]) -> Array:
    """Equal-pressure / equal-Gibbs residual in ``(ln delta_liquid, ln delta_vapor)``."""
    fluid, t = params
    rho_liquid = jnp.exp(x[0]) * fluid.rho_reducing
    rho_vapor = jnp.exp(x[1]) * fluid.rho_reducing
    p_scale = psat_ancillary(fluid, t)
    r_mech = (pressure(fluid, rho_liquid, t) - pressure(fluid, rho_vapor, t)) / p_scale
    r_chem = (gibbs_energy(fluid, rho_liquid, t) - gibbs_energy(fluid, rho_vapor, t)) / (
        fluid.gas_constant * t
    )
    return jnp.stack([r_mech, r_chem])


@jax.jit
def saturation_densities(fluid: HelmholtzFluid, t: ArrayLike) -> tuple[Array, Array]:
    """Coexisting molar densities ``(rho_liquid, rho_vapor)`` at ``t`` (K).

    Valid for ``t_triple <= t < t_critical`` (callers should clip; the Newton
    Jacobian degenerates at the critical point where both densities merge).
    Differentiable in ``t`` and the EOS coefficients.
    """
    t = jnp.asarray(t, dtype=float)
    x0 = jnp.stack(
        [
            jnp.log(rho_liquid_ancillary(fluid, t) / fluid.rho_reducing),
            jnp.log(rho_vapor_ancillary(fluid, t) / fluid.rho_reducing),
        ]
    )
    x_star = newton_system(_coexistence_residual, x0, (fluid, t), 1e-12, 60)
    rho = jnp.exp(x_star) * fluid.rho_reducing
    return rho[0], rho[1]


@jax.jit
def saturation_pressure(fluid: HelmholtzFluid, t: ArrayLike) -> Array:
    """Saturation pressure (Pa) from the Maxwell construction at ``t`` (K)."""
    rho_liquid, rho_vapor = saturation_densities(fluid, t)
    return 0.5 * (pressure(fluid, rho_liquid, t) + pressure(fluid, rho_vapor, t))


def _tsat_seed(fluid: HelmholtzFluid, p: ArrayLike) -> Array:
    """Invert the *ancillary* saturation line for an initial temperature."""

    def residual(t: Array, params: tuple[HelmholtzFluid, Array]) -> Array:
        anc_fluid, target = params
        return jnp.log(psat_ancillary(anc_fluid, t) / target)

    lo = jnp.asarray(fluid.t_triple)
    hi = jnp.asarray(T_SAT_MAX_FRACTION * fluid.t_critical)
    return bracketed_root(residual, (fluid, jnp.asarray(p, dtype=float)), lo, hi, 1e-9, 200)


def _boiling_residual(x: Array, params: tuple[HelmholtzFluid, Array]) -> Array:
    """Three-unknown residual ``(ln delta_l, ln delta_v, T/t_critical)`` at fixed ``p``."""
    fluid, p = params
    rho_liquid = jnp.exp(x[0]) * fluid.rho_reducing
    rho_vapor = jnp.exp(x[1]) * fluid.rho_reducing
    t = x[2] * fluid.t_critical
    r_liquid = pressure(fluid, rho_liquid, t) / p - 1.0
    r_vapor = pressure(fluid, rho_vapor, t) / p - 1.0
    r_chem = (gibbs_energy(fluid, rho_liquid, t) - gibbs_energy(fluid, rho_vapor, t)) / (
        fluid.gas_constant * t
    )
    return jnp.stack([r_liquid, r_vapor, r_chem])


@jax.jit
def saturation_temperature(fluid: HelmholtzFluid, p: ArrayLike) -> Array:
    """Saturation (boiling) temperature (K) at pressure ``p`` (Pa).

    Valid for ``p_triple <= p < p_critical`` (callers should clip).
    Differentiable in ``p`` and the EOS coefficients.
    """
    p = jnp.asarray(p, dtype=float)
    t0 = _tsat_seed(fluid, p)
    x0 = jnp.stack(
        [
            jnp.log(rho_liquid_ancillary(fluid, t0) / fluid.rho_reducing),
            jnp.log(rho_vapor_ancillary(fluid, t0) / fluid.rho_reducing),
            t0 / fluid.t_critical,
        ]
    )
    x_star = newton_system(_boiling_residual, x0, (fluid, p), 1e-12, 60)
    return x_star[2] * fluid.t_critical


@dataclass(frozen=True)
class SaturationState:
    """The full vapor-liquid coexistence state of a pure fluid.

    Attributes:
        t: Saturation temperature (K).
        p: Saturation pressure (Pa).
        rho_liquid: Saturated-liquid molar density (mol/m^3).
        rho_vapor: Saturated-vapor molar density (mol/m^3).
        h_liquid: Saturated-liquid molar enthalpy (J/mol).
        h_vapor: Saturated-vapor molar enthalpy (J/mol).
        s_liquid: Saturated-liquid molar entropy (J/mol/K).
        s_vapor: Saturated-vapor molar entropy (J/mol/K).
        h_vaporization: Latent heat of vaporization (J/mol).
    """

    t: Array
    p: Array
    rho_liquid: Array
    rho_vapor: Array
    h_liquid: Array
    h_vapor: Array
    s_liquid: Array
    s_vapor: Array
    h_vaporization: Array


jax.tree_util.register_dataclass(
    SaturationState,
    data_fields=[
        "t",
        "p",
        "rho_liquid",
        "rho_vapor",
        "h_liquid",
        "h_vapor",
        "s_liquid",
        "s_vapor",
        "h_vaporization",
    ],
    meta_fields=[],
)


def saturation_state(
    fluid: HelmholtzFluid, *, t: ArrayLike | None = None, p: ArrayLike | None = None
) -> SaturationState:
    """The coexistence state at a given temperature *or* pressure.

    Exactly one of ``t`` (K) or ``p`` (Pa) must be supplied; the other is
    solved from the Maxwell construction. All returned fields are
    differentiable with respect to the given specification.
    """
    if (t is None) == (p is None):
        raise ValueError("specify exactly one of t or p")
    if t is None:
        t = saturation_temperature(fluid, jnp.asarray(p, dtype=float))
    return _saturation_state_at(fluid, jnp.asarray(t, dtype=float))


@jax.jit
def _saturation_state_at(fluid: HelmholtzFluid, t: Array) -> SaturationState:
    rho_liquid, rho_vapor = saturation_densities(fluid, t)
    h_liquid = enthalpy(fluid, rho_liquid, t)
    h_vapor = enthalpy(fluid, rho_vapor, t)
    return SaturationState(
        t=t,
        p=0.5 * (pressure(fluid, rho_liquid, t) + pressure(fluid, rho_vapor, t)),
        rho_liquid=rho_liquid,
        rho_vapor=rho_vapor,
        h_liquid=h_liquid,
        h_vapor=h_vapor,
        s_liquid=entropy(fluid, rho_liquid, t),
        s_vapor=entropy(fluid, rho_vapor, t),
        h_vaporization=h_vapor - h_liquid,
    )


__all__ = [
    "T_SAT_MAX_FRACTION",
    "SaturationState",
    "psat_ancillary",
    "rho_liquid_ancillary",
    "rho_vapor_ancillary",
    "saturation_densities",
    "saturation_pressure",
    "saturation_state",
    "saturation_temperature",
]
