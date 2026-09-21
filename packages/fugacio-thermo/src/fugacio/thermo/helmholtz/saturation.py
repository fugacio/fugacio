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
Clausius-Clapeyron relation ``dP/dT = h_vap / (T dv)`` to machine precision,
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

from fugacio.thermo.diagnostics import SolveResult
from fugacio.thermo.helmholtz.fluids import Ancillary, HelmholtzFluid
from fugacio.thermo.helmholtz.props import enthalpy, entropy, gibbs_energy, pressure
from fugacio.thermo.implicit import bracketed_root, newton_system_with_info

ArrayLike = Array | float

#: Fraction of ``t_critical`` beyond which the two-density Newton is considered
#: degenerate (the Jacobian is singular at the critical point itself).
T_SAT_MAX_FRACTION = 0.99999

#: Coexistence residuals subtract large Helmholtz terms, so their floating-point
#: floor can sit just above a strict tolerance. A Newton iteration that can no
#: longer reduce a residual below this is at that floor, not diverging.
_ROUNDOFF_FLOOR = 1e-9


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
    # Dense-liquid pressure subtracts large Helmholtz terms. Normalizing its
    # roundoff by a small vapor pressure demands sub-nanopascal accuracy at
    # ambient water conditions. A 1 MPa floor permits 1 microPa at tol=1e-12
    # while retaining a strict chemical-potential residual.
    p_scale = jnp.maximum(psat_ancillary(fluid, t), 1e6)
    r_mech = (pressure(fluid, rho_liquid, t) - pressure(fluid, rho_vapor, t)) / p_scale
    r_chem = (gibbs_energy(fluid, rho_liquid, t) - gibbs_energy(fluid, rho_vapor, t)) / (
        fluid.gas_constant * t
    )
    return jnp.stack([r_mech, r_chem])


@jax.jit
def saturation_densities_with_info(fluid: HelmholtzFluid, t: ArrayLike) -> SolveResult:
    """Coexisting molar densities ``[rho_liquid, rho_vapor]`` at ``t`` (K), with a report.

    Valid for ``t_triple <= t < t_critical`` (the Newton Jacobian degenerates at
    the critical point where both densities merge). A converged value is
    differentiable in ``t`` and the EOS coefficients.
    """
    t = jnp.asarray(t, dtype=float)
    x0 = jnp.stack(
        [
            jnp.log(rho_liquid_ancillary(fluid, t) / fluid.rho_reducing),
            jnp.log(rho_vapor_ancillary(fluid, t) / fluid.rho_reducing),
        ]
    )
    solved = newton_system_with_info(
        _coexistence_residual, x0, (fluid, t), 1e-12, 60, stall_tol=_ROUNDOFF_FLOOR
    )
    return SolveResult(jnp.exp(solved.value) * fluid.rho_reducing, solved.report)


def saturation_densities(fluid: HelmholtzFluid, t: ArrayLike) -> tuple[Array, Array]:
    """Best coexisting densities ``(rho_liquid, rho_vapor)`` at ``t`` (K).

    Callers should clip ``t`` into ``[t_triple, T_SAT_MAX_FRACTION * t_critical)``.
    An unconverged solve keeps its best iterate but has nonfinite derivatives;
    use `saturation_densities_with_info` to accept or reject it.
    """
    rho = saturation_densities_with_info(fluid, t).value
    return rho[0], rho[1]


@jax.jit
def saturation_pressure_with_info(fluid: HelmholtzFluid, t: ArrayLike) -> SolveResult:
    """Saturation pressure (Pa) from the Maxwell construction at ``t`` (K), with a report."""
    t = jnp.asarray(t, dtype=float)
    solved = saturation_densities_with_info(fluid, t)
    rho_liquid, rho_vapor = solved.value[0], solved.value[1]
    return SolveResult(
        0.5 * (pressure(fluid, rho_liquid, t) + pressure(fluid, rho_vapor, t)), solved.report
    )


def saturation_pressure(fluid: HelmholtzFluid, t: ArrayLike) -> Array:
    """Best saturation pressure (Pa) at ``t`` (K); see `saturation_pressure_with_info`."""
    return saturation_pressure_with_info(fluid, t).value


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
    r_liquid = (pressure(fluid, rho_liquid, t) - p) / jnp.maximum(p, 1e6)
    r_vapor = pressure(fluid, rho_vapor, t) / p - 1.0
    r_chem = (gibbs_energy(fluid, rho_liquid, t) - gibbs_energy(fluid, rho_vapor, t)) / (
        fluid.gas_constant * t
    )
    return jnp.stack([r_liquid, r_vapor, r_chem])


@jax.jit
def saturation_temperature_with_info(fluid: HelmholtzFluid, p: ArrayLike) -> SolveResult:
    """Saturation (boiling) temperature (K) at pressure ``p`` (Pa), with a report.

    Valid for ``p_triple <= p < p_critical``. A converged value is
    differentiable in ``p`` and the EOS coefficients.
    """
    p = jnp.asarray(p, dtype=float)
    t0 = jax.lax.stop_gradient(_tsat_seed(fluid, p))
    x0 = jnp.stack(
        [
            jnp.log(rho_liquid_ancillary(fluid, t0) / fluid.rho_reducing),
            jnp.log(rho_vapor_ancillary(fluid, t0) / fluid.rho_reducing),
            t0 / fluid.t_critical,
        ]
    )
    solved = newton_system_with_info(
        _boiling_residual, x0, (fluid, p), 1e-12, 60, stall_tol=_ROUNDOFF_FLOOR
    )
    return SolveResult(solved.value[2] * fluid.t_critical, solved.report)


def saturation_temperature(fluid: HelmholtzFluid, p: ArrayLike) -> Array:
    """Best saturation temperature (K) at ``p`` (Pa); see `saturation_temperature_with_info`.

    Callers should clip ``p`` into ``[p_triple, p_critical)``.
    """
    return saturation_temperature_with_info(fluid, p).value


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
    "saturation_densities_with_info",
    "saturation_pressure",
    "saturation_pressure_with_info",
    "saturation_state",
    "saturation_temperature",
    "saturation_temperature_with_info",
]
