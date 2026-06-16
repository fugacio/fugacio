"""Steam-table style state functions for reference fluids.

Process specifications arrive as ``(T, P)``, ``(P, h)``, ``(P, s)`` or a
saturation quality -- not as the ``(rho, T)`` a Helmholtz EOS natively speaks.
This module resolves those specifications into a :class:`FluidState` snapshot
of every property, handling the two-phase dome:

* :func:`state_tp` -- single-phase states (a ``(T, P)`` specification only
  pins a two-phase mixture on the saturation line itself, a measure-zero set);
* :func:`state_ph` / :func:`state_ps` -- the energy-balance workhorses; inside
  the dome they return the saturation temperature and vapor quality ``q``
  (this is exactly the "steam tables" calculation, e.g. finding the wetness at
  a steam-turbine exhaust);
* :func:`state_tq` / :func:`state_pq` -- states on the dome by quality.

All solves are wrapped in implicit-differentiation rules, so the returned
state is exactly differentiable with respect to the specification: the
gradient of :func:`state_ph` temperature with respect to pressure inside the
dome *is* the Clausius-Clapeyron slope, with no finite differencing anywhere.

In a two-phase state the bulk ``cp``, ``cv`` and speed of sound are undefined
(the mixture is not a single thermodynamic phase); those fields are ``nan`` by
construction there, while ``t``, ``p``, ``rho``, ``u``, ``h``, ``s``, ``g``
and ``q`` remain well-defined mixture values.

Because every state function embeds Newton/bisection solver loops whose eager
re-tracing would dwarf the numerical work, the public functions here (and the
solver-backed functions in the sibling modules) are jit-compiled with the
fluid as a pytree argument: the first call for a given fluid pays a one-time
compilation, after which calls cost microseconds and remain fully
differentiable and jit/vmap-composable.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.helmholtz import props
from fugacio.thermo.helmholtz.density import (
    liquid_density_guess,
    log_delta_residual,
    vapor_density_guess,
)
from fugacio.thermo.helmholtz.fluids import HelmholtzFluid
from fugacio.thermo.helmholtz.saturation import (
    T_SAT_MAX_FRACTION,
    saturation_densities,
    saturation_temperature,
)
from fugacio.thermo.implicit import bracketed_root, newton_root

ArrayLike = Array | float


@dataclass(frozen=True)
class FluidState:
    """A resolved thermodynamic state of a pure reference fluid.

    Attributes:
        t: Temperature (K).
        p: Pressure (Pa).
        rho: Bulk molar density (mol/m^3).
        z: Compressibility factor ``P/(rho R T)``.
        u: Molar internal energy (J/mol).
        h: Molar enthalpy (J/mol).
        s: Molar entropy (J/mol/K).
        g: Molar Gibbs energy (J/mol).
        cv: Molar isochoric heat capacity (J/mol/K); ``nan`` in the dome.
        cp: Molar isobaric heat capacity (J/mol/K); ``nan`` in the dome.
        w: Speed of sound (m/s); ``nan`` in the dome.
        q: Vapor quality (mole basis); ``nan`` for single-phase states.
        two_phase: Whether the state lies inside the vapor-liquid dome.
    """

    t: Array
    p: Array
    rho: Array
    z: Array
    u: Array
    h: Array
    s: Array
    g: Array
    cv: Array
    cp: Array
    w: Array
    q: Array
    two_phase: Array


jax.tree_util.register_dataclass(
    FluidState,
    data_fields=["t", "p", "rho", "z", "u", "h", "s", "g", "cv", "cp", "w", "q", "two_phase"],
    meta_fields=[],
)


def _single_phase_state(fluid: HelmholtzFluid, t: Array, p: Array, rho: Array) -> FluidState:
    return FluidState(
        t=t,
        p=p,
        rho=rho,
        z=p / (rho * fluid.gas_constant * t),
        u=props.internal_energy(fluid, rho, t),
        h=props.enthalpy(fluid, rho, t),
        s=props.entropy(fluid, rho, t),
        g=props.gibbs_energy(fluid, rho, t),
        cv=props.isochoric_heat_capacity(fluid, rho, t),
        cp=props.isobaric_heat_capacity(fluid, rho, t),
        w=props.speed_of_sound(fluid, rho, t),
        q=jnp.asarray(jnp.nan),
        two_phase=jnp.asarray(False),
    )


def _mixture_state(fluid: HelmholtzFluid, t: Array, p: Array, q: Array) -> FluidState:
    rho_liquid, rho_vapor = saturation_densities(fluid, t)
    volume = (1.0 - q) / rho_liquid + q / rho_vapor
    h = (1.0 - q) * props.enthalpy(fluid, rho_liquid, t) + q * props.enthalpy(fluid, rho_vapor, t)
    s = (1.0 - q) * props.entropy(fluid, rho_liquid, t) + q * props.entropy(fluid, rho_vapor, t)
    u = (1.0 - q) * props.internal_energy(fluid, rho_liquid, t) + q * props.internal_energy(
        fluid, rho_vapor, t
    )
    nan = jnp.asarray(jnp.nan)
    return FluidState(
        t=t,
        p=p,
        rho=1.0 / volume,
        z=p * volume / (fluid.gas_constant * t),
        u=u,
        h=h,
        s=s,
        g=h - t * s,
        cv=nan,
        cp=nan,
        w=nan,
        q=q,
        two_phase=jnp.asarray(True),
    )


def _select(two_phase: Array, mixture: FluidState, single: FluidState) -> FluidState:
    """Branch-free selection between a dome state and a single-phase state."""

    def pick(a: Array, b: Array) -> Array:
        return jnp.where(two_phase, a, b)

    return FluidState(
        t=pick(mixture.t, single.t),
        p=pick(mixture.p, single.p),
        rho=pick(mixture.rho, single.rho),
        z=pick(mixture.z, single.z),
        u=pick(mixture.u, single.u),
        h=pick(mixture.h, single.h),
        s=pick(mixture.s, single.s),
        g=pick(mixture.g, single.g),
        cv=pick(mixture.cv, single.cv),
        cp=pick(mixture.cp, single.cp),
        w=pick(mixture.w, single.w),
        q=pick(mixture.q, single.q),
        two_phase=two_phase,
    )


def _newton_density_from(fluid: HelmholtzFluid, t: Array, p: Array, rho_init: Array) -> Array:
    x0 = jnp.log(rho_init / fluid.rho_reducing)
    x_star = newton_root(log_delta_residual, (fluid, t, p), x0, 1e-13, 80)
    return jnp.exp(x_star) * fluid.rho_reducing


def _bracketed_density_at(fluid: HelmholtzFluid, t: Array, p: Array) -> Array:
    lo = jnp.log(jnp.asarray(1e-10))
    hi = jnp.log(jnp.asarray(fluid.rho_max / fluid.rho_reducing))
    x_star = bracketed_root(log_delta_residual, (fluid, t, p), lo, hi, 1e-14, 300)
    return jnp.exp(x_star) * fluid.rho_reducing


@partial(jax.jit, static_argnames=("phase",))
def _state_tp(fluid: HelmholtzFluid, t: Array, p: Array, phase: str) -> FluidState:
    if phase == "liquid":
        rho = _newton_density_from(fluid, t, p, liquid_density_guess(fluid, t))
    elif phase == "vapor":
        rho = _newton_density_from(fluid, t, p, vapor_density_guess(fluid, t, p))
    elif phase == "supercritical":
        rho = _bracketed_density_at(fluid, t, p)
    else:  # auto: stability against the solved saturation line, branch-free.
        t_sat = jnp.clip(t, fluid.t_triple, T_SAT_MAX_FRACTION * fluid.t_critical)
        rho_l_sat, rho_v_sat = saturation_densities(fluid, t_sat)
        psat = props.pressure(fluid, rho_v_sat, t_sat)
        subcritical = t < fluid.t_critical
        liquid_like = subcritical & (p > psat)
        seed = jnp.where(liquid_like, rho_l_sat, vapor_density_guess(fluid, t, p))
        rho_newton = _newton_density_from(fluid, t, p, seed)
        rho_bracketed = _bracketed_density_at(fluid, t, p)
        rho = jnp.where(subcritical, rho_newton, rho_bracketed)
    return _single_phase_state(fluid, t, p, rho)


def state_tp(
    fluid: HelmholtzFluid, t: ArrayLike, p: ArrayLike, *, phase: str = "auto"
) -> FluidState:
    """The single-phase state at temperature ``t`` (K) and pressure ``p`` (Pa).

    With ``phase="auto"`` the stable branch is chosen by comparing ``p``
    against the solved saturation pressure (subcritical) or by a bracketed
    density solve (supercritical). Pass ``"liquid"`` / ``"vapor"`` /
    ``"supercritical"`` to skip the saturation solve when the branch is known,
    or to evaluate a metastable branch on purpose.
    """
    if phase not in ("auto", "liquid", "vapor", "supercritical"):
        raise ValueError(f"unknown phase {phase!r}")
    return _state_tp(fluid, jnp.asarray(t, dtype=float), jnp.asarray(p, dtype=float), phase)


@jax.jit
def _state_tq(fluid: HelmholtzFluid, t: Array, q: Array) -> FluidState:
    rho_liquid, rho_vapor = saturation_densities(fluid, t)
    p = 0.5 * (props.pressure(fluid, rho_liquid, t) + props.pressure(fluid, rho_vapor, t))
    return _mixture_state(fluid, t, p, q)


def state_tq(fluid: HelmholtzFluid, t: ArrayLike, q: ArrayLike) -> FluidState:
    """The two-phase state at saturation temperature ``t`` (K) and quality ``q``."""
    return _state_tq(fluid, jnp.asarray(t, dtype=float), jnp.asarray(q, dtype=float))


@jax.jit
def _state_pq(fluid: HelmholtzFluid, p: Array, q: Array) -> FluidState:
    t = saturation_temperature(fluid, p)
    return _mixture_state(fluid, t, p, q)


def state_pq(fluid: HelmholtzFluid, p: ArrayLike, q: ArrayLike) -> FluidState:
    """The two-phase state at saturation pressure ``p`` (Pa) and quality ``q``."""
    return _state_pq(fluid, jnp.asarray(p, dtype=float), jnp.asarray(q, dtype=float))


def _inverse_state(fluid: HelmholtzFluid, p: Array, target: Array, prop: str) -> FluidState:
    """Shared ``(P, h)`` / ``(P, s)`` resolution; ``prop`` is ``"h"`` or ``"s"``."""
    value = props.enthalpy if prop == "h" else props.entropy

    # Saturation bracket at this pressure (clipped into the subcritical band;
    # the clipped solve also runs -- and is discarded -- for supercritical p).
    p_sat = jnp.clip(p, fluid.p_triple, 0.9999 * fluid.p_critical)
    t_sat = saturation_temperature(fluid, p_sat)
    rho_liquid_sat, rho_vapor_sat = saturation_densities(
        fluid, jnp.clip(t_sat, fluid.t_triple, T_SAT_MAX_FRACTION * fluid.t_critical)
    )
    value_liquid = value(fluid, rho_liquid_sat, t_sat)
    value_vapor = value(fluid, rho_vapor_sat, t_sat)

    subcritical = p < fluid.p_critical
    two_phase = subcritical & (target >= value_liquid) & (target <= value_vapor)
    q = jnp.clip((target - value_liquid) / (value_vapor - value_liquid), 0.0, 1.0)

    scale = fluid.gas_constant * (fluid.t_reducing if prop == "h" else 1.0)

    def liquid_residual(t: Array, params: tuple[HelmholtzFluid, Array, Array]) -> Array:
        f, pp, tt = params
        rho = _newton_density_from(f, t, pp, liquid_density_guess(f, t))
        return (value(f, rho, t) - tt) / scale

    def vapor_residual(t: Array, params: tuple[HelmholtzFluid, Array, Array]) -> Array:
        f, pp, tt = params
        rho = _newton_density_from(f, t, pp, vapor_density_guess(f, t, pp))
        return (value(f, rho, t) - tt) / scale

    def supercritical_residual(t: Array, params: tuple[HelmholtzFluid, Array, Array]) -> Array:
        f, pp, tt = params
        rho = _bracketed_density_at(f, t, pp)
        return (value(f, rho, t) - tt) / scale

    t_floor = jnp.asarray(fluid.t_triple)
    t_ceiling = jnp.asarray(fluid.t_max)
    params = (fluid, p, target)
    t_liquid = bracketed_root(liquid_residual, params, t_floor, t_sat, 1e-9, 200)
    t_vapor = bracketed_root(vapor_residual, params, t_sat, t_ceiling, 1e-9, 200)
    t_super = bracketed_root(supercritical_residual, params, t_floor, t_ceiling, 1e-9, 200)

    liquid_side = target < value_liquid
    t_single = jnp.where(subcritical, jnp.where(liquid_side, t_liquid, t_vapor), t_super)
    rho_liquid = _newton_density_from(fluid, t_single, p, liquid_density_guess(fluid, t_single))
    rho_vapor = _newton_density_from(fluid, t_single, p, vapor_density_guess(fluid, t_single, p))
    rho_super = _bracketed_density_at(fluid, t_single, p)
    rho_single = jnp.where(subcritical, jnp.where(liquid_side, rho_liquid, rho_vapor), rho_super)

    single = _single_phase_state(fluid, t_single, p, rho_single)
    mixture = _mixture_state(fluid, t_sat, p, q)
    return _select(two_phase, mixture, single)


@jax.jit
def _state_ph(fluid: HelmholtzFluid, p: Array, h: Array) -> FluidState:
    return _inverse_state(fluid, p, h, "h")


def state_ph(fluid: HelmholtzFluid, p: ArrayLike, h: ArrayLike) -> FluidState:
    """The state at pressure ``p`` (Pa) and molar enthalpy ``h`` (J/mol).

    Inside the vapor-liquid dome the result has ``two_phase=True``, the
    saturation temperature, and the vapor quality ``q = (h - h')/(h'' - h')``;
    outside it the temperature solves ``h(T, P) = h`` on the stable branch
    (enthalpy is strictly increasing in ``T`` at fixed ``P``, so the bracketed
    solve is unconditionally convergent within the EOS range).
    """
    return _state_ph(fluid, jnp.asarray(p, dtype=float), jnp.asarray(h, dtype=float))


@jax.jit
def _state_ps(fluid: HelmholtzFluid, p: Array, s: Array) -> FluidState:
    return _inverse_state(fluid, p, s, "s")


def state_ps(fluid: HelmholtzFluid, p: ArrayLike, s: ArrayLike) -> FluidState:
    """The state at pressure ``p`` (Pa) and molar entropy ``s`` (J/mol/K).

    The isentropic twin of :func:`state_ph` -- the building block of ideal
    compressor/turbine outlets. Same dome semantics.
    """
    return _state_ps(fluid, jnp.asarray(p, dtype=float), jnp.asarray(s, dtype=float))


__all__ = [
    "FluidState",
    "state_ph",
    "state_pq",
    "state_ps",
    "state_tp",
    "state_tq",
]
