"""Real-fluid molar properties: ideal-gas integrals plus EOS residual functions.

This is the bridge that assembles ``M_real = M_ideal_gas + M_residual`` for a
real mixture, combining `fugacio.thermo.ideal` (the temperature-dependent
ideal-gas backbone) with `fugacio.thermo.departure` (the pressure- and
composition-dependent residual). The outputs (molar enthalpy, entropy, Gibbs
energy, and heat capacity) are exactly what energy balances need, and they are
differentiable with respect to ``T``, ``P``, composition, and every model
parameter just like the rest of the engine.

Enthalpy and entropy are reported relative to an **ideal-gas reference state** at
``T_REF`` and ``P_REF`` (the same reference the ideal-gas integrals use). Only
*differences* are physically meaningful, and any consistent reference cancels in
a balance; callers that need an absolute (formation-based) enthalpy (a chemical
reactor, say) add the standard enthalpies of formation themselves.

Each function accepts an explicit ``phase`` (``"vapor"`` or ``"liquid"``) or
``"auto"``, which evaluates both cubic roots and selects the one with the lower
molar Gibbs energy (the thermodynamically stable single phase). The ``"auto"``
branch is differentiable (a `jax.numpy.where` blend), with the usual
caveat that the gradient is one-sided exactly at a phase boundary.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import P_REF, T_REF, R
from fugacio.thermo.departure import residual_cp, residual_properties
from fugacio.thermo.eos import PR, CubicEOS
from fugacio.thermo.ideal import cp_ig, enthalpy_ig_mixture, entropy_ig_mixture

ArrayLike = Array | float
CpCoeffs = tuple[Array, Array, Array, Array, Array]


def molar_enthalpy(
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    *,
    eos: CubicEOS = PR,
    phase: str = "auto",
    kij: Array | None = None,
    t_ref: float = T_REF,
) -> Array:
    """Real-fluid molar enthalpy ``H(T, P, x)`` (J/mol, vs. ideal gas at ``t_ref``)."""
    a, b, c, d, e = cp
    h_ig = enthalpy_ig_mixture(t, x, a, b, c, d, e, t_ref=t_ref)
    if phase == "auto":
        rv = residual_properties(eos, t, p, x, tc, pc, omega, phase="vapor", kij=kij)
        rl = residual_properties(eos, t, p, x, tc, pc, omega, phase="liquid", kij=kij)
        h_res = jnp.where(rv.gibbs <= rl.gibbs, rv.enthalpy, rl.enthalpy)
    else:
        h_res = residual_properties(eos, t, p, x, tc, pc, omega, phase=phase, kij=kij).enthalpy
    return h_ig + h_res


def molar_entropy(
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    *,
    eos: CubicEOS = PR,
    phase: str = "auto",
    kij: Array | None = None,
    t_ref: float = T_REF,
    p_ref: float = P_REF,
) -> Array:
    """Real-fluid molar entropy ``S(T, P, x)`` (J/mol/K, vs. ideal gas reference)."""
    a, b, c, d, e = cp
    s_ig = entropy_ig_mixture(t, p, x, a, b, c, d, e, t_ref=t_ref, p_ref=p_ref)
    if phase == "auto":
        rv = residual_properties(eos, t, p, x, tc, pc, omega, phase="vapor", kij=kij)
        rl = residual_properties(eos, t, p, x, tc, pc, omega, phase="liquid", kij=kij)
        s_res = jnp.where(rv.gibbs <= rl.gibbs, rv.entropy, rl.entropy)
    else:
        s_res = residual_properties(eos, t, p, x, tc, pc, omega, phase=phase, kij=kij).entropy
    return s_ig + s_res


def molar_gibbs(
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    *,
    eos: CubicEOS = PR,
    phase: str = "auto",
    kij: Array | None = None,
    t_ref: float = T_REF,
    p_ref: float = P_REF,
) -> Array:
    """Real-fluid molar Gibbs energy ``G = H - T S`` (J/mol)."""
    h = molar_enthalpy(t, p, x, tc, pc, omega, cp, eos=eos, phase=phase, kij=kij, t_ref=t_ref)
    s = molar_entropy(
        t, p, x, tc, pc, omega, cp, eos=eos, phase=phase, kij=kij, t_ref=t_ref, p_ref=p_ref
    )
    return h - jnp.asarray(t, dtype=float) * s


def molar_cp(
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    *,
    eos: CubicEOS = PR,
    phase: str = "auto",
    kij: Array | None = None,
) -> Array:
    """Real-fluid molar heat capacity ``Cp(T, P, x)`` (J/mol/K)."""
    a, b, c, d, e = cp
    x = jnp.asarray(x)
    cp_ig_mix = jnp.sum(x * cp_ig(t, a, b, c, d, e))
    if phase == "auto":
        rv = residual_properties(eos, t, p, x, tc, pc, omega, phase="vapor", kij=kij)
        rl = residual_properties(eos, t, p, x, tc, pc, omega, phase="liquid", kij=kij)
        cp_res_v = residual_cp(eos, t, p, x, tc, pc, omega, phase="vapor", kij=kij)
        cp_res_l = residual_cp(eos, t, p, x, tc, pc, omega, phase="liquid", kij=kij)
        cp_res = jnp.where(rv.gibbs <= rl.gibbs, cp_res_v, cp_res_l)
    else:
        cp_res = residual_cp(eos, t, p, x, tc, pc, omega, phase=phase, kij=kij)
    return cp_ig_mix + cp_res


def stable_phase(
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
) -> str:
    """Label the thermodynamically stable single phase (``"vapor"`` or ``"liquid"``).

    The stable root is the one with the lower molar Gibbs energy; when only one
    real root exists (a compressed liquid or a supercritical fluid, where both
    cubic-root selections collapse onto it) the two Gibbs energies tie, so the
    phase is then classified by the stable root's compressibility: liquid-like
    below the cubic critical scale ``Z = 1/3``, vapour-like above it. This is a
    host-side convenience that materialises a Python ``bool``; the differentiable
    path is the ``phase="auto"`` option on the property functions.
    """
    rv = residual_properties(eos, t, p, x, tc, pc, omega, phase="vapor", kij=kij)
    rl = residual_properties(eos, t, p, x, tc, pc, omega, phase="liquid", kij=kij)
    z_star = float(rl.z) if float(rl.gibbs) < float(rv.gibbs) else float(rv.z)
    return "liquid" if z_star < 1.0 / 3.0 else "vapor"


def speed_of_sound_ideal(
    t: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    mw: Array,
) -> Array:
    """Ideal-gas speed of sound ``sqrt(gamma R T / M)`` (m/s) for the mixture.

    A lightweight, always-defined acoustic estimate (``gamma = Cp/Cv`` with the
    ideal-gas ``Cv = Cp - R``); the full real-fluid sound speed follows once the
    residual ``Cv`` and ``(dP/dV)_T`` are wired in.
    """
    a, b, c, d, e = cp
    x = jnp.asarray(x)
    cp_ig_mix = jnp.sum(x * cp_ig(t, a, b, c, d, e))
    gamma = cp_ig_mix / (cp_ig_mix - R)
    m = jnp.sum(x * jnp.asarray(mw)) * 1.0e-3
    return jnp.sqrt(gamma * R * jnp.asarray(t, dtype=float) / m)
