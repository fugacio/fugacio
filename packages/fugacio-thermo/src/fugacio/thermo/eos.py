"""Differentiable cubic equations of state (van der Waals, RK, SRK, Peng-Robinson).

All four classic cubics share one pressure-explicit form::

    P = R*T / (V - b)  -  a(T) / (V**2 + u*b*V + w*b**2)

and differ only in the constants ``(Omega_a, Omega_b, u, w)`` and the
temperature dependence of the attractive term ``a(T) = a_c * alpha(T_r)``. This
module implements that single generalized cubic and exposes the four families as
ready-made `PR`, `SRK`, `RK`, `VDW` specifications.

In dimensionless form the molar compressibility ``Z = P V / (R T)`` is the root
of::

    Z**3 - (1 + B - u*B) Z**2 + (A + w*B**2 - u*B - u*B**2) Z
        - (A*B + w*B**2 + w*B**3) = 0

with ``A = a P / (R T)**2`` and ``B = b P / (R T)``.

**Differentiability.** Selecting the physically correct root (smallest for a
liquid, largest for a vapour) is inherently non-smooth, so rather than
differentiate through the closed-form cubic solution we compute ``Z`` by the
analytic trigonometric/Cardano method and attach exact derivatives through the
*implicit function theorem* applied to the cubic residual (a `jax.custom_jvp`
rule). The result is clean, cheap gradients of ``Z`` -- and therefore of every
downstream property and fugacity coefficient -- with respect to ``T``, ``P``,
composition, and the EOS parameters themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import R

ArrayLike = Array | float


@dataclass(frozen=True)
class CubicEOS:
    """Specification of a two-parameter cubic equation of state.

    Attributes:
        name: Human-readable name.
        omega_a: Attraction constant ``Omega_a`` in ``a_c = Omega_a R**2 Tc**2 / Pc``.
        omega_b: Repulsion constant ``Omega_b`` in ``b = Omega_b R Tc / Pc``.
        u: Linear coefficient of the cubic denominator ``V**2 + u b V + w b**2``.
        w: Constant coefficient of the cubic denominator.
        alpha_kind: Which ``alpha(T_r, omega)`` law to use
            (``"one"``, ``"rk"``, ``"soave"`` or ``"pr"``).
    """

    name: str
    omega_a: float
    omega_b: float
    u: float
    w: float
    alpha_kind: str


#: Van der Waals (1873): the original cubic; no temperature dependence in ``a``.
VDW = CubicEOS("van der Waals", 27.0 / 64.0, 1.0 / 8.0, 0.0, 0.0, "one")
#: Redlich-Kwong (1949).
RK = CubicEOS("Redlich-Kwong", 0.42748, 0.08664, 1.0, 0.0, "rk")
#: Soave-Redlich-Kwong (1972): RK with an acentric-factor alpha function.
SRK = CubicEOS("Soave-Redlich-Kwong", 0.42748, 0.08664, 1.0, 0.0, "soave")
#: Peng-Robinson (1976): the workhorse for vapour-liquid equilibria.
PR = CubicEOS("Peng-Robinson", 0.45724, 0.07780, 2.0, -1.0, "pr")


def alpha(eos: CubicEOS, t: ArrayLike, tc: ArrayLike, omega: ArrayLike) -> Array:
    """Temperature-dependent ``alpha(T_r)`` factor of the attractive term."""
    tr = jnp.asarray(t) / tc
    if eos.alpha_kind == "one":
        return jnp.ones_like(tr)
    if eos.alpha_kind == "rk":
        return tr**-0.5
    if eos.alpha_kind == "soave":
        m = 0.480 + 1.574 * omega - 0.176 * omega**2
        return (1.0 + m * (1.0 - jnp.sqrt(tr))) ** 2
    if eos.alpha_kind == "pr":
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        return (1.0 + kappa * (1.0 - jnp.sqrt(tr))) ** 2
    raise ValueError(f"unknown alpha_kind {eos.alpha_kind!r}")


def a_pure(eos: CubicEOS, t: ArrayLike, tc: ArrayLike, pc: ArrayLike, omega: ArrayLike) -> Array:
    """Attractive parameter ``a(T)`` for one or more pure components (J m^3 / mol^2)."""
    a_c = eos.omega_a * R**2 * jnp.asarray(tc) ** 2 / pc
    return a_c * alpha(eos, t, tc, omega)


def b_pure(eos: CubicEOS, tc: ArrayLike, pc: ArrayLike) -> Array:
    """Repulsive (co-volume) parameter ``b`` for one or more pure components (m^3/mol)."""
    return eos.omega_b * R * jnp.asarray(tc) / pc


def _cubic_coeffs(
    a_dimless: Array, b_dimless: Array, u: float, w: float
) -> tuple[Array, Array, Array]:
    """Return ``(p2, p1, p0)`` of ``Z**3 + p2 Z**2 + p1 Z + p0`` for given ``A``, ``B``."""
    big_a = a_dimless
    big_b = b_dimless
    p2 = -(1.0 + big_b - u * big_b)
    p1 = big_a + w * big_b**2 - u * big_b - u * big_b**2
    p0 = -(big_a * big_b + w * big_b**2 + w * big_b**3)
    return p2, p1, p0


def _real_root(big_a: Array, big_b: Array, u: float, w: float, largest: bool) -> Array:
    """Analytically select a real root of the compressibility cubic.

    ``largest=True`` returns the vapour-like (largest) root; otherwise the
    liquid-like (smallest root exceeding ``B``). The expression is NaN-safe in
    whichever branch is unused so it can be traced by JAX.
    """
    p2, p1, p0 = _cubic_coeffs(big_a, big_b, u, w)
    shift = p2 / 3.0
    p_dep = p1 - p2 * p2 / 3.0
    q_dep = 2.0 * p2**3 / 27.0 - p2 * p1 / 3.0 + p0
    half_q = q_dep / 2.0
    disc = half_q**2 + (p_dep / 3.0) ** 3

    # One real root (disc > 0): Cardano.
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))

    def cbrt(x: Array) -> Array:
        return jnp.sign(x) * jnp.abs(x) ** (1.0 / 3.0)

    z_cardano = cbrt(-half_q + sqrt_disc) + cbrt(-half_q - sqrt_disc) - shift

    # Three real roots (disc <= 0): trigonometric method.
    p_neg = jnp.minimum(p_dep, -1e-300)
    amp = 2.0 * jnp.sqrt(-p_neg / 3.0)
    cos_arg = jnp.clip((3.0 * q_dep) / (2.0 * p_neg) * jnp.sqrt(-3.0 / p_neg), -1.0, 1.0)
    theta = jnp.arccos(cos_arg)
    roots = jnp.stack(
        [
            amp * jnp.cos(theta / 3.0) - shift,
            amp * jnp.cos(theta / 3.0 - 2.0 * jnp.pi / 3.0) - shift,
            amp * jnp.cos(theta / 3.0 - 4.0 * jnp.pi / 3.0) - shift,
        ]
    )
    if largest:
        z_trig = jnp.max(roots, axis=0)
    else:
        above_b = jnp.where(roots > big_b, roots, jnp.inf)
        z_trig = jnp.min(above_b, axis=0)

    z = jnp.where(disc > 0.0, z_cardano, z_trig)

    # Two Newton polish steps on the exact cubic to tighten the analytic root.
    for _ in range(2):
        f = z**3 + p2 * z**2 + p1 * z + p0
        fp = 3.0 * z**2 + 2.0 * p2 * z + p1
        fp = jnp.where(jnp.abs(fp) < 1e-12, jnp.sign(fp) * 1e-12 + 1e-12, fp)
        z = z - f / fp
    return z


@partial(jax.custom_jvp, nondiff_argnums=(2, 3, 4))
def compress_factor(big_a: Array, big_b: Array, u: float, w: float, largest: bool) -> Array:
    """Compressibility factor ``Z`` from the dimensionless ``A``, ``B`` (implicitly diff'd)."""
    return _real_root(big_a, big_b, u, w, largest)


@compress_factor.defjvp
def _compress_factor_jvp(
    u: float,
    w: float,
    largest: bool,
    primals: tuple[Array, Array],
    tangents: tuple[Array, Array],
) -> tuple[Array, Array]:
    big_a, big_b = primals
    a_dot, b_dot = tangents
    z = compress_factor(big_a, big_b, u, w, largest)
    p2, p1, _ = _cubic_coeffs(big_a, big_b, u, w)
    f_z = 3.0 * z**2 + 2.0 * p2 * z + p1
    f_a = z - big_b
    f_b = (
        z**2 * (u - 1.0)
        + z * (2.0 * w * big_b - u - 2.0 * u * big_b)
        - (big_a + 2.0 * w * big_b + 3.0 * w * big_b**2)
    )
    z_dot = -(f_a * a_dot + f_b * b_dot) / f_z
    return z, z_dot


def _departure_g(z: Array, big_b: Array, eos: CubicEOS) -> Array:
    """The ``g`` factor in ``ln(phi) = ... - (A/B) * g * (...)``.

    For the van der Waals family (where the cubic denominator is a perfect
    square) this is the limit ``g = B / Z``; otherwise it is the standard
    logarithmic term.
    """
    disc = eos.u**2 - 4.0 * eos.w
    if disc == 0.0:
        return big_b / z
    root = math.sqrt(disc)
    sigma = (eos.u + root) / 2.0
    epsilon = (eos.u - root) / 2.0
    return (1.0 / root) * jnp.log((z + sigma * big_b) / (z + epsilon * big_b))


def _ab_mixture(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    kij: Array | None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    a_i = a_pure(eos, t, tc, pc, omega)
    b_i = b_pure(eos, tc, pc)
    if kij is None:
        kij = jnp.zeros((a_i.shape[0], a_i.shape[0]))
    a_ij = (1.0 - kij) * jnp.sqrt(a_i[:, None] * a_i[None, :])
    a_mix = x @ a_ij @ x
    b_mix = x @ b_i
    rt = R * jnp.asarray(t)
    big_a = a_mix * jnp.asarray(p) / rt**2
    big_b = b_mix * jnp.asarray(p) / rt
    return big_a, big_b, a_ij, a_mix, b_mix, b_i


def compressibility(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    phase: str = "vapor",
    kij: Array | None = None,
) -> Array:
    """Mixture compressibility factor ``Z`` for the requested ``phase``."""
    big_a, big_b, *_ = _ab_mixture(eos, t, p, x, tc, pc, omega, kij)
    return compress_factor(big_a, big_b, eos.u, eos.w, phase == "vapor")


def ln_phi_mixture(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    phase: str = "vapor",
    kij: Array | None = None,
) -> tuple[Array, Array]:
    """Log fugacity coefficients ``ln(phi_i)`` of every component, and ``Z``.

    Uses the van der Waals one-fluid mixing rule with binary interaction
    parameters ``kij`` (defaults to all zeros).

    Returns:
        ``(ln_phi, Z)`` where ``ln_phi`` is a 1-D array aligned with ``x``.
    """
    x = jnp.asarray(x)
    big_a, big_b, a_ij, a_mix, b_mix, b_i = _ab_mixture(eos, t, p, x, tc, pc, omega, kij)
    z = compress_factor(big_a, big_b, eos.u, eos.w, phase == "vapor")
    g = _departure_g(z, big_b, eos)
    sum_term = a_ij @ x
    b_ratio = b_i / b_mix
    ln_phi = (
        b_ratio * (z - 1.0)
        - jnp.log(z - big_b)
        - (big_a / big_b) * g * (2.0 * sum_term / a_mix - b_ratio)
    )
    return ln_phi, z


def ln_phi_pure(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    tc: ArrayLike,
    pc: ArrayLike,
    omega: ArrayLike,
    *,
    phase: str = "vapor",
) -> tuple[Array, Array]:
    """Log fugacity coefficient and ``Z`` of a single pure component."""
    ln_phi, z = ln_phi_mixture(
        eos,
        t,
        p,
        jnp.asarray([1.0]),
        jnp.asarray([tc]),
        jnp.asarray([pc]),
        jnp.asarray([omega]),
        phase=phase,
    )
    return ln_phi[0], z


def molar_volume(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    phase: str = "vapor",
    kij: Array | None = None,
) -> Array:
    """Mixture molar volume ``V = Z R T / P`` (m^3/mol)."""
    z = compressibility(eos, t, p, x, tc, pc, omega, phase=phase, kij=kij)
    return z * R * jnp.asarray(t) / jnp.asarray(p)


def pressure(
    eos: CubicEOS,
    t: ArrayLike,
    v: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    kij: Array | None = None,
) -> Array:
    """Pressure from the explicit cubic EOS at given ``T`` and molar volume ``V``."""
    x = jnp.asarray(x)
    a_i = a_pure(eos, t, tc, pc, omega)
    b_i = b_pure(eos, tc, pc)
    if kij is None:
        kij = jnp.zeros((a_i.shape[0], a_i.shape[0]))
    a_ij = (1.0 - kij) * jnp.sqrt(a_i[:, None] * a_i[None, :])
    a_mix = x @ a_ij @ x
    b_mix = x @ b_i
    v = jnp.asarray(v)
    return R * jnp.asarray(t) / (v - b_mix) - a_mix / (v**2 + eos.u * b_mix * v + eos.w * b_mix**2)
