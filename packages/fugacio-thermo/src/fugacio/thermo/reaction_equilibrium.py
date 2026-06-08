"""Chemical-reaction equilibrium: solve for the equilibrium composition.

Given one or more reactions, a feed (in moles), and ``(T, P)``, this module finds
the extents of reaction that make each reaction's activity quotient equal to its
equilibrium constant ``K_j(T)`` (from :mod:`fugacio.thermo.reactions`). For a gas
phase the activity of species ``i`` is

* ``a_i = y_i P / P_ref`` for an ideal gas (``basis="ideal-gas"``), or
* ``a_i = y_i phi_i P / P_ref`` for a real gas, with fugacity coefficients
  ``phi_i`` from the cubic EOS (``basis="phi"``),

so the equilibrium condition for reaction ``j`` is
``sum_i nu_ij ln a_i = ln K_j(T)``.

The unknowns are the reaction extents ``xi`` (one per reaction); the full
composition is ``n = n_feed + xi . Nu``. A single reaction is solved by a robust
bracketed root over the physically feasible extent range; several simultaneous
reactions are solved by a damped Newton system. Both solvers differentiate the
*converged* extent with respect to ``T``, ``P``, and the feed by implicit
differentiation (see :mod:`fugacio.thermo.implicit`), so conversions and yields
carry exact gradients.

The standard state is the ideal gas at ``P_ref`` (1 bar), matching the tabulated
formation data used to build ``K(T)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import P_REF, R
from fugacio.thermo.eos import PR, CubicEOS, ln_phi_mixture
from fugacio.thermo.implicit import bracketed_root, newton_system
from fugacio.thermo.reactions import Reaction, delta_g_rxn, reaction_arrays

ArrayLike = Array | float


class EquilibriumResult(NamedTuple):
    """Outcome of a reaction-equilibrium calculation.

    Attributes:
        extent: Equilibrium extent of each reaction (mol), shape ``(R,)``.
        moles: Equilibrium moles of each component, shape ``(n,)``.
        y: Equilibrium mole fractions, shape ``(n,)``.
        k: Equilibrium constant of each reaction at ``T``, shape ``(R,)``.
    """

    extent: Array
    moles: Array
    y: Array
    k: Array


def _as_list(reactions: Reaction | Sequence[Reaction]) -> list[Reaction]:
    return [reactions] if isinstance(reactions, Reaction) else list(reactions)


def _stack(reactions: Sequence[Reaction]) -> tuple[tuple[str, ...], Array]:
    components = reactions[0].components
    for r in reactions:
        if r.components != components:
            raise ValueError("all reactions must share the same component ordering")
    nu = jnp.stack([jnp.asarray(r.nu) for r in reactions])
    return components, nu


def _ln_activity(
    n: Array,
    t: ArrayLike,
    p: ArrayLike,
    *,
    basis: str,
    eos: CubicEOS,
    tc: Array | None,
    pc: Array | None,
    omega: Array | None,
    kij: Array | None,
) -> Array:
    """Log activities ``ln a_i`` of a gas mixture with composition moles ``n``."""
    total = jnp.sum(n)
    y = n / total
    ln_y = jnp.log(jnp.clip(y, 1e-300, None))
    ln_a = ln_y + jnp.log(jnp.asarray(p) / P_REF)
    if basis == "phi":
        if tc is None or pc is None or omega is None:
            raise ValueError("basis='phi' requires tc, pc, omega")
        ln_phi, _ = ln_phi_mixture(eos, t, p, y, tc, pc, omega, phase="vapor", kij=kij)
        ln_a = ln_a + ln_phi
    return ln_a


def _ln_k(nu_row: Array, t: ArrayLike, hf: Array, gf: Array, coeffs: tuple[Array, ...]) -> Array:
    a, b, c, d, e = coeffs
    return -delta_g_rxn(nu_row, t, hf, gf, a, b, c, d, e) / (R * jnp.asarray(t))


def equilibrium(
    reactions: Reaction | Sequence[Reaction],
    n_feed: Array,
    t: ArrayLike,
    p: ArrayLike,
    *,
    basis: str = "ideal-gas",
    eos: CubicEOS = PR,
    tc: Array | None = None,
    pc: Array | None = None,
    omega: Array | None = None,
    kij: Array | None = None,
    tol: float = 1e-11,
    max_iter: int = 60,
) -> EquilibriumResult:
    """Solve for the equilibrium composition of a reacting gas mixture.

    Args:
        reactions: A single :class:`~fugacio.thermo.reactions.Reaction` or several
            sharing the same component ordering.
        n_feed: Feed amounts per component (mol), shape ``(n,)``.
        t: Temperature (K).
        p: Pressure (Pa).
        basis: ``"ideal-gas"`` (``a_i = y_i P/P_ref``) or ``"phi"`` (EOS fugacity
            coefficients).
        eos: Cubic EOS used when ``basis="phi"``.
        tc, pc, omega, kij: Component constants for the EOS (required for ``"phi"``).
        tol, max_iter: Solver tolerances.

    Returns:
        An :class:`EquilibriumResult` with extents, moles, mole fractions, and
        ``K(T)``. Differentiable in ``t``, ``p``, and ``n_feed``.
    """
    rxns = _as_list(reactions)
    components, nu = _stack(rxns)
    n_feed = jnp.asarray(n_feed, dtype=float)
    hf, gf, coeffs = reaction_arrays(components)

    def moles_of(extent: Array) -> Array:
        return n_feed + extent @ nu

    theta = (jnp.asarray(t, dtype=float), jnp.asarray(p, dtype=float), n_feed)

    if len(rxns) == 1:
        nu_row = nu[0]

        def residual_scalar(xi: Array, th: tuple[Array, Array, Array]) -> Array:
            t_, p_, nf = th
            n = nf + xi * nu_row
            ln_a = _ln_activity(n, t_, p_, basis=basis, eos=eos, tc=tc, pc=pc, omega=omega, kij=kij)
            return jnp.sum(nu_row * ln_a) - _ln_k(nu_row, t_, hf, gf, coeffs)

        reactant = nu_row < 0.0
        product = nu_row > 0.0
        xi_hi = jnp.min(jnp.where(reactant, n_feed / jnp.where(reactant, -nu_row, 1.0), jnp.inf))
        xi_lo = -jnp.min(jnp.where(product, n_feed / jnp.where(product, nu_row, 1.0), jnp.inf))
        span = xi_hi - xi_lo
        lo = xi_lo + 1e-7 * span
        hi = xi_hi - 1e-7 * span
        xi_star = bracketed_root(residual_scalar, theta, lo, hi, tol)
        extent = jnp.reshape(xi_star, (1,))
    else:

        def residual_vec(extent: Array, th: tuple[Array, Array, Array]) -> Array:
            t_, p_, nf = th
            n = nf + extent @ nu
            ln_a = _ln_activity(n, t_, p_, basis=basis, eos=eos, tc=tc, pc=pc, omega=omega, kij=kij)
            ln_k = jnp.stack([_ln_k(nu[j], t_, hf, gf, coeffs) for j in range(nu.shape[0])])
            return nu @ ln_a - ln_k

        # Interior start: advance each reaction a small, feasibility-safe step.
        reactant = nu < 0.0
        cap = jnp.min(
            jnp.where(reactant, n_feed[None, :] / jnp.where(reactant, -nu, 1.0), jnp.inf), axis=1
        )
        extent = newton_system(residual_vec, 0.1 * cap, theta, tol, max_iter)

    n_eq = moles_of(extent)
    y_eq = n_eq / jnp.sum(n_eq)
    k = jnp.stack([jnp.exp(_ln_k(nu[j], t, hf, gf, coeffs)) for j in range(nu.shape[0])])
    return EquilibriumResult(extent=extent, moles=n_eq, y=y_eq, k=k)


def conversion(result: EquilibriumResult, n_feed: Array, component_index: int) -> Array:
    """Fractional conversion of a feed component, ``(n0 - n_eq) / n0``."""
    n_feed = jnp.asarray(n_feed, dtype=float)
    n0 = n_feed[component_index]
    return (n0 - result.moles[component_index]) / n0


__all__ = [
    "EquilibriumResult",
    "conversion",
    "equilibrium",
]
