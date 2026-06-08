"""UNIFAC group-contribution model for predictive activity coefficients.

UNIFAC predicts liquid-phase activity coefficients with *no* mixture-specific
data: a molecule is decomposed into functional subgroups, and the only adjustable
quantities are group volume/area parameters (``R_k``, ``Q_k``) and a matrix of
group-group interaction energies (``a_mn``) regressed once, globally. This is how
Fugacio fills in activity-coefficient parameters for pairs where curated binary
data are proprietary -- exactly the role the project README assigns to UNIFAC.

``ln(gamma_i) = ln(gamma_i^comb) + ln(gamma_i^res)``; the combinatorial part is
the UNIQUAC size/shape term and the residual part is a sum over group residual
activity coefficients ``Gamma_k`` evaluated in the mixture and in the pure fluid.

The bundled :data:`SUBGROUPS`, :data:`INTERACTIONS`, and :data:`COMPONENT_GROUPS`
tables live in :mod:`fugacio.thermo.groupcontrib._unifac_data`, generated from the
public UNIFAC (Hansen VLE) parameters and DDBST group assignments for the curated
component database (see ``scripts/gen_parameters.py``). The numerical kernel
:func:`unifac_ln_gamma` is general and differentiable, so it works with any
parameter tables you supply.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.groupcontrib._unifac_data import (
    COMPONENT_GROUPS,
    INTERACTIONS,
    SUBGROUPS,
)

ArrayLike = Array | float

#: Lattice coordination number.
_Z = 10.0

__all__ = [
    "COMPONENT_GROUPS",
    "INTERACTIONS",
    "SUBGROUPS",
    "unifac_activity",
    "unifac_ln_gamma",
]


def _main_interaction_matrix() -> tuple[list[int], Array]:
    main_ids = sorted({mg for _, mg, _, _ in SUBGROUPS.values()})
    index = {mg: i for i, mg in enumerate(main_ids)}
    size = len(main_ids)
    a = [[0.0] * size for _ in range(size)]
    for (m, n), value in INTERACTIONS.items():
        a[index[m]][index[n]] = value
    return main_ids, jnp.asarray(a)


_MAIN_IDS, _A_MAIN = _main_interaction_matrix()
_MAIN_INDEX = {mg: i for i, mg in enumerate(_MAIN_IDS)}


def _ln_group_gamma(theta: Array, q: Array, psi: Array) -> Array:
    """Group residual activity coefficients ``ln(Gamma_k)`` for a group fraction."""
    s = theta @ psi  # S_k = sum_m theta_m Psi_mk
    return q * (1.0 - jnp.log(s) - psi @ (theta / s))


def unifac_ln_gamma(
    x: Array,
    nu: Array,
    r: Array,
    q: Array,
    main_index: Array,
    a_main: Array,
    t: ArrayLike,
) -> Array:
    """Log activity coefficients from UNIFAC, given explicit group tables.

    Args:
        x: Mole fractions, shape ``(ncomp,)``.
        nu: Subgroup counts, integer-valued array of shape ``(ncomp, ngroup)``.
        r: Subgroup ``R_k`` values, shape ``(ngroup,)``.
        q: Subgroup ``Q_k`` values, shape ``(ngroup,)``.
        main_index: Index of each subgroup's main group into ``a_main``, shape ``(ngroup,)``.
        a_main: Main-group interaction matrix ``a_mn`` (K), shape ``(nmain, nmain)``.
        t: Temperature (K).

    Returns:
        ``ln(gamma)`` of shape ``(ncomp,)``.
    """
    x = jnp.asarray(x)
    nu = jnp.asarray(nu)
    t = jnp.asarray(t)

    # Combinatorial part (UNIQUAC size/shape term).
    r_i = nu @ r
    q_i = nu @ q
    phi = r_i * x / jnp.sum(r_i * x)
    theta_c = q_i * x / jnp.sum(q_i * x)
    ell = (_Z / 2.0) * (r_i - q_i) - (r_i - 1.0)
    ln_gamma_c = (
        jnp.log(phi / x)
        + (_Z / 2.0) * q_i * jnp.log(theta_c / phi)
        + ell
        - (phi / x) * jnp.sum(x * ell)
    )

    # Residual part via group residual activity coefficients.
    psi = jnp.exp(-a_main[main_index[:, None], main_index[None, :]] / t)

    x_groups = x @ nu  # total of each group in the mixture
    big_x = x_groups / jnp.sum(x_groups)
    theta = q * big_x / jnp.sum(q * big_x)
    ln_gamma_group_mix = _ln_group_gamma(theta, q, psi)

    def pure_group_ln_gamma(nu_i: Array) -> Array:
        big_x_i = nu_i / jnp.sum(nu_i)
        theta_i = q * big_x_i / jnp.sum(q * big_x_i)
        return _ln_group_gamma(theta_i, q, psi)

    ln_gamma_group_pure = jax.vmap(pure_group_ln_gamma)(nu)
    ln_gamma_r = jnp.sum(nu * (ln_gamma_group_mix[None, :] - ln_gamma_group_pure), axis=1)
    return ln_gamma_c + ln_gamma_r


def _assemble(components: list[str]) -> tuple[Array, Array, Array, Array, Array]:
    unknown = [c for c in components if c not in COMPONENT_GROUPS]
    if unknown:
        raise KeyError(f"no UNIFAC group assignment for: {unknown}")
    subgroup_ids = sorted({sg for c in components for sg in COMPONENT_GROUPS[c]})
    nu = jnp.asarray(
        [[float(COMPONENT_GROUPS[c].get(sg, 0)) for sg in subgroup_ids] for c in components]
    )
    r = jnp.asarray([SUBGROUPS[sg][2] for sg in subgroup_ids])
    q = jnp.asarray([SUBGROUPS[sg][3] for sg in subgroup_ids])
    main_index = jnp.asarray([_MAIN_INDEX[SUBGROUPS[sg][1]] for sg in subgroup_ids])
    return nu, r, q, main_index, _A_MAIN


def unifac_activity(components: list[str], x: Array, t: ArrayLike) -> Array:
    """Predict ``ln(gamma)`` for named database components using bundled tables.

    Args:
        components: Component names with UNIFAC assignments in :data:`COMPONENT_GROUPS`.
        x: Mole fractions aligned with ``components``.
        t: Temperature (K).

    Returns:
        ``ln(gamma)`` of shape ``(len(components),)``.
    """
    nu, r, q, main_index, a_main = _assemble(components)
    return unifac_ln_gamma(x, nu, r, q, main_index, a_main, t)
