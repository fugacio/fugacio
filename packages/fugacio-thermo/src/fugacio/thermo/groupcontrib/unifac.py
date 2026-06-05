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
tables are a curated subset of the public Hansen VLE parameter set, sufficient
for alkane / aromatic / alcohol / water / ketone systems. The numerical kernel
:func:`unifac_ln_gamma` is general and differentiable, so it works with any
parameter tables you supply.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float

#: Lattice coordination number.
_Z = 10.0


# subgroup id -> (name, main-group id, R_k, Q_k). Public Hansen VLE values.
SUBGROUPS: dict[int, tuple[str, int, float, float]] = {
    1: ("CH3", 1, 0.9011, 0.848),
    2: ("CH2", 1, 0.6744, 0.540),
    3: ("CH", 1, 0.4469, 0.228),
    4: ("C", 1, 0.2195, 0.000),
    9: ("ACH", 3, 0.5313, 0.400),
    10: ("AC", 3, 0.3652, 0.120),
    11: ("ACCH3", 4, 1.2663, 0.968),
    12: ("ACCH2", 4, 1.0396, 0.660),
    13: ("ACCH", 4, 0.8121, 0.348),
    14: ("OH", 5, 1.0000, 1.200),
    15: ("CH3OH", 6, 1.4311, 1.432),
    16: ("H2O", 7, 0.9200, 1.400),
    19: ("CH3CO", 9, 1.6724, 1.488),
    20: ("CH2CO", 9, 1.4457, 1.180),
}

# Main-group interaction parameters a_mn (Kelvin); curated subset of the public
# Hansen VLE table. Missing ordered pairs default to 0 (athermal interaction).
INTERACTIONS: dict[tuple[int, int], float] = {
    (1, 3): 61.13,
    (3, 1): -11.12,
    (1, 4): 76.50,
    (4, 1): -69.70,
    (1, 5): 986.5,
    (5, 1): 156.4,
    (1, 6): 697.2,
    (6, 1): 16.51,
    (1, 7): 1318.0,
    (7, 1): 300.0,
    (1, 9): 476.4,
    (9, 1): 26.76,
    (3, 4): 167.0,
    (4, 3): -146.8,
    (3, 5): 636.1,
    (5, 3): 89.60,
    (3, 7): 903.8,
    (7, 3): 362.3,
    (3, 9): 25.77,
    (9, 3): 140.1,
    (4, 5): 803.2,
    (5, 4): 25.82,
    (4, 7): 5695.0,
    (7, 4): 377.6,
    (5, 6): -137.1,
    (6, 5): 249.1,
    (5, 7): 353.5,
    (7, 5): -229.1,
    (5, 9): 84.0,
    (9, 5): 164.5,
    (6, 7): 289.6,
    (7, 6): -181.0,
    (6, 9): 108.7,
    (9, 6): 23.39,
    (7, 9): -195.4,
    (9, 7): 472.5,
}

# Component -> {subgroup id: count}. Canonical names match the component database.
COMPONENT_GROUPS: dict[str, dict[int, int]] = {
    "propane": {1: 2, 2: 1},
    "n-butane": {1: 2, 2: 2},
    "isobutane": {1: 3, 3: 1},
    "n-pentane": {1: 2, 2: 3},
    "n-hexane": {1: 2, 2: 4},
    "n-heptane": {1: 2, 2: 5},
    "n-octane": {1: 2, 2: 6},
    "cyclohexane": {2: 6},
    "benzene": {9: 6},
    "toluene": {9: 5, 11: 1},
    "water": {16: 1},
    "methanol": {15: 1},
    "ethanol": {1: 1, 2: 1, 14: 1},
    "2-propanol": {1: 2, 3: 1, 14: 1},
    "acetone": {1: 1, 19: 1},
}


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
