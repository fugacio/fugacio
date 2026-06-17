"""Modified UNIFAC (Dortmund): predictive activity coefficients with T-dependence.

The Dortmund revision of UNIFAC improves dilute-region and temperature behaviour
with two changes to classic UNIFAC (`fugacio.thermo.groupcontrib.unifac`):

* a temperature-dependent group interaction,
  ``psi_mn = exp(-(a_mn + b_mn T + c_mn T^2) / T)``; and
* a modified combinatorial term using ``V'_i = r_i^{3/4} / sum_j x_j r_j^{3/4}``
  for the leading volume contribution.

The kernel `modified_unifac_ln_gamma` is general and differentiable; the
bundled Dortmund subgroup/interaction tables and DDBST assignments live in
`fugacio.thermo.groupcontrib._dortmund_data` (see ``scripts/gen_parameters.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.groupcontrib._dortmund_data import (
    DO_COMPONENT_GROUPS,
    DO_INTERACTIONS,
    DO_SUBGROUPS,
)

ArrayLike = Array | float

#: Lattice coordination number.
_Z = 10.0


def _main_matrices() -> tuple[list[int], Array, Array, Array]:
    main_ids = sorted({mg for _, mg, _, _ in DO_SUBGROUPS.values()})
    index = {mg: i for i, mg in enumerate(main_ids)}
    size = len(main_ids)
    a = [[0.0] * size for _ in range(size)]
    b = [[0.0] * size for _ in range(size)]
    c = [[0.0] * size for _ in range(size)]
    for (m, n), (a_mn, b_mn, c_mn) in DO_INTERACTIONS.items():
        a[index[m]][index[n]] = a_mn
        b[index[m]][index[n]] = b_mn
        c[index[m]][index[n]] = c_mn
    return main_ids, jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)


_MAIN_IDS, _A_MAIN, _B_MAIN, _C_MAIN = _main_matrices()
_MAIN_INDEX = {mg: i for i, mg in enumerate(_MAIN_IDS)}


def _ln_group_gamma(theta: Array, q: Array, psi: Array) -> Array:
    """Group residual activity coefficients ``ln(Gamma_k)`` for a group fraction."""
    s = theta @ psi
    return q * (1.0 - jnp.log(s) - psi @ (theta / s))


def modified_unifac_ln_gamma(
    x: Array,
    nu: Array,
    r: Array,
    q: Array,
    main_index: Array,
    a_main: Array,
    b_main: Array,
    c_main: Array,
    t: ArrayLike,
) -> Array:
    """Log activity coefficients from modified UNIFAC (Dortmund), given group tables.

    Args:
        x: Mole fractions, shape ``(ncomp,)``.
        nu: Subgroup counts, shape ``(ncomp, ngroup)``.
        r: Subgroup ``R_k`` values, shape ``(ngroup,)``.
        q: Subgroup ``Q_k`` values, shape ``(ngroup,)``.
        main_index: Each subgroup's main-group index, shape ``(ngroup,)``.
        a_main: Constant interaction-coefficient matrix, shape ``(nmain, nmain)``.
        b_main: Linear-in-``T`` interaction-coefficient matrix, shape ``(nmain, nmain)``.
        c_main: Quadratic-in-``T`` interaction-coefficient matrix, shape ``(nmain, nmain)``.
        t: Temperature (K).

    Returns:
        ``ln(gamma)`` of shape ``(ncomp,)``.
    """
    x = jnp.asarray(x)
    nu = jnp.asarray(nu)
    t = jnp.asarray(t)

    # Modified combinatorial part (3/4-power leading volume term).
    r_i = nu @ r
    q_i = nu @ q
    v = r_i / jnp.sum(x * r_i)
    f = q_i / jnp.sum(x * q_i)
    r34 = r_i**0.75
    v_prime = r34 / jnp.sum(x * r34)
    ln_gamma_c = (
        1.0 - v_prime + jnp.log(v_prime) - (_Z / 2.0) * q_i * (1.0 - v / f + jnp.log(v / f))
    )

    # Residual part with the temperature-dependent psi.
    a = a_main[main_index[:, None], main_index[None, :]]
    b = b_main[main_index[:, None], main_index[None, :]]
    c = c_main[main_index[:, None], main_index[None, :]]
    psi = jnp.exp(-(a + b * t + c * t**2) / t)

    x_groups = x @ nu
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


def _assemble(components: list[str]) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    unknown = [c for c in components if c not in DO_COMPONENT_GROUPS]
    if unknown:
        raise KeyError(f"no modified-UNIFAC group assignment for: {unknown}")
    subgroup_ids = sorted({sg for c in components for sg in DO_COMPONENT_GROUPS[c]})
    nu = jnp.asarray(
        [[float(DO_COMPONENT_GROUPS[c].get(sg, 0)) for sg in subgroup_ids] for c in components]
    )
    r = jnp.asarray([DO_SUBGROUPS[sg][2] for sg in subgroup_ids])
    q = jnp.asarray([DO_SUBGROUPS[sg][3] for sg in subgroup_ids])
    main_index = jnp.asarray([_MAIN_INDEX[DO_SUBGROUPS[sg][1]] for sg in subgroup_ids])
    return nu, r, q, main_index, _A_MAIN, _B_MAIN, _C_MAIN


def modified_unifac_activity(components: list[str], x: Array, t: ArrayLike) -> Array:
    """Predict ``ln(gamma)`` for named database components with modified UNIFAC.

    Args:
        components: Component names with assignments in `DO_COMPONENT_GROUPS`.
        x: Mole fractions aligned with ``components``.
        t: Temperature (K).

    Returns:
        ``ln(gamma)`` of shape ``(len(components),)``.
    """
    nu, r, q, main_index, a_main, b_main, c_main = _assemble(components)
    return modified_unifac_ln_gamma(x, nu, r, q, main_index, a_main, b_main, c_main, t)
