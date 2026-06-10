"""Surface tension of reference fluids (Mulero-Cachadina-IAPWS correlations).

Every vendored fluid carries the recommended surface-tension correlation

    sigma(T) = sum_i a_i * (1 - T/Tc)**e_i        (N/m)

from the Mulero, Cachadina & Parra compilation (J. Phys. Chem. Ref. Data,
2012) -- for water this is identical to the IAPWS R1-76 release. The
correlation is differentiable in ``T`` (guarded at the critical point, where
``sigma -> 0`` with an infinite slope for exponents below one).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.helmholtz.fluids import HelmholtzFluid

ArrayLike = Array | float


def surface_tension(fluid: HelmholtzFluid, t: ArrayLike) -> Array:
    """Vapor-liquid surface tension (N/m) at ``t`` (K); zero at and above ``Tc``."""
    theta = jnp.clip(1.0 - jnp.asarray(t, dtype=float) / fluid.sigma_tc, 0.0, 1.0)
    positive = theta > 0.0
    powered = jnp.where(positive, theta, 1.0) ** fluid.sigma_e
    return jnp.sum(fluid.sigma_a * jnp.where(positive, powered, 0.0))


__all__ = ["surface_tension"]
