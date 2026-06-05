"""Activity-coefficient models for non-ideal liquid mixtures.

The package collects excess-Gibbs / activity-coefficient models that share a
common, differentiable ``jax.numpy`` implementation:

* :mod:`~fugacio.thermo.activity.margules` and
  :mod:`~fugacio.thermo.activity.vanlaar` -- two-parameter binary models;
* :mod:`~fugacio.thermo.activity.wilson`,
  :mod:`~fugacio.thermo.activity.nrtl` and
  :mod:`~fugacio.thermo.activity.uniquac` -- multicomponent local-composition
  models.

Each model exposes ``*_ln_gamma`` and ``*_gamma`` functions; the local-
composition models additionally provide builders that assemble their interaction
matrices from temperature and physical parameters.
"""

from fugacio.thermo.activity.margules import (
    margules_excess_gibbs,
    margules_gamma,
    margules_ln_gamma,
)
from fugacio.thermo.activity.nrtl import (
    nrtl_excess_gibbs,
    nrtl_g,
    nrtl_gamma,
    nrtl_ln_gamma,
    nrtl_tau,
)
from fugacio.thermo.activity.uniquac import (
    uniquac_gamma,
    uniquac_ln_gamma,
    uniquac_tau,
)
from fugacio.thermo.activity.vanlaar import van_laar_gamma, van_laar_ln_gamma
from fugacio.thermo.activity.wilson import (
    wilson_excess_gibbs,
    wilson_gamma,
    wilson_lambda,
    wilson_ln_gamma,
)

__all__ = [
    "margules_excess_gibbs",
    "margules_gamma",
    "margules_ln_gamma",
    "nrtl_excess_gibbs",
    "nrtl_g",
    "nrtl_gamma",
    "nrtl_ln_gamma",
    "nrtl_tau",
    "uniquac_gamma",
    "uniquac_ln_gamma",
    "uniquac_tau",
    "van_laar_gamma",
    "van_laar_ln_gamma",
    "wilson_excess_gibbs",
    "wilson_gamma",
    "wilson_lambda",
    "wilson_ln_gamma",
]
