"""Activity-coefficient models for non-ideal liquid mixtures.

The package collects excess-Gibbs / activity-coefficient models that share a
common, differentiable ``jax.numpy`` implementation:

* `margules` and
  `vanlaar`: two-parameter binary models;
* `wilson`,
  `nrtl` and
  `uniquac`: multicomponent local-composition
  models;
* `regular_solution`: Scatchard-Hildebrand
  regular-solution and Flory-Huggins models from pure-component descriptors.

Each model exposes ``*_ln_gamma`` and ``*_gamma`` functions; the local-
composition models additionally provide builders that assemble their interaction
matrices from temperature and physical parameters.

For a uniform, object-oriented interface (used by the gamma-phi equilibrium
engine and the parameter regressor), `models`
wraps each kernel in a differentiable model object implementing
`ActivityModel`.
"""

from fugacio.thermo.activity.margules import (
    margules_excess_gibbs,
    margules_gamma,
    margules_ln_gamma,
)
from fugacio.thermo.activity.models import (
    NRTL,
    UNIQUAC,
    ActivityModel,
    FloryHuggins,
    Hildebrand,
    Margules,
    RegularSolution,
    VanLaar,
    Wilson,
    excess_gibbs,
    gamma,
    margules,
    nrtl,
    nrtl_from_energies,
    uniquac,
    uniquac_from_energies,
    van_laar,
)
from fugacio.thermo.activity.nrtl import (
    nrtl_excess_gibbs,
    nrtl_g,
    nrtl_gamma,
    nrtl_ln_gamma,
    nrtl_tau,
)
from fugacio.thermo.activity.regular_solution import (
    flory_huggins_gamma,
    flory_huggins_ln_gamma,
    hildebrand_ln_gamma,
    regular_solution_gamma,
    regular_solution_ln_gamma,
    volume_fractions,
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
    "NRTL",
    "UNIQUAC",
    "ActivityModel",
    "FloryHuggins",
    "Hildebrand",
    "Margules",
    "RegularSolution",
    "VanLaar",
    "Wilson",
    "excess_gibbs",
    "flory_huggins_gamma",
    "flory_huggins_ln_gamma",
    "gamma",
    "hildebrand_ln_gamma",
    "margules",
    "margules_excess_gibbs",
    "margules_gamma",
    "margules_ln_gamma",
    "nrtl",
    "nrtl_excess_gibbs",
    "nrtl_from_energies",
    "nrtl_g",
    "nrtl_gamma",
    "nrtl_ln_gamma",
    "nrtl_tau",
    "regular_solution_gamma",
    "regular_solution_ln_gamma",
    "uniquac",
    "uniquac_from_energies",
    "uniquac_gamma",
    "uniquac_ln_gamma",
    "uniquac_tau",
    "van_laar",
    "van_laar_gamma",
    "van_laar_ln_gamma",
    "volume_fractions",
    "wilson_excess_gibbs",
    "wilson_gamma",
    "wilson_lambda",
    "wilson_ln_gamma",
]
