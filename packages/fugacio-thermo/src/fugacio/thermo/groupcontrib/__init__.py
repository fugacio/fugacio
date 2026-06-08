"""Group-contribution methods for parameter-free property prediction.

* :mod:`~fugacio.thermo.groupcontrib.unifac` predicts liquid-phase activity
  coefficients from functional-group interactions.
* :mod:`~fugacio.thermo.groupcontrib.joback` estimates pure-component constants
  (critical properties, boiling point, formation properties, ideal-gas Cp) from
  a count of functional groups.

These let Fugacio cover mixtures and species for which curated, fitted parameters
are unavailable -- the role the project README assigns to group contribution.
"""

from fugacio.thermo.groupcontrib.dortmund import (
    modified_unifac_activity,
    modified_unifac_ln_gamma,
)
from fugacio.thermo.groupcontrib.joback import GROUPS, JobackGroup, joback_estimate
from fugacio.thermo.groupcontrib.unifac import (
    COMPONENT_GROUPS,
    INTERACTIONS,
    SUBGROUPS,
    unifac_activity,
    unifac_ln_gamma,
)

__all__ = [
    "COMPONENT_GROUPS",
    "GROUPS",
    "INTERACTIONS",
    "SUBGROUPS",
    "JobackGroup",
    "joback_estimate",
    "modified_unifac_activity",
    "modified_unifac_ln_gamma",
    "unifac_activity",
    "unifac_ln_gamma",
]
