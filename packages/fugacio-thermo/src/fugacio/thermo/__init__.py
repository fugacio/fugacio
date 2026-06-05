"""Differentiable thermodynamics and physical-property engine for Fugacio.

Everything here is written against :mod:`jax.numpy`, so gradients flow cleanly
through the rest of the Fugacio stack (``fugacio.sim``, ``fugacio.copilot``).

The public surface is grouped as:

* **data** -- :class:`Component`, the curated :data:`DATABASE`, and helpers;
* **ideal gas** -- :func:`cp_ig`, :func:`enthalpy_ig`, :func:`entropy_ig`;
* **equations of state** -- :data:`PR`, :data:`SRK`, :data:`RK`, :data:`VDW`
  plus fugacity-coefficient and volume routines;
* **activity models** -- Margules, van Laar, Wilson, NRTL, UNIQUAC;
* **group contribution** -- :func:`unifac_activity`, :func:`joback_estimate`;
* **phase equilibrium** -- :func:`flash_pt`, :func:`psat_eos`,
  :func:`bubble_pressure_eos`, :func:`dew_pressure_eos`,
  :func:`stability_analysis`;
* **validation** -- consistency laws and finite-difference gradient checks.
"""

from fugacio.thermo.activity import (
    margules_excess_gibbs,
    margules_gamma,
    margules_ln_gamma,
    nrtl_gamma,
    nrtl_ln_gamma,
    nrtl_tau,
    uniquac_gamma,
    uniquac_ln_gamma,
    uniquac_tau,
    van_laar_gamma,
    van_laar_ln_gamma,
    wilson_gamma,
    wilson_lambda,
    wilson_ln_gamma,
)
from fugacio.thermo.components import (
    DATABASE,
    AntoineCoeffs,
    Component,
    CpIdeal,
    component_arrays,
    get,
    names,
)
from fugacio.thermo.constants import P_REF, T_REF, R
from fugacio.thermo.eos import (
    PR,
    RK,
    SRK,
    VDW,
    CubicEOS,
    compressibility,
    ln_phi_mixture,
    ln_phi_pure,
    molar_volume,
    pressure,
)
from fugacio.thermo.equilibrium import (
    FlashResult,
    StabilityResult,
    bubble_pressure_eos,
    dew_pressure_eos,
    flash_pt,
    psat_eos,
    rachford_rice,
    stability_analysis,
    wilson_k,
)
from fugacio.thermo.groupcontrib import joback_estimate, unifac_activity
from fugacio.thermo.ideal import cp_ig, enthalpy_ig, entropy_ig, gibbs_ig

__all__ = [
    "DATABASE",
    "PR",
    "P_REF",
    "RK",
    "SRK",
    "T_REF",
    "VDW",
    "AntoineCoeffs",
    "Component",
    "CpIdeal",
    "CubicEOS",
    "FlashResult",
    "R",
    "StabilityResult",
    "bubble_pressure_eos",
    "component_arrays",
    "compressibility",
    "cp_ig",
    "dew_pressure_eos",
    "enthalpy_ig",
    "entropy_ig",
    "flash_pt",
    "get",
    "gibbs_ig",
    "joback_estimate",
    "ln_phi_mixture",
    "ln_phi_pure",
    "margules_excess_gibbs",
    "margules_gamma",
    "margules_ln_gamma",
    "molar_volume",
    "names",
    "nrtl_gamma",
    "nrtl_ln_gamma",
    "nrtl_tau",
    "pressure",
    "psat_eos",
    "rachford_rice",
    "stability_analysis",
    "unifac_activity",
    "uniquac_gamma",
    "uniquac_ln_gamma",
    "uniquac_tau",
    "van_laar_gamma",
    "van_laar_ln_gamma",
    "wilson_gamma",
    "wilson_k",
    "wilson_lambda",
    "wilson_ln_gamma",
]

__version__ = "0.0.1"
