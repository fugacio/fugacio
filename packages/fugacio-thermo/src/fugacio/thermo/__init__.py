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
from fugacio.thermo.departure import (
    ResidualProperties,
    residual_cp,
    residual_enthalpy,
    residual_entropy,
    residual_gibbs,
    residual_properties,
)
from fugacio.thermo.energy import (
    EnergyFlashResult,
    flash_ph,
    flash_ps,
    mixture_enthalpy,
    mixture_entropy,
)
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
from fugacio.thermo.ideal import (
    cp_ig,
    enthalpy_ig,
    enthalpy_ig_mixture,
    entropy_ig,
    entropy_ig_mixture,
    gibbs_ig,
    gibbs_ig_mixture,
    ideal_gas_coeffs,
)
from fugacio.thermo.properties import (
    molar_cp,
    molar_enthalpy,
    molar_entropy,
    molar_gibbs,
    speed_of_sound_ideal,
    stable_phase,
)

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
    "EnergyFlashResult",
    "FlashResult",
    "R",
    "ResidualProperties",
    "StabilityResult",
    "bubble_pressure_eos",
    "component_arrays",
    "compressibility",
    "cp_ig",
    "dew_pressure_eos",
    "enthalpy_ig",
    "enthalpy_ig_mixture",
    "entropy_ig",
    "entropy_ig_mixture",
    "flash_ph",
    "flash_ps",
    "flash_pt",
    "get",
    "gibbs_ig",
    "gibbs_ig_mixture",
    "ideal_gas_coeffs",
    "joback_estimate",
    "ln_phi_mixture",
    "ln_phi_pure",
    "margules_excess_gibbs",
    "margules_gamma",
    "margules_ln_gamma",
    "mixture_enthalpy",
    "mixture_entropy",
    "molar_cp",
    "molar_enthalpy",
    "molar_entropy",
    "molar_gibbs",
    "molar_volume",
    "names",
    "nrtl_gamma",
    "nrtl_ln_gamma",
    "nrtl_tau",
    "pressure",
    "psat_eos",
    "rachford_rice",
    "residual_cp",
    "residual_enthalpy",
    "residual_entropy",
    "residual_gibbs",
    "residual_properties",
    "speed_of_sound_ideal",
    "stability_analysis",
    "stable_phase",
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
