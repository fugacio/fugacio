"""PC-SAFT: a molecular-based, differentiable equation of state.

The perturbed-chain statistical associating fluid theory (PC-SAFT) of Gross &
Sadowski is the third major class of thermodynamic method in Fugacio, alongside
the cubic equations of state (`fugacio.thermo.eos`) and the gamma-phi activity
models (`fugacio.thermo.gammaphi`). It models a fluid as chains of spherical
segments with dispersion attraction and optional short-range association, which
captures associating species (water, alcohols, amines), polymers, and refrigerant
mixtures far better than a cubic EOS.

As with the reference multiparameter EOS (`fugacio.thermo.helmholtz`), PC-SAFT
is *one scalar reduced Helmholtz energy*
(`fugacio.thermo.saft.pcsaft.alpha_residual`) and every property is an autodiff
derivative of it, so the whole model, including the Wertheim association
site-fraction solve, is differentiable in both state and parameters.

The public surface is:

* **parameters**: `SaftParameters`, `saft_parameters`,
  `saft_parameters_for` (named components from the curated bank);
* **energy & properties**: `alpha_residual`, `compressibility_factor`,
  `pressure`, `molar_density`, `ln_fugacity_coefficients`,
  `residual_properties`, and `site_fractions`;
* **phase equilibrium**: `flash_pt_saft`, `bubble_pressure_saft`,
  `dew_pressure_saft`, `psat_saft`, each with a checked ``_with_info`` form;
* **regression**: `fit_saft_pure`, `fit_saft_kij`.

Use `fugacio.thermo.package.SAFTPackage` (or ``package_for(..., "pcsaft")``
in `fugacio.sim`) to run PC-SAFT inside unit operations and flowsheets.
"""

from fugacio.thermo.saft.association import (
    alpha_association,
    association_strength,
    site_fractions,
)
from fugacio.thermo.saft.equilibrium import (
    bubble_pressure_saft,
    bubble_pressure_saft_with_info,
    dew_pressure_saft,
    dew_pressure_saft_with_info,
    flash_pt_saft,
    flash_pt_saft_with_info,
    psat_saft,
    psat_saft_with_info,
)
from fugacio.thermo.saft.parameters import (
    SaftParameters,
    saft_parameters,
    saft_parameters_for,
    segment_diameter,
)
from fugacio.thermo.saft.pcsaft import (
    alpha_dispersion,
    alpha_hard_chain,
    alpha_residual,
)
from fugacio.thermo.saft.properties import (
    ResidualProperties,
    compressibility_factor,
    ln_fugacity_coefficients,
    molar_density,
    pressure,
    residual_properties,
)
from fugacio.thermo.saft.regression import fit_saft_kij, fit_saft_pure

__all__ = [
    "ResidualProperties",
    "SaftParameters",
    "alpha_association",
    "alpha_dispersion",
    "alpha_hard_chain",
    "alpha_residual",
    "association_strength",
    "bubble_pressure_saft",
    "bubble_pressure_saft_with_info",
    "compressibility_factor",
    "dew_pressure_saft",
    "dew_pressure_saft_with_info",
    "fit_saft_kij",
    "fit_saft_pure",
    "flash_pt_saft",
    "flash_pt_saft_with_info",
    "ln_fugacity_coefficients",
    "molar_density",
    "pressure",
    "psat_saft",
    "psat_saft_with_info",
    "residual_properties",
    "saft_parameters",
    "saft_parameters_for",
    "segment_diameter",
    "site_fractions",
]
