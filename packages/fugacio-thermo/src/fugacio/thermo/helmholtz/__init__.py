"""Reference multiparameter (Helmholtz-energy) equations of state.

This subpackage gives Fugacio *reference-grade* pure-fluid properties: the
same class of model behind NIST REFPROP and CoolProp -- IAPWS-95 for
water/steam, Span & Wagner for CO2, Setzmann & Wagner for methane, and the
short technical formulations for the other vendored process fluids
(:func:`reference_fluid_names` lists all 26).

The entire model is one scalar function, the reduced Helmholtz energy
``alpha(delta, tau)``; every property is an exact :func:`jax.grad` derivative
of it (:mod:`~fugacio.thermo.helmholtz.terms`,
:mod:`~fugacio.thermo.helmholtz.props`). Saturation lines come from a
differentiable Maxwell construction
(:mod:`~fugacio.thermo.helmholtz.saturation`), ``(T, P)`` / ``(P, h)`` /
``(P, s)`` / quality states from steam-table style resolvers
(:mod:`~fugacio.thermo.helmholtz.states`), and water carries the full IAPWS
transport formulations with autodiff critical enhancements
(:mod:`~fugacio.thermo.helmholtz.iapws`).

Validation is layered like the rest of Fugacio: hermetic unit tests pin the
published IAPWS-95/Span-Wagner check tables, consistency tests assert
first-principles identities through the solvers (Clausius-Clapeyron along the
*solved* saturation line, Maxwell relations, ``cp - cv``), and the opt-in
oracle suite grades dense property grids for every fluid against CoolProp.

Example::

    from fugacio.thermo.helmholtz import reference_fluid, saturation_state, state_ph
    import jax

    steam = reference_fluid("water")
    sat = saturation_state(steam, p=10e5)          # 10 bar saturation
    sat.t, sat.h_vaporization                      # 453.03 K, 2014.6 kJ/kg * M

    # d(Tsat)/dP along the solved line -- the Clausius-Clapeyron slope:
    jax.grad(lambda p: saturation_state(steam, p=p).t)(10e5)
"""

from __future__ import annotations

from fugacio.thermo.helmholtz.density import molar_density
from fugacio.thermo.helmholtz.fluids import (
    Ancillary,
    HelmholtzFluid,
    has_reference_fluid,
    reference_fluid,
    reference_fluid_names,
)
from fugacio.thermo.helmholtz.iapws import water_thermal_conductivity, water_viscosity
from fugacio.thermo.helmholtz.props import (
    compressibility_factor,
    enthalpy,
    entropy,
    fugacity,
    gibbs_energy,
    helmholtz_energy,
    internal_energy,
    isobaric_expansivity,
    isobaric_heat_capacity,
    isochoric_heat_capacity,
    isothermal_compressibility,
    joule_thomson,
    ln_fugacity_coefficient,
    pressure,
    second_virial,
    speed_of_sound,
    third_virial,
)
from fugacio.thermo.helmholtz.saturation import (
    SaturationState,
    psat_ancillary,
    rho_liquid_ancillary,
    rho_vapor_ancillary,
    saturation_densities,
    saturation_pressure,
    saturation_state,
    saturation_temperature,
)
from fugacio.thermo.helmholtz.states import (
    FluidState,
    state_ph,
    state_pq,
    state_ps,
    state_tp,
    state_tq,
)
from fugacio.thermo.helmholtz.surface import surface_tension
from fugacio.thermo.helmholtz.terms import (
    AlphaDerivatives,
    alpha_derivatives,
    ideal_alpha,
    residual_alpha,
)

__all__ = [
    "AlphaDerivatives",
    "Ancillary",
    "FluidState",
    "HelmholtzFluid",
    "SaturationState",
    "alpha_derivatives",
    "compressibility_factor",
    "enthalpy",
    "entropy",
    "fugacity",
    "gibbs_energy",
    "has_reference_fluid",
    "helmholtz_energy",
    "ideal_alpha",
    "internal_energy",
    "isobaric_expansivity",
    "isobaric_heat_capacity",
    "isochoric_heat_capacity",
    "isothermal_compressibility",
    "joule_thomson",
    "ln_fugacity_coefficient",
    "molar_density",
    "pressure",
    "psat_ancillary",
    "reference_fluid",
    "reference_fluid_names",
    "residual_alpha",
    "rho_liquid_ancillary",
    "rho_vapor_ancillary",
    "saturation_densities",
    "saturation_pressure",
    "saturation_state",
    "saturation_temperature",
    "second_virial",
    "speed_of_sound",
    "state_ph",
    "state_pq",
    "state_ps",
    "state_tp",
    "state_tq",
    "surface_tension",
    "third_virial",
    "water_thermal_conductivity",
    "water_viscosity",
]
