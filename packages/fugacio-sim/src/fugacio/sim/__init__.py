"""Differentiable process-simulation layer for Fugacio (depends on ``fugacio.thermo``).

The layer provides:

* :class:`Stream` -- the differentiable material stream passed between units;
* a stream property bridge (:func:`enthalpy_flow`, :func:`entropy_flow`,
  :func:`molar_enthalpy`, :func:`molar_entropy`, :func:`mass_flow`) that gives any
  stream a two-phase-aware enthalpy/entropy via :mod:`fugacio.thermo`;
* unit operations with rigorous material *and* energy balances
  (:func:`flash_drum`, :func:`heater`, :func:`valve`, :func:`pump`,
  :func:`compressor`, :func:`turbine`, :func:`mix`, :func:`splitter`,
  :func:`component_separator`);
* a thermodynamic-model bridge (:func:`eos_model_for`, :func:`nrtl_model_for`,
  :func:`uniquac_model_for`, :func:`unifac_model_for`) that turns component names
  into a ready EOS or gamma-phi :class:`~fugacio.thermo.EquilibriumModel`;
* non-ideal separation units (:func:`flash_vle`, :func:`decanter`,
  :func:`three_phase_flash`) and binary diagram / azeotrope / residue-curve helpers
  (:func:`pxy_diagram`, :func:`txy_diagram`, :func:`azeotrope_pressure`,
  :func:`azeotrope_temperature`, :func:`residue_curve`, :func:`residue_curve_map`);
* reactor unit operations (:func:`equilibrium_reactor`,
  :func:`stoichiometric_reactor`, :func:`cstr`, :func:`pfr`,
  :func:`batch_reactor`) and reactive separations (:func:`reactive_flash`,
  :func:`reactive_distillation`);
* a differentiable optimization toolkit (:func:`minimize`, :func:`argmin`,
  :func:`least_squares`) that differentiates *through the optimum* by the
  implicit function theorem;
* design specifications and set-point controllers (:func:`meet_spec`,
  :func:`solve_design`, :func:`controller`) that adjust a degree of freedom to
  hit a target, plus end-to-end flowsheet optimization (:func:`optimize_flowsheet`);
* equipment sizing, Turton bare-module costing, utility pricing, and financial
  metrics (:func:`heat_exchanger_area`, :func:`bare_module_cost`,
  :func:`utility_cost`, :func:`total_annual_cost`, :func:`npv`) for money-valued
  design objectives;
* the original lightweight modified-Raoult helpers (:func:`bubble_pressure`,
  :func:`antoine_psat`).
"""

from fugacio.sim.column import (
    ColumnResult,
    ShortcutResult,
    fenske_min_stages,
    gilliland_stages,
    kirkbride_feed_stage,
    relative_volatility,
    shortcut_column,
    solve_column,
    underwood_min_reflux,
)
from fugacio.sim.design import (
    DesignSpec,
    FlowsheetOptResult,
    SpecResult,
    controller,
    meet_spec,
    optimize_flowsheet,
    solve_design,
)
from fugacio.sim.diagrams import (
    AzeotropeResult,
    PxyDiagram,
    ResidueCurve,
    TxyDiagram,
    azeotrope_pressure,
    azeotrope_temperature,
    pxy_diagram,
    residue_curve,
    residue_curve_map,
    txy_diagram,
)
from fugacio.sim.economics import (
    CEPCI_DEFAULT,
    CEPCI_REF,
    EquipmentCost,
    Utility,
    annualized_capital,
    bare_module_cost,
    capital_recovery_factor,
    column_diameter,
    column_height,
    cylinder_volume,
    discounted_payback,
    heat_exchanger_area,
    installed_capital,
    lmtd,
    npv,
    pressure_factor,
    purchased_cost,
    total_annual_cost,
    utility_cost,
    vapor_molar_volume_ideal,
    vessel_volume,
)
from fugacio.sim.flowsheet import Flowsheet, tear_solve
from fugacio.sim.models import (
    UnifacModel,
    eos_model_for,
    nrtl_model_for,
    unifac_model_for,
    uniquac_model_for,
)
from fugacio.sim.optimize import (
    OptimizeResult,
    argmin,
    least_squares,
    minimize,
)
from fugacio.sim.properties import (
    enthalpy_flow,
    entropy_flow,
    mass_flow,
    molar_enthalpy,
    molar_entropy,
    molar_mass,
)
from fugacio.sim.reactive import (
    ReactiveColumnResult,
    ReactiveFlashResult,
    reactive_distillation,
    reactive_flash,
)
from fugacio.sim.reactors import (
    ReactorResult,
    batch_reactor,
    conversion,
    cstr,
    equilibrium_reactor,
    pfr,
    stoichiometric_reactor,
)
from fugacio.sim.separations import decanter, flash_vle, three_phase_flash
from fugacio.sim.stream import Stream
from fugacio.sim.units import (
    HeaterResult,
    PumpResult,
    WorkResult,
    component_separator,
    compressor,
    flash_drum,
    heater,
    mix,
    pump,
    splitter,
    turbine,
    valve,
)
from fugacio.sim.vle import antoine_psat, bubble_pressure

__all__ = [
    "CEPCI_DEFAULT",
    "CEPCI_REF",
    "AzeotropeResult",
    "ColumnResult",
    "DesignSpec",
    "EquipmentCost",
    "Flowsheet",
    "FlowsheetOptResult",
    "HeaterResult",
    "OptimizeResult",
    "PumpResult",
    "PxyDiagram",
    "ReactiveColumnResult",
    "ReactiveFlashResult",
    "ReactorResult",
    "ResidueCurve",
    "ShortcutResult",
    "SpecResult",
    "Stream",
    "TxyDiagram",
    "UnifacModel",
    "Utility",
    "WorkResult",
    "annualized_capital",
    "antoine_psat",
    "argmin",
    "azeotrope_pressure",
    "azeotrope_temperature",
    "bare_module_cost",
    "batch_reactor",
    "bubble_pressure",
    "capital_recovery_factor",
    "column_diameter",
    "column_height",
    "component_separator",
    "compressor",
    "controller",
    "conversion",
    "cstr",
    "cylinder_volume",
    "decanter",
    "discounted_payback",
    "enthalpy_flow",
    "entropy_flow",
    "eos_model_for",
    "equilibrium_reactor",
    "fenske_min_stages",
    "flash_drum",
    "flash_vle",
    "gilliland_stages",
    "heat_exchanger_area",
    "heater",
    "installed_capital",
    "kirkbride_feed_stage",
    "least_squares",
    "lmtd",
    "mass_flow",
    "meet_spec",
    "minimize",
    "mix",
    "molar_enthalpy",
    "molar_entropy",
    "molar_mass",
    "npv",
    "nrtl_model_for",
    "optimize_flowsheet",
    "pfr",
    "pressure_factor",
    "pump",
    "purchased_cost",
    "pxy_diagram",
    "reactive_distillation",
    "reactive_flash",
    "relative_volatility",
    "residue_curve",
    "residue_curve_map",
    "shortcut_column",
    "solve_column",
    "solve_design",
    "splitter",
    "stoichiometric_reactor",
    "tear_solve",
    "three_phase_flash",
    "total_annual_cost",
    "turbine",
    "txy_diagram",
    "underwood_min_reflux",
    "unifac_model_for",
    "uniquac_model_for",
    "utility_cost",
    "valve",
    "vapor_molar_volume_ideal",
    "vessel_volume",
]

__version__ = "0.0.1"
