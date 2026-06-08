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
from fugacio.sim.flowsheet import Flowsheet, tear_solve
from fugacio.sim.models import (
    UnifacModel,
    eos_model_for,
    nrtl_model_for,
    unifac_model_for,
    uniquac_model_for,
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
    "AzeotropeResult",
    "ColumnResult",
    "Flowsheet",
    "HeaterResult",
    "PumpResult",
    "PxyDiagram",
    "ReactiveColumnResult",
    "ReactiveFlashResult",
    "ReactorResult",
    "ResidueCurve",
    "ShortcutResult",
    "Stream",
    "TxyDiagram",
    "UnifacModel",
    "WorkResult",
    "antoine_psat",
    "azeotrope_pressure",
    "azeotrope_temperature",
    "batch_reactor",
    "bubble_pressure",
    "component_separator",
    "compressor",
    "conversion",
    "cstr",
    "decanter",
    "enthalpy_flow",
    "entropy_flow",
    "eos_model_for",
    "equilibrium_reactor",
    "fenske_min_stages",
    "flash_drum",
    "flash_vle",
    "gilliland_stages",
    "heater",
    "kirkbride_feed_stage",
    "mass_flow",
    "mix",
    "molar_enthalpy",
    "molar_entropy",
    "molar_mass",
    "nrtl_model_for",
    "pfr",
    "pump",
    "pxy_diagram",
    "reactive_distillation",
    "reactive_flash",
    "relative_volatility",
    "residue_curve",
    "residue_curve_map",
    "shortcut_column",
    "solve_column",
    "splitter",
    "stoichiometric_reactor",
    "tear_solve",
    "three_phase_flash",
    "turbine",
    "txy_diagram",
    "underwood_min_reflux",
    "unifac_model_for",
    "uniquac_model_for",
    "valve",
]

__version__ = "0.0.1"
