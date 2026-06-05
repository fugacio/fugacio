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
from fugacio.sim.flowsheet import Flowsheet, tear_solve
from fugacio.sim.properties import (
    enthalpy_flow,
    entropy_flow,
    mass_flow,
    molar_enthalpy,
    molar_entropy,
    molar_mass,
)
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
    "ColumnResult",
    "Flowsheet",
    "HeaterResult",
    "PumpResult",
    "ShortcutResult",
    "Stream",
    "WorkResult",
    "antoine_psat",
    "bubble_pressure",
    "component_separator",
    "compressor",
    "enthalpy_flow",
    "entropy_flow",
    "fenske_min_stages",
    "flash_drum",
    "gilliland_stages",
    "heater",
    "kirkbride_feed_stage",
    "mass_flow",
    "mix",
    "molar_enthalpy",
    "molar_entropy",
    "molar_mass",
    "pump",
    "relative_volatility",
    "shortcut_column",
    "solve_column",
    "splitter",
    "tear_solve",
    "turbine",
    "underwood_min_reflux",
    "valve",
]

__version__ = "0.0.1"
