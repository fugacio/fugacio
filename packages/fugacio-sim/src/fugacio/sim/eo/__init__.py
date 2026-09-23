"""Simultaneous process equations using common physical unit kernels.

EOFlowsheet declares unit blocks, custom residual equations, and design
specifications. The shared graph validates connectivity and assembles local
Jacobians into sparse Newton and implicit derivative systems. Built-in units
retain their compiled thermodynamic solves and resolved phase inventories.
The optimization API supports nested and full-space formulations.
"""

from fugacio.sim.eo.blocks import (
    Block,
    Column,
    ComponentSeparator,
    Compressor,
    Context,
    Flash,
    Heater,
    HeatExchanger,
    Mixer,
    Pump,
    Scales,
    Splitter,
    StoichiometricReactor,
    Turbine,
    Valve,
)
from fugacio.sim.eo.flowsheet import (
    DOFReport,
    EOFlowsheet,
    EOSolution,
)
from fugacio.sim.eo.optimize import (
    Decision,
    EOOptResult,
    Objective,
    optimize_flowsheet_eo,
)

__all__ = [
    "Block",
    "Column",
    "ComponentSeparator",
    "Compressor",
    "Context",
    "DOFReport",
    "Decision",
    "EOFlowsheet",
    "EOOptResult",
    "EOSolution",
    "Flash",
    "HeatExchanger",
    "Heater",
    "Mixer",
    "Objective",
    "Pump",
    "Scales",
    "Splitter",
    "StoichiometricReactor",
    "Turbine",
    "Valve",
    "optimize_flowsheet_eo",
]
