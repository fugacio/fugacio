"""Equation-oriented (EO) flowsheeting for Fugacio.

The sequential-modular engine (`fugacio.sim.flowsheet`) evaluates units in order
and tears recycles; this package instead assembles every unit's equations, the
stream connectivity, the recycles, and any design specs into one residual system
and solves it **simultaneously** by Newton's method. The Jacobian, the one hard
ingredient of a classical EO solver, comes exactly from JAX autodiff, and the
converged flowsheet is differentiable with respect to its parameters and feeds by
the implicit function theorem, so whole-plant gradient optimization needs no tear
and no nested loops.

The public surface is:

* `EOFlowsheet`: build a flowsheet from feeds, `Block` units, and design specs,
  run a degrees-of-freedom check (`DOFReport`), and `EOFlowsheet.solve` it to an
  `EOSolution`;
* unit blocks mirroring the sequential-modular units, each written as residual
  equations: `Mixer`, `Splitter`, `Heater`, `Valve`, `Pump`, `Compressor`,
  `Turbine`, `Flash`, `ComponentSeparator`;
* `Scales` / `Context` for residual conditioning and shared solve data;
* `optimize_flowsheet_eo` (with `EOOptResult`): nested or full-space simultaneous
  optimization over named decision variables.
"""

from fugacio.sim.eo.blocks import (
    Block,
    ComponentSeparator,
    Compressor,
    Context,
    Flash,
    Heater,
    Mixer,
    Pump,
    Scales,
    Splitter,
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
    "ComponentSeparator",
    "Compressor",
    "Context",
    "DOFReport",
    "Decision",
    "EOFlowsheet",
    "EOOptResult",
    "EOSolution",
    "Flash",
    "Heater",
    "Mixer",
    "Objective",
    "Pump",
    "Scales",
    "Splitter",
    "Turbine",
    "Valve",
    "optimize_flowsheet_eo",
]
