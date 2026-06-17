"""Differentiable dynamic (time-domain) process simulation for Fugacio.

The `fugacio.sim` blocks elsewhere are steady-state; this subpackage adds the
time dimension while keeping everything end-to-end differentiable:

* `integrate`: the ODE integration core: a fixed
  output-grid `odeint` (explicit and stiff implicit steppers, differentiable
  in both modes) and an adaptive `integrate` with a continuous-adjoint
  ``custom_vjp``;
* `units`: dynamic unit operations carried as ODEs in
  conserved holdups (buffer/level tanks, mixing tanks, dynamic CSTR, dynamic
  flash, lumped thermal mass / heat exchanger, gas receiver), reusing the
  steady-state thermodynamics for the instantaneous constitutive relations;
* `flowsheet`: `DynamicFlowsheet`, a
  declarative assembly of dynamic units and controllers into one global ODE that
  is simulated over time and differentiated through;
* `optimize`: dynamic optimization and estimation
  (optimal control over input trajectories, time-series parameter estimation),
  composing the existing optimizers with the integrator.
"""

from __future__ import annotations

from fugacio.sim.dynamics.flowsheet import (
    DynamicFlowsheet,
    DynamicResult,
    simulate,
)
from fugacio.sim.dynamics.integrate import (
    FIXED_METHODS,
    IMPLICIT_METHODS,
    ODEResult,
    integrate,
    odeint,
    odeint_final,
)
from fugacio.sim.dynamics.optimize import (
    DynamicEstimateResult,
    OptimalControlResult,
    estimate_dynamics,
    optimal_control,
    tune_pid,
)
from fugacio.sim.dynamics.units import (
    DynamicCSTR,
    DynamicFlash,
    DynamicUnit,
    GasReceiver,
    LevelTank,
    MixingTank,
    ThermalMass,
)

__all__ = [
    "FIXED_METHODS",
    "IMPLICIT_METHODS",
    "DynamicCSTR",
    "DynamicEstimateResult",
    "DynamicFlash",
    "DynamicFlowsheet",
    "DynamicResult",
    "DynamicUnit",
    "GasReceiver",
    "LevelTank",
    "MixingTank",
    "ODEResult",
    "OptimalControlResult",
    "ThermalMass",
    "estimate_dynamics",
    "integrate",
    "odeint",
    "odeint_final",
    "optimal_control",
    "simulate",
    "tune_pid",
]
