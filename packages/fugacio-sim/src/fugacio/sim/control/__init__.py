"""Classical and differentiable process control for Fugacio.

This subpackage supplies the control side of dynamic simulation:

* `pid`: a realizable, anti-windup `PID`
  controller whose gains are a differentiable pytree (so they can be tuned by
  gradient descent), carried as ODE states inside a dynamic flowsheet;
* `blocks`: linear blocks (first/second order, FOPDT,
  lead-lag) with both analytic step responses and state-space realizations, plus
  static actuator nonlinearities;
* `metrics`: step-response performance metrics
  (overshoot, rise/settling time, IAE/ISE/ITAE) for reporting and as tuning
  objectives;
* `tuning`: FOPDT model identification and the
  classical tuning rules (Ziegler-Nichols, Cohen-Coon, IMC/lambda, AMIGO);
* `linearize`: exact autodiff linearization of a
  nonlinear plant into a `StateSpace`, with poles, Bode response, and
  controllability/observability.
"""

from __future__ import annotations

from fugacio.sim.control.blocks import (
    dead_band,
    first_order_ss,
    first_order_step,
    fopdt_step,
    lead_lag,
    rate_limit,
    saturate,
    second_order_ss,
    second_order_step,
)
from fugacio.sim.control.linearize import (
    StateSpace,
    bode,
    controllability,
    dc_gain,
    frequency_response,
    is_controllable,
    is_observable,
    is_stable,
    linearize,
    observability,
    poles,
)
from fugacio.sim.control.metrics import (
    StepInfo,
    iae,
    ise,
    itae,
    overshoot,
    peak_time,
    rise_time,
    settling_time,
    steady_state_error,
    step_info,
)
from fugacio.sim.control.pid import PID, PIDState, p_only, pi
from fugacio.sim.control.tuning import (
    FOPDTModel,
    amigo,
    cohen_coon,
    fit_fopdt,
    imc_tuning,
    ziegler_nichols,
)

__all__ = [
    "PID",
    "FOPDTModel",
    "PIDState",
    "StateSpace",
    "StepInfo",
    "amigo",
    "bode",
    "cohen_coon",
    "controllability",
    "dc_gain",
    "dead_band",
    "first_order_ss",
    "first_order_step",
    "fit_fopdt",
    "fopdt_step",
    "frequency_response",
    "iae",
    "imc_tuning",
    "is_controllable",
    "is_observable",
    "is_stable",
    "ise",
    "itae",
    "lead_lag",
    "linearize",
    "observability",
    "overshoot",
    "p_only",
    "peak_time",
    "pi",
    "poles",
    "rate_limit",
    "rise_time",
    "saturate",
    "second_order_ss",
    "second_order_step",
    "settling_time",
    "steady_state_error",
    "step_info",
    "ziegler_nichols",
]
