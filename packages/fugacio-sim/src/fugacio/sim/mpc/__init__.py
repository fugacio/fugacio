"""Differentiable model predictive control and state estimation for Fugacio.

Advanced process control is where a differentiable engine earns its keep: every
controller here is built from the same autodiff primitives as the rest of Fugacio,
so a closed-loop performance index has exact gradients with respect to the
*controller's own tuning*, and the controllers compose with the dynamic flowsheet
they regulate. The subpackage is layered bottom-up:

* :mod:`~fugacio.sim.mpc.riccati` -- differentiable Riccati solvers (DARE/CARE),
  the infinite-horizon LQR gain (:func:`dlqr`, :func:`lqr`) and the steady-state
  Kalman gain (:func:`kalman_gain`), via fixed-iteration structured-doubling /
  matrix-sign recursions that backprop cleanly.
* :mod:`~fugacio.sim.mpc.qp` -- a small dense convex QP solver
  (:func:`solve_qp`) in the OSQP/ADMM style with an active-set polish and an
  implicit-function-theorem ``custom_vjp``, the differentiable core of linear MPC.
* :mod:`~fugacio.sim.mpc.linear` -- condensed-QP linear and offset-free MPC
  (:func:`linear_mpc`) with an LQR terminal cost, a disturbance-observer for
  zero steady-state offset, and hard input / soft output constraints.
* :mod:`~fugacio.sim.mpc.estimation` -- the Kalman filter, the extended and
  unscented Kalman filters, and optimization-based moving-horizon estimation
  (:func:`moving_horizon_estimate`), all differentiable.
* :mod:`~fugacio.sim.mpc.nonlinear` -- nonlinear and economic MPC
  (:func:`nonlinear_mpc`) by direct single-shooting over the true nonlinear model,
  optimized with :func:`fugacio.sim.argmin` (differentiating through the optimum).
* :mod:`~fugacio.sim.mpc.simulate` -- a one-``scan`` closed-loop harness
  (:func:`simulate_closed_loop`) and gradient-based weight tuning
  (:func:`tune_mpc`).
"""

from __future__ import annotations

from fugacio.sim.mpc.estimation import (
    ExtendedKalmanFilter,
    GaussianState,
    KalmanFilter,
    MHEResult,
    UnscentedKalmanFilter,
    moving_horizon_estimate,
)
from fugacio.sim.mpc.linear import (
    LinearMPC,
    MPCResult,
    MPCState,
    c2d,
    linear_mpc,
)
from fugacio.sim.mpc.nonlinear import (
    NMPCResult,
    NonlinearMPC,
    Transition,
    discretize,
    nonlinear_mpc,
    quadratic_tracking,
)
from fugacio.sim.mpc.qp import (
    QPSettings,
    QPSolution,
    solve_qp,
    solve_qp_canonical,
)
from fugacio.sim.mpc.riccati import (
    care,
    dare,
    dlqr,
    kalman_gain,
    lqr,
    riccati_residual_continuous,
    riccati_residual_discrete,
)
from fugacio.sim.mpc.simulate import (
    ClosedLoop,
    closed_loop_cost,
    constant_setpoint,
    linear_feedback,
    nonlinear_feedback,
    simulate_closed_loop,
    tune_mpc,
)

__all__ = [
    "ClosedLoop",
    "ExtendedKalmanFilter",
    "GaussianState",
    "KalmanFilter",
    "LinearMPC",
    "MHEResult",
    "MPCResult",
    "MPCState",
    "NMPCResult",
    "NonlinearMPC",
    "QPSettings",
    "QPSolution",
    "Transition",
    "UnscentedKalmanFilter",
    "c2d",
    "care",
    "closed_loop_cost",
    "constant_setpoint",
    "dare",
    "discretize",
    "dlqr",
    "kalman_gain",
    "linear_feedback",
    "linear_mpc",
    "lqr",
    "moving_horizon_estimate",
    "nonlinear_feedback",
    "nonlinear_mpc",
    "quadratic_tracking",
    "riccati_residual_continuous",
    "riccati_residual_discrete",
    "simulate_closed_loop",
    "solve_qp",
    "solve_qp_canonical",
    "tune_mpc",
]
