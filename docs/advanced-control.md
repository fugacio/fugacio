# Advanced process control (MPC & state estimation)

The [dynamics & control](dynamics.md) layer closes single PID loops around a
plant. Real plants are **multivariable** and **constrained** (interacting loops,
actuators that saturate, products that must stay on spec), and that's the
province of *model predictive control*. At every sample MPC solves a small
optimal-control problem over a receding horizon, honoring the constraints
explicitly; paired with a **state estimator** it reconstructs the unmeasured
state (and unmeasured disturbances) the controller needs.

The `fugacio.sim.mpc` subpackage builds this from the same autodiff primitives as
the rest of the engine, so two things hold that are unusual for a control
library. First, every solver (the Riccati recursion, the QP, the nonlinear
program, the moving-horizon estimator) is **differentiable through its own
solution**, so a closed-loop performance index has *exact* gradients with respect
to the controller's tuning. Second, the controllers compose with the
[dynamic flowsheet](dynamics.md) they regulate. The package is layered bottom-up,
and so is this page.

## Riccati, LQR & the steady-state Kalman gain

The bedrock of linear control is the algebraic Riccati equation. `dare` and
`care` solve the discrete and continuous equations by fixed-iteration
**structured-doubling** / matrix-sign recursions, no Python-level convergence
branch, so the whole solve is a clean `jax.lax.scan` that backpropagates. From
them, `dlqr` / `lqr` return the infinite-horizon **LQR** state-feedback gain
`u = -K x` and its cost-to-go, and `kalman_gain` returns the dual: the
steady-state **Kalman** filter gain and prior error covariance.

```python
import jax, jax.numpy as jnp
from fugacio.sim import dlqr, kalman_gain

dt = 0.1                                          # a discrete double integrator
a = jnp.array([[1.0, dt], [0.0, 1.0]])           # state = (position, velocity)
b = jnp.array([[0.5 * dt**2], [dt]])
q = jnp.diag(jnp.array([1.0, 0.0]))              # penalize position error
r = jnp.array([[0.1]])

k, p_cost = dlqr(a, b, q, r)                      # optimal gain u = -K x
c = jnp.array([[1.0, 0.0]])                       # measure position only
gain, sigma = kalman_gain(a, c, 1e-3 * jnp.eye(2), jnp.array([[1e-2]]))
```

Because the solve is differentiable, the *sensitivity* of any LQR/Kalman quantity
to the design weights is one `jax.grad` away, useful for control-aware design,
where the plant and its controller are tuned together:

```python
# How the optimal cost-to-go responds to the input weight, through the Riccati solve.
dP_dr = jax.grad(lambda w: dlqr(a, b, q, jnp.array([[w]]))[1][0, 0])(0.1)
```

## A differentiable quadratic program

Linear MPC is, at heart, a convex **quadratic program** solved every sample.
`solve_qp` is a small dense QP solver in the OSQP/ADMM style: a fixed number of
ADMM iterations (a differentiable `scan`) followed by an active-set **polish**,
and, crucially, a hand-written `custom_vjp` that differentiates the *solution*
through the KKT system (the implicit-function theorem), not the iteration. So you
can put a QP anywhere inside a differentiable program.

```python
import jax, jax.numpy as jnp
from fugacio.sim import solve_qp

p = jnp.array([[2.0, 0.0], [0.0, 2.0]])
q = jnp.array([-2.0, -5.0])
g = jnp.array([[1.0, 1.0]])                       # x0 + x1 <= 1
sol = solve_qp(p, q, g_ineq=g, h_ineq=jnp.array([1.0]), lb=jnp.zeros(2))
sol.x, sol.obj, bool(sol.converged)

# Differentiate the optimizer: how the solution moves with the constraint bound.
dx_dh = jax.jacobian(
    lambda h: solve_qp(p, q, g_ineq=g, h_ineq=jnp.array([h]), lb=jnp.zeros(2)).x
)(1.0)
```

## Linear & offset-free MPC

`linear_mpc` assembles a ready controller from a `StateSpace` model and weights.
It uses the **condensed** QP (decision variable = the input sequence), an **LQR
terminal cost** (the DARE cost-to-go, so a short horizon inherits the stability of
the infinite-horizon law), hard input magnitude/rate limits, and *soft* output
limits. By default it carries an **output-disturbance observer**, the textbook
recipe for **offset-free** tracking: a step load or plant/model mismatch is
absorbed into an estimated disturbance and driven out of the steady-state error.

The controller mirrors the [`PID`](dynamics.md) interface (`init_state` then
`step(state, y_meas, r) -> (u, state)`), so it drops straight into a loop.

```python
import jax.numpy as jnp
from fugacio.sim import StateSpace, linear_mpc

dt = 0.1
ss = StateSpace(
    a=jnp.array([[1.0, dt], [0.0, 1.0]]),
    b=jnp.array([[0.5 * dt**2], [dt]]),
    c=jnp.array([[1.0, 0.0]]),
    d=jnp.zeros((1, 1)),
)

mpc = linear_mpc(ss, q=10.0, r=0.1, horizon=20, u_min=-1.0, u_max=1.0, du_max=0.3)

state = mpc.init_state(jnp.zeros(2))
u, state = mpc.step(state, jnp.array([0.0]), jnp.array([1.0]))   # first optimal move
```

Pass a *continuous* model with `discretize=True` and a `dt` to ZOH-discretize it
first (`c2d` exposes the same conversion standalone). The terminal mode
(`terminal="lqr"` / `"none"`) and the disturbance model (`disturbance="output"` /
`"none"`) are both switchable.

## State estimation: KF, EKF, UKF & MHE

MPC needs the state; estimation supplies it. All four estimators share the
`GaussianState(mean, cov)` belief and a uniform `predict` / `update` / `step` /
`filter` interface (`filter` runs a whole sequence with one `scan`).

* **`KalmanFilter`**: the optimal linear-Gaussian recursion, with a numerically
  robust **Joseph-form** covariance update.
* **`ExtendedKalmanFilter`**: the same recursion on the *exact autodiff
  linearization* of an arbitrary nonlinear transition/measurement (`A = ∂f/∂x`,
  `C = ∂h/∂x` via `jax.jacobian`, no finite differences).
* **`UnscentedKalmanFilter`**: a derivative-free sigma-point filter for strongly
  nonlinear models.
* **`moving_horizon_estimate`**: optimization-based estimation over a sliding
  window, the estimation dual of MPC: it fits the most recent measurements
  subject to the dynamics, with an arrival cost summarizing older data.

```python
import jax.numpy as jnp
from fugacio.sim import ExtendedKalmanFilter, GaussianState

def f(x, u):                                      # nonlinear pendulum, one RK-free step
    theta, omega = x
    return jnp.array([theta + 0.05 * omega,
                      omega - 0.05 * (9.81 * jnp.sin(theta) - u[0])])

ekf = ExtendedKalmanFilter(f=f, h=lambda x: x[:1],     # measure the angle only
                           q=1e-4 * jnp.eye(2), r=jnp.array([[1e-2]]))
belief = GaussianState(mean=jnp.array([0.3, 0.0]), cov=0.1 * jnp.eye(2))
belief = ekf.step(belief, jnp.array([0.0]), jnp.array([0.32]))   # predict + correct
```

## Nonlinear & economic MPC

A linear MPC about one operating point is only locally valid. `nonlinear_mpc`
optimizes over the *true* nonlinear model each step by direct **single shooting**:
the prediction is a roll-out of a discrete transition `f(x, u, theta)`, the cost
is a sum over that roll-out, and the open-loop problem is handed to
[`argmin`](optimization.md), which differentiates *through the optimum* and is
**warm-started** from the previous step's shifted plan. `discretize` turns a
continuous right-hand side into the one-step transition; `quadratic_tracking`
builds the usual stage/terminal costs, or supply an arbitrary stage cost for
**economic** MPC (optimize energy or product value directly).

```python
import jax.numpy as jnp
from fugacio.sim import discretize, nonlinear_mpc, quadratic_tracking

def rhs(t, x, u, theta):                          # continuous pendulum
    theta_, omega = x
    return jnp.array([omega, -9.81 * jnp.sin(theta_) + u[0]])

f = discretize(rhs, dt=0.05)                      # -> f(x, u, theta)
stage, terminal = quadratic_tracking(q=jnp.array([10.0, 1.0]), r=jnp.array([0.1]))
mpc = nonlinear_mpc(f, stage_cost=stage, terminal_cost=terminal,
                    horizon=25, n_input=1, u_min=-5.0, u_max=5.0)

u, warm = mpc.step(jnp.array([3.0, 0.0]), jnp.zeros(1), theta={"r": jnp.array([jnp.pi, 0.0])})
```

## Closed-loop simulation & gradient-based tuning

`simulate_closed_loop` marches a plant and controller together as a single
`jax.lax.scan` (measurement → control → plant step, with optional process and
measurement noise) and returns the full state/output/input trajectory. The
`linear_feedback` and `nonlinear_feedback` adapters wrap an MPC into the harness's
`(state, measurement, setpoint) -> (input, state)` protocol, and
`constant_setpoint` builds a setpoint program.

Because the entire loop is differentiable (*through the controller's own QP*),
`tune_mpc` descends a closed-loop performance index (`closed_loop_cost`, i.e.
tracking ISE plus optional effort/move penalties) directly on the MPC **weights**.
This is exact first-order tuning, not a grid search: the gradient knows how
nudging `Q` or `R` changes the resulting trajectory.

```python
import jax.numpy as jnp
from fugacio.sim import (
    StateSpace, linear_mpc, linear_feedback, simulate_closed_loop,
    constant_setpoint, closed_loop_cost, tune_mpc,
)

ss = StateSpace(a=jnp.array([[0.9]]), b=jnp.array([[0.1]]),
                c=jnp.array([[1.0]]), d=jnp.zeros((1, 1)))

def simulate(log_w):                              # build controller from the params, run the loop
    mpc = linear_mpc(ss, q=jnp.exp(log_w[0]), r=jnp.exp(log_w[1]), horizon=15)
    step, ctrl0 = linear_feedback(mpc, jnp.zeros(1))
    return simulate_closed_loop(lambda x, u: ss.a @ x + ss.b @ u, step,
                                jnp.zeros(1), ctrl0, constant_setpoint(1.0, 40),
                                measure=lambda x: ss.c @ x)

res = tune_mpc(simulate, jnp.array([0.0, 0.0]),    # tune (log q, log r)
               performance=lambda loop: closed_loop_cost(loop, effort_weight=1e-3))
res.x        # the tuned log-weights (gradients flowed through every QP solve)
```

## The AI copilot for advanced control

`fugacio.copilot` exposes the layer to an LLM design agent as deterministic,
JSON-in/JSON-out tools over a linear state-space plant:

* **`lqr_design`**: the LQR gain, Riccati cost-to-go and closed-loop poles for
  given weights (discrete or continuous);
* **`kalman_design`**: the steady-state Kalman gain, error covariance and
  estimator poles for given noise covariances;
* **`simulate_mpc`**: run a constrained, offset-free linear MPC in closed loop to
  a setpoint (optionally with an unmeasured output disturbance) and report the
  trajectories and step metrics;
* **`tune_mpc_weights`**: descend the closed-loop tracking cost on the MPC
  weights, exploiting the differentiability of the controller's own QP.

`summarize_lqr_design` and `summarize_mpc_simulation` render the results as
Markdown for a notebook or a chat reply.

```python
from fugacio.copilot import call_tool, summarize_mpc_simulation

out = call_tool("simulate_mpc", {
    "a": [[0.9]], "b": [[0.1]], "c": [[1.0]],
    "q": 1.0, "r": 0.01, "setpoint": [1.0], "disturbance": [0.3],
})
print(summarize_mpc_simulation(out))     # reaches the setpoint with zero offset
```
