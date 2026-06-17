# Dynamics & process control

Everything else in `fugacio.sim` is **steady state**: a flash, a column, a
recycle loop is the solution of an algebraic system. The `fugacio.sim.dynamics`
and `fugacio.sim.control` layers add the missing dimension, **time**, while
keeping the whole stack end-to-end differentiable. You can simulate how a plant
*gets* to steady state, close PID loops around it, and take a gradient of any
trajectory feature (a settling time, an off-spec integral, a peak temperature)
with respect to controller gains, setpoints, feed schedules, or equipment
parameters.

## Differentiable ODE integration

Two complementary integrators sit at the core, and the split is deliberate.

`odeint` integrates `dy/dt = f(t, y, theta)` on a **fixed output grid** with a
`jax.lax.scan`, so it returns the whole trajectory and is differentiable in both
forward and reverse mode out of the box. Steppers include explicit Euler, classic
RK4, a fixed Dormand–Prince 5(4), and the A-stable implicit Euler / trapezoidal
methods (each implicit step solved by a short, unrolled Newton iteration so the
march stays reverse-differentiable) for stiff systems. The state `y` and the
parameters `theta` are arbitrary JAX pytrees.

```python
import jax, jax.numpy as jnp
from fugacio.sim import odeint

# A decaying oscillator; integrate on a uniform grid and differentiate the result.
def rhs(t, y, theta):
    omega, zeta = theta
    return jnp.array([y[1], -omega**2 * y[0] - 2 * zeta * omega * y[1]])

ts = jnp.linspace(0.0, 20.0, 401)
traj = odeint(rhs, jnp.array([1.0, 0.0]), ts, (jnp.asarray(1.5), jnp.asarray(0.2)),
              method="dopri5", substeps=4)

# d(final position) / d(damping ratio), straight through the solve:
g = jax.grad(lambda zeta: odeint(rhs, jnp.array([1.0, 0.0]), ts,
                                 (1.5, zeta), method="rk4")[-1, 0])(0.2)
```

`integrate` is an **adaptive-step** Dormand–Prince 5(4) with a PI step-size
controller, for when only the final state matters and efficiency or stiffness
control does. Its data-dependent step count rules out naive reverse-mode, so
gradients come from the **continuous-adjoint** method (a hand-written
`custom_vjp` that integrates the adjoint ODE backwards), exactly the
"differentiate the converged solution, not the iteration" philosophy used for the
algebraic solvers elsewhere in the stack. The cost of the gradient is one adjoint
solve, independent of how many forward steps the controller took.

```python
from fugacio.sim import integrate

res = integrate(rhs, jnp.array([1.0, 0.0]), 0.0, 20.0, (1.5, 0.2),
                rtol=1e-8, atol=1e-10)
res.y, res.n_accepted, res.success
```

## A differentiable PID controller

`PID` is a registered JAX pytree, so the *gains themselves* are differentiable. It
is written to live **inside** a flowsheet ODE: its integral action and filtered
derivative are carried as ODE states, so a closed loop is just a larger ODE. Two
things make it production-grade rather than a toy: a realizable, **filtered
derivative** (acting on the measurement, so no derivative kick on setpoint
changes), and **back-calculation anti-windup** that unwinds the integrator while
the output is saturated.

```python
import jax.numpy as jnp
from fugacio.sim import pi, odeint, iae

controller = pi(kc=1.2, tau_i=8.0, u_min=0.0, u_max=100.0)
kp, taup, sp = 2.0, 5.0, 1.0          # a first-order plant and a unit setpoint

def loop(t, st, theta):
    y, ctrl = st["y"], st["c"]
    u = controller.output(ctrl, sp, y)
    return {"y": (-y + kp * u) / taup, "c": controller.derivative(ctrl, sp, y)}

ts = jnp.linspace(0.0, 40.0, 401)
st0 = {"y": jnp.asarray(0.0), "c": controller.init_state(0.0)}
pv = odeint(loop, st0, ts, method="rk4", substeps=4)["y"]
iae(ts, pv, sp)                       # integral absolute error of the response
```

## Linear blocks, response metrics & linearization

The classical building blocks come with closed-form **step responses**
(`first_order_step`, `fopdt_step`, `second_order_step`, `lead_lag`) and
**state-space realizations** (`first_order_ss`, `second_order_ss`), plus the
static actuator nonlinearities (`saturate`, `dead_band`, `rate_limit`). Response
**metrics** (`overshoot`, `rise_time`, `settling_time`, the error integrals
`iae` / `ise` / `itae`, and the `step_info` bundle) turn a trajectory into the
figures of merit you actually tune for; the error integrals are smooth, so they
are the natural objectives for gradient-based tuning.

`linearize` reduces any nonlinear `f(y, u, theta)` to a local `StateSpace`
`(A, B, C, D)` by autodiff (no finite differences), and from there you get
`poles`, `is_stable`, `dc_gain`, `frequency_response` / `bode`, and the
`controllability` / `observability` matrices and rank tests.

```python
import jax.numpy as jnp
from fugacio.sim.control import linearize, poles, dc_gain, second_order_ss

a, b, c, d = second_order_ss(gain=3.0, wn=2.0, zeta=0.4)
ss = linearize(lambda x, u, th: a @ x + b @ jnp.atleast_1d(u),
               jnp.zeros(2), jnp.zeros(1), output=lambda x, u, th: c @ x)
poles(ss)          # -0.8 +/- 1.83j  (=> stable, underdamped)
dc_gain(ss)        # 3.0
```

## PID tuning

Classical tuning is a two-step recipe: reduce the process to a first-order-plus-
dead-time (FOPDT) model, then apply a correlation. `fit_fopdt` identifies
`(K, tau, L)` from a measured step response by differentiable least squares, and
`ziegler_nichols`, `cohen_coon`, `imc_tuning` (lambda tuning) and `amigo` turn an
FOPDT model into a ready `PID`.

```python
from fugacio.sim import fit_fopdt
from fugacio.sim.control import imc_tuning

model = fit_fopdt(t_data, y_data)            # -> FOPDTModel(gain, tau, dead_time)
controller = imc_tuning(model, controller="PI", tau_c=2.0)
```

For a closed-loop, performance-index-optimal tune that exploits the fully
differentiable plant, `tune_pid` descends an IAE / ISE / ITAE objective directly
on the gains: gradients flow through the whole simulated loop, so it's exact
first-order, not a grid search.

```python
from fugacio.sim import tune_pid

res = tune_pid(loop_response, {"kc": 0.5, "tau_i": 8.0}, setpoint=1.0, ts=ts,
               bounds=({"kc": 0.05, "tau_i": 0.5}, {"kc": 20.0, "tau_i": 50.0}))
res.x        # the tuned gains
```

## Dynamic unit operations

A steady-state unit maps inlets to outlets instantaneously; a **dynamic** unit has
memory (inventory of material and energy), so its outlets depend on accumulated
holdup, and its behaviour is an ODE. Every model follows the same recipe: the
**state** is a conserved holdup (component moles, an energy state) with a clean
balance `d(holdup)/dt = in − out + generation`, while the **constitutive
relations** (phase split, density, reaction rate, pressure) are evaluated
instantaneously from the holdup with the steady-state `fugacio.thermo` kernels, so
a dynamic flash reuses `flash_pt`, a dynamic reactor reuses the reaction
thermochemistry and rate laws, and so on.

| Unit | State | Captures |
| --- | --- | --- |
| `LevelTank` | component holdup | liquid level & composition (pump or Torricelli valve outlet) |
| `MixingTank` | mole fractions | constant-holdup blending (composition lag at the residence time) |
| `ThermalMass` | temperature | heated/cooled stirred tank, sensible + ambient loss |
| `DynamicCSTR` | concentrations + T | non-isothermal reactor (ignition/extinction, limit cycles) |
| `GasReceiver` | gas holdup | surge-drum pressure (ideal-gas, flow or valve outlet) |
| `DynamicFlash` | liquid holdup | separator composition with equilibrium vapour draw |

## Assembling a dynamic flowsheet

`DynamicFlowsheet` is the dynamic counterpart of `Flowsheet`: register feeds
(constant or `(t, theta) -> Stream`), connect dynamic units port-to-port, and
close control loops, then `simulate` over a horizon. Internally every unit holdup
*and* every controller state is concatenated into one global state with a single
right-hand side, handed to `odeint`. Holdups break algebraic recycle loops, so a
dynamic flowsheet needs **no tear solver**. The whole simulation is differentiable
in `theta` and the initial state.

```python
import jax.numpy as jnp
from fugacio.sim import Stream, pi
from fugacio.sim.dynamics import DynamicFlowsheet, ThermalMass

feed = Stream(n=jnp.array([5.0]), t=jnp.asarray(290.0), p=jnp.asarray(101325.0),
              components=("water",))
tank = ThermalMass(name="H", components=("water",), holdup=200.0, ua=20.0,
                   t_ambient=290.0, t0=300.0)

fs = DynamicFlowsheet()
fs.feed("feed", feed)
fs.add(tank, inputs=["feed"])
fs.control(pi(kc=300.0, tau_i=120.0, u_min=0.0, u_max=5e4),
           measurement=("H", "temperature"), setpoint=330.0, actuator=("H", "duty"))

result = fs.simulate(ts=jnp.linspace(0.0, 6000.0, 601))
result.measurement("H", "temperature")[-1]    # -> 330 K (loop holds the setpoint)
result.controls["H.duty"]                      # the manipulated duty trajectory
```

## Dynamic optimization & estimation

Because the integrator is end-to-end differentiable, the three classic
"optimization over a dynamic model" problems collapse to gradients through a
simulation composed with the existing optimizers:

* `optimal_control` chooses a piecewise-constant input trajectory to minimize a
  running-plus-terminal cost (minimum-energy transitions, set-point moves);
* `estimate_dynamics` fits model parameters (and optionally the initial state) to
  time-series measurements by Levenberg–Marquardt through the integrator
  (parameter estimation / data reconciliation);
* `tune_pid` descends a closed-loop performance index on the controller gains.

```python
import jax.numpy as jnp
from fugacio.sim.dynamics import estimate_dynamics

# Recover a first-order rate constant from a measured decay.
ts = jnp.linspace(0.0, 6.0, 60)
data = jnp.exp(-0.7 * ts)
fit = estimate_dynamics(lambda t, y, th: -th["k"] * y, jnp.asarray(1.0), ts, data,
                        {"k": jnp.asarray(0.2)})
fit.theta["k"]      # -> 0.7
```

## The AI copilot, dynamically

`fugacio.copilot` exposes the layer to an LLM design agent as deterministic,
JSON-in/JSON-out tools: `identify_fopdt` (fit an FOPDT model from step data),
`tune_pid` (gains by a named rule, with the resulting closed-loop metrics),
`closed_loop_response` (simulate a servo response for explicit gains), and
`recommend_pid_tuning` (compare the rules and pick the lowest-IAE one).
`summarize_pid_tuning` renders the comparison as a Markdown table.

```python
from fugacio.copilot import call_tool, summarize_pid_tuning

rec = call_tool("recommend_pid_tuning",
                {"gain": 2.0, "tau": 5.0, "dead_time": 1.5, "controller": "PI"})
print(summarize_pid_tuning(rec))     # ranked rules + the recommended one
```
