# Equation-oriented flowsheeting

The `fugacio.sim.eo` layer solves a whole flowsheet as **one** system of
equations. Where the sequential-modular engine
([Flowsheet & recycle](api/sim/flowsheet.md)) evaluates units in order and
converges recycles by tearing, the equation-oriented (EO) engine collects every
unit's equations, the stream connectivity, the recycles, and any design specs
into a single residual system `F(x, theta) = 0` and solves it simultaneously by
Newton's method. There's no tear stream and no unit ordering: a recycle is just a
stream two blocks happen to share, and the global solve closes it like any other
equation.

This is the formulation a differentiable core is built for. The one expensive
ingredient of a classical EO solver, the Jacobian `dF/dx`, comes *exactly* from
JAX autodiff instead of finite differences or hand-coded analytic blocks, and the
converged solution is itself differentiable in the parameters `theta` (operating
conditions, feeds, prices, model parameters) by the implicit function theorem.
A scalar product spec, duty, or cost uses one process adjoint solve, together
with the local unit derivatives and their internal implicit solves.

## Build and solve a flowsheet

An `EOFlowsheet` is a small declarative builder: register feeds, add `Block`
units that name their inlet and outlet streams, then `solve`. Built-in blocks
call the common physical unit kernels and constrain their outlet stream
coordinates. Custom blocks can provide residual equations and auxiliary
unknowns. Local Jacobians assemble into a checked sparse CPU solve; see the
[shared runtime](process-runtime.md).
`EOFlowsheet()` uses Peng-Robinson over the feeds' components;
`EOFlowsheet(model=pkg)` runs every block on another
[property package](property-packages.md).

```python
import jax.numpy as jnp
from fugacio.sim import Stream
from fugacio.sim.eo import EOFlowsheet, Compressor, Heater, Valve

feed = Stream.from_fractions(
    ("methane", "propane", "n-pentane"),
    jnp.array([0.80, 0.15, 0.05]), flow=100.0, t=330.0, p=40e5,
)

fs = EOFlowsheet()
fs.feed("feed", feed)
fs.add(Compressor(inlets=("feed",), outlets=("c",), p_out=80e5, efficiency=0.8))
fs.add(Heater(inlets=("c",), outlets=("h",), t_out=360.0, dp=0.0))
fs.add(Valve(inlets=("h",), outlets=("v",), p_out=15e5))

sol = fs.solve()
sol["v"].t, sol["v"].p          # solved valve outlet state
sol.residual_norm               # max-norm of the scaled residual (~0)
```

`solve` checks its report by default: a concrete failure raises
`ConvergenceError` naming the block equation or design specification with the
largest residual, and a compiled failure returns NaN streams. `check=False`
returns the best iterate with its `report` for diagnosis.

The compressor, heater, and valve are solved **together**, not one after the
other. The `EOSolution` exposes every named stream (`sol["c"]`, `sol["h"]`,
`sol["v"]`, and the feed), custom block auxiliary unknowns (`sol.aux`, empty
for these built-ins), any freed design-spec values
(`sol.specs`), and the converged residual norm. Solved the same chain unit by
unit, the sequential-modular `compressor`, `heater`, and `valve` reproduce these
streams to solver tolerance: the EO and sequential-modular engines share the same
physics, so they must agree on any flowsheet both can express.

## Operating conditions are differentiable parameters

A block spec is either a literal or a string key. A string is looked up in the
parameter mapping you pass to `solve`, so an operating condition becomes a
differentiable parameter just by naming it. The flash drum below reads its
temperature and pressure from `theta`:

```python
from fugacio.sim.eo import EOFlowsheet, Flash

fs = EOFlowsheet()
fs.feed("feed", feed)
fs.add(Flash(inlets=("feed",), outlets=("vap", "liq"), t="T", p="P"))

sol = fs.solve({"T": 315.0, "P": 18e5})
sol["vap"].n + sol["liq"].n     # closes to the feed flows
```

## Recycles need no tearing

A recycle is an internal stream that an upstream block reads and a downstream
block writes; the global Newton solve closes it. Below, a splitter sends part of
the flash liquid back to the mixer, and nothing about the loop needs special
handling: there's no tear stream, no guess to converge, and no Wegstein
acceleration.

```python
from fugacio.sim.eo import EOFlowsheet, Mixer, Flash, Splitter

fresh = Stream.from_fractions(
    ("methane", "propane", "n-pentane"),
    jnp.array([0.5, 0.3, 0.2]), flow=100.0, t=320.0, p=20e5,
)

fs = EOFlowsheet()
fs.feed("fresh", fresh)
fs.add(Mixer(inlets=("fresh", "recycle"), outlets=("mixed",), t=320.0))
fs.add(Flash(inlets=("mixed",), outlets=("vapor", "liquid"), t="T", p="P"))
fs.add(Splitter(inlets=("liquid",), outlets=("recycle", "purge"), fractions="r"))

sol = fs.solve({"T": 320.0, "P": 20e5, "r": jnp.array([0.5, 0.5])})

# Overall balance: fresh feed leaves as vapor product plus purge (recycle cancels).
sol["fresh"].n - (sol["vapor"].n + sol["purge"].n)   # ~0
```

The process adjoint is independent of how many Newton iterations the forward
solve took:

```python
import jax

def methane_recovered(temp):
    sol = fs.solve({"T": temp, "P": 20e5, "r": jnp.array([0.5, 0.5])})
    return sol["vapor"].n[0]

jax.grad(methane_recovered)(322.0)        # exact, matches a finite difference
```

## Degrees of freedom

`degrees_of_freedom` reports the unknown/equation balance, the EO analogue of a
specification check. A square system (`degrees_of_freedom == 0`) has balanced
equation counts; its Jacobian can still be singular. Use `diagnose` to inspect
local rank and `solve().report` to check convergence. A positive count is
under-specified (add a spec), and a negative one is over-specified. `solve` runs this check by default and raises a
descriptive error when the flowsheet isn't square.

```python
fs = EOFlowsheet()
fs.feed("feed", feed)
fs.add(Flash(inlets=("feed",), outlets=("vap", "liq"), t="T", p="P"))

report = fs.degrees_of_freedom()
report.n_unknowns, report.n_equations, report.degrees_of_freedom   # 16, 16, 0
report.per_block                                                   # {"vap": 16}
```

Each stream has 2C + 2 coordinates: C total component flows, temperature,
pressure, and C vapor component flows. This three-component flash has two
outlets, so it contributes 16 unknowns and 16 equations.

## Design specs

In EO form a design spec is simply one more equation and one more unknown, solved
*simultaneously* with the flowsheet. `spec` frees a manipulated parameter (it
becomes an unknown, seeded at `init`) and adds the equation
`measure(streams) - target = 0`. Coupled specs converge together, with no nested
loop.

```python
fs = EOFlowsheet()
fs.feed("feed", feed)
fs.add(Heater(inlets=("feed",), outlets=("out",), duty="Q", dp=0.0))
# Free the duty Q so the outlet temperature reaches 360 K.
fs.spec("Q", lambda s: s["out"].t, target=360.0, init=1.0e5)

sol = fs.solve({})
sol["out"].t          # 360.0
sol.specs["Q"]        # the duty that achieves it
```

## Property packages and the plant-scale blocks

`EOFlowsheet(model=...)` accepts any [property package](property-packages.md),
and every block draws its enthalpies, volumes, and fugacities from it, so the
whole simultaneous solve can run on NRTL, PC-SAFT, or a reference fluid as well
as on a cubic. Three blocks cover the units that turn a chain of exchangers
into a plant:

* `HeatExchanger(inlets=(hot_in, cold_in), outlets=(hot_out, cold_out), ...)`
  couples two streams with one thermal specification (`duty`,
  `t_hot_out`, `t_cold_out`, or `ua` with the common segmented rating model).
* `StoichiometricReactor(nu=..., key=..., conversion=..., t_out=... | duty=...)`
  applies fixed conversions with an energy balance on absolute ideal-gas
  enthalpies (formation enthalpy plus the ideal-gas heat-capacity integral), so
  the heat of reaction is carried automatically.
* `Column(inlets=(feed,), outlets=(distillate, bottoms), feed_stages=(6,),
  n_stages=12, p=..., specs=(("reflux_ratio", "R"), ("distillate_rate", 50.0)))`
  embeds a converged [rigorous MESH column](distillation.md) as one block, so a
  column can sit inside a recycle loop that the global Newton closes.

```python
from fugacio.sim import package_for
from fugacio.sim.eo import Column, EOFlowsheet, HeatExchanger

fs = EOFlowsheet(model=package_for(("benzene", "toluene"), "srk"))
fs.feed("cold", cold_feed)
fs.feed("hot", hot_utility)
fs.add(HeatExchanger(inlets=("hot", "cold"), outlets=("hot_out", "warm"), t_cold_out=360.0))
fs.add(Column(inlets=("warm",), outlets=("d", "b"), feed_stages=(6,), n_stages=12,
              p=1.013e5, specs=(("reflux_ratio", "R"), ("distillate_rate", 50.0))))
sol = fs.solve({"R": 2.5})
```

The exchanger and column blocks reproduce their sequential-modular
counterparts (`heat_exchanger`, `rigorous_column`) to solver tolerance, which
the test suite checks. The sequential-modular `stoichiometric_reactor` adds the
package's residual enthalpy to the same formation basis, so the two reactors
agree only where residual enthalpies are negligible, as in a low-pressure gas.

## Flowsheet optimization

`optimize_flowsheet_eo` minimizes an objective read off the solved streams over
named decision variables, each given as `(init, lower, upper)`. Two equivalent
formulations are offered.

**Nested** (the robust default) treats each decision as a flowsheet parameter:
every objective evaluation solves the flowsheet to convergence, and the optimizer
descends the resulting reduced objective. The flowsheet solve is differentiable,
so the reduced gradient is exact.

```python
from fugacio.sim.eo import optimize_flowsheet_eo

fs = EOFlowsheet()
fs.feed("feed", fresh)
fs.add(Flash(inlets=("feed",), outlets=("vap", "liq"), t="T", p="P"))

res = optimize_flowsheet_eo(
    fs,
    lambda s: (jnp.sum(s["vap"].n) - 60.0) ** 2,   # hit a target vapor flow
    {"T": (312.0, 305.0, 335.0)},                  # (init, lower, upper)
    params={"P": 20e5},
)
res.decision["T"], res.objective, res.converged
```

**Simultaneous / full-space** (`simultaneous=True`) optimizes the decision
variables *and* the flowsheet state together, with the flowsheet equations
imposed as equality constraints through an augmented-Lagrangian solve. No inner
loop converges the flowsheet; feasibility and optimality are reached at once, the
classic equation-oriented optimization paradigm. The two formulations agree at
the optimum.

```python
fs = EOFlowsheet()
fs.feed("feed", feed)
fs.add(Heater(inlets=("feed",), outlets=("out",), duty="Q", dp=0.0))

res = optimize_flowsheet_eo(
    fs,
    lambda s: (s["out"].t - 360.0) ** 2,
    {"Q": (5.0e4, -5.0e6, 5.0e6)},
    simultaneous=True,
)
res.decision["Q"], res.constraint_violation        # the duty; feasibility ~0
```

## How it stays well conditioned and finite

Two implementation details make the solve trustworthy across unit systems and
phase regimes:

- **Scaling.** Every unknown and residual is carried in non-dimensional form
  (see `Scales`): material balances by a flow scale, energy balances by an
  enthalpy scale, pressures by a pressure scale, with the equifugacity relations
  left dimensionless. The decision variables of a full-space optimization are
  scaled the same way, so the augmented-Lagrangian step moves a weakly coupled
  duty as readily as a strongly coupled temperature.
- **Phase-safe derivatives.** A stream's bulk enthalpy and entropy blend the
  flashed vapor and liquid contributions. Differentiating that blend naively
  fails in a single-phase region, where the absent phase forces a cubic root that
  doesn't exist there and whose derivative is `NaN`. A `jax.lax.switch`
  differentiates only the phase(s) that exist, so the gradient stays finite for a
  subcooled liquid, a superheated vapor, and a two-phase stream alike.

The first solve of each distinct flowsheet pays a JIT compilation; a structural
plan is then cached on the flowsheet, so repeated solves, the forward sweeps of a
finite-difference check, and the inner solves of an optimization all reuse one
compilation. The property package enters that compilation as a dynamic input,
so new package parameters don't recompile either.

See [reliability and diagnostics](reliability.md) for phase-preserving streams,
checked failures, warm starts, continuation, and derivative limits.
