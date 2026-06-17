# Fugacio

**Open, differentiable thermodynamics and process simulation — with an AI design copilot.**

Fugacio is an open-source successor to closed, expensive process simulators. The
numerical core is written in [JAX](https://github.com/jax-ml/jax), so an entire
flowsheet is *end-to-end differentiable*: you can take gradients through phase
equilibrium and recycle loops for optimization, parameter estimation, and tight
ML integration.

Fugacio treats physical correctness as the baseline, not a stretch goal — and
because its reference data and models are open, that correctness stays
*continuously machine-checkable* as the engine grows one model at a time. Every
model is graded against free, authoritative oracles: differential testing against
open reference codes ([CoolProp](https://github.com/CoolProp/CoolProp),
[`thermo`](https://github.com/CalebBell/thermo),
[Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl), and
[Cantera](https://github.com/Cantera/cantera)); first-principles consistency laws
that need no external data (Gibbs-Duhem, Maxwell relations, mass- and
energy-balance closure, equifugacity, and phase-stability tangent-plane tests);
open experimental measurements from the [NIST ThermoML
Archive](https://www.nist.gov/mml/acmd/trc/thermoml/thermoml-archive); and —
uniquely for a differentiable core — automatic-differentiation gradients checked
against finite differences, with group-contribution methods (UNIFAC, Joback)
covering parameters where curated industrial datasets remain proprietary.
Together these oracles act as an executable *acceptance harness* for physics —
turning "is this simulator correct?" into thousands of small, automatically
graded checks: a fast, unambiguous feedback loop that makes the core tractable to
grow incrementally, including via long-running, AI-assisted development.

It ships as three layered packages in a single [`uv`](https://docs.astral.sh/uv/)
workspace:

| Package | Import | Responsibility |
| --- | --- | --- |
| `fugacio-thermo` | `fugacio.thermo` | Differentiable properties + phase equilibrium: EOS & γ–φ activity models, reference multiparameter Helmholtz EOS (IAPWS-95 water/steam, Span–Wagner CO₂, 26 fluids) with steam-table state functions and IAPWS transport, energy/PT-PH-PS flashes, liquid & transport properties (density, viscosity, conductivity, surface tension, diffusivity), rigorous LLE/VLLE, parameter regression with a bundled ThermoML parameter bank, and reaction thermochemistry, equilibrium & kinetics (the foundation). |
| `fugacio-sim` | `fugacio.sim` | Flowsheet engine: energy-balanced unit ops, a differentiable recycle/tear solver, distillation columns, binary/residue-curve diagrams, reactors, reactive separations, optimization/design/economics, time-domain **dynamics & process control** (differentiable ODE integrators, PID, dynamic units, `DynamicFlowsheet`), **advanced control** (differentiable QP, offset-free linear MPC, Kalman/EKF/UKF/moving-horizon estimation, nonlinear & economic MPC, gradient-based tuning), and **heat integration & pinch analysis** (minimum-utility/pinch targets, composite curves, area/cost supertargeting, network synthesis) (depends on `thermo`). |
| `fugacio-copilot` | `fugacio.copilot` | LLM design agent: a JSON tool registry over the engine plus gradient-based optimizers (depends on `sim`). |

The dependency direction is strict — **`thermo` < `sim` < `copilot`** — and is
enforced in CI by [import-linter](https://github.com/seddonym/import-linter).
All three distributions share the `fugacio`
[PEP 420 namespace](https://peps.python.org/pep-0420/), so they publish to PyPI
independently yet import under one roof.

## Quickstart

```bash
# Install uv: https://docs.astral.sh/uv/
uv sync --all-packages   # venv + lockfile; install all three packages (editable)
uv run pytest            # run the test suite
```

A flash drum, and its gradient — differentiated straight through the
equation-of-state phase equilibrium:

```python
import jax
import jax.numpy as jnp
from fugacio.sim import Stream, flash_drum

feed = Stream.from_fractions(
    ("methane", "propane", "n-pentane"),
    jnp.array([0.5, 0.3, 0.2]),
    flow=100.0, t=320.0, p=20e5,
)
vapor, liquid = flash_drum(feed, 320.0, 20e5)   # rigorous Peng-Robinson flash

# Sensitivity of vapour product flow to drum temperature (exact, via implicit diff):
d_vapor_dT = jax.grad(lambda T: flash_drum(feed, T, 20e5)[0].total)
d_vapor_dT(320.0)
```

Unit operations close rigorous energy balances, and recycle loops are solved to a
fixed point and differentiated by the implicit function theorem — so gradients
flow through the *converged* flowsheet, not the iteration:

```python
from fugacio.sim import mix, splitter, tear_solve

def one_pass(recycle, theta):                      # mixer -> flash -> recycle split
    mixed = mix([feed, recycle], t=320.0)          # adiabatic, energy-balanced
    _vapor, liquid = flash_drum(mixed, theta["T"], theta["P"])
    recycled, _purge = splitter(liquid, jnp.array([theta["r"], 1.0 - theta["r"]]))
    return recycled

guess = Stream.from_fractions(feed.components, jnp.array([0.1, 0.3, 0.6]), 30.0, 320.0, 20e5)
recycle = tear_solve(one_pass, guess, {"T": 320.0, "P": 20e5, "r": 0.5})
```

Because the converged flowsheet is differentiable, optimization, design specs, and
process economics are just more differentiable layers. `argmin` returns the
*solution* of a constrained problem and differentiates it through the optimality
conditions (implicit function theorem), so a real screening economics objective —
Turton bare-module capital plus utilities — optimizes end to end, gradients and
all (see [the optimization & economics guide](docs/optimization.md)):

```python
import jax
from fugacio.sim import heat_exchanger_area, bare_module_cost, total_annual_cost, utility_cost

# Size a cooler from its duty (LMTD), cost it (Turton), and get the total annual cost —
# then the exact sensitivity of TAC to the temperature approach, by autodiff.
def tac(dt_cold):
    area = heat_exchanger_area(duty=1.0e6, u=500.0, dt_hot=60.0, dt_cold=dt_cold)
    capex = bare_module_cost("heat_exchanger", area).bare_module
    return total_annual_cost(capex, utility_cost(1.0e6, "cooling_water"))

tac(40.0), jax.grad(tac)(40.0)   # $/yr and d(TAC)/d(approach)
```

Pure-fluid properties come at *reference grade* where it matters: water/steam is
the full IAPWS-95 formulation (the same model behind REFPROP and CoolProp),
implemented as one scalar Helmholtz energy whose every property — and every
*solver*, density to Maxwell construction — is an exact `jax.grad` derivative
(see [the reference-fluids guide](docs/reference-fluids.md)):

```python
import jax
from fugacio.thermo import reference_fluid, saturation_state
from fugacio.sim import steam_heating, steam_turbine

steam = reference_fluid("water")                     # IAPWS-95
sat = saturation_state(steam, p=10e5)                # 10 bar: 453.03 K, Δh_vap 2014.6 kJ/kg

# d(Tsat)/dP through the solved Maxwell construction — the Clausius-Clapeyron
# slope by autodiff, not finite differences:
jax.grad(lambda p: saturation_state(steam, p=p).t)(10e5)

steam_heating(2.5e6, pressure=11e5).mass_flow        # kg/s of MP steam for a reboiler
steam_turbine(10.0, p_in=40e5, t_in=723.15, p_out=1e5).power  # Rankine shaft power
```

Steady state is only half of a plant. The `fugacio.sim.dynamics` and
`fugacio.sim.control` layers add **time** while keeping everything
differentiable: a `jax.lax.scan` ODE integrator (and an adaptive one with a
continuous-adjoint `custom_vjp`), a filtered anti-windup `PID` whose gains are a
differentiable pytree, dynamic unit operations carried as holdup ODEs, and a
`DynamicFlowsheet` that assembles units and control loops into one global ODE. You
can take a gradient of a closed-loop performance index straight through the
simulated loop — so tuning is exact first-order, not a grid search (see
[the dynamics & control guide](docs/dynamics.md)):

```python
import jax
import jax.numpy as jnp
from fugacio.sim import pi, odeint, iae

# A PI loop on a first-order plant, integrated as one ODE (plant state + controller state).
kp, taup, sp = 2.0, 5.0, 1.0
ts = jnp.linspace(0.0, 40.0, 401)

def response(gains):
    c = pi(kc=gains["kc"], tau_i=gains["tau_i"], u_min=-10.0, u_max=10.0)
    def loop(t, st, _):
        y, ctrl = st["y"], st["c"]
        u = c.output(ctrl, sp, y)
        return {"y": (-y + kp * u) / taup, "c": c.derivative(ctrl, sp, y)}
    st0 = {"y": jnp.asarray(0.0), "c": c.init_state(0.0)}
    return odeint(loop, st0, ts, method="rk4", substeps=3)["y"]

# Gradient of the closed-loop IAE with respect to the PID gains — through the whole sim:
jax.grad(lambda g: iae(ts, response(g), sp))({"kc": jnp.asarray(0.5), "tau_i": jnp.asarray(8.0)})
```

Single PID loops are not the whole control story. The `fugacio.sim.mpc` layer adds
**model predictive control** and **state estimation**, and keeps them
differentiable through their own solvers. A differentiable OSQP-style QP (with an
implicit-function-theorem `custom_vjp`) backs a condensed, offset-free **linear
MPC** — LQR terminal cost, hard input / soft output constraints, and a
disturbance observer for zero steady-state offset; Kalman / extended / unscented
filters and moving-horizon estimation reconstruct the state; and **nonlinear &
economic MPC** optimize over the true model via `argmin`. Because the QP itself is
differentiable, `tune_mpc` descends a closed-loop index on the controller weights
— exact first-order tuning of the optimizer (see
[the advanced-control guide](docs/advanced-control.md)):

```python
import jax.numpy as jnp
from fugacio.sim import StateSpace, linear_mpc

# Constrained, offset-free MPC on a discrete double integrator (position, velocity).
dt = 0.1
ss = StateSpace(a=jnp.array([[1.0, dt], [0.0, 1.0]]), b=jnp.array([[0.5 * dt**2], [dt]]),
                c=jnp.array([[1.0, 0.0]]), d=jnp.zeros((1, 1)))
mpc = linear_mpc(ss, q=10.0, r=0.1, horizon=20, u_min=-1.0, u_max=1.0, du_max=0.3)

state = mpc.init_state(jnp.zeros(2))
u, state = mpc.step(state, jnp.array([0.0]), jnp.array([1.0]))   # first constrained, optimal move
```

The energy bill of a plant is fixed before any exchanger is drawn, so
`fugacio.sim.integration` brings the whole **pinch-technology** workflow — and
keeps it differentiable. The problem table algorithm gives the minimum hot/cold
utility targets and the pinch; composite and grand composite curves give the
T–H picture; a Bath-formula area target, a minimum-units target, and utility
pricing combine into a total annual cost whose minimum over `dt_min` is the
cost-optimal design point (**supertargeting**); and `synthesize_network` builds —
and independently verifies — a minimum-utility heat-exchanger network by the pinch
design method (see [the heat-integration guide](docs/heat-integration.md)):

```python
import jax
from fugacio.sim import make_stream, pinch_analysis, optimal_dt_min, synthesize_network, minimum_utilities

streams = [
    make_stream(20.0, 135.0, cp=2.0, name="C1"), make_stream(170.0, 60.0, cp=3.0, name="H1"),
    make_stream(80.0, 140.0, cp=4.0, name="C2"), make_stream(150.0, 30.0, cp=1.5, name="H2"),
]

res = pinch_analysis(streams, dt_min=10.0)
res.hot_utility, res.cold_utility, res.hot_pinch_temperature   # 20.0 W, 60.0 W, pinch at 90 K

# Differentiable target: d(min hot utility) / d(dt_min), exact and free:
jax.grad(lambda dt: minimum_utilities(streams, dt)[0])(10.0)   # 0.5 W/K

optimal_dt_min(streams, bounds=(1.0, 40.0)).dt_min             # cost-optimal approach (~5.6 K)
net = synthesize_network(streams, dt_min=10.0)                 # MER network, verified
net.feasible, net.achieves_mer, net.n_units
```

The `fugacio.copilot` agent exposes all of this — properties, steam tables,
unit ops, distillation, reactors, optimization, sizing, costing, FOPDT
identification / PID tuning, LQR & Kalman design, constrained MPC simulation &
weight tuning, and heat-integration targeting & network synthesis — as a JSON
tool registry, driven by a vendor-neutral provider layer (OpenAI / Anthropic /
mock) through a multi-turn, tool-calling loop.

## Development

```bash
just         # list available tasks
just fmt     # ruff format + autofix
just check   # lint + types + import boundaries + tests (exactly what CI runs)
```

| Task | Command |
| --- | --- |
| Format / autofix | `just fmt` |
| Lint | `just lint` |
| Types | `just types` |
| Import boundaries | `just imports` |
| Tests | `just test` |
| Oracle differential tests (opt-in) | `just oracles` |

`just test` runs the fast, hermetic unit suite. The differential-testing oracles
(graded against [CoolProp](https://github.com/CoolProp/CoolProp) and
[`chemicals`](https://github.com/CalebBell/chemicals) for pure-fluid properties
and the reference Helmholtz EOS / IAPWS transport implementations,
[`thermo`](https://github.com/CalebBell/thermo) /
[Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) for activity
coefficients, and [Cantera](https://github.com/Cantera/cantera) for reaction
equilibrium) are marked `oracle` and excluded from the default run; install those
optional packages and run them explicitly with `just oracles`. CI runs the same
oracle suite on every pull request, on pushes to `main`, and on a weekly
schedule (`.github/workflows/oracles.yml`).

## Layout

```text
fugacio/
├── packages/
│   ├── fugacio-thermo/    # fugacio.thermo  (no internal deps)
│   ├── fugacio-sim/       # fugacio.sim     (-> thermo)
│   └── fugacio-copilot/   # fugacio.copilot (-> sim)
├── pyproject.toml         # uv workspace root + shared ruff/mypy/pytest config
├── .importlinter          # enforced layer boundaries
└── .github/workflows/     # CI + trusted-publishing release
```

## License

[Apache-2.0](LICENSE).
