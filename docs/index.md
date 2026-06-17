# Fugacio

Open, differentiable thermodynamics and process simulation, with an AI design
copilot. See the [project README](https://github.com/owenthcarey/fugacio) for the
full overview.

Fugacio is built as three layered packages (strict direction
**`thermo` < `sim` < `copilot`**, enforced in CI):

- **`fugacio.thermo`**, differentiable properties + phase equilibrium: a curated
  open component database, ideal-gas correlations, cubic equations of state
  (vdW / RK / SRK / PR) with fugacity coefficients,
  [reference multiparameter Helmholtz EOS](reference-fluids.md) (IAPWS-95
  water/steam, Span–Wagner CO₂, 26 fluids, with steam-table state functions and
  IAPWS transport), real-fluid energy properties
  (residual/departure functions, enthalpy/entropy, isenthalpic & isentropic
  flashes), [liquid & transport properties](physical-properties.md) (density,
  viscosity, thermal conductivity, surface tension, diffusivity, both pure and
  mixture), activity-coefficient models (Margules, van Laar, Wilson, NRTL,
  UNIQUAC), group contribution (UNIFAC + Dortmund, Joback), EOS *and* γ–φ
  equilibrium solvers (Rachford-Rice, PT flash, saturation, bubble/dew), rigorous
  [LLE / VLLE and tangent-plane stability](phase-equilibrium.md), parameter
  regression with a bundled [ThermoML parameter
  bank](physical-properties.md#the-thermoml-parameter-bank), and [reaction
  thermochemistry, equilibrium, and kinetics](reactions.md).
- **`fugacio.sim`**, flowsheet / unit-operation engine (depends on `thermo`):
  a differentiable `Stream` pytree, energy-balanced unit operations (`flash_drum`,
  `heater`, `valve`, `pump`, `compressor`, `turbine`, `mix`, `splitter`), a
  recycle/tear solver with implicit-diff gradients (`tear_solve`, `Flowsheet`),
  distillation columns (shortcut FUG and a rigorous equilibrium-stage model),
  binary diagrams / azeotropes / residue-curve maps,
  [reactors](reactions.md) (equilibrium, stoichiometric, CSTR, PFR, batch),
  reactive separations (reactive flash & distillation),
  [steam & cooling-water utilities](reference-fluids.md#steam-cooling-water-utilities-fugaciosim)
  on IAPWS-95,
  [differentiable optimization, design specs & process economics](optimization.md)
  (constrained NLP with implicit-diff `argmin`, controllers, Turton costing, TAC/NPV),
  [time-domain dynamics & process control](dynamics.md) (differentiable ODE
  integrators with a continuous adjoint, a filtered anti-windup PID, dynamic unit
  operations, `DynamicFlowsheet`, and gradient-based controller tuning),
  [advanced control](advanced-control.md) (a differentiable OSQP-style QP,
  condensed offset-free linear MPC, Kalman / extended / unscented / moving-horizon
  estimation, nonlinear & economic MPC, and gradient-based weight tuning), and
  [heat integration & pinch analysis](heat-integration.md) (differentiable
  minimum-utility/pinch targets, composite & grand composite curves, area/cost
  supertargeting for the optimal `dt_min`, and heat-exchanger-network synthesis).
- **`fugacio.copilot`**, LLM design agent (depends on `sim`): a JSON tool
  registry over the engine (properties, [steam tables &
  reference fluids](reference-fluids.md), unit operations, distillation, reactors,
  reaction equilibrium, [optimization, sizing & costing](optimization.md),
  [FOPDT identification & PID tuning](dynamics.md#the-ai-copilot-dynamically),
  [LQR & Kalman design, MPC simulation & tuning](advanced-control.md#the-ai-copilot-for-advanced-control),
  [heat-integration targets & network synthesis](heat-integration.md#the-ai-copilot-integrated)),
  plus a vendor-neutral provider layer (OpenAI / Anthropic / mock) and a
  multi-turn, tool-calling [agent loop](optimization.md#the-ai-design-copilot).

Everything is written in [JAX](https://github.com/jax-ml/jax) and the iterative
solvers carry implicit-function-theorem gradient rules, so an entire flowsheet is
end-to-end differentiable, including through phase equilibrium.

```python
import jax
import jax.numpy as jnp
from fugacio.sim import Stream, flash_drum

feed = Stream.from_fractions(
    ("methane", "propane", "n-pentane"),
    jnp.array([0.5, 0.3, 0.2]),
    flow=100.0, t=320.0, p=20e5,
)
vapor, liquid = flash_drum(feed, 320.0, 20e5)

# Exact sensitivity of vapour product flow to drum temperature:
jax.grad(lambda T: flash_drum(feed, T, 20e5)[0].total)(320.0)
```

## Correctness as an executable harness

Physical correctness is continuously machine-checked: first-principles
consistency laws that need no external data (Gibbs-Duhem, equifugacity,
fugacity-pressure identity, phase stability), automatic-differentiation gradients
checked against finite differences, and opt-in differential testing against open
reference codes: [CoolProp](https://github.com/CoolProp/CoolProp) and
[`chemicals`](https://github.com/CalebBell/chemicals) for pure-fluid properties
and the reference Helmholtz EOS layer (IAPWS-95, Span–Wagner),
[`thermo`](https://github.com/CalebBell/thermo) /
[Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) for activity
coefficients, and [Cantera](https://github.com/Cantera/cantera) for
reaction equilibrium and standard-state thermochemistry.
