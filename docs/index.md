# Fugacio

Open, differentiable thermodynamics and process simulation, with an AI design
copilot. See the [project README](https://github.com/owenthcarey/fugacio) for the
full overview.

Fugacio is built as three layered packages (strict direction
**`thermo` < `sim` < `copilot`**, enforced in CI):

- **`fugacio.thermo`** — differentiable properties + phase equilibrium: a curated
  open component database, ideal-gas correlations, cubic equations of state
  (vdW / RK / SRK / PR) with fugacity coefficients, real-fluid energy properties
  (residual/departure functions, enthalpy/entropy, isenthalpic & isentropic
  flashes), activity-coefficient models (Margules, van Laar, Wilson, NRTL,
  UNIQUAC), group contribution (UNIFAC + Dortmund, Joback), EOS *and* γ–φ
  equilibrium solvers (Rachford-Rice, PT flash, saturation, bubble/dew), rigorous
  [LLE / VLLE and tangent-plane stability](phase-equilibrium.md), parameter
  regression, and [reaction thermochemistry, equilibrium, and
  kinetics](reactions.md).
- **`fugacio.sim`** — flowsheet / unit-operation engine (depends on `thermo`):
  a differentiable `Stream` pytree, energy-balanced unit operations (`flash_drum`,
  `heater`, `valve`, `pump`, `compressor`, `turbine`, `mix`, `splitter`), a
  recycle/tear solver with implicit-diff gradients (`tear_solve`, `Flowsheet`),
  distillation columns (shortcut FUG and a rigorous equilibrium-stage model),
  binary diagrams / azeotropes / residue-curve maps,
  [reactors](reactions.md) (equilibrium, stoichiometric, CSTR, PFR, batch), and
  reactive separations (reactive flash & distillation).
- **`fugacio.copilot`** — LLM design agent (depends on `sim`): a JSON tool
  registry over the engine — properties, unit operations, distillation, reactors,
  reaction equilibrium, and gradient-based optimizers — plus a model-agnostic
  agent loop.

Everything is written in [JAX](https://github.com/jax-ml/jax) and the iterative
solvers carry implicit-function-theorem gradient rules, so an entire flowsheet is
end-to-end differentiable — including through phase equilibrium.

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
reference codes — [CoolProp](https://github.com/CoolProp/CoolProp) and
[`chemicals`](https://github.com/CalebBell/chemicals) for pure-fluid properties,
[`thermo`](https://github.com/CalebBell/thermo) /
[Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) for activity
coefficients, and [Cantera](https://github.com/Cantera/cantera) for
reaction equilibrium and standard-state thermochemistry.
