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
| `fugacio-thermo` | `fugacio.thermo` | Differentiable properties + phase equilibrium: EOS & γ–φ activity models, energy/PT-PH-PS flashes, rigorous LLE/VLLE, parameter regression, and reaction thermochemistry, equilibrium & kinetics (the foundation). |
| `fugacio-sim` | `fugacio.sim` | Flowsheet engine: energy-balanced unit ops, a differentiable recycle/tear solver, distillation columns, binary/residue-curve diagrams, reactors, and reactive separations (depends on `thermo`). |
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

The `fugacio.copilot` agent exposes all of this — properties, unit ops,
distillation, reactors, optimization, sizing, and costing — as a JSON tool
registry, driven by a vendor-neutral provider layer (OpenAI / Anthropic / mock)
through a multi-turn, tool-calling loop.

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
[`chemicals`](https://github.com/CalebBell/chemicals) for pure-fluid properties,
[`thermo`](https://github.com/CalebBell/thermo) /
[Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) for activity
coefficients, and [Cantera](https://github.com/Cantera/cantera) for reaction
equilibrium) are marked `oracle` and excluded from the default run; install those
optional packages and run them explicitly with `just oracles`.

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
