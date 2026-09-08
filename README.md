# Fugacio

**Open, differentiable thermodynamics and process simulation, with an AI design copilot.**

Fugacio is building an open-source alternative to closed, expensive process
simulators. Its numerical core uses [JAX](https://github.com/jax-ml/jax) to
support gradients through physical properties, phase equilibrium, and converged
flowsheets for optimization, parameter estimation, and machine learning.

The project is in pre-alpha. Model availability and numerical convergence don't
establish experimental validity; see [validation and qualification](#validation-and-qualification)
for the current evidence and limitations.

[Documentation](https://fugacio.com/) · [API reference](docs/api/index.md) ·
[Quickstart](#quickstart) · [Development](#development)

## Project goals

- **Make process simulation open and accessible.** Provide an inspectable,
  extensible foundation for thermodynamics, process design, and control without
  dependence on proprietary simulators or datasets.
- **Make the full calculation differentiable.** Carry gradients through
  equilibrium, recycle loops, and design calculations so engineers can optimize
  processes, fit parameters, and integrate machine learning with physical models.
- **Make physical correctness continuously checkable.** Treat conservation laws,
  thermodynamic consistency, reference comparisons, experimental evidence, and
  gradient checks as acceptance criteria as each model is added.
- **Make AI-assisted design accountable to the engine.** Give a provider-neutral
  copilot access to simulation and optimization tools, with numerical reports and
  physical acceptance checks to support its results.

## Quickstart

Use Python 3.11 or later and [uv](https://docs.astral.sh/uv/). From a checkout of
this repository, install the workspace packages and development dependencies:

```bash
uv sync --locked --all-packages
```

Run the following example with `uv run python`. It separates a hydrocarbon feed
with a Peng-Robinson flash and computes the sensitivity of vapor flow to drum
temperature. Temperature is in kelvins, pressure in pascals, and flow in moles per
second.

```python
import jax
import jax.numpy as jnp

from fugacio.sim import Stream, flash_drum

feed = Stream.from_fractions(
    ("methane", "propane", "n-pentane"),
    jnp.array([0.5, 0.3, 0.2]),
    flow=100.0,
    t=320.0,
    p=20e5,
)
vapor, liquid = flash_drum(feed, 320.0, 20e5)

# Differentiate vapor flow through the converged phase-equilibrium solve.
d_vapor_dT = jax.grad(lambda t: flash_drum(feed, t, 20e5)[0].total)(320.0)

print(f"Vapor: {vapor.total:.3f} mol/s; liquid: {liquid.total:.3f} mol/s")
print(f"Vapor-flow sensitivity: {d_vapor_dT:.3f} mol/(s K)")
```

Equilibrium and recycle solvers use implicit differentiation of converged
solutions. The [reliability guide](docs/reliability.md) explains solve reports,
phase inventories, warm starts, continuation, and derivative limits.

### Saved process workflow

For a saved process workflow, export an example, solve it, and keep an audited
run and engineering report:

```bash
uv run fugacio example heater heater.json
uv run fugacio run heater.json --output run.json --report report.md
```

The [process cases guide](docs/process-cases.md) covers case revisions, both
flowsheet backends, bounded design specifications, sweeps, constrained
optimization, gradient checks, and copilot submissions tied to computed runs.
It includes a rigorous depropanizer with heat recovery and a measured NRTL
heater whose independent holdout is reevaluated when the case is loaded.

## Packages

The [uv workspace](pyproject.toml) contains three computational layers and an
umbrella package:

| Package | Import | Role |
| --- | --- | --- |
| [`fugacio-thermo`](packages/fugacio-thermo/) | `fugacio.thermo` | Physical properties, phase equilibrium, parameter regression, and reaction thermochemistry. |
| [`fugacio-sim`](packages/fugacio-sim/) | `fugacio.sim` | Unit operations, flowsheets, optimization, dynamics, control, and heat integration. |
| [`fugacio-copilot`](packages/fugacio-copilot/) | `fugacio.copilot` | AI design agent with a JSON tool registry and provider-neutral tool-calling loop. |
| [`fugacio`](packages/fugacio/) | No additional module | Installs all three component packages together. |

`fugacio-copilot` depends on `fugacio-sim`, which depends on `fugacio-thermo`.
[Import Linter](https://github.com/seddonym/import-linter) enforces these boundaries
in CI. The component packages share the `fugacio` namespace and are published as
separate distributions with coordinated versions.

## Capabilities and guides

The [documentation site](https://fugacio.com/) provides worked examples and an
API reference generated from package docstrings.

- **Thermodynamics:** A common [property-package interface](docs/property-packages.md)
  covers cubic equations of state, activity-coefficient models, molecular
  [PC-SAFT](docs/molecular-saft.md), and [reference Helmholtz equations of state](docs/reference-fluids.md),
  including IAPWS-95 water and steam. Guides also cover
  [phase equilibrium and stability](docs/phase-equilibrium.md),
  [physical and transport properties](docs/physical-properties.md), and
  parameter regression. UNIFAC and Joback provide group-contribution estimates
  where curated parameters aren't available.
- **Steady-state simulation:** Energy-balanced units, automatic recycle
  partitioning and tear selection, and two-sided heat exchangers support
  [flowsheeting](docs/flowsheeting.md). The engine also includes
  [rigorous MESH distillation](docs/distillation.md),
  [reactors and reactive separations](docs/reactions.md), and simultaneous
  [equation-oriented flowsheet solving](docs/equation-oriented.md).
- **Process design:** Differentiable constrained optimization, design
  specifications, equipment sizing, and screening economics are covered in the
  [optimization guide](docs/optimization.md).
  [Heat integration](docs/heat-integration.md) adds pinch analysis, utility
  targets, area and cost targeting, and heat-exchanger network synthesis.
- **Dynamics and control:** [Dynamic simulation](docs/dynamics.md) combines
  differentiable ODE integrators, dynamic units, PID control, and controller
  tuning. [Advanced control](docs/advanced-control.md) adds linear, nonlinear,
  and economic model predictive control, plus state estimation.
- **AI design copilot:** A [tool registry](docs/api/copilot/tools.md) exposes
  calculations across the stack through a multi-turn
  [agent loop](docs/api/copilot/agent.md), with OpenAI, Anthropic, and mock
  [providers](docs/api/copilot/providers.md).
  The [accountable design loop](docs/process-cases.md#accountable-copilot)
  requires an accepted current-case run and generates its final report from
  recorded metrics.

## Validation and qualification

Physical correctness is a project requirement. Fugacio uses an executable
acceptance harness with four complementary sources of evidence:

1. **Physical consistency:** Mass and energy balances, Gibbs-Duhem and Maxwell
   relations, equifugacity, and tangent-plane phase-stability tests.
2. **Reference implementations:** Differential tests against open codes,
   including [CoolProp](https://github.com/CoolProp/CoolProp),
   [chemicals](https://github.com/CalebBell/chemicals),
   [thermo](https://github.com/CalebBell/thermo),
   [Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl), and
   [Cantera](https://github.com/Cantera/cantera).
3. **Experimental measurements:** Data from the
   [NIST ThermoML Archive](https://www.nist.gov/mml/acmd/trc/thermoml/thermoml-archive),
   with source provenance, parameter fitting, and independent holdout validation.
4. **Derivative checks:** Automatic-differentiation gradients compared with
   finite differences.

These checks provide repeatable feedback for incremental development, including
long-running, AI-assisted development. Numerical convergence, physical
acceptance, and agreement with measurements are assessed separately.

The [measured qualification workflow](docs/qualification.md) covers 447
experimental conditions across 20 binary systems from seven NIST ThermoML
records. The [committed matrix](benchmarks/thermodynamic-qualification.json)
contains two qualified property cases, two unqualified liquid-liquid equilibrium
cases, and 18 cases without curated NRTL parameters. Qualification applies to
specific systems and properties; it doesn't extend to every model family.

Parameter evidence distinguishes measured fits, curated data, predictive methods,
and explicit zero-interaction assumptions. `package_for` rejects missing NRTL
or UNIQUAC pairs unless the caller explicitly permits them. The older bundled
parameter bank remains synthetic demonstration data, separate from the measured
corpus.

To rebuild the matrix and run the ethanol-water example, which fits one
publication, validates against a second, and checks an energy-balanced heater:

```bash
just qualify
```

## Development

Install [just](https://github.com/casey/just) to use the repository's task runner,
then run `just` to list available tasks.

| Task | Command |
| --- | --- |
| Format and autofix | `just fmt` |
| Lint and check formatting | `just lint` |
| Check types | `just types` |
| Check import boundaries | `just imports` |
| Run tests without external reference packages | `just test` |
| Run lint, types, import boundaries, tests, and qualification | `just check` |
| Run reference comparison tests | `just oracles` |
| Rebuild measured qualification and process example | `just qualify` |
| Run the pinned Julia/Clapeyron comparison | `just clapeyron-oracles` |
| Preview documentation | `just docs-serve` |
| Build documentation with warnings treated as errors | `just docs-build` |

`just test` runs `uv run pytest`, excluding tests marked `oracle`.
`just oracles` installs the optional reference dependencies through uv and runs
those tests explicitly. CI runs reference comparisons on pull requests, pushes
to `main`, and a weekly schedule. The separate Julia comparison requires Julia
1.10.10 and uses Clapeyron 0.6.25 with a committed dependency manifest.

Documentation sources live in [`docs/`](docs/). Public APIs use Google-style
docstrings, enforced by Ruff; keep their argument, return, and exception
descriptions accurate. `just docs-serve` provides live preview.
`just docs-build` matches the strict CI build and requires Cairo and Pango for
social-card rendering.

## License

Fugacio is licensed under the [Apache License 2.0](LICENSE).
