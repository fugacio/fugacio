# Saved process cases and design studies

`fugacio.sim.cases` turns a process definition into a reusable engineering
artifact. A case declares its component basis, property package, feeds, units,
parameters, measurements, specifications, and optional economics. The same
definition can be saved as JSON, solved with either flowsheet backend, studied,
and reopened by the copilot.

## Run a saved case

```bash
uv run fugacio example heater heater.json
uv run fugacio validate heater.json
uv run fugacio run heater.json --output run.json --report report.md
```

The CLI saves immutable case revisions and run artifacts in `.fugacio-cases`
by default. Use `--workspace PATH` to choose another directory. A run prints
its complete JSON artifact, including `artifact_id`. Use that ID to inspect,
compare, or replay results:

```bash
uv run fugacio inspect RUN_ID --report
uv run fugacio replay RUN_ID
uv run fugacio compare BASELINE_ID CANDIDATE_ID
```

The command exits with status 0 on success, 1 for an input or execution error,
and 2 when a completed run or study fails its checks. A failed calculation is
still saved. Input validation errors don't fabricate a run.

The Python equivalent is:

```python
from fugacio.sim.cases import CaseRunner, CaseWorkspace, ProcessCase
from fugacio.sim.cases.examples import example_case

example_case("heater").save("heater.json")
case = ProcessCase.load("heater.json")
runner = CaseRunner(case)
baseline = runner.run(check=True)
workspace = CaseWorkspace(".fugacio-cases")
workspace.save_run(baseline)

candidate = runner.run({"temperature": {"value": 360, "unit": "K"}}, check=True)
workspace.save_run(candidate)
print(candidate.markdown())
```

`run(check=False)`, the default, returns failures for inspection.
`run(check=True)` and `run.check()` raise `CaseAcceptanceError`, which carries
the complete run in its `run` attribute.

## Format and units

Schema version 1 accepts these top-level fields:

| Field | Contents |
| --- | --- |
| `schema_version`, `name` | Version and portable identifier |
| `description` | Optional explanatory text |
| `components` | Ordered, canonicalized component names |
| `property_package` | Method, explicit options, optional inline measured fit and holdout IDs |
| `parameters` | Named scalar defaults, declared units, optional bounds |
| `feeds` | Named flow, composition, pressure, and temperature or molar enthalpy |
| `units` | Registered unit kinds, names, ports, and settings |
| `metrics` | Named, dimension-checked result expressions and display units |
| `specifications` | Bounded manipulated parameters and target metrics |
| `economics` | Explicit utility prices, operating time, and optional capital estimates |

Dimensional literals require `{"value": 16, "unit": "bar"}`. A parameter
reference is `{"parameter": "pressure"}`. Parameter declarations use
`{"value": 16, "unit": "bar", "lower": 14, "upper": 18}`; defaults and bounds
share the declared unit. Numerical kernels use SI. Parameters and metric
results retain their original display units.

Use `K`, `degC`, or `degF` for absolute temperature and `delta_K`, `delta_degC`,
or `delta_degF` for intervals. Pressure is absolute. Flow is molar, and feed
`z` is a mole-fraction vector in the case's component order. Bare numbers are
allowed only for dimensionless literals. The closed unit vocabulary includes
engineering pressure, flow, power, energy, area, volume, and currency units;
`fugacio registry` and the copilot's `case_format` describe the available inputs.

JSON loading rejects duplicate keys, nonfinite literals, unsupported versions,
unknown fields, executable expressions, and oversized documents. Case IDs are
SHA-256 hashes of normalized content. `case.with_parameters(...)` creates a new
revision; it never mutates an existing case or rewrites an earlier run.

Each stream has one producer and at most one consumer. Introduce a splitter
for material branches. Feeds must connect to the process, every unit must have
a path from a feed, and the case must have an external product. Recycles are
allowed and partitioned by the existing flowsheet engine.

## Units and backends

The initial registry supports mixers, splitters, heaters/coolers, valves,
pumps, compressors, turbines, PT flashes, ideal component separators,
two-sided heat exchangers, rigorous MESH columns, and single-reaction
stoichiometric reactors.

Columns retain all product streams, side draws, condenser and reboiler duties,
stage temperatures and pressures, phase compositions, K-values, and traffic.
Column specifications use component names and explicit physical quantities.
Stage numbers start at 1. Reactions declare a stoichiometric vector in component
order and a named key reactant; the case validator checks element conservation.
Reaction energy includes standard formation enthalpies with the package's
sensible and residual terms. The reference-fluid energy datum isn't supported
for these reaction cases.

```python
from fugacio.sim.cases import CaseRunner, SolverOptions

sequential = CaseRunner(case, options=SolverOptions(recycle_method="broyden"))
simultaneous = CaseRunner(case, options=SolverOptions(backend="eo"))
```

Both backends delegate unit physics to the same kernels. The case EO adapter
solves simultaneous stream equations with nested implicit unit solves; it
doesn't expand a column into global MESH unknowns. Pure-fluid EO streams retain
enthalpy coordinates so quality isn't lost on the saturation line. Failed
solves retain their reports, and no backend silently substitutes another.
Execution compiles each complete registered unit, including feed-property
preparation and retained outputs, with dynamic operating values. Studies reuse
one local linearization per operating point, selecting forward or reverse
directions from the input/output counts. They retain the individual unit kernels.
Fixed case topology and dynamic recycle parameters let optimizer trials reuse
the compiled maps. Saved
profiles remain available without becoming study derivative outputs.
Separate CLI invocations can reuse JAX's persistent compilation cache with
`export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"` before running commands.
The [performance guide](performance.md) covers structured column solves,
flowsheet coloring, reusable derivatives, profiling artifacts, and isolated
benchmarks. Timings depend on hardware, JAX version, and cache state.
For machines with limited memory, pass `release_caches=True` to sensitivities
or optimization (or `"release_caches": true` in a CLI request). This explicitly
clears process-wide in-memory JAX compilation caches between the baseline,
derivative/optimization, and final audit phases. Optimization retains compiled
kernels across points and collects unreachable Python objects between them.
It retains the disk cache and all completed artifacts. Clearing caches doesn't eliminate the peak memory
required by each compilation. The older dense implementation couldn't complete
the plant derivative in a 7 GB Linux validation container; the dedicated
performance workflow now checks the structured implementation against an
explicit process memory limit and retains the observed result.

The registry is deliberately finite. Dynamic units, multiphase liquid
inventories, custom Python functions, and arbitrary Python EO blocks remain
available through their existing APIs; they aren't serialized as version 1
cases. A new portable kind needs a validator, kernel adapter, retained result
definition, and tests.

## Metrics and design specifications

A metric combines an expression with a display unit:

```json
{
  "reboiler_duty": {
    "expression": {"unit": "column", "property": "reboiler_duty"},
    "unit": "MW"
  },
  "purity": {
    "expression": {"stream": "distillate", "property": "mole_fraction", "component": "propane"},
    "unit": "%"
  }
}
```

Expressions can reference parameters, streams, retained unit properties,
column profiles, plant totals, or other metrics. Arithmetic operations are
`add`, `subtract`, `multiply`, `divide`, `negate`, `abs`, `square`, `min`, and
`max`, with their operands in `args`. Cycles and incompatible dimensions are
rejected. No Python evaluation is involved. An undefined quantity, such as the
composition of an empty stream, is unavailable and fails a declared metric.

An outer design specification frees one bounded parameter to meet one metric:

```json
{
  "name": "target_duty",
  "parameter": "temperature",
  "metric": "duty",
  "target": {"value": 2.5, "unit": "kW"},
  "tolerance": {"value": 0.001, "unit": "W"}
}
```

Put this object in the case's `specifications` array. Coupled specifications
are solved together within normalized parameter bounds. Acceptance checks the
actual target error and the final bounds. Implicit derivatives follow the
converged specification, not the manipulated parameter's initial guess.

## Sweeps, sensitivities, and optimization

```python
from fugacio.sim.cases import optimize, sensitivities, sweep

grid = sweep(
    runner,
    {"temperature": [{"value": t, "unit": "K"} for t in (340, 350, 360)]},
    workspace=workspace,
)
gradient = sensitivities(runner, ["temperature"], ["duty"], workspace=workspace)
design = optimize(
    runner,
    ["temperature"],
    "annual_cost",
    constraints=[{
        "metric": "product_temperature",
        "lower": {"value": 360, "unit": "K"},
        "tolerance": {"value": 1e-5, "unit": "delta_K"},
    }],
    workspace=workspace,
)
```

Sweeps evaluate a bounded Cartesian grid in deterministic order. Failed points
remain in the manifest with a failed run or an input error. Study manifests
reference immutable run IDs; `StudyResult.save` writes all runs before the
manifest. When a workspace is provided, completed audited runs are also saved
as the study progresses, so an interrupted study retains its completed work.
An unacceptable baseline produces a failed study with the full baseline run;
it never produces a derivative or optimization claim.
Baseline comparisons align metrics by expression and dimension and
show changes in parameters, packages, solver settings, and audit policy.

Sensitivity studies compare JAX derivatives with centered differences of
independently accepted runs. They reject boundary points without a centered
interval, failed perturbations, nonfinite derivatives, and changes between
liquid, vapor, two-phase, or empty stream regimes. Reported derivatives include
SI values and display-unit conversions. This is a local consistency test, not
an uncertainty estimate or a guarantee of global smoothness.

Optimization uses host SLSQP with exact JAX derivatives and declared finite
variable bounds. Constraints accept `lower`/`upper` or `equal`, plus an explicit
tolerance. Trial points receive numerical checks; the baseline and final point
receive full physical audits. An optimizer's success flag alone can't produce
an accepted study. Final feasibility, finite derivatives, physical acceptance,
and cost-correlation ranges must also pass. Results are local candidates, with
no global-optimum claim. A parameter controlled by a design specification can't
also be an independent study variable.

Derivative evaluations and optimizer trials reuse the accepted baseline's
recycle streams and column stage states as detached initial guesses. Failed
trials never replace that seed. Each study records its baseline run ID.
Finite-difference points and final candidates start cold, and optimization
acceptance requires the final objective and constraint metrics to agree with
the warm-start trial within a scaled relative tolerance of 1e-6.

The CLI takes study arguments in a JSON request file:

```bash
uv run fugacio sweep heater.json sweep-request.json
uv run fugacio optimize heater.json optimization-request.json
uv run fugacio sensitivities heater.json sensitivity-request.json
```

For example, a sweep request is
`{"grid": {"temperature": [{"value": 340, "unit": "K"}, {"value": 360, "unit": "K"}]}}`.
Optimization requests use `variables`, `objective`, optional `sense`, and
`constraints`. Sensitivity requests use `parameters` and `metrics`. Optional
`overrides` set the base operating point for optimization and sensitivities.

## Depropanizer and measured heater

```bash
uv run fugacio example depropanizer depropanizer.json
uv run fugacio run depropanizer.json --recycle-method broyden --report depropanizer.md
uv run fugacio example measured-heater measured-heater.json
uv run fugacio run measured-heater.json --report measured-heater.md
```

The depropanizer reproduces the existing 16-stage propane/butane/pentane plant
case. It meets 95% propane purity and 98% recovery, transfers bottoms heat to
the feed, and exposes pressure and exchanger approach as bounded design
variables. Its economics include separately accumulated heating and cooling,
an exchanger sized from retained duty and temperature differences, and explicit
capital and utility assumptions. The supplied prices and CEPCI are illustrative
screening inputs, not current quotes. Optimize `annual_cost` over `pressure`
and `approach`; the column's purity and recovery remain internal specifications.

The measured heater embeds the ethanol-water NRTL fit from the reproducible
qualification workflow. Loading verifies training observation IDs and raw
source hashes, then reevaluates the declared independent publication holdout.
The stored fit coefficients don't supply a trusted qualification flag. The
example stays within observed training bounds. The example evidence can be
regenerated with `scripts/qualify.py`; its source IDs and hashes remain in JSON.

## Acceptance and reproducibility

Every run retains separate evidence:

1. Numerical solver and unit reports, plus final outlet replay checks.
2. Stream phase-state checks and unit/plant component and energy closure.
   Reactive boundaries also check elements. Empty streams have no equilibrium
   composition and are explicitly identified.
3. Operating bounds, specification residuals, and finite named metrics.
4. Economics, including correlation size ranges even when the underlying
   screening function clips its numerical argument.
5. Parameter sources, model assumptions, observed applicability bounds, and
   independently evaluated empirical qualification.

Heat and shaft work are positive into the fluid. Heating and cooling utilities
are accumulated separately; a column's condenser and reboiler don't cancel out
of utility cost. Recovered shaft power is reported separately and receives no
automatic electricity-sale credit. Annual cost uses USD/s internally and
`USD/yr` for display, based on a 365.25-day year and declared operating hours.

An isothermal flash, temperature-specified mixer, or ideal component separator
infers required external heat from its specified states. Its report says so;
balance closure doesn't establish available equipment capacity. Unit profiles
are retained, but stream audits don't independently validate every internal
stage's stability. A finite-start stability search isn't a global proof.

Artifacts include the exact case, requested and solved parameter values,
solver options, acceptance policy, and runtime versions. Files are strict JSON
and are written atomically. Hashes detect alteration and identify revisions;
they aren't signatures and don't prove that an arbitrary external author ran
a calculation. Replays may differ numerically across JAX versions or hardware.
Unknown empirical ranges remain unknown, and accepted process balances don't
qualify an entire plant or model family.

The audit also checks declared equipment roles: pumps and compressors raise
pressure, valves and turbines lower it, and liquid pumps require liquid feeds.
Pressure-specified mixers, flashes, and columns require sufficient feed pressure.
Reported metrics distinguish process expressions from literal input values.

## Accountable copilot

```python
from fugacio.copilot import run_design_agent
from fugacio.sim.cases import CaseWorkspace

result = run_design_agent(
    "Build a heater case and minimize annual utility cost while delivering at least 360 K.",
    provider,
    workspace=CaseWorkspace(".fugacio-cases"),
)
print(result.answer)
```

`provider` implements the existing provider-neutral interface. The design loop
can obtain examples, create or load cases, update a complete revision, set
parameter defaults, run cases, perform studies, and inspect stored evidence.
Case mutations invalidate a pending submission. A stale revision, failed run,
unknown metric, or externally loaded run that hasn't been recomputed in the
session can't be submitted.

Completion requires `submit_design(run_id, metrics, baseline_id=...)`. It takes
recorded metric names, not model-authored numbers or a free-form report. The
selected metrics must depend on process results; an input literal isn't computed
performance. The
final answer is rendered deterministically from the accepted current-case run.
Unsupported provider text remains visible in the transcript and prompts the
model to continue computing. Exhausted budgets return an explicit incomplete
result. The general `run_llm_agent` remains available for explanatory questions;
it doesn't enforce this stricter design-submission contract.
