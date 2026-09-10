# Process performance and reusable derivatives

Fugacio separates physical acceptance from performance observations. The
structured solvers solve the same residual equations as their dense references.
A short runtime doesn't establish convergence, derivative accuracy, or measured
qualification. Failed runs and resource limits remain visible in saved evidence.

## Numerical choices

```python
from fugacio.sim.cases import CaseRunner, SolverOptions
from fugacio.sim.cases.examples import example_case

runner = CaseRunner(
    example_case("depropanizer"),
    options=SolverOptions(
        recycle_method="broyden",
        column_solver="block",
        eo_jacobian="colored",
    ),
)
structure = runner.diagnose_structure()
run = runner.run(check=True)
```

`column_solver="block"` is the default for saved cases. The direct
`rigorous_column` API calls this option `linear_solver`. Set either option to
`"dense"` to use the reference assembly and solve. `eo_jacobian="colored"`
reduces global flowsheet differentiation work; `"dense"` retains the original
assembly. These are numerical choices recorded in runs, independent of case
identity. Replaying an older run without these fields selects its original
dense algorithms.

### Rigorous columns

For N stages and C components, each stage has b = 2C + 1 unknowns.
The condenser and reboiler add a border of size k, at most two. A stage's MESH
equations depend on its own and adjacent stages. Specifications form the
unrestricted border. Total-condenser and reboiler duties exchange coordinates
with endpoint flow variables so the local blocks can be eliminated. This
permutation doesn't change any physical equation or specification.

Three stage colors assemble the local Jacobian in at most 3b + k forward
directions, plus k reverse directions for the specification rows. A
three-component, 16-stage column has 114 unknowns but needs 23 forward and two
reverse directions. Increasing it to 64 stages gives 450 unknowns with the same
direction count. Storage for the normal block representation grows linearly in
stage count, with fixed component count.

Block elimination pivots within each stage. Every solve checks its linear
residual against the original structured matrix. It records the infinity-norm
backward error, `norm(Ax - b) / (norm(A) * norm(x) + norm(b))`, and also requires
`norm(Ax - b) / norm(b)` to pass. Each right-hand side has its own scale and an
underflow guard. The latter check catches inaccurate Newton directions whose
backward error looks small because the solution has very large components.
Both checks use complete vectors, so tiny individual equations and exactly
zero solution components don't impose spurious componentwise relative accuracy.
They keep adjoint acceptance independent of whether a metric is expressed in
watts or megawatts. If the block result is nonfinite or inaccurate, a pivoted
dense solve is checked against the same equations. An unresolved result returns
nonfinite sensitivities. The fallback
can still require quadratic storage; this isn't a guarantee of bounded memory
for every ill-conditioned system.

Forward and transposed solves use implicit differentiation. Pivot decisions,
iteration histories, and warm-start selection don't carry derivatives. The
converged matrix and factors can be retained by a local linearization for later
directions. Tests compare dense and block assembly, cross-block pivot fallback,
material and energy balances, finite differences, and higher derivatives.

`result.solver_info()` describes the selected strategy and direction count. It
isn't a count of actual fallback events. A direct
`BorderedBlockJacobian.solve_with_info` call returns its actual fallback flag,
backward error, and relative residual. Full stage component flows are retained
for exact warm starts, including columns with side products.

### Flowsheet assembly and diagnostics

`EOFlowsheet.diagnose_structure()` and `CaseRunner.diagnose_structure()` inspect
declared incidence without a nonlinear solve. Reports include equation and
variable counts, maximum matching, unmatched labels, density, and coloring
cost. They also identify duplicate auxiliary owners and freed variables.
Matching is an upper bound on numerical rank. `EOFlowsheet.diagnose()` adds
state-dependent rank and conditioning checks.

Built-in EO blocks declare their stream and auxiliary dependencies. Custom
blocks, including subclasses of built-ins, default to all unknowns unless
they explicitly implement `residual_dependencies`. Declaring dependencies is
a contract over the model domain; observed zeros at one state aren't enough.
Global EO matrices still use dense pivoted factorization. Registered columns
and flashes retain their nested implicit solves; the case adapter doesn't
expand a column into global MESH unknowns.

The modular recycle solver keeps the initial unit pass outside its iteration
executable. Broyden backtracking uses one trial-evaluation site, including the
full step, so the compiler doesn't receive separate copies of the entire unit
train for full and shortened steps. Its residual checks and accepted step
sequence retain the same search policy.
Recycle sensitivities retain a dense global matrix but assemble its directions
sequentially. This avoids multiplying a column's internal tangent batch by
the number of tear coordinates. State and parameter directions share one
residual linearization, so the matrix and its right-hand side don't repeat
the same nested unit linearizations.
Scalar implicit roots and ordinary dense implicit systems also share residual
data between state and parameter directions. This matters when an exchanger
duty root contains energy flashes and equilibrium roots: separate residual
evaluations at each level otherwise repeat work throughout that stack.
Phase-classification locators detach their state and package inputs before
calling a flash. Their derivatives were already excluded from the physical
model; this avoids constructing those unused derivatives in the first place.
The same rule applies to numerical LU factors: the linear solve differentiates
its original matrix equation, while factor and pivot calculations stay detached.
Exchanger sides with matching model structures and array shapes share one
sequential PH-flash body. Each side retains its own composition and model
parameters. Different models or component bases retain separate bodies;
phase-regime branches remain conditional within each position.
Mixture outlets reuse the curve endpoints and resolve only their phase
inventories. Pure-fluid outlets retain a PH solve because temperature alone
doesn't determine saturation quality. This avoids repeating two complete
mixture energy solves after the profiles have already converged.

```bash
uv run fugacio diagnose examples/process-cases/depropanizer.json
```

Case runners share compiled unit templates when their numerical structure and
fixed settings match. Each instance binds its own parameter values; renaming
units, streams, or parameters doesn't require another executable. Scalar inputs
keep their precision but use consistent JAX scalar types across cold starts,
recycles, and derivative replays. Structural diagnostics report the instance
and template counts.

## Derivative studies

Sensitivities and optimization build one local plant linearization per operating
point. They reuse it across derivative directions and retain the compiled unit
boundaries. `derivative_mode="auto"` chooses forward mode when there are no more
inputs than outputs, and reverse mode otherwise. The choice minimizes direction
count; it isn't a hardware-specific timing prediction. Explicit `"forward"` and
`"reverse"` modes support comparisons and unusual models.

```python
from fugacio.sim.cases import sensitivities
from fugacio.sim.cases.examples import heater_bank_case

bank = CaseRunner(heater_bank_case(24))
study = sensitivities(
    bank,
    list(bank.parameters),
    ["duty"],
    derivative_mode="auto",
    derivative_batch_size=1,
)
assert study.accepted
assert study.artifact["derivatives"]["directions"] == 1
```

The default batch size of one limits tangent storage. Larger batches may trade
memory for throughput. Each new operating point gets a new linearization;
there's no global point cache that could return a stale derivative. Study
artifacts record the selected orientation, dimensions, and direction count.
Finite differences still use independently audited perturbations, and an
optimization's final candidate still starts cold and receives its physical
audit.

For array-valued numerical kernels, `fugacio.thermo.sensitivity.linearize`
exposes the reusable map directly:

```python
import jax.numpy as jnp
from fugacio.thermo.sensitivity import linearize

local = linearize(lambda x: jnp.sum(jnp.sin(x)), jnp.array([0.2, 0.5]))
value = local.value
direction = local.jvp(jnp.array([1.0, 0.0]))
adjoint = local.vjp(jnp.array(1.0))
jacobian = local.jacobian(mode="reverse")
```

The map belongs to that point and the model captured by the function. Release
it when those inputs change to release its retained residuals. `has_aux=True`
keeps numerical reports separate from differentiated outputs.

## Recorded performance observations

`profile` saves timings in a separate content-addressed artifact. Ordinary run
identities don't acquire timestamps or timing fields.

```python
from fugacio.sim.cases import CaseWorkspace, profile

observed = profile(
    runner,
    parameters=["pressure", "approach"],
    metrics=["annual_cost"],
    warm_repeats=2,
    workspace=CaseWorkspace(".fugacio-cases"),
)
```

The artifact records the first and repeated audited runs, first and repeated
linearizations, Jacobian applications, environment, solver configuration, and
process peak RSS. Operations synchronize device results before stopping their
timers. RSS is the process-lifetime high-water mark, including compiled
executables; it isn't a per-phase allocation count. A profile's finite
derivatives and repeatability don't replace the independent finite-difference
checks in `sensitivities`.

The Python `sensitivities` and `optimize` APIs also accept a
`PerformanceRecorder` through `recorder`. It measures each linearization and
Jacobian application while keeping timings outside ordinary study identities.
The standalone benchmark includes these phases in its progress checkpoints.

The CLI accepts `fugacio profile CASE REQUEST.json`, and the design copilot's
`study_case` accepts `kind="profile"`. Its `diagnose_case` tool inspects
structure without creating a trusted run. Solver options and study derivative
options are available through the same CLI and copilot routes as ordinary runs.

## Measured behavior

The September 9, 2026, observations are recorded in
`benchmarks/process-performance.json`, including source digests, environments,
phase timings, acceptance checks, and failures. The following measurements
used Linux, JAX 0.10.1, float64, four CPUs, and `MALLOC_ARENA_MAX=2`. Each row
used a fresh process and empty persistent cache. A macOS NRTL study shared
the physical host during these runs, so elapsed times aren't isolated hardware
comparisons. GB means 1,000,000,000 bytes.

| Workload | Cold total, seconds | Peak RSS, GB | Acceptance evidence |
| --- | ---: | ---: | --- |
| 32-stage binary column, block | 114.8 | 1.392 | Balances and finite differences |
| 32-stage binary column, dense | 94.5 | 1.109 | Balances and finite differences |
| 64-stage binary column, block | 110.7 | 1.408 | Balances and finite differences |
| 24 independent heater variables | 43.5 | 0.861 | Audited runs and repeatable reverse derivatives |

The 32-stage column outputs agree between algorithms. The block assembler
uses 17 forward and two border reverse directions for 162 unknowns, and keeps
that direction count at 64 stages and 322 unknowns. Its warmed derivative
executions took 14–15 ms at 32 stages, compared with 25–26 ms for dense, and
105–106 ms at 64 stages. Dense compilation was faster for the 32-stage case.
The 64-stage block solve also agrees with its dense reference and converges
in seven Newton iterations. All column finite-difference relative errors
were below 0.00003. These observations don't establish a universal crossover
or a 3x cold-gradient speedup against v0.6.0. Dense fallback can change the
cost of a structured solve, so direction counts alone don't predict runtime.

The heater bank uses one compiled template for 24 independently bound
temperatures and one reverse direction for its total-duty derivative. The
profile checks finite, repeatable derivatives; separate regression tests check
the template bindings against independently audited finite differences.

The full depropanizer optimization passed in a Linux container with a hard
7,000,000,000-byte memory limit and no swap allowance, using JAX 0.10.1,
four CPUs, and `MALLOC_ARENA_MAX=2` at process startup. Its cold total was
832.9 seconds and peak RSS was 6.590 GB. SLSQP
completed six iterations; every point had accepted numerical derivatives,
and the final candidate passed its independent cold-start physical audit.
Modeled annual cost fell 7.9% and reboiler duty fell 10.6%, with propane purity
at 95% and recovery at 98%. These are screening-economics results for the
saved case, not measured operating savings.

The first plant linearization took 506.0 seconds and its Jacobian application
took 43.3 seconds. Subsequent linearizations took 3.1–3.8 seconds and Jacobian
applications took 0.8–0.9 seconds. Neither other hosts nor additional study
outputs inherit a 7 GB guarantee. The default-allocator cold run reached
7.006 GB at its fifth optimizer point; warm-cache runs also exceeded 7 GB.
Failed resource observations remain in the benchmark record. A warm persistent
cache can reduce compilation time without reducing peak resident memory.

The Linux memory-budget workflow limits glibc's allocation arenas to reduce
retained allocation memory. Set `MALLOC_ARENA_MAX=2` before starting Python to
reproduce that runtime configuration. This changes allocation behavior, not
the process equations or their acceptance tolerances. Profiles record this
setting alongside thread and XLA settings. See the
[glibc allocation tunables](https://sourceware.org/glibc/manual/latest/html_node/Memory-Allocation-Tunables.html).

The separate two-parameter, two-output depropanizer sensitivity study passed
all four independent finite-difference comparisons. With an existing cache,
it took 270.8 seconds and peaked at 7.654 GB. Its container allowed 7.8 GB of
RAM and 9 GB of RAM plus swap, within a local VM with about 8.2 GB of RAM and
1.1 GB of swap. That observation uses a larger allowance than the optimization
target; it isn't evidence that every study fits within 7 GB.

The 12-stage ethanol/water NRTL train passed all seven physical audits and all
six finite-difference comparisons across approach temperature, pressure, and
reflux. The final numerical implementation took 2,184.5 seconds with an existing
persistent cache on macOS with JAX 0.4.38 and peaked at 9.246 GB. Its largest
derivative relative error was 0.000055. An earlier cold run passed in 2,820.2
seconds at 8.234 GB, before the linear acceptance check changed from rowwise
to global normwise scaling. The benchmark record labels those implementations
separately; these aren't a controlled cold/warm comparison. Both observations
use the train's separate 12 GB budget and existing curated thermodynamic data.

## Reproducible benchmarks

The benchmark runner starts a fresh subprocess and uses an empty persistent
cache in cold mode. It samples peak RSS every 0.2 seconds and stops a worker
above its declared budget. This sampled limit can overshoot briefly. The parent
also checks the completed worker's high-water mark, so an over-budget final
allocation can't pass between samples. The parent enforces elapsed time and
records termination, including the last completed
phase. On Linux, it also records available cgroup memory-event counters, which
can identify an OOM kill before the worker's watchdog records its own peak.
A successful benchmark requires accepted numerical/physical evidence
and completion within the resource limits.

```bash
# Linux/glibc setting used by the memory-budget workflow.
export MALLOC_ARENA_MAX=2

uv run python scripts/benchmark_process.py \
  --scenario column --stages 32 --column-solver block \
  --output artifacts/performance/column32-block

uv run python scripts/benchmark_process.py \
  --scenario column --stages 32 --column-solver dense \
  --output artifacts/performance/column32-dense

uv run python scripts/benchmark_process.py \
  --scenario depropanizer --study optimization --release-caches \
  --max-rss-gb 7 --timeout-seconds 3000 \
  --output artifacts/performance/depropanizer-optimization

uv run python scripts/benchmark_process.py \
  --scenario ethanol-train --stages 12 --study sensitivities --release-caches \
  --max-rss-gb 12 --timeout-seconds 3000 \
  --output artifacts/performance/ethanol-train

uv run python scripts/benchmark_process.py \
  --scenario heater-bank --variables 24 \
  --output artifacts/performance/many-variable
```

Output directories must be empty. They retain the report, incremental progress,
worker log, cache, and any audited case artifacts. To measure persistent-cache
reuse, select `--cache warm --cache-dir PATH_TO_PRIOR_CACHE` and a new output
directory. Reports include the revision, dirty-tree status, and a source digest
covering code, bundled model data, and dependency declarations, captured before
starting the worker. Benchmark a stable checkout and avoid
competing workloads when comparing timings.
`--release-caches` applies the study API's explicit cache-release option
between baseline, derivative, and final-audit phases. Optimization retains its
compiled kernels across design points and collects unreachable Python objects
after each completed point. The host value/Jacobian pair remains cached for
repeated optimizer callbacks at that same point. Clearing compiler caches
between individual points increased both compilation time and peak memory in
the Linux depropanizer experiment. The option retains the persistent cache and
recorded runs. The constrained-memory CI studies enable it.

The standalone column benchmark separates tracing and lowering, compilation,
first execution, and repeated execution for both the primal and derivative
kernels. Its derivative is compared with centered finite differences. Modular
plant profiles report their actual phase boundaries; the first plant
linearization can compile individual units, so its time mustn't be labeled
pure execution time.

The process-performance CI workflow runs 32- and 64-stage columns, the saved
depropanizer optimization, the NRTL train, and a 24-variable study in separate
jobs. The depropanizer, columns, and heater bank have a 7 GB process limit;
the NRTL sensitivity train has a separate 12 GB budget. All these Linux jobs
set `MALLOC_ARENA_MAX=2` before process startup. The NRTL job needs enough
additional RAM for the runner and operating system. GitHub documents 16 GB
for public-repository Linux runners; private forks need to select a runner
with sufficient memory. See the
[GitHub-hosted runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
The workflow preserves failed observations. Timing
observations are hardware dependent; the CI gates completion and correctness,
not a universal speedup ratio. The NRTL train uses existing curated parameters
and doesn't expand the measured qualification matrix.
