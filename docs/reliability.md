# Reliable solves and phase-preserving streams

A finite outlet temperature doesn't establish that a calculation converged.
Fugacio provides numerical reports for roots, recycles, rigorous columns,
equation-oriented flowsheets, energy-balanced units, and optimization. Use these
reports when accepting a design or presenting a result.

## The failure contract

No public call returns a plausible number from a failed solve. Every
equilibrium calculation comes in two forms:

- A checked `*_with_info` form returns the best iterate and a `SolveReport`.
  Its derivatives are nonfinite unless the report converged, so an optimizer
  can't consume a failed sensitivity.
- A value-only form returns NaN whenever that report fails.

This covers the cubic, gamma-phi, liquid-liquid, three-phase, and PC-SAFT
flashes and saturation calls, reaction equilibrium, and every
[property-package](property-packages.md) method (`flash_pt`, `bubble_pressure`,
`dew_temperature`, `flash_ph`, and so on).

```python
import jax
import jax.numpy as jnp
from fugacio.sim import package_for

pkg = package_for(("methane", "n-butane"))
x = jnp.array([0.6, 0.4])

solved = pkg.bubble_pressure_with_info(330.0, x)
solved.report.to_dict()["status"]     # "trivial": the iteration found y = x
pkg.bubble_pressure(330.0, x).value   # nan

# Above the cricondenbar there's no bubble point, and no usable derivative.
jax.grad(lambda p: pkg.bubble_temperature(p, x).value)(200e5)   # nan
```

`SolveStatus` names the outcome:

| Status | Meaning |
| --- | --- |
| `CONVERGED` (0) | The independently checked residual passed its tolerance. |
| `MAX_ITERATIONS` (1) | The iteration cap was reached first. |
| `NONFINITE` (2) | A NaN or infinite value appeared. |
| `SINGULAR` (3) | A linear system was singular. |
| `STALLED` (4) | The residual stopped decreasing. |
| `INVALID_INPUT` (5) | The specification is invalid or violates an operating limit. |
| `INFEASIBLE` (6) | No physical solution satisfies the specification, such as one that needs negative amounts. |
| `OUT_OF_DOMAIN` (7) | A well-posed request lies outside the model's domain, such as a vapor pressure above the critical temperature. |
| `TRIVIAL` (8) | An equilibrium iteration collapsed onto identical phases where a distinct phase was required. |

Unit operations share one contract (see `fugacio.sim.units`). Every unit is a
compiled kernel that returns its outlets with a report, and a failed kernel's
outlets are NaN:

- An eager call raises `ConvergenceError` for a failed solve, or `ValueError`
  for a violated operating limit from `fugacio.sim.unit_limits`: a valve,
  mixer, drum, or turbine that would raise pressure, a pump or compressor that
  would lower it, split fractions outside `[0, 1]` or not summing to one, or a
  pump with a vapor inlet.
- A traced call (inside `jax.jit`, a flowsheet recycle, or an optimizer)
  can't raise. It returns NaN outlets with nonfinite derivatives; the unit's
  `report`, and each NaN outlet's `Stream.report`, record the failure.

```python
from fugacio.sim import Stream, valve

feed = Stream.from_fractions(("methane", "propane"), jnp.array([0.9, 0.1]), 10.0, 300.0, 20e5)
valve(feed, 30e5)                            # ValueError: a valve can't raise pressure

jax.jit(lambda p: valve(feed, p).t)(30e5)    # nan: the traced limit fails the report
```

The saved-case audit (`fugacio.sim.cases`) checks solved streams against the
same `unit_limits` predicates, so a limit can't be enforced in one place and
forgotten in another.

`rigorous_column`, `stoichiometric_reactor`, the flow reactors
(`equilibrium_reactor`, `cstr`, and `pfr`), and `reactive_flash` follow the same
contract; a failed reactor also keeps its report, balances, and profiles for
diagnosis. Where a unit offers `check=False`, it returns its best iterate
instead, with a failed `report` and nonfinite derivatives, so check
`result.report` (or `result.converged`) before using it.

## Read a solve report

```python
import jax.numpy as jnp
from fugacio.thermo import newton_system_with_info, require_converged

result = newton_system_with_info(
    lambda x, target: x**2 - target,
    jnp.array([1.0]),
    jnp.array([4.0]),
    lower=jnp.array([0.0]),
)
require_converged(result.report, "design equation", ("target relation",))
result.value                       # [2.0]
result.report.to_dict()             # Strict JSON on the host
```

Reports contain a status, an iteration count, the maximum absolute scaled
residual, the last scaled step, and the index of the largest residual. A small
step alone doesn't establish convergence. `ConvergenceError` retains the report
and calculation context. Units that independently verify an energy balance
report zero numerical iterations for that verification; this isn't a count of
their internal flash iterations. `nan_unless_converged(value, report)` applies
the value-only contract to your own calculations, and
`fugacio.thermo.diagnostics.with_status` overrides a report's status where a
condition fails.

`newton_system_with_info` accepts characteristic variable and equation scales and
optional bounds. Its Newton search reduces the actual residual and keeps trial
variables within the bounds. Bounds aid the search; they don't replace any
original equation. The scalar `bracketed_root_with_info` checks both a valid
bracket and the resulting residual: a bracket without a sign change is
`INVALID_INPUT`, and a sign change across a pole, whose residual stays large as
the bracket shrinks, fails. `scanned_root_with_info` first scans a wide bracket
for the first finite sign change, so a residual that's undefined over part of
the bracket still works. `bracketed_root` returns the same checked root without
its report; its derivative is nonfinite when the check fails, so use the
reporting variant to accept a value.

Access results by attribute: most carry a `report` field alongside their
values, and a unit result also exposes its outlets as `outlets`.

## Test stability first

A converged two-phase flash can still be wrong: it considers one liquid and one
vapor, so a feed that splits into two liquids can return a converged
vapor-liquid answer. Fugacio checks stability with one shared tangent-plane
search, `fugacio.thermo.stability.tpd_search`, which starts from the feed, the
Wilson-like estimates, and an enrichment toward every component, on both the
liquid and the vapor branch:

- `pkg.stability(t, p, z)` on every property package returns a
  `StabilityResult(stable, tpd, trial, branch, converged)`.
- `flash_lle` and `decanter` test stability before they split a liquid.
- An eager `flash_drum` warns with `PhysicalAcceptanceWarning` when an outlet
  is unstable, and `fugacio.sim.acceptance.flash_drum_checked` audits both
  outlets; its `check()` raises `PhysicalAcceptanceError`.
- `flash_pt_checked`, `audit_stream`, and flowsheet audits reject a state with a
  negative tangent-plane distance, and `PhysicalReport.failures()` explains
  each rejection in plain language.

A finite-start search can't prove a global minimum. A negative distance
establishes instability; a verdict of stability also requires every trial to
reach a stationary point (`converged`).

## Keep a stream's phase state

Temperature and pressure don't determine the quality of a pure saturated fluid.
`Stream` therefore carries optional vapor component flows, `vapor_n`, in addition
to total component flows. Heater, valve, machine, flash, and rigorous-column
outputs retain their resolved phase inventories. Splitters scale both inventories.
Stream enthalpy, entropy, and volume use those inventories instead of repeating a
PT flash that would discard saturation quality.

```python
from fugacio.sim import Stream, package_for, splitter

steam = package_for(["water"], "iapws")
wet = Stream.from_ph(
    ("water",), jnp.ones(1), 2.0, 1e5, 25000.0, model=steam,
)
a, b = splitter(wet, jnp.array([0.3, 0.7]))
wet.check()
```

`Stream.from_ps` accepts pressure and molar entropy. For known saturated endpoints,
`Stream.from_fractions(..., phase="liquid")` or `phase="vapor"` selects the branch.
The default constructor retains PT behavior. Empty streams have zero extensive
properties and use a finite trial composition during property evaluation.

Use the same property package and energy reference throughout an energy balance.
The stream doesn't store a property package. Named package factories bind the
component order and reject a mismatched stream. `stream.reordered(components)`
reorders material and phase inventories together. Packages constructed directly
from arrays can't infer component names; the caller must keep their order aligned.

Pure-fluid PH and PS flashes on mixture packages solve quality and temperature
together inside the saturation region. The reference-fluid package uses its
published PH and PS state functions. Equation-oriented pure-fluid streams use
molar enthalpy as their thermal coordinate so that quality remains an unknown
when saturation temperature is fixed.

For the same reason, `heater(feed, t_out=...)` on a pure fluid raises
`ValueError` when `t_out` is exactly its saturation temperature at the outlet
pressure: the quality is undetermined. Specify `vapor_fraction` (0 for
saturated liquid, 1 for saturated vapor, or any quality in between) or `duty`
instead. `adiabatic_flash(feed, p, duty=...)` separates a letdown or a heated
pure fluid on its saturation line, where a temperature can't fix the phase
amounts.

## Accept a flowsheet calculation

```python
result = fs.solve_with_info(parameters)
result.check()
streams = result.streams
for name, report in result.reports.items():
    print(name, report.to_dict())
```

Sequential flowsheet reports identify recycle blocks, units, and invalid named
streams, and `result.units` keeps the heat, work, and report each unit returned.
`fs.solve()` checks those reports by default. `solve_with_info` retains best
iterates for diagnosis; a unit that fails outside a recycle iteration can still
raise a `ConvergenceError`, naming the unit, before a complete flowsheet result
exists. Recycles converge with Broyden's method unless you pass another
`method`.

With a property package, `Flowsheet(model=pkg)` audits every solved stream for
equilibrium, stability, and parameter applicability after convergence.
`result.accepted` combines convergence with those audits, `result.audits` holds
each stream's `PhysicalReport`, and `fs.solve()` raises
`PhysicalAcceptanceError` for a stream that fails. The audits run once, outside
the recycle iterations, so they don't slow the solve itself.

`EOFlowsheet.solve()` returns an `EOSolution` with a report and checks concrete
failures by default. `check=False` exposes its best iterate. Errors identify the
responsible block equation or design specification. EO flash blocks impose zero
flow for an absent phase instead of imposing equifugacity on a phase that doesn't
exist. Rigorous-column errors identify stage material, equilibrium, or energy
rows and column specifications.

Heat exchangers retain their documented second-law cap for an infeasible request.
Their `report` marks a missed specification as infeasible, and `result.check()`
raises. This lets an interactive caller inspect the capped result explicitly.
The copilot checks this report and returns structured errors instead of using the
capped result as a successful specification. Tool inputs and outputs must be
strict JSON; nonfinite numbers cause an error.

Exchanger temperature curves preserve the supplied inlet states. Pressure drops
are distributed linearly along each stream's flow direction, and each outlet's
PH state closes the shared duty. Single-phase bulk properties differentiate the
existing phase directly, so an absent phase's equilibrium derivative can't
invalidate an otherwise regular property sensitivity.

Single-phase PH and PS results retain absent-phase trial compositions and K
values for inspection, but those trial values don't carry derivatives. Present
phase amounts follow the material balance, and energy sensitivities follow the
existing phase's property equations.

## Improve convergence and inspect specifications

EO solves reuse their most recent concrete converged state by default.
`warm_start=False` disables reuse, and `guess=...` supplies named stream guesses.
The property-package parameters remain dynamic inputs to the cached solve.
Sequential solves accept `guess=previous.streams` for their recycle seeds.
A rigorous column's `warm_start()` method returns the full stage state accepted
by its `guess` argument. If a direct NRTL column solve fails, the solver can
start at ideal activity and restore the activity parameters in increments.
`homotopy=False` disables this fallback. Each increment has its own Newton
iteration limit; the final report verifies the requested model's equations.

```python
path = fs.solve_path(start_parameters, target_parameters)
path.check()
endpoint = path.value
```

Both flowsheet engines support this host-side continuation method. Failed trials
reduce the step and retain the previous accepted state; successful steps advance
toward the requested endpoint. The result records every trial and its report.
If the endpoint isn't reached, the result isn't marked converged, even when the
last accepted intermediate point solved successfully. `continuation_solve`
accepts a custom solver callback for other problems. `solve_path` accepts
recycle methods and tolerances through `solve_options`.
Differentiate a separate endpoint solve, using the accepted state as its guess;
the adaptive path selection itself isn't differentiable.

`EOFlowsheet.degrees_of_freedom()` checks equation counts. A square system can
still be singular. `EOFlowsheet.diagnose(parameters, guess=...)` evaluates the
scaled Jacobian and reports numerical rank, conditioning, zero equation rows, and
unconstrained variables. This is a local numerical diagnostic at the supplied
state, not a global proof of structural solvability.

## Differentiate only valid solutions

The shared vector Newton, fixed-point, and recycle solves use implicit JVP rules.
JAX transposes the linearized residual solve for reverse derivatives; the adjoint
doesn't repeat a possibly divergent fixed-point iteration. The rules support JVP,
VJP, JIT, batching, and higher derivatives at smooth, nonsingular solutions.
Nested thermodynamic models must also support the requested transformation.

`argmin` differentiates a fixed-size KKT system containing equalities, active
inequalities, and active bounds. Inactive multiplier equations have identity
rows, preventing inactive constraints from making the matrix singular. Optimizer
reports check constrained stationarity and feasibility. Failed optimization can't
supply a valid solution sensitivity.

Host-side checked calculations raise on failure. Inside compiled calculations,
checked flowsheet values and failed implicit sensitivities become nonfinite, and
reporting APIs retain the failure status. Carry the report through JIT and inspect
it before accepting results. For an acceptance check of your own, made after a
solve, `fugacio.thermo.implicit.gate_derivative(value, valid)` keeps the value
but makes its derivative nonfinite unless `valid`; `gate_tree` applies it to
every floating leaf of a pytree. Phase transitions, active-set changes, redundant
constraints, and singular roots can make a derivative undefined even when the
primal residual is small. A successful residual check doesn't prove smoothness
or global optimality.

## Run the plant acceptance cases

The plant tests exercise public stream properties across unit boundaries. They
check the converged process, including its external material and energy balances.

| Case | Acceptance checks |
| --- | --- |
| HDA with hydrogen recycle | Carbon and hydrogen closure, reaction formation enthalpy, furnace and cooler duties, pressure letdown, and benzene recovery. |
| Depropanizer with feed preheat | Purity and recovery specifications, column and exchanger energy closure, recycle closure, and heat-recovery sensitivity against finite differences. |
| Ethanol/water with NRTL | A rigorous column inside a converged economizer loop, product composition, heat recovery, and the whole-train energy balance. |
| Phase-changing water utility | Preserved quality through heating and letdown, condensation and pumping energy closure on PR and IAPWS, and an operating-point optimization checked against finite differences. |

```bash
uv run pytest packages/fugacio-sim/tests/test_plants.py
uv run pytest packages/fugacio-sim/tests/test_utility_reliability.py
```

These cases carry the `plant` marker and run in CI. Their nested JAX solvers can
take several minutes to compile on the first run. The HDA case uses a specified
conversion and an ideal component separator; its energy accounting includes the
heat required by that separator.
