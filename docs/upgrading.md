# Upgrading to 0.9

Fugacio 0.9 rebuilds the engine around two rules:

- **No public call returns a silent wrong answer.** A failed solve returns NaN,
  raises, or carries a failed report. It never returns a plausible number.
- **No public call recompiles for a new operating point.** A unit compiles
  once per property-package structure and specification kind, then reuses that
  executable.

To get there, this release removes older interfaces instead of deprecating
them. This page maps each removed or renamed interface to its replacement and
lists the behavior changes that can affect working code. Fugacio is pre-1.0, so
minor releases can break compatibility.

## Behavior changes

### Failed solves return NaN or raise

Value-only equilibrium calls return NaN when their solve fails, including when
the request lies outside the model's domain. This covers `flash_pt`,
`psat_eos`, `bubble_pressure_eos`, `dew_pressure_eos`, the gamma-phi flash and
saturation calls, `flash_lle`, `flash_vlle`, the PC-SAFT equilibrium calls,
`saturation_pressures`, reaction `equilibrium`, and the equivalent
property-package methods (`flash_pt`, `bubble_pressure`, `dew_temperature`,
`flash_ph`, and so on).

Each has a `*_with_info` twin that returns the best iterate and a
`SolveReport`. Derivatives of that iterate are nonfinite unless it converged, so
an optimizer can't consume a failed sensitivity. Two statuses are new:
`SolveStatus.OUT_OF_DOMAIN` (7) marks a request outside the model's domain, such
as a vapor pressure above the critical temperature, and `SolveStatus.TRIVIAL`
(8) marks an iteration that collapsed onto identical phases.

```python
from fugacio.thermo import PR, SolveStatus, get, psat_eos, psat_eos_with_info

methane = get("methane")
psat_eos(PR, 300.0, methane.tc, methane.pc, methane.omega)       # nan (supercritical)
solved = psat_eos_with_info(PR, 300.0, methane.tc, methane.pc, methane.omega)
SolveStatus(int(solved.report.status))                           # SolveStatus.OUT_OF_DOMAIN
```

Unit operations follow one contract (see the [reliability guide](reliability.md)):

- An eager call raises `ConvergenceError` for a failed solve and `ValueError`
  for a violated operating limit, such as a valve that raises pressure,
  splitter fractions that don't sum to one, or a pump with a vapor inlet.
- A traced call (inside `jax.jit`, a flowsheet recycle, or an optimizer)
  returns NaN outlets with nonfinite derivatives.

Reactors and columns raise the same way in eager code. A reactor solve that
can't converge, such as an adiabatic CSTR on irreversible kinetics for an
equilibrium-limited reaction, raises `ConvergenceError`; earlier releases could
return negative flows.
`fugacio.thermo.reaction_equilibrium.equilibrium` returns an
`EquilibriumResult` with a `report`, NaN fields on failure, and negative
equilibrium moles reported as `INFEASIBLE`.

### Stability comes first

Every property package has `stability(t, p, z)`, a tangent-plane search on
both its liquid and vapor branches. The removed Michelsen routines searched
fewer trial phases and could miss a second liquid. Consequences:

- An eager `flash_drum` warns with
  `fugacio.thermo.acceptance.PhysicalAcceptanceWarning` when an outlet is
  unstable, for example a vapor-liquid answer for a feed that splits into two
  liquids. `fugacio.sim.acceptance.flash_drum_checked` audits the outlets
  instead, and its `check()` raises `PhysicalAcceptanceError`.
- `flash_lle` and `decanter` test stability before they split a liquid.
- `Flowsheet(model=pkg)` audits every solved stream. `solve()` raises
  `PhysicalAcceptanceError` for a stream that fails its audit, and
  `solve_with_info()` returns the audits for inspection.

### Recycles default to Broyden

`tear_solve`, `tear_solve_with_info`, `Flowsheet.solve`, and saved cases
(`SolverOptions.recycle_method`) now default to `"broyden"`, matching the CLI's
`--recycle-method`. Pass `method="wegstein"` (or
`SolverOptions(recycle_method="wegstein")`) to select the previous default. A
recycle block with no upstream stream to seed its tear raises `ValueError`;
supply a guess with `Flowsheet.tear`.

### Kernels compile once

Units are compiled kernels with the property package and every operating value
as dynamic arguments, so a new temperature, pressure, or package parameter
reuses the executable. Property-package solve methods (`flash_pt`,
`stability`, the bubble and dew calls, `mixture_enthalpy`, `flash_ph`, and so
on) run through kernels cached per method, with the package as a dynamic
argument. Flowsheets cache their recycle maps per partition, and saved-case
unit templates are shared across runners in a process. To reuse compiled
programs across processes, call
`fugacio.sim.enable_compilation_cache(directory)` or pass `--jax-cache DIR` to
the CLI. See the [performance guide](performance.md#compile-once-kernels).

## `fugacio.thermo`

### Removed modules

| Removed | Replacement |
| --- | --- |
| `fugacio.thermo.phase`: `EquilibriumModel`, `EOSModel`, `GammaPhiModel`, `eos_model`, `gamma_phi_model` | `fugacio.thermo.package`: the `PropertyPackage` protocol, `cubic_package` (`CubicPackage`), and `gamma_phi_package` (`GammaPhiPackage`) |
| `fugacio.thermo.saft.model`: `SAFTModel`, `saft_model` | `saft_package` (`SAFTPackage`), or `fugacio.sim.package_for(components, "pcsaft")` |
| `fugacio.thermo.properties`: `molar_enthalpy`, `molar_entropy`, `molar_gibbs`, `molar_cp` | Package methods `enthalpy`, `entropy`, `gibbs`, and `heat_capacity`, each with an explicit `phase=` |
| `fugacio.thermo.properties.stable_phase` | `pkg.flash_pt(t, p, z).beta` (0 for liquid, 1 for vapor), or compare `pkg.gibbs` on both branches |
| `fugacio.thermo.properties.speed_of_sound_ideal` | No replacement |
| `P_REF` and `T_REF` from `fugacio.thermo.properties` | `fugacio.thermo.constants` (also exported from `fugacio.thermo`) |

The package constructors take the ideal-gas heat-capacity coefficients that the
old equilibrium models didn't need:

```python
import jax.numpy as jnp
from fugacio.thermo import component_arrays, cubic_package, get, ideal_gas_coeffs

names = ["methane", "propane", "n-pentane"]
arr = component_arrays(names)
cp = ideal_gas_coeffs([get(c) for c in names])
z = jnp.array([0.5, 0.3, 0.2])

# Before: eos_model(arr["tc"], arr["pc"], arr["omega"]).flash_pt(320.0, 20e5, z)
pkg = cubic_package(arr["tc"], arr["pc"], arr["omega"], cp)
res = pkg.flash_pt(320.0, 20e5, z)

# Before: molar_enthalpy(320.0, 20e5, z, tc, pc, omega, cp, phase="vapor")
h = pkg.enthalpy(320.0, 20e5, z, phase="vapor")
```

### Removed functions and renamed types

| Removed | Replacement |
| --- | --- |
| `stability_analysis(eos, t, p, z, tc, pc, omega)` (Michelsen) | `pkg.stability(t, p, z)` on any package |
| `stability_saft` | `saft_package(...).stability(t, p, z)` |
| `stability_analysis_general(ln_coeff_fn, z, trials)` | `fugacio.thermo.stability.tpd_search(branches, d, support, starts)` |
| `TangentPlaneResult` (`stable`, `tpd`, `split`) and `equilibrium.StabilityResult` (`stable`, `tpd`) | `fugacio.thermo.stability.StabilityResult` (`stable`, `tpd`, `trial`, `branch`, `converged`) |
| `flash_ph`, `flash_ps`, `mixture_enthalpy`, `mixture_entropy` (cubic-only functions in `fugacio.thermo.energy`) | `pkg.flash_ph(p, h, z)`, `pkg.flash_ps(p, s, z)`, `pkg.mixture_enthalpy(t, p, z)`, `pkg.mixture_entropy(t, p, z)` |
| `flash_ph_with_info(pkg, ...)`, `flash_ps_with_info(pkg, ...)`, `fugacio.thermo.package.flash_pt_with_info(pkg, ...)` | `pkg.flash_ph_with_info(p, h, z)`, `pkg.flash_ps_with_info(p, s, z)`, `pkg.flash_pt_with_info(t, p, z)` |
| `fugacio.thermo.acceptance.accepted_value(value, accepted)` | `fugacio.thermo.implicit.gate_derivative(value, valid)`, or `gate_tree(tree, valid)` for a pytree |
| PC-SAFT association scheme `"1A"` | None; an unsupported scheme raises `ValueError` |

Bubble and dew calls return `SaturationResult(value, composition)`, a named
tuple, so `p, y = pkg.bubble_pressure(t, x)` still unpacks. New in
`fugacio.thermo.consistency`, `gibbs_helmholtz_residual` and
`fugacity_enthalpy_residual` report how consistently a package's enthalpy
matches its entropy and its fugacity coefficients; the default gamma-phi
package deviates by about 1% in the second check unless it's built with
`poynting=True` and `phi_saturation=True` (see
[property packages](property-packages.md#consistency-checks)).
`bracketed_root` is a thin wrapper over the checked `bracketed_root_with_info`,
which rejects a bracket without a sign change and a residual that stays large
across a pole. The new `scanned_root_with_info` scans a bracket for the first
finite sign change. `PhysicalReport.failures(policy)` lists failed criteria in
plain language, and `PhysicalAcceptanceError` messages include them.

## `fugacio.sim`

### One property argument

Every simulation call takes one property argument, `model=`: a
`PropertyPackage`, or `None` for the Peng-Robinson default over the stream's
components. The `eos=` and `kij=` arguments are gone from the unit operations,
the stream property helpers (`molar_enthalpy`, `vapor_density`,
`vapor_volumetric_flow`, ...), `heat_exchanger`, `rigorous_column`, `absorber`,
`stripper`, `equilibrium_reactor`, `EOFlowsheet`, and the EO blocks.
`column_diameter_for` gained `model=`. `model` is keyword-only on every unit in
`fugacio.sim.units`, and a `model` that isn't a package raises `TypeError`.

```python
import jax.numpy as jnp
from fugacio.sim import EOFlowsheet, Stream, flash_drum, package_for

feed = Stream.from_fractions(
    ("methane", "propane", "n-pentane"), jnp.array([0.5, 0.3, 0.2]), 100.0, 320.0, 20e5
)

# Before: flash_drum(feed, 320.0, 20e5, eos=SRK)
srk = package_for(feed.components, "srk")
vapor, liquid = flash_drum(feed, 320.0, 20e5, model=srk)

# Before: EOFlowsheet(eos=PR)
fs = EOFlowsheet()                 # Peng-Robinson default
fs = EOFlowsheet(model=srk)        # any package
```

| Removed | Replacement |
| --- | --- |
| `eos_model_for`, `nrtl_model_for`, `uniquac_model_for`, `unifac_model_for`, `saft_model_for` | `package_for(components, method)` with `"pr"`, `"srk"`, `"nrtl"`, `"uniquac"`, `"unifac"`, `"dortmund"`, or `"pcsaft"` |
| `unifac_model_for(components, dortmund=True)` | `package_for(components, "dortmund")` |
| `as_package(model, components)` | Build a package directly; `resolve_package` accepts only packages |
| `package_for(..., strict=False)` | `package_for(..., parameter_policy="allow_ideal")` |
| `flash_vle(feed, t, p, model)` with a positional model | `flash_drum(feed, t, p, model=package_for(components, "nrtl"))` |
| `relative_volatility(eos, t, p, z, tc, pc, omega, ref=, kij=)` | `relative_volatility(model, t, p, z, ref=)` |

`package_for` defaults to `parameter_policy="strict"`, which raises `KeyError`
for a missing NRTL or UNIQUAC pair. The removed `nrtl_model_for` and
`uniquac_model_for` silently used zero interactions, so choose
`"allow_ideal"` explicitly if you relied on that. An option that a method
doesn't accept raises `TypeError`.

### Unit results

`flash_drum` still returns `(vapor, liquid)`, and `valve` and `mix` return a
`Stream`. The other units return a result with a `report`:
`flash_drum_with_info` and `adiabatic_flash` return `FlashDrumResult(vapor,
liquid, duty, report)`, `heater` returns `HeaterResult(outlet, duty, report)`,
`pump` returns `PumpResult(outlet, work, report)`, and `compressor` and
`turbine` return `WorkResult(outlet, work, ideal_work, report)`. Use
`.outlet` where you need the stream.

Every result a flowsheet unit may return exposes `outlets` and, where it has
one, `heat`: `FlashDrumResult`, `HeaterResult`, `PumpResult`, `WorkResult`,
`ReactionResult`, `ReactiveFlashResult`, `StoichiometricResult`,
`RigorousColumnResult` (outlets `(distillate, bottoms, *side_draws)`, heat from
the condenser, reboiler, and the new `stage_duties` field), and
`HeatExchangerResult` (outlets only). A flowsheet unit function can return the
result itself, and the flowsheet keeps its heat, work, and report in
`FlowsheetResult.units`.

`heater` also accepts `vapor_fraction` (any quality for a pure fluid, 0 or 1 for
a mixture). A pure fluid specified by `t_out` exactly at its saturation
temperature raises `ValueError`, since its quality is undetermined; specify
`vapor_fraction` or `duty` instead. `adiabatic_flash(feed, p, duty=0.0)` is a
pressure-and-duty separator. `decanter(feed, model)` and `three_phase_flash(feed, t, p, model)`
require a `GammaPhiPackage`. A feed that isn't three-phase raises with a hint
naming `flash_drum` or `decanter`.

### Columns

| Removed | Replacement |
| --- | --- |
| `column.solve_column(feed, n_stages, feed_stage, reflux, distillate_rate)` | `rigorous_column([ColumnFeed(feed, feed_stage + 1)], n_stages + 1, p=feed.p, specs=[reflux_ratio(reflux), distillate_rate(d)])` |
| `ColumnResult` | `RigorousColumnResult` |
| `reactive_distillation(...)` and `ReactiveColumnResult` | `rigorous_column(..., reactions=system, reaction_volumes=volumes)` or `reactive_column(feeds, n_stages, system, volumes, ...)` |

`solve_column` placed its total condenser above stage 1, while
`rigorous_column` counts the condenser as stage 1, so both the stage count and
the feed stage grow by one. `rigorous_column` closes stage energy balances
instead of assuming constant molar overflow, so its profile differs from the
removed column's. `column.py` keeps only the shortcut (FUG) methods. See
[reactive process workflows](reactive-workflows.md) for reaction sets and
reacting volumes.

### Reactors

| Before | After |
| --- | --- |
| `ReactorResult(outlet, duty, extent)` | `ReactionResult` from `equilibrium_reactor`, `cstr`, and `pfr`; `StoichiometricResult` from `stoichiometric_reactor`; `BatchResult(contents, heat, extent)` from `batch_reactor` |
| `adiabatic=True` on flow reactors | `duty=0.0`; any other `duty` is a specified heat input |
| `cstr(feed, reactions, rate_laws, volume)`, `pfr(...)` | `cstr(feed, reactions, volume, rate_laws)`: the volume is now the third positional argument |
| `equilibrium_reactor(..., basis=, eos=, kij=)` | `equilibrium_reactor(..., model=pkg)`; activities come from the package's fugacities |
| `stoichiometric_reactor(..., adiabatic=, t_lo=, t_hi=)` | `stoichiometric_reactor(..., duty=0.0, model=pkg)`; the outlet temperature comes from a PH flash |

The flow reactors wrap the checked `reaction_reactor`, which solves a
homogeneous phase on the property package (Peng-Robinson by default):
equilibrium uses its fugacity coefficients, concentrations its phase molar
volume, and energy its enthalpy plus ideal-gas formation enthalpies.
`stoichiometric_reactor` raises `ValueError` for an extent that consumes more
of a reactant than the feed holds. `batch_reactor(..., adiabatic=True)`
remains, and returns `BatchResult(contents, heat, extent)` with the moles in
`contents`: a closed, rigid vessel conserves internal energy rather than
enthalpy, the batch model now does, and it reports the final ideal-gas
pressure.

### Flowsheets

`Flowsheet` gains a `model` field, and `Flowsheet.solve_with_info(theta, *,
guess=None, audit=True, **tear_kwargs)` returns a
`FlowsheetResult(streams, reports, units, audits)`. The new `units` holds a
`UnitRecord(heat, work, report)` for each unit, `audits` holds each stream's
`PhysicalReport` when the flowsheet has a `model`, and the new `accepted`
combines `converged` with those audits. `check()` also raises
`PhysicalAcceptanceError` for a failed audit.
`fugacio.sim.cases.backends.CaseFlowsheet` is removed; saved cases use
`Flowsheet`, which now caches compiled recycle maps itself.

`fugacio.sim.acceptance.BalanceBoundary` gains `units`: a boundary's heat and
work are its explicit `heat` and `work` plus those the solved result retained
for the named units. `audit_flowsheet` also accepts an equation-oriented
`EOSolution`.

## Process cases and the CLI

| Before | After |
| --- | --- |
| `flash` unit settings: `t` and `p` | `p` plus exactly one of `t` (isothermal) or `duty` (a specified duty; zero is an adiabatic drum) |
| `heater` settings: one of `t_out` or `duty` | One of `t_out`, `duty`, or `vapor_fraction` |
| `fugacio inspect ID --report` for runs only | Renders any artifact: a run gets its full report, and a study or comparison gets a field summary |
| `export JAX_COMPILATION_CACHE_DIR=...` | `--jax-cache DIR` on `run`, `sweep`, `optimize`, `sensitivities`, `profile`, and `replay`, or the `FUGACIO_JAX_CACHE` environment variable |
| `SolverOptions.recycle_method` default `"wegstein"` | `"broyden"`, the CLI's existing default |

Saved JSON files now respect the process umask instead of always being private.
The Markdown run report's stream table adds mass flow (kg/s), vapor fraction,
and one mole-fraction column per component; failed checks name their reasons.
The new `jt-separator` example demonstrates a Joule-Thomson valve followed by an
adiabatic drum:

```bash
uv run fugacio example jt-separator jt-separator.json
uv run fugacio run jt-separator.json --jax-cache .jax_cache --report jt-separator.md
```

## `fugacio.copilot`

| Before | After |
| --- | --- |
| `AnthropicProvider()` default model `claude-3-5-sonnet-latest` | `claude-opus-5` |
| `OpenAIProvider()` default model `gpt-4o-mini` | `gpt-5-mini` |
| `chat(..., temperature=0.0, max_tokens=1024)` | `temperature=None` (not sent unless set) and `max_tokens=16000` |
| `run_llm_agent(..., temperature=0.0, max_tokens=1024)` and `llm_planner(..., temperature=0.0, max_tokens=1024)` | No `temperature` argument; `max_tokens=16000` |
| `run_design_agent(..., max_tokens=4096)` | `max_tokens=16000` |
| `AgentResult.stop_reason`: `"answer"` or `"budget"` | Adds `"refusal"` and `"max_tokens"`; neither reply becomes an answer |

`ChatResponse` gains `stop_reason` and `provider_blocks`, and `Message` gains
`is_error` and `provider_blocks`. The agent loops resend a `"pause_turn"`
reply so the model can resume it. The Anthropic provider sends all tool results
for one assistant turn in a single user message, flags failed calls with
`is_error`, replays thinking blocks verbatim, and requests server-side refusal
fallbacks by default; pass `fallbacks=None` on platforms without them. The
OpenAI provider turns tool arguments that aren't a JSON object into a
`ToolCall.invalid` call, which the loops answer with an error result. PC-SAFT
and property tools raise on a failed or out-of-domain solve, so the model
receives a structured error rather than an unconverged number.
