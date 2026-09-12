# Reactive process workflows

A `ReactionSet` supplies one component basis, stoichiometry, thermochemistry,
kinetics, and reacting phase to reactors, reactive flashes, and MESH columns.
Saved cases carry the same definition through simulation, sensitivity studies,
bounded optimization, persistence, and copilot submission.

## Run the complete examples

```bash
uv run fugacio example reactive-recycle reactive-recycle.json
uv run fugacio run reactive-recycle.json --output recycle-run.json --report recycle-report.md
uv run fugacio example reactive-separation reactive-separation.json
uv run fugacio run reactive-separation.json --output separation-run.json --report separation-report.md
```

The committed cases and study requests are in `examples/process-cases/`.
`reactive-recycle` uses Peng-Robinson properties for vapor-phase butane
isomerization in a CSTR, followed by a specified component separator and a
recycle. The separator's component splits and required heat are explicit
assumptions. `reactive-separation` uses six MESH stages, four liquid reaction
volumes, a total condenser, and a kettle reboiler. These cases demonstrate
numerical workflows. Their kinetic coefficients and equipment sizes are
illustrative; neither case establishes catalyst performance or empirical
kinetic qualification.

```python
from fugacio.sim.cases import CaseRunner, CaseWorkspace, optimize, sensitivities
from fugacio.sim.cases.examples import example_case

runner = CaseRunner(example_case("reactive-recycle"))
workspace = CaseWorkspace(".fugacio-cases")
baseline = runner.run(check=True)
workspace.save_run(baseline)

sensitivity = sensitivities(
    runner, ["volume", "kinetic_rate"], ["extent"], workspace=workspace
)
design = optimize(
    runner, ["volume"], "product_isobutane", sense="max", workspace=workspace
)
assert sensitivity.accepted and design.accepted
```

Sensitivity acceptance compares automatic derivatives with centered differences
of independently accepted runs. Optimization acceptance additionally requires
optimizer termination, bounds and constraints, and an accepted final process.
The volume-only demonstration maximizes product flow within declared bounds;
it doesn't represent a capital-cost optimum. The same cases accept
`SolverOptions(backend="eo")` for equation-oriented flowsheet solving.

## Reaction references and rate units

`ReactionSet.from_reactions` checks the ordered components, element conservation,
independent reaction vectors, kinetic dimensions implied by the input convention,
and the shapes of the bundled reference-rate coefficients. Each reaction must
have reactants and products. A homogeneous set declares `phase="vapor"` or
`phase="liquid"`.

The standard chemical potential uses ideal-gas formation data at 298.15 K and
1 bar, corrected with the component heat-capacity correlations. Activities for
chemical equilibrium and activity-based kinetics are dimensionless fugacities:

```text
ln(a_i) = ln(z_i) + ln(phi_i) + ln(P / P_REF)
ln(K_r) = -delta_G_r(T) / (R T)
```

The package supplies phase-specific fugacities, density, and sensible/residual
enthalpy. Liquid activity inputs include the liquid reference fugacity; they
aren't simply `x * gamma`. Energy balances add the component formation enthalpies
exactly once. Reference Helmholtz packages with a different energy datum are
rejected for these reactions.

`ReferenceRate` uses a reference-temperature Arrhenius expression. Its forward
and reverse coefficients have units of `mol/(m3 s)` at unit dimensionless inputs.
It accepts these two input conventions:

| Convention | Input | Required reference |
| --- | --- | --- |
| `activity` | Component fugacity divided by 1 bar | Thermochemical standard state |
| `normalized_concentration` | `z / (molar_volume * reference_concentration)` | Positive concentration in `mol/m3` |

Independent reverse coefficients specify empirical kinetics. They don't imply
thermodynamic consistency. With `detailed_balance=True`, the reverse coefficient
is derived from `K(T)` and the forward coefficient. This requires fugacity
activities and elementary stoichiometric orders. Saved cases reject separately
specified reverse coefficients in this mode.

Python also accepts existing `PowerLaw`, `MassActionReversible`, and `LHHW`
objects with `rate_basis="concentration"`. Their dimensional concentration
inputs are in `mol/m3`, and coefficient dimensions depend on reaction order.
Saved cases use `ReferenceRate` and explicit dimensionless input conventions so
coefficient units remain unambiguous.

## Common-package reactor API

```python
import jax
import jax.numpy as jnp
from fugacio.sim import ReactionSet, ReferenceRate, Stream, reaction_reactor
from fugacio.thermo import Reaction

jax.config.update("jax_enable_x64", True)
components = ("n-butane", "isobutane")
reaction = Reaction(components, jnp.array([-1.0, 1.0]))
law = ReferenceRate(
    k_forward=jnp.asarray(1.0), ea_forward=jnp.asarray(20000.0),
    forward_orders=jnp.array([1.0, 0.0]),
    k_reverse=jnp.asarray(0.0), ea_reverse=jnp.asarray(0.0),
    reverse_orders=jnp.array([0.0, 1.0]),
    reference_temperature=jnp.asarray(400.0), detailed_balance=True,
)
system = ReactionSet.from_reactions(
    reaction, [law], phase="vapor", rate_basis="activity", names=["isomerize"]
)
feed = Stream.from_fractions(components, [0.7, 0.3], 10.0, 400.0, 1e6)
result = reaction_reactor(feed, system, kind="cstr", volume=1.0, t_out=400.0)
result.check()
```

`kind` selects `equilibrium`, `cstr`, or `pfr`. The default package is
Peng-Robinson; `model` accepts the common package interface. Supply `t_out` for
isothermal operation, or `duty` in W for a specified heat input. `duty=0` is
adiabatic. These specifications are mutually exclusive. When both are omitted,
the reactor retains the inlet temperature. Positive `dp` represents pressure
loss. PFR pressure loss and specified heat are distributed uniformly in volume.

`ReactionResult` retains the outlet, extent vector, component generation,
external duty, solver report, normalized material/element/energy errors,
homogeneous-phase error, and component/temperature/pressure/rate profiles.
CSTR and equilibrium profiles contain inlet and outlet points. Rate profiles
have an empty reaction axis when no kinetic laws are supplied. The isothermal
PFR starts its profile at the specified reactor temperature; the reported duty
also includes conditioning the incoming feed to that temperature.

PFRs compare RK4 integrations on `steps` and `2 * steps` meshes. The returned
profile uses the refined mesh. `integration_error` is the maximum normalized
step-doubling error divided by 15 and must be at most one. Intermediate RK
inventories must remain nonnegative, and the final energy balance is audited
separately. Increase `steps` when refinement fails. This implementation doesn't
provide an adaptive stiff integrator or certify phase behavior between mesh
points.

These reactors require the declared homogeneous phase at the audited states.
Use a reactive flash or column when reaction and phase separation occur
together. `check=False` retains failed primals and reports. Failed outputs and
profiles have nonfinite derivatives; callers must inspect acceptance before
using a design gradient. A finite PT flash alone isn't a global stability proof.
Saved runs add the existing finite-start stability audit to process streams.

## Reactive MESH and flash

`rigorous_column(..., reactions=system, reaction_volumes=volumes)` adds
volumetric sources to its existing material and energy equations. The
`reactive_column` convenience function delegates to the same solver. Each stage
uses its declared liquid or vapor composition and temperature. Material sources
are `volume * rate @ nu`; reaction heat is `-generation @ formation_enthalpy`.
This supports net mole changes without a constant-molar-overflow assumption.

A scalar volume applies to interior stages; the end stages have zero reacting
volume. An explicit array supplies one volume per stage, including the ends.
Volumes must be nonnegative. A total condenser cannot have reacting vapor
volume. Reboiler and liquid-condenser reactions require explicit end volumes.
The case format always requires the full array, with physical volume units.

Reaction sources remain local to each stage, preserving the existing bordered
block solver and implicit derivatives. Dense solving remains available for
comparison. Nonlinear homotopy can reduce reaction volumes before solving the
full system. Returned profiles include generation, rates, reaction heat, and
reacting volumes as well as the usual MESH quantities. The column report checks
the solved equations; saved cases independently audit products and overall
balances. Internal stages don't receive an independent global-stability proof.

`reactive_flash` solves chemical equilibrium together with the package's PT
flash. Its quotient uses a phase that's present, and its two products preserve
phase inventories. The result contains duty, generation, a combined acceptance
report, and the final PT `phase_report`. Property and thermochemical parameters
pass explicitly through the implicit root, including under JIT differentiation.

The older `reactive_distillation` function remains a constant-molar-overflow
approximation with molar holdup and liquid `x * gamma` kinetic inputs. Its
holdup isn't a volume, and its material closure doesn't establish energy
closure. Use the MESH API for the saved workflows described here. Legacy
`equilibrium_reactor`, `cstr`, and `pfr` calls without a package or `ReactionSet`
retain their ideal-gas behavior. Supplying either selects the checked common
implementation. The batch API remains separate.

## Portable definitions and measurements

A case declares reusable sets under `reaction_sets`, then references a set by
name in each unit's `settings.reaction_set`. Each set contains `phase`,
`rate_basis`, optional `reference_concentration`, and a list of named reactions
with `nu` vectors. An optional `rate` object holds the `ReferenceRate` fields.
Kinetic units require rates for every reaction; equilibrium units can omit them.
Every set is validated, including unused definitions.

Registered kinds are `equilibrium_reactor`, `cstr`, `pfr`, and `reactive_flash`.
Columns retain kind `column` and add `reaction_set` and `reaction_volumes`.
PFR cases may specify a fixed `steps` count. Parameters can reference rate
coefficients, activation energies, reference temperatures/concentrations,
orders, operating temperatures, and volumes. Their bindings remain dynamic
when compiled units share a template.

```json
{"unit": "reactor", "property": "extent", "reaction": "isomerize"}
```

Extent measurements require a reaction name. Component generation measurements
use `"property": "generation"` with a component name. Both use mol/s. Column and
reactor rate profiles use `"profile": "reaction_rates"`, a one-based `stage`, and
a reaction name, with units `mol/(m3 s)`. For PFRs, `stage` indexes a retained axial
sample, from 1 through `2 * steps + 1`. Component-flow, temperature, and pressure
profiles use the same indexing. These expressions work in metrics, studies,
and copilot submissions.

Run artifacts include `reaction_evidence` separately from property-package
qualification. It records the chemical reference, declared phase/input basis,
and the absence of measured kinetic qualification. Accepted numerics don't
turn illustrative coefficients into measured data.
