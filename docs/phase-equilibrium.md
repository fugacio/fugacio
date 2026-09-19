# Non-ideal phase equilibrium

Beyond cubic equations of state, Fugacio carries a full **γ–φ property system**:
liquid-phase activity coefficients combined with a vapor-phase fugacity model,
plus rigorous liquid-liquid and vapor-liquid-liquid equilibria. This is the route
that captures azeotropes, partial miscibility, and strongly non-ideal mixtures,
the regime where ideal-K and single-EOS methods give wrong answers.

## The γ–φ property package

A `GammaPhiPackage` pairs an activity model (NRTL, UNIQUAC, Wilson, Margules, van
Laar, UNIFAC, …) with a pure-component reference fugacity (saturation pressure,
Poynting correction, and saturation φ) and a vapor model. It's a
[property package](property-packages.md), so it exposes the *same* `flash_pt`,
bubble, dew, `stability`, and energy calls as a cubic `CubicPackage`, and the
rest of the stack switches thermodynamic method by swapping one differentiable
object. `fugacio.sim.package_for(components, "nrtl")` (or `"uniquac"`,
`"unifac"`, `"dortmund"`) builds one straight from component names; the
lower-level `fugacio.thermo.gamma_phi_package` builds one from an activity model
and component constants.

The saturation-based reference exists only below each component's critical
temperature. A state in which a present component is supercritical is outside
the package's domain: checked calls report `SolveStatus.OUT_OF_DOMAIN`, and the
value-only calls return NaN.

```python
import jax.numpy as jnp
from fugacio.thermo import bubble_pressure_gamma, component_arrays, nrtl

arr = component_arrays(["ethanol", "water"])
model = nrtl(                       # NRTL with b_ij/T interactions (K), alpha = 0.3
    a=jnp.zeros((2, 2)),
    b=jnp.array([[0.0, 670.0], [310.0, 0.0]]),
    alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]),
)
p, y = bubble_pressure_gamma(model, 350.0, jnp.array([0.3, 0.7]),
                             arr["tc"], arr["pc"], arr["omega"])
# p, y are differentiable w.r.t. T, x, *and* the NRTL parameters.
```

The activity-based phase-equilibrium entry points are `flash_pt_gamma`,
`bubble_pressure_gamma` / `bubble_temperature_gamma`, `dew_pressure_gamma` /
`dew_temperature_gamma`, and the low-level `gamma_phi_k_values`. Each has a
checked `*_with_info` form that returns the best iterate and a `SolveReport`;
the value-only form returns NaN when that report fails. The reference state is
assembled from `liquid_reference_fugacity`, `poynting_factor`,
`saturation_fugacity_coefficient`, and (for dissolved gases) `henry_constant`.

## Liquid-liquid & three-phase equilibria

- `flash_lle`: isoactivity liquid-liquid flash, with `tie_line`, `binodal_curve`,
  and `binary_binodal` for the miscibility envelope. It's stability-first: a
  liquid established as stable returns one phase (`psi = 0`), and an unstable
  one seeds the split from its most negative trial phase, so the iteration
  doesn't settle on the trivial solution. A collapse onto identical phases is
  reported as `TRIVIAL`.
- `flash_vlle`: three-phase vapor-liquid-liquid flash, plus the binary
  `heterogeneous_azeotrope` solver.
- **Tangent-plane stability**: `tpd_search` is the one search behind every
  stability test. It runs damped successive substitution from several starts
  on each trial branch, keeps absent components exactly absent, and returns a
  `StabilityResult(stable, tpd, trial, branch, converged)`. Every property
  package exposes it as `pkg.stability(t, p, z)`, searching liquid and vapor
  trial phases, and `liquid_stability(model, t, z)` tests an activity-model
  liquid for a second liquid. A finite-start search can't prove a global
  minimum: a negative `tpd` establishes instability, but a verdict of
  stability holds only when `converged` is also true.

```python
import jax.numpy as jnp
from fugacio.thermo import flash_lle, nrtl

# A partially miscible binary splits into two liquid phases.
model = nrtl(
    a=jnp.zeros((2, 2)),
    b=jnp.array([[0.0, 1200.0], [1300.0, 0.0]]),
    alpha=jnp.array([[0.0, 0.2], [0.2, 0.0]]),
)
res = flash_lle(model, 298.15, jnp.array([0.5, 0.5]))
res.x_i, res.x_ii   # the two conjugate liquid compositions (a tie line)
```

The same search catches a vapor-liquid flash that should have found two
liquids. Water and n-hexane are nearly immiscible, and a cubic package's
stability test says so:

```python
import jax.numpy as jnp
from fugacio.sim import package_for

pkg = package_for(("water", "n-hexane"))                      # Peng-Robinson
result = pkg.stability(300.0, 101325.0, jnp.array([0.5, 0.5]))
result.stable, result.tpd   # False, and a strongly negative distance
```

A two-phase flash considers one liquid and one vapor, so test stability
wherever a second liquid can form. `fugacio.sim.flash_drum` warns when an
outlet is unstable, and `fugacio.thermo.acceptance.flash_pt_checked` rejects
such a state.

## Group contribution & predicted parameters

When no fitted binary parameters are available, predict them. `unifac_activity`
and the Dortmund `modified_unifac_activity` give activity coefficients straight
from molecular structure, and the regression layer turns a UNIFAC γ-grid into
binary NRTL / UNIQUAC parameters: `predict_nrtl_from_unifac`,
`predict_uniquac_from_unifac`. Curated interaction parameters are available from
the database (`nrtl_from_database`, `uniquac_from_database`, `kij_from_database`,
`pr_kij`) where present.

## Parameter regression

Fit activity-model parameters to data by differentiable optimisation: a
self-contained `levenberg_marquardt` (and `gradient_descent`) over arbitrary
parameter pytrees, with residual builders (`bubble_pressure_residuals`,
`activity_residuals`, `lle_residuals`) and ready fitters `fit_nrtl_binary` /
`fit_uniquac_binary`. Experimental VLE/LLE can be read from the open
[NIST ThermoML Archive](https://www.nist.gov/mml/acmd/trc/thermoml/thermoml-archive)
with `read_thermoml` / `load_sample`.

## Binary diagrams, azeotropes & residue curves

The `fugacio.sim` layer builds the classic non-ideal diagrams from any binary
property package: `pxy_diagram`, `txy_diagram`, and the `azeotrope_pressure` /
`azeotrope_temperature` finders. For ternary screening, `residue_curve` integrates
a single open-evaporation trajectory and `residue_curve_map` sweeps a family of
them, the standard tool for laying out distillation boundaries.

```python
import jax.numpy as jnp
from fugacio.sim import package_for, residue_curve

model = package_for(("acetone", "methanol", "water"), "nrtl")
curve = residue_curve(model, jnp.array([0.4, 0.4, 0.2]), 101325.0)
curve.x   # liquid-composition trajectory toward the high-boiling node (water)
```

## Validation

The activity kernels are cross-checked in the opt-in oracle suite (`just oracles`)
against the [`thermo`](https://github.com/CalebBell/thermo) library (NRTL, UNIQUAC,
classic & Dortmund UNIFAC, Wilson) and, when a Julia install is present, against
[Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl), passing identical
interaction parameters so a discrepancy isolates the kernel rather than the inputs.
