# Non-ideal phase equilibrium

Beyond cubic equations of state, Fugacio carries a full **γ–φ property system**:
liquid-phase activity coefficients combined with a vapour-phase fugacity model,
plus rigorous liquid-liquid and vapour-liquid-liquid equilibria. This is the route
that captures azeotropes, partial miscibility, and strongly non-ideal mixtures —
the regime where ideal-K and single-EOS methods quietly give wrong answers.

## The γ–φ property model

A `GammaPhiModel` pairs an activity model (NRTL, UNIQUAC, Wilson, Margules, van
Laar, UNIFAC, …) with a pure-component reference fugacity (saturation pressure,
Poynting correction, and saturation φ) and a vapour model. It exposes the *same*
`flash_pt`, bubble, and dew calls as an `EOSModel`, so the rest of the stack
switches thermodynamic method by swapping one differentiable object. The
`fugacio.sim` helpers `eos_model_for`, `nrtl_model_for`, `uniquac_model_for`, and
`unifac_model_for` build a ready model straight from component names.

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
`dew_temperature_gamma`, and the low-level `gamma_phi_k_values`. The reference
state is assembled from `liquid_reference_fugacity`, `poynting_factor`,
`saturation_fugacity_coefficient`, and (for dissolved gases) `henry_constant`.

## Liquid-liquid & three-phase equilibria

- `flash_lle` — isoactivity liquid-liquid flash, with `tie_line`, `binodal_curve`,
  and `binary_binodal` for the miscibility envelope.
- `flash_vlle` — three-phase vapour-liquid-liquid flash, plus the binary
  `heterogeneous_azeotrope` solver.
- **Tangent-plane stability** (TPD): `tangent_plane_distance`, `liquid_stability`,
  and the general `stability_analysis_general` decide how many phases are actually
  present before a flash is attempted.

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
model: `pxy_diagram`, `txy_diagram`, and the `azeotrope_pressure` /
`azeotrope_temperature` finders. For ternary screening, `residue_curve` integrates
a single open-evaporation trajectory and `residue_curve_map` sweeps a family of
them — the standard tool for laying out distillation boundaries.

```python
import jax.numpy as jnp
from fugacio.sim import nrtl_model_for, residue_curve

model = nrtl_model_for(("acetone", "methanol", "water"))
curve = residue_curve(model, jnp.array([0.4, 0.4, 0.2]), 101325.0)
curve.x   # liquid-composition trajectory toward the high-boiling node
```

## Validation

The activity kernels are cross-checked in the opt-in oracle suite (`just oracles`)
against the [`thermo`](https://github.com/CalebBell/thermo) library (NRTL, UNIQUAC,
classic & Dortmund UNIFAC, Wilson) and, when a Julia install is present, against
[Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) — passing identical
interaction parameters so a discrepancy isolates the kernel rather than the inputs.
