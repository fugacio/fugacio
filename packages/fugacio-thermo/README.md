# fugacio-thermo

Differentiable thermodynamics and physical-property engine for the
[Fugacio](https://github.com/owenthcarey/fugacio) stack. Every model is written
in [JAX](https://github.com/jax-ml/jax), so any output (a fugacity coefficient,
a saturation pressure, a flash result) is differentiable with respect to
temperature, pressure, composition, *and* model parameters. The iterative solvers
(cubic-EOS root, flash, saturation, bubble/dew) carry hand-written
implicit-function-theorem rules, so gradients flow exactly through them rather
than through unrolled iterations.

## What's inside

- **Curated open component database** (`DATABASE`, `get`, `component_arrays`):
  critical constants, acentric factors, Antoine coefficients, and ideal-gas heat
  capacities for common species.
- **Ideal-gas properties**: `cp_ig`, `enthalpy_ig`, `entropy_ig`, `gibbs_ig`
  (plus mixture variants).
- **Cubic equations of state**: van der Waals, Redlich-Kwong, SRK, Peng-Robinson
  (`VDW`, `RK`, `SRK`, `PR`), with mixing rules, a differentiable compressibility
  solver, fugacity coefficients (`ln_phi_mixture`, `ln_phi_pure`), and molar
  volume.
- **Real-fluid energy properties**: residual/departure functions
  (`residual_enthalpy`, `residual_entropy`, `residual_gibbs`, `residual_cp`),
  real-fluid molar properties (`molar_enthalpy`, `molar_entropy`, `molar_gibbs`,
  `molar_cp`, `stable_phase`), two-phase `mixture_enthalpy` / `mixture_entropy`,
  and energy-specified flashes `flash_ph` (isenthalpic) and `flash_ps`
  (isentropic), the backbone of adiabatic units, valves, compressors, and
  turbines.
- **Activity-coefficient models**: Margules, van Laar, Wilson, NRTL, UNIQUAC, and
  predictive regular-solution / Flory-Huggins, available both as functions and
  as differentiable `ActivityModel` objects (`nrtl`, `uniquac`, ...) whose
  parameters are themselves gradient leaves.
- **Group contribution**: predictive `unifac_activity` and `joback_estimate`
  (pure-component constants from a structure).
- **Reference state**: pure-liquid reference fugacity (`liquid_reference_fugacity`),
  the `poynting_factor`, saturation fugacity coefficient, and `henry_constant`.
- **EOS phase equilibrium**: `rachford_rice`, `flash_pt`, `psat_eos`,
  `bubble_pressure_eos`, `dew_pressure_eos`, and Michelsen `stability_analysis`.
- **Non-ideal (gamma-phi) VLE**: `flash_pt_gamma`, `bubble_pressure_gamma`,
  `dew_pressure_gamma`, and the temperature duals, the route that captures
  azeotropes and strongly polar mixtures.
- **Liquid-liquid & three-phase equilibria**: isoactivity `flash_lle` with
  `tie_line` / `binodal_curve`, three-phase `flash_vlle`, the binary
  `heterogeneous_azeotrope` solver, and a general tangent-plane stability test
  (`stability_analysis_general`, `liquid_stability`).
- **Unified model interface**: `EOSModel` and `GammaPhiModel` expose the same
  `flash_pt` / bubble / dew calls, so the rest of the stack switches
  thermodynamic method by swapping one (differentiable) object.
- **Parameter regression & prediction**: a self-contained `levenberg_marquardt`
  over arbitrary parameter pytrees with residual builders
  (`bubble_pressure_residuals`, `activity_residuals`, `lle_residuals`), ready
  fitters (`fit_nrtl_binary`, `fit_uniquac_binary`), and UNIFAC-to-binary
  prediction (`predict_nrtl_from_unifac`, `predict_uniquac_from_unifac`) for
  mixtures without fitted parameters.
- **Reactions, equilibrium & kinetics**: stoichiometry and standard-state
  thermochemistry (`Reaction`, `reaction_properties`, `delta_g_rxn`,
  `equilibrium_constant`), chemical-reaction `equilibrium` (single or simultaneous,
  ideal-gas or EOS-`phi` basis), and differentiable rate laws (`PowerLaw`,
  `MassActionReversible`, `LHHW`, `Arrhenius`).
- **Validation harness**: first-principles consistency checks (Gibbs-Duhem,
  equifugacity, the `(d ln phi / dP)_T` identity), an AD-vs-finite-difference
  checker, and optional differential-testing oracles: CoolProp / `chemicals`
  (pure-fluid properties), `thermo` / Clapeyron.jl (activity coefficients), and
  Cantera (reaction equilibrium and standard-state thermochemistry).

## Example: a differentiable flash

```python
import jax
import jax.numpy as jnp
from fugacio.thermo import PR, component_arrays, flash_pt

arr = component_arrays(["methane", "propane", "n-pentane"])
z = jnp.array([0.5, 0.3, 0.2])

result = flash_pt(PR, 320.0, 20e5, z, arr["tc"], arr["pc"], arr["omega"])
result.beta      # vapour fraction (~0.75)
result.x, result.y  # liquid / vapour compositions

# Gradient of the vapour fraction w.r.t. pressure, straight through the solver:
dbeta_dP = jax.grad(
    lambda p: flash_pt(PR, 320.0, p, z, arr["tc"], arr["pc"], arr["omega"]).beta
)
dbeta_dP(20e5)
```

## Example: a non-ideal (gamma-phi) bubble point

```python
import jax.numpy as jnp
from fugacio.thermo import bubble_pressure_gamma, component_arrays, nrtl

arr = component_arrays(["ethanol", "water"])
# NRTL with 1/T interaction coefficients (K); alpha = 0.3.
model = nrtl(
    a=jnp.zeros((2, 2)),
    b=jnp.array([[0.0, 670.0], [310.0, 0.0]]),
    alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]),
)
P, y = bubble_pressure_gamma(model, 350.0, jnp.array([0.3, 0.7]),
                             arr["tc"], arr["pc"], arr["omega"])
# P, y are differentiable w.r.t. T, x, *and* the NRTL parameters.
```

Part of the `fugacio` namespace; installs independently:
`pip install fugacio-thermo`.
