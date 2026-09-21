# fugacio-thermo

Differentiable thermodynamics and physical-property engine for the
[Fugacio](https://github.com/fugacio/fugacio) stack. Every model is written
in [JAX](https://github.com/jax-ml/jax), so any output (a fugacity coefficient,
a saturation pressure, a flash result) is differentiable with respect to
temperature, pressure, composition, *and* model parameters. The iterative solvers
(cubic-EOS root, flash, saturation, bubble/dew) carry hand-written
implicit-function-theorem rules, so gradients flow exactly through them rather
than through unrolled iterations.

No call returns a plausible number from a failed solve. Each iterative
calculation has a checked `*_with_info` form that returns the best iterate and a
`SolveReport` (with statuses such as `OUT_OF_DOMAIN` and `TRIVIAL`), and its
value-only form returns NaN when that report fails. Derivatives of a failed
solve are nonfinite.

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
- **Residual properties**: departure functions (`residual_enthalpy`,
  `residual_entropy`, `residual_gibbs`, `residual_cp`, `residual_properties`)
  on the cubic equations of state.
- **Property packages** (`PropertyPackage`, `cubic_package`, `gamma_phi_package`,
  `saft_package`, `helmholtz_package`): the one model type, an object that owns
  phase equilibrium *and* energy for a component set, implemented for every
  method class. Each package provides single-phase `enthalpy`, `entropy`,
  `volume`, `gibbs`, and `heat_capacity`; two-phase `mixture_enthalpy` /
  `mixture_entropy` / `mixture_volume`; PT, PH (isenthalpic), and PS
  (isentropic) flashes, the backbone of adiabatic units, valves, compressors,
  and turbines; a TV flash; bubble and dew points; and `stability`. The gamma-phi package
  gets its heat of mixing from autodiff of the excess Gibbs energy
  (`excess_enthalpy`, `excess_entropy`), so enthalpy stays consistent with the
  activity coefficients. Solve methods run through compiled kernels with the
  package as a dynamic argument. Importing `fugacio.thermo` enables 64-bit JAX
  arithmetic (set `FUGACIO_X64=0` to opt out).
- **Activity-coefficient models**: Margules, van Laar, Wilson, NRTL, UNIQUAC, and
  predictive regular-solution / Flory-Huggins, available both as functions and
  as differentiable `ActivityModel` objects (`nrtl`, `uniquac`, ...) whose
  parameters are themselves gradient leaves.
- **Group contribution**: predictive `unifac_activity` and `joback_estimate`
  (pure-component constants from a structure).
- **Molecular PC-SAFT**: the perturbed-chain SAFT equation of state
  (`SaftParameters`, `saft_parameters_for`, `alpha_residual`, and the
  `SAFTPackage` from `saft_package`) with a curated Gross-Sadowski parameter
  bank, Wertheim TPT1 association for hydrogen-bonding fluids (`2B`, `3B`, and
  `4C` schemes), fugacity / density / residual properties by autodiff, the
  equilibrium routines `flash_pt_saft`, `bubble_pressure_saft`,
  `dew_pressure_saft`, and `psat_saft` (each with a `_with_info` form), and
  differentiable parameter regression (`fit_saft_pure`, `fit_saft_kij`).
- **Reference state**: pure-liquid reference fugacity (`liquid_reference_fugacity`),
  the `poynting_factor`, saturation fugacity coefficient, and `henry_constant`.
- **EOS phase equilibrium**: `rachford_rice`, `flash_pt`, `psat_eos`,
  `bubble_pressure_eos`, and `dew_pressure_eos`, each with a `_with_info` form.
- **Non-ideal (gamma-phi) VLE**: `flash_pt_gamma`, `bubble_pressure_gamma`,
  `dew_pressure_gamma`, and the temperature duals, the route that captures
  azeotropes and strongly polar mixtures.
- **Liquid-liquid & three-phase equilibria**: stability-first isoactivity
  `flash_lle` with `tie_line` / `binodal_curve`, three-phase `flash_vlle`, and
  the binary `heterogeneous_azeotrope` solver.
- **Phase stability**: one shared tangent-plane search, `tpd_search`, with
  liquid and vapor trial branches and multiple starts. Every package exposes it
  as `stability`, and `liquid_stability` tests an activity-model liquid for a
  second liquid.
- **Physical acceptance** (`fugacio.thermo.acceptance`): `flash_pt_checked`,
  `flash_ph_checked`, and `flash_ps_checked` grade material balance,
  equifugacity, stability, and parameter applicability independently of the
  solver, and
  `PhysicalReport.failures()` names each failed criterion.
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
  equifugacity, the `(d ln phi / dP)_T` identity, and the Gibbs-Helmholtz and
  fugacity-enthalpy consistency of property packages), an AD-vs-finite-difference
  checker, and optional differential-testing oracles: CoolProp / `chemicals`
  (pure-fluid properties), `thermo` / Clapeyron.jl (activity coefficients and
  PC-SAFT), and Cantera (reaction equilibrium and standard-state thermochemistry).

## Example: a differentiable flash

```python
import jax
import jax.numpy as jnp
from fugacio.thermo import PR, component_arrays, flash_pt

arr = component_arrays(["methane", "propane", "n-pentane"])
z = jnp.array([0.5, 0.3, 0.2])

result = flash_pt(PR, 320.0, 20e5, z, arr["tc"], arr["pc"], arr["omega"])
result.beta      # vapor fraction (~0.75); NaN if the flash had failed
result.x, result.y  # liquid / vapor compositions

# Gradient of the vapor fraction w.r.t. pressure, straight through the solver:
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
