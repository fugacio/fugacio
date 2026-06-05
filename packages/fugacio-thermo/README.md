# fugacio-thermo

Differentiable thermodynamics and physical-property engine for the
[Fugacio](https://github.com/owenthcarey/fugacio) stack. Every model is written
in [JAX](https://github.com/jax-ml/jax), so any output — a fugacity coefficient,
a saturation pressure, a flash result — is differentiable with respect to
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
- **Cubic equations of state** — van der Waals, Redlich-Kwong, SRK, Peng-Robinson
  (`VDW`, `RK`, `SRK`, `PR`) — with mixing rules, a differentiable compressibility
  solver, fugacity coefficients (`ln_phi_mixture`, `ln_phi_pure`), and molar
  volume.
- **Real-fluid energy properties**: residual/departure functions
  (`residual_enthalpy`, `residual_entropy`, `residual_gibbs`, `residual_cp`),
  real-fluid molar properties (`molar_enthalpy`, `molar_entropy`, `molar_gibbs`,
  `molar_cp`, `stable_phase`), two-phase `mixture_enthalpy` / `mixture_entropy`,
  and energy-specified flashes `flash_ph` (isenthalpic) and `flash_ps`
  (isentropic) — the backbone of adiabatic units, valves, compressors, and
  turbines.
- **Activity-coefficient models**: Margules, van Laar, Wilson, NRTL, UNIQUAC.
- **Group contribution**: predictive `unifac_activity` and `joback_estimate`
  (pure-component constants from a structure).
- **Phase equilibrium**: `rachford_rice`, `flash_pt`, `psat_eos`,
  `bubble_pressure_eos`, `dew_pressure_eos`, and Michelsen `stability_analysis`.
- **Validation harness**: first-principles consistency checks (Gibbs-Duhem,
  equifugacity, the `(d ln phi / dP)_T` identity), an AD-vs-finite-difference
  checker, and optional differential-testing oracles (CoolProp, `chemicals`).

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

Part of the `fugacio` namespace; installs independently:
`pip install fugacio-thermo`.
