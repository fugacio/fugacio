# Molecular SAFT (PC-SAFT)

Cubic equations of state and the gamma-phi property system both struggle with the
same physics: strong, specific association (hydrogen bonding in water, alcohols,
and amines), chain-length effects in long or polymeric molecules, and the wide
density range between a light gas and a dense liquid. Fugacio's third class of
thermodynamic method, **PC-SAFT** (perturbed-chain statistical associating fluid
theory, Gross and Sadowski 2001), is built for exactly that regime. It models a
fluid as chains of spherical segments with dispersion attraction and optional
short-range association, and it plugs into the rest of the stack through the same
`EquilibriumModel` interface as everything else.

Like the [reference Helmholtz fluids](reference-fluids.md), PC-SAFT is *one scalar
reduced residual Helmholtz energy* and every property is an autodiff derivative of
it, so the whole model, including the Wertheim association site-fraction solve, is
differentiable in both the thermodynamic state and the molecular parameters.

## Parameters

A non-associating species needs three parameters: the segment number `m`, the
segment diameter `sigma`, and the dispersion energy `epsilon/k`. Associating
species carry two more, the association energy `epsilon_AB/k` and volume
`kappa_AB`, plus a Huang-Radosz site scheme. `saft_parameters_for` assembles a
differentiable `SaftParameters` pytree straight from component names, drawing on
the curated Gross and Sadowski parameter bank and filling binary corrections
`k_ij` from the database where available.

```python
from fugacio.thermo import saft_parameters_for

params = saft_parameters_for(["ethanol", "water"])  # both associating (2B scheme)
params.m            # segment numbers, shape (2,)
params.associating  # True: a Wertheim association term is active
```

You can also build a set from raw arrays with `saft_parameters` (pass
`sigma_in_angstrom=True` to use the literature unit), which is what the regression
layer perturbs when it fits parameters to data.

## Properties

Every property is a derivative of `alpha_residual(params, rho, T, x)`. The
`(T, P, x)` entry points first solve `P(rho, T, x) = P` for the molar density on
the requested phase branch, then differentiate the energy:

```python
import jax.numpy as jnp
from fugacio.thermo import saft_parameters_for
from fugacio.thermo.saft import (
    compressibility_factor,
    ln_fugacity_coefficients,
    molar_density,
    residual_properties,
)

params = saft_parameters_for(["water"])
x = jnp.ones(1)
rho = molar_density(params, 298.15, 1e5, x, phase="liquid")   # mol/m^3
z = compressibility_factor(params, rho, 298.15, x)
ln_phi = ln_fugacity_coefficients(params, 298.15, 1e5, x, phase="liquid")
res = residual_properties(params, 298.15, 1e5, x, phase="liquid")
res.enthalpy, res.entropy, res.cp   # departure functions (J/mol, J/mol/K)
```

The `phase` argument selects the density branch: `"liquid"` and `"vapor"` seed a
Newton solve from a dense packing or the ideal gas, while `"stable"` returns the
root with the lower molar Gibbs energy when more than one branch exists.

## Phase equilibrium

`SAFTModel` wraps a parameter set and the critical constants (used only to seed
Wilson K-values) behind the unified model interface: `flash_pt`,
`bubble_pressure` / `bubble_temperature`, `dew_pressure` / `dew_temperature`, and a
tangent-plane `stability` test. The `fugacio.sim` helper `saft_model_for` builds
one from component names.

```python
import jax.numpy as jnp
from fugacio.sim import saft_model_for

model = saft_model_for(["propane", "n-butane"])
res = model.flash_pt(320.0, 8e5, jnp.array([0.5, 0.5]))
res.beta, res.x, res.y   # vapour fraction and phase compositions

p_bub, y = model.bubble_pressure(320.0, jnp.array([0.4, 0.6]))
# beta, p_bub, y are differentiable w.r.t. T, P, z, *and* the PC-SAFT parameters.
```

For a pure component, `psat_saft` solves the saturation pressure by equifugacity
from an initial guess (a Wilson or Antoine value is fine):

```python
from fugacio.thermo import saft_parameters_for
from fugacio.thermo.saft import psat_saft

water = saft_parameters_for(["water"])
psat = psat_saft(water, 373.15, 1.0e5)   # ~ 1 atm
```

## Differentiable parameter regression

Because every PC-SAFT property is differentiable with respect to the molecular
parameters, fitting them to data is plain gradient-based least squares straight
through the saturation and bubble-point solvers, with no finite-difference
parameter sweeps. `fit_saft_pure` regresses the pure `(m, sigma, epsilon)` to
saturation-pressure and liquid-density data, and `fit_saft_kij` regresses a binary
correction `k_ij` to isothermal bubble pressures.

```python
import jax.numpy as jnp
from fugacio.thermo import saft_parameters_for
from fugacio.thermo.saft import fit_saft_pure

base = saft_parameters_for(["n-pentane"])
temperatures = jnp.array([280.0, 300.0, 320.0, 340.0])
psat_exp = jnp.array([36_000.0, 73_000.0, 137_000.0, 235_000.0])       # Pa
rho_liquid_exp = jnp.array([8_800.0, 8_550.0, 8_280.0, 7_980.0])       # mol/m^3
fitted, cost = fit_saft_pure(base, temperatures, psat_exp, rho_liquid_exp)
```

The same differentiability is what lets a `SAFTModel` sit inside a flowsheet and
still expose gradients of a downstream objective with respect to the thermodynamic
model's own parameters.

## Copilot tools

The design copilot exposes PC-SAFT through deterministic, JSON-in/JSON-out tools:
`saft_flash`, `saft_density`, `saft_saturation_pressure`, `saft_bubble_pressure`,
and `saft_residual_enthalpy`. They accept the same component names as the rest of
the registry but route the calculation through the molecular EOS, the method of
choice when the agent reasons about associating fluids.

## Validation

PC-SAFT is checked the same way as the rest of Fugacio. A first-principles
consistency suite verifies exact thermodynamic identities through autodiff: the
pressure as the density derivative of the residual energy, the mole-fraction
weighted log-fugacity equal to the residual Gibbs energy, the Gibbs-Helmholtz
relation for the residual enthalpy, and AD-versus-finite-difference agreement for
the density, saturation, flash, and Wertheim site-fraction solvers. The opt-in
oracle suite (`just oracles`) additionally grades pressures and saturation
pressures against [Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl),
an independent PC-SAFT implementation, when a Julia install is present.
