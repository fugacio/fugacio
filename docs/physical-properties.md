# Physical & transport properties

Equipment sizing runs on densities, viscosities, conductivities, and surface
tension, not just fugacities. Fugacio carries a curated, differentiable layer of
**pure-component correlations**, **corresponding-states estimators**, and
**mixture combining rules** for exactly these properties, sourced from the open
literature (Poling, Prausnitz & O'Connell, *The Properties of Gases and
Liquids*, 5th ed.) and cross-checked against CoolProp and `chemicals` in the
oracle suite.

The design follows the rest of `fugacio.thermo`: correlation *kernels* are pure
JAX functions of scalars and arrays (differentiable, `jit`/`vmap`-compatible),
and *dispatchers* accept component names and pick the best available method:
curated DIPPR-form coefficients when the database has them, a
corresponding-states estimate otherwise.

## Pure-component correlations

The component database bundles fitted DIPPR-form coefficients (forms 100, 101,
102, 105, 106) for vapour pressure, liquid density, heat of vaporization,
liquid heat capacity, gas viscosity, and liquid/gas thermal conductivity, with
validity ranges. The generic kernels are exposed directly (`dippr100`,
`dippr101`, `dippr102`, `dippr105`, `dippr106`), and the estimators fill the
gaps:

- **Heat of vaporization**: `heat_of_vaporization` (DIPPR-106 where curated,
  `pitzer_hvap` otherwise), plus `watson_hvap` for re-scaling a known value to
  another temperature.
- **Liquid heat capacity**: `liquid_heat_capacity` (DIPPR-100 where curated,
  `rowlinson_bondi_cp` from the ideal-gas Cp otherwise).
- **Surface tension**: `surface_tensions` (Mulero-Cachadiña / DIPPR-106 fits
  where curated, `brock_bird_surface_tension` otherwise).

```python
import jax
from fugacio.thermo import heat_of_vaporization, liquid_heat_capacity

hvap = heat_of_vaporization(["water", "ethanol"], 298.15)   # J/mol
cp_l = liquid_heat_capacity(["water", "ethanol"], 298.15)   # J/mol/K

# Differentiable in T like everything else:
jax.grad(lambda t: heat_of_vaporization(["water"], t)[0])(298.15)
```

## Liquid & vapour density

- `liquid_molar_volumes` / `liquid_density`: saturated-liquid volumes from
  curated DIPPR-105 fits, falling back to COSTALD (`costald_volume`) and
  Rackett (`rackett_volume`); mixtures use Amagat averaging
  (`mixture_liquid_volume`).
- `vapor_density`: mass density from any cubic EOS at `(T, P, y)`.
- **Volume translation**: `peneloux_shift` / `translated_molar_volume` apply
  the Péneloux correction to cubic-EOS liquid volumes (`zra_estimate` supplies
  the Rackett compressibility), typically halving raw PR/SRK liquid-density
  error.
- `tyn_calus_vb`: molar volume at the normal boiling point, the input the
  diffusivity estimators need.

## Transport properties

Gas phase (dilute, kinetic-theory based):

- `gas_viscosities`: Chung's method (`chung_viscosity_gas`) with the Neufeld
  collision integral; mixtures by Wilke (`gas_mixture_viscosity`).
- `gas_thermal_conductivities`: Chung's thermal-conductivity form
  (`chung_thermal_conductivity_gas`); mixtures by Wassiljewa / Mason-Saxena
  (`gas_mixture_thermal_conductivity`).

Liquid phase (corresponding-states):

- `liquid_viscosities`: Letsou-Stiel (`letsou_stiel_viscosity`); mixtures by
  Grunberg-Nissan (`liquid_mixture_viscosity`).
- `liquid_thermal_conductivities`: Sato-Riedel
  (`sato_riedel_thermal_conductivity`); mixtures by the DIPPR9H power-law rule
  (`liquid_mixture_thermal_conductivity`).
- `mixture_surface_tension`: Winterfeld-Scriven-Davis combining rule over the
  pure-component values.

Binary diffusion coefficients:

- `gas_diffusivity`: Fuller-Schettler-Giddings at `(T, P)` from tabulated
  atomic diffusion volumes (`diffusion_volume`).
- `liquid_diffusivity`: Wilke-Chang at infinite dilution, with solvent
  association factors and `tyn_calus_vb` for the solute volume.

```python
import jax.numpy as jnp
from fugacio.thermo import (
    gas_diffusivity, liquid_mixture_viscosity, mixture_surface_tension,
)

x = jnp.array([0.5, 0.5])
mu = liquid_mixture_viscosity(["benzene", "toluene"], 298.15, x)   # Pa*s
sigma = mixture_surface_tension(["benzene", "toluene"], 298.15, x) # N/m
d_ab = gas_diffusivity("ethanol", "air", 313.15, 101325.0)         # m^2/s
```

## Stream-level properties & sizing

The `fugacio.sim` layer evaluates all of these at a stream's own state:
`liquid_density`, `vapor_density`, `liquid_volumetric_flow`,
`vapor_volumetric_flow`, `liquid_viscosity`, `vapor_viscosity`,
`liquid_thermal_conductivity`, `vapor_thermal_conductivity`, and
`surface_tension` each take a `Stream`. `column_diameter_for` chains them into
a Souders-Brown column/drum diameter sized from actual stream densities, and
because everything is JAX, the diameter is differentiable with respect to feed
conditions through the flash *and* the property correlations.

```python
import jax.numpy as jnp
from fugacio.sim import Stream, column_diameter_for, flash_drum, liquid_density

feed = Stream.from_fractions(
    ("methane", "propane", "n-pentane"),
    jnp.array([0.5, 0.3, 0.2]),
    flow=100.0, t=320.0, p=20e5,
)
vap, liq = flash_drum(feed, 320.0, 20e5)
rho_l = liquid_density(liq)                 # kg/m^3 at the drum state
d = column_diameter_for(vap, liq)           # Souders-Brown diameter (m)
```

The copilot exposes the same capability as JSON tools: `physical_properties`
(densities, viscosities, conductivities, surface tension, Hvap, liquid Cp for a
mixture at `T`, `P`) and `binary_diffusivity` (Fuller + Wilke-Chang).

## The ThermoML parameter bank

Fugacio bundles isothermal binary VLE datasets in
[NIST ThermoML](https://www.nist.gov/mml/acmd/trc/thermoml/thermoml-archive)
format and a batch regression driver that fits NRTL parameters to every bundled
dataset (`fit_vle_dataset`, `fit_bundled_samples`). The results ship as a
**parameter bank**, a lookup table of fitted binary interaction parameters
with their fit quality:

```python
from fugacio.thermo import ParameterBank

bank = ParameterBank.bundled()
entry = bank.lookup("ethanol", "water")     # orientation-aware
entry.b, entry.alpha                        # fitted NRTL b_ij (K) and alpha
entry.rms_pct                               # pressure RMS error of the fit (%)
```

`ParameterBank.lookup` handles component-order swaps (transposing the parameter
matrices), and the bank serializes to JSON for regeneration via
`scripts/gen_parameter_bank.py`.

## Validation

Every correlation is covered at three levels:

1. **Unit tests** (default CI): literature spot values, pure-component limits
   of every mixture rule, gradient and `jit` checks.
2. **Oracle tests** (`just oracles`): differential testing against
   [CoolProp](https://github.com/CoolProp/CoolProp) reference equations of
   state (densities, viscosities, conductivities, surface tension, Hvap, Cp)
   and [`chemicals`](https://github.com/CalebBell/chemicals) kernel-for-kernel
   (identical inputs isolate the implementation, not the method).
3. **ThermoML regression**: the parameter bank's fits are checked to reproduce
   the bundled experimental bubble pressures within their recorded RMS.
