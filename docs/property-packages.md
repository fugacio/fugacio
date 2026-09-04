# Property packages

A process simulator needs two things from its thermodynamics: *what splits*
(fugacities, K-values, flashes) and *how much energy* (enthalpy, entropy,
volume). Fugacio has long answered the first question for several method
classes, but until this release every energy balance in `fugacio.sim` was wired
to a cubic equation of state alone: a heater, a mixer, an isentropic compressor,
or a column reboiler could only be evaluated on Peng-Robinson or SRK, whatever
model had been used to flash the stream.

A **property package** closes that gap. `fugacio.thermo.PropertyPackage` is one
object that owns both phase equilibrium and energy for a set of components, and
it's the single interface every energy-balanced unit, the rigorous column, the
two-sided heat exchanger, and the equation-oriented engine consume through their
`model` argument. It's implemented for all four method classes Fugacio carries:

| Package | Route | Energy side |
| --- | --- | --- |
| `CubicPackage` | phi-phi on PR, SRK, RK, or vdW | residual functions of the cubic (the historical default) |
| `GammaPhiPackage` | activity-coefficient liquid (NRTL, UNIQUAC, Wilson, UNIFAC, Dortmund) with an ideal or EOS vapor | pure-liquid enthalpies plus the excess enthalpy from autodiff of \(g^E\) |
| `SAFTPackage` | PC-SAFT with Wertheim association | temperature derivatives of the residual Helmholtz energy |
| `HelmholtzPackage` | one pure reference fluid (IAPWS-95 water, Span-Wagner CO2, and the other vendored formulations) | the published multiparameter equation itself |

## Build one

`fugacio.sim.package_for` is the constructor a flowsheet author reaches for. It
takes component names and a method keyword from `fugacio.sim.METHODS`:

```python
from fugacio.sim import package_for

pr = package_for(("propane", "n-butane", "n-pentane"))            # Peng-Robinson, the default
srk = package_for(("methane", "ethane"), "srk", use_database_kij=True)
nrtl = package_for(("ethanol", "water"), "nrtl")                   # curated binaries
unifac = package_for(("acetone", "chloroform"), "unifac")          # predictive
saft = package_for(("methanol", "n-hexane"), "pcsaft")
steam = package_for(("water",), "iapws")                           # reference fluid
```

Method-specific options (`kij`, `alpha_default`, `vapor="eos"`, `poynting`, and
so on) are forwarded to the underlying model factory; see `package_for` in the
[API reference](api/sim/models.md). If you already hold a bare model object (an
`EOSModel`, a `GammaPhiModel`, or a `SAFTModel`), any `model=` argument in
`fugacio.sim` accepts it directly and upgrades it to the matching package.

## One interface

Every package exposes the same calls:

* Single-phase properties at \((T, P, x)\) with an explicit `phase`:
  `ln_phi`, `enthalpy`, `entropy`, `volume`, and `gibbs`.
* Phase equilibrium: `flash_pt`, `bubble_pressure`, `dew_pressure`,
  `bubble_temperature`, `dew_temperature`, `k_values`, and `k_seed` (the
  method's natural initial K-value estimate, which the column solver uses).
* Two-phase-aware bulk properties, implemented once on top of the flash:
  `mixture_enthalpy`, `mixture_entropy`, `mixture_volume`.
* Energy-specified flashes, also generic: `flash_ph`, `flash_ps`, `flash_tv`.

```python
import jax.numpy as jnp

pkg = package_for(("ethanol", "water"), "nrtl")
x = jnp.array([0.4, 0.6])

h_liq = pkg.enthalpy(350.0, 1.013e5, x, phase="liquid")   # J/mol, ideal-gas reference
res = pkg.flash_pt(360.0, 1.013e5, x)                      # beta, x, y, K
state = pkg.flash_ph(1.013e5, h_liq + 20_000.0, x)         # T and phase split at fixed P, H
```

Enthalpy and entropy are relative to the ideal-gas reference at `T_REF` and
`P_REF` from `fugacio.thermo.properties`, except for `HelmholtzPackage`, which
keeps the reference of the published formulation. Only differences are physical
and any consistent reference cancels in a balance, so don't mix packages with
different references across one energy balance (a `HelmholtzPackage` steam side
and a cubic process side inside one *exchanger* is fine, because each side is
balanced separately).

## Heat of mixing for free

The gamma-phi package doesn't need a separate heat-of-mixing correlation. The
excess enthalpy follows from the Gibbs-Helmholtz relation,

\[
h^E = -R T^2 \,\frac{\partial}{\partial T}\!\left(\frac{g^E}{RT}\right)
    = -R T^2 \sum_i x_i \,\frac{\partial \ln\gamma_i}{\partial T},
\]

and the temperature derivative is taken by automatic differentiation of the very
`ln_gamma` the activity model already provides. The heat of mixing is therefore
exactly consistent with the activity coefficients that decide the VLE, and it
inherits any temperature dependence the parameters carry. `excess_enthalpy` and
`excess_entropy` are exported from `fugacio.thermo` for direct use:

```python
from fugacio.thermo import excess_enthalpy

h_e = excess_enthalpy(nrtl.activity, jnp.array([0.3, 0.7]), 320.0)  # J/mol
```

## Everything takes `model=`

The energy-balanced units in `fugacio.sim.units` (`heater`, `valve`, `pump`,
`compressor`, `turbine`, `mix`, `flash_drum`), the rigorous column, the two-sided
`heat_exchanger`, and the `EOFlowsheet` all accept `model=`. Omitting it keeps
the historical Peng-Robinson default, so existing flowsheets are unchanged.

```python
from fugacio.sim import Stream, heater, compressor

feed = Stream.from_fractions(("ethanol", "water"), jnp.array([0.1, 0.9]), 100.0, 300.0, 1.5e5)
warm = heater(feed, t_out=350.0, model=nrtl)        # NRTL liquid enthalpy, heat of mixing included
```

A package is a registered JAX pytree whose model parameters are leaves, so a
flowsheet built on any of them is differentiable with respect to the
thermodynamic parameters as well as the operating conditions: `jax.grad` of a
column duty with respect to a `kij` or an NRTL \(\tau_{ij}\) works the same way
as a gradient with respect to a reflux ratio.

## 64-bit arithmetic is on by default

Cubic-root selection, Newton iterations to \(10^{-10}\) residuals, and the
implicit-function-theorem gradients throughout Fugacio assume double precision.
Importing `fugacio.thermo` now enables `jax_enable_x64` automatically. Set the
environment variable `FUGACIO_X64=0` before importing if you deliberately want
single precision and accept the loss of digits.
