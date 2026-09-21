# Property packages

A process simulator needs two things from its thermodynamics: *what splits*
(fugacities, K-values, flashes) and *how much energy* (enthalpy, entropy,
volume). A **property package** answers both. `fugacio.thermo.PropertyPackage`
is one object that owns phase equilibrium and energy for a set of components,
and it's the single model type that every unit operation, the rigorous column,
the two-sided heat exchanger, the reactors, and the equation-oriented engine
consume through their `model` argument. It's implemented for all four method
classes Fugacio carries:

| Package | Route | Energy side |
| --- | --- | --- |
| `CubicPackage` | phi-phi on PR, SRK, RK, or vdW | residual functions of the cubic (Peng-Robinson is the default package) |
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

Method-specific options (`kij`, `use_database_kij`, `alpha_default`,
`vapor="eos"`, `poynting`, `phi_saturation`, and so on) are forwarded to the
underlying factory; see `package_for` in the [API reference](api/sim/models.md).
An option the method doesn't accept raises `TypeError`. NRTL and UNIQUAC
packages raise `KeyError` for a pair without curated parameters unless you pass
`parameter_policy="allow_ideal"`, which accepts zero interactions and records
that assumption in the package's evidence. The lower-level factories in
`fugacio.thermo` build a package without the name lookup: `cubic_package`,
`gamma_phi_package`, and `saft_package` from component constants and model
parameters, and `helmholtz_package` from a reference fluid.

Every `model=` argument in `fugacio.sim` takes a package, or `None` for
Peng-Robinson over the stream's components; anything else raises `TypeError`. A
package built by name also rejects a stream whose components differ from its
own or come in another order.

## One interface

Every package exposes the same calls:

* Single-phase properties at \((T, P, x)\) with an explicit `phase`:
  `ln_phi`, `enthalpy`, `entropy`, `volume`, `gibbs`, and `heat_capacity`.
* Phase equilibrium: `flash_pt`, `bubble_pressure`, `dew_pressure`,
  `bubble_temperature`, `dew_temperature`, `k_values`, and `k_seed` (the
  method's natural initial K-value estimate, which the column solver uses). The
  saturation calls return `SaturationResult(value, composition)`, a named
  tuple, so `p, y = pkg.bubble_pressure(t, x)` unpacks.
* Stability: `stability(t, p, z)`, the shared tangent-plane search on the
  package's liquid and vapor branches, returning a `StabilityResult`.
* Two-phase-aware bulk properties, implemented once on top of the flash:
  `mixture_enthalpy`, `mixture_entropy`, `mixture_volume`.
* Energy-specified flashes, also generic: `flash_ph`, `flash_ps`, `flash_tv`.

Each iterative calculation also has a checked `*_with_info` form
(`flash_pt_with_info`, `bubble_pressure_with_info`, `dew_temperature_with_info`,
`flash_ph_with_info`, and so on) that returns the best state and a
`SolveReport`. The value-only form returns NaN whenever that report fails,
including for a state outside the package's domain (`in_domain`), such as a
gamma-phi state in which a present component is supercritical. See the
[reliability guide](reliability.md#the-failure-contract) for the full contract.

```python
import jax.numpy as jnp

pkg = package_for(("ethanol", "water"), "nrtl")
x = jnp.array([0.4, 0.6])

h_liq = pkg.enthalpy(350.0, 1.013e5, x, phase="liquid")   # J/mol, ideal-gas reference
res = pkg.flash_pt(360.0, 1.013e5, x)                      # beta, x, y, K

# T and phase split at fixed P and H: about 355.9 K with 48% vapor.
solved = pkg.flash_ph_with_info(1.013e5, h_liq + 20_000.0, x, t_min=300.0, t_max=450.0)
solved.value.t, solved.value.beta, solved.report.converged
```

`flash_ph` and `flash_ps` search `[t_min, t_max]`, 50 K to 1500 K by default. A
gamma-phi package is defined only below its components' critical temperatures,
so give it a bracket inside that range; a search that ends outside the domain
reports `OUT_OF_DOMAIN`.

The solve methods run through compiled kernels cached per method and static
options, with the package itself as a dynamic argument, so a new state or new
parameter values reuse the compiled program. A package whose pytree holds a
non-array leaf, such as an unregistered activity-model object, runs eagerly
instead.

Enthalpy and entropy are relative to the ideal-gas reference at `T_REF` and
`P_REF` from `fugacio.thermo.constants`, except for `HelmholtzPackage`, which
keeps the reference of the published formulation. Only differences are physical
and any consistent reference cancels in a balance, so don't mix packages with
different references across one energy balance (a `HelmholtzPackage` steam side
and a cubic process side inside one *exchanger* is fine, because each side is
balanced separately).

## Associating mixtures on PC-SAFT

A `SAFTPackage` gives the same interface to [PC-SAFT](molecular-saft.md), whose
Wertheim association term handles hydrogen-bonding components such as methanol:

```python
import jax.numpy as jnp
from fugacio.sim import package_for

saft = package_for(("methanol", "n-hexane"), "pcsaft")   # methanol uses the 2B scheme
x = jnp.array([0.4, 0.6])

bubble = saft.bubble_pressure_with_info(320.0, x)
bubble.report.converged                                  # True
p, y = bubble.value                                      # about 77.7 kPa, incipient vapor
h = saft.enthalpy(320.0, p, x, phase="liquid")           # J/mol, association included
```

The package carries no curated `k_ij` for this pair, so it's a prediction from
the pure-component parameters.

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

## Consistency checks

`fugacio.thermo.consistency` has two data-free checks that tie a package's
energy side to its equilibrium side. Each returns a relative residual that
should be zero; neither raises.

* `gibbs_helmholtz_residual(pkg, t, p, x, *, phase)` checks
  \(H = -T^2 \left(\partial (G/T) / \partial T\right)_P\) on one branch, from
  the package's own enthalpy and entropy.
* `fugacity_enthalpy_residual(pkg, t, p, x)` checks
  \(H^V - H^L = -R T^2 \, \partial / \partial T \sum_i x_i (\ln\phi_i^V - \ln\phi_i^L)\),
  which ties the fugacity coefficients that decide the phase split to the
  enthalpies that close energy balances. Evaluate it where both branches exist.

Cubic, PC-SAFT, and reference-fluid packages satisfy both to about \(10^{-10}\)
or better. A gamma-phi package built with `poynting=True` and
`phi_saturation=True` satisfies them to a few parts per million. The default
gamma-phi package, with neither correction, deviates by about 1% in
`fugacity_enthalpy_residual`, because its liquid fugacity and liquid enthalpy
use different reference states. When energy balances matter, build the package
with both corrections:

```python
from fugacio.thermo.consistency import fugacity_enthalpy_residual

nrtl_hx = package_for(("ethanol", "water"), "nrtl", poynting=True, phi_saturation=True)
fugacity_enthalpy_residual(nrtl_hx, 350.0, 1.013e5, jnp.array([0.4, 0.6]))
```

## Everything takes `model=`

The unit operations in `fugacio.sim.units` (`flash_drum`, `adiabatic_flash`,
`heater`, `valve`, `pump`, `compressor`, `turbine`, and `mix`), the stream
property helpers (`molar_enthalpy`, `enthalpy_flow`, `vapor_density`,
`column_diameter_for`, and the rest), the reactors, the rigorous column, the
two-sided `heat_exchanger`, and `EOFlowsheet` all accept `model=`. Omitting it
selects Peng-Robinson over the stream's components. `Flowsheet(model=...)`
audits every solved stream on that package. The liquid-liquid separators,
`decanter` and `three_phase_flash`, take a gamma-phi package as a positional
argument.

```python
from fugacio.sim import Stream, heater

feed = Stream.from_fractions(("ethanol", "water"), jnp.array([0.1, 0.9]), 100.0, 300.0, 1.5e5)
warm = heater(feed, t_out=350.0, model=nrtl)        # NRTL liquid enthalpy, heat of mixing included
warm.outlet, warm.duty
```

A package is a registered JAX pytree whose model parameters are leaves, so a
flowsheet built on any of them is differentiable with respect to the
thermodynamic parameters as well as the operating conditions: `jax.grad` of a
column duty with respect to a `kij` or an NRTL \(\tau_{ij}\) works the same way
as a gradient with respect to a reflux ratio.

## 64-bit arithmetic is on by default

Cubic-root selection, Newton iterations to \(10^{-10}\) residuals, and the
implicit-function-theorem gradients throughout Fugacio assume double precision.
Importing `fugacio.thermo` enables `jax_enable_x64` automatically. Set the
environment variable `FUGACIO_X64=0` before importing if you deliberately want
single precision and accept the loss of digits.
