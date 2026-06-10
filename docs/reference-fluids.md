# Reference fluids & steam tables

Cubic equations of state are fast and general, but when a design lives or dies
on water, CO₂, ammonia, or a refrigerant, engineers reach for *reference*
equations of state — the multiparameter Helmholtz formulations behind NIST
REFPROP and CoolProp. `fugacio.thermo.helmholtz` implements that model class
natively in JAX: **IAPWS-95** for water/steam, **Span–Wagner** for CO₂,
**Setzmann–Wagner** for methane, and the recommended formulations for 23 more
process fluids (`reference_fluid_names()` lists them — light hydrocarbons
through n-octane, H₂/N₂/O₂/Ar/CO, H₂S, SO₂, ammonia, ethanol, benzene, toluene,
R134a, R32, R1234yf).

## One scalar function, every property, by autodiff

A multiparameter EOS is a single scalar field: the reduced Helmholtz energy
`α(δ, τ) = α⁰(δ, τ) + αʳ(δ, τ)` in reduced density and inverse reduced
temperature. Every measurable property is an algebraic combination of its
partial derivatives. Reference implementations hand-derive those derivatives
term family by term family — hundreds of lines of error-prone calculus per
fluid. Fugacio stores only the published coefficient tables and evaluates the
scalar `α`; **every derivative is `jax.grad`**, including the third-order ones
inside solver Jacobians:

```python
from fugacio.thermo import reference_fluid
from fugacio.thermo.helmholtz import pressure, isobaric_heat_capacity, speed_of_sound

water = reference_fluid("water")          # frozen dataclass, a JAX pytree
rho, t = 838.025 / water.molar_mass, 500.0    # mol/m^3, K

pressure(water, rho, t)                   # 10.0003858 MPa  (IAPWS-95 check value)
isobaric_heat_capacity(water, rho, t)     # J/mol/K
speed_of_sound(water, rho, t)             # 1271.28 m/s
```

The hermetic test suite pins the IAPWS-95 release's printed `α` derivatives and
single-phase pressures, the IAPWS viscosity/conductivity check tables, and
textbook steam-table anchors; the opt-in oracle suite grades dense grids for
all 26 fluids against CoolProp at ~1e-9 relative — two independent
implementations of the same published equations agreeing to solver precision.

## Saturation by Maxwell construction — differentiable

Coexistence is solved as equal pressure and equal Gibbs energy in
`(ln δ', ln δ'')` with a damped Newton, seeded by the published ancillary
equations and wrapped in an implicit-function-theorem `custom_vjp`. The solved
saturation line is therefore *exactly differentiable*:

```python
import jax
from fugacio.thermo import reference_fluid, saturation_state
from fugacio.thermo.helmholtz import saturation_pressure

water = reference_fluid("water")
sat = saturation_state(water, t=450.0)    # p, rho', rho'', h', h'', s', s'', Δh_vap

# Clausius-Clapeyron, both sides computed independently:
dp_dt = jax.grad(lambda t: saturation_pressure(water, t))(450.0)
dv = 1.0 / sat.rho_vapor - 1.0 / sat.rho_liquid
dp_dt, sat.h_vaporization / (450.0 * dv)  # agree to ~1e-12 relative
```

That gradient flows through the EOS coefficients too — `d(psat)/d(n_k)` for a
published correlation coefficient is one `jax.grad` away, which is what
sensitivity analysis and EOS refitting need.

## Steam-table state functions

Process specifications arrive as `(T, P)`, `(P, h)`, `(P, s)`, or a quality —
not as the `(ρ, T)` a Helmholtz EOS natively speaks. The `state_*` family
resolves them with two-phase dome handling (`q`, mixture properties, `nan`
heat capacities where they are undefined), and stays differentiable through
every embedded solve:

```python
from fugacio.thermo import reference_fluid, state_tp, state_ph, state_ps, state_pq

water = reference_fluid("water")
inlet = state_tp(water, 723.15, 40e5)         # superheated steam, auto phase pick
outlet = state_ps(water, 1e5, inlet.s)        # isentropic expansion -> wet steam
outlet.two_phase, outlet.q                    # True, 0.9306 (turbine exhaust wetness)
```

The solver-backed functions are jit-compiled with the fluid as a pytree
argument — the first call per fluid pays a one-time compilation, after which
calls cost microseconds and compose with `jit`, `vmap`, and `grad`.

## IAPWS transport with autodiff critical enhancements

Water carries the full IAPWS formulations for viscosity (R12-08) and thermal
conductivity (R15-11) — including the **critical enhancement** terms that other
open transcriptions make the caller parameterize or approximate, because they
need `(∂ρ/∂P)_T` at the state *and* at 1.5 T_c. Here those are exact autodiff
compressibilities of IAPWS-95, so the scientific formulation evaluates
everywhere, on one differentiable graph:

```python
from fugacio.thermo import water_viscosity, water_thermal_conductivity

rho = 322.0 / 0.018015268                      # critical density, mol/m^3
water_viscosity(647.35, rho)                   # 42.96 µPa·s (IAPWS check value)
water_thermal_conductivity(647.35, rho)        # 1.4438 W/m/K (enhancement peak)
```

Surface tension uses each fluid's recommended `σ(T)` correlation
(`reference_surface_tension`, IAPWS/Mulero family).

## Steam & cooling-water utilities (`fugacio.sim`)

The simulation layer turns duties into utility balances on real IAPWS-95
water — real latent heat at the header pressure, real liquid enthalpies, real
isentropic enthalpy drops — all differentiable for utility-system optimization:

```python
from fugacio.sim import (
    STEAM_LEVELS, steam_heating, cooling_water, steam_turbine,
    steam_quality_after_letdown, condensate_flash_fraction,
)

steam_heating(2.5e6, pressure=STEAM_LEVELS["mp"]).mass_flow   # reboiler steam, kg/s
cooling_water(3.2e6).mass_flow                                # condenser CW, kg/s
steam_turbine(10.0, p_in=40e5, t_in=723.15, p_out=1e5).power  # W of shaft work
condensate_flash_fraction(42e5, 5e5)                          # 21.9 % flash steam
```

The copilot exposes the same capabilities as JSON tools: `steam_state`
(steam-table lookups by `P` + one of `T`/`q`/`h`/`s`), `reference_fluid_state`,
`reference_saturation`, `steam_utility_requirements` (physical sizing + priced
annual cost), and `steam_turbine`.

## Provenance & regeneration

Coefficient tables are vendored into
`fugacio/thermo/helmholtz/_data.py` by `scripts/gen_helmholtz.py`, which
extracts and normalizes them from CoolProp's JSON fluid library (BibTeX keys of
the original publications are kept per fluid). The generator is run by hand;
the vendored data is deterministic and ships with the package, so runtime needs
no CoolProp.
