# fugacio-sim

Differentiable process-simulation layer for the
[Fugacio](https://github.com/fugacio/fugacio) stack: flowsheet and
unit-operation models built on top of `fugacio.thermo`.

The core abstraction is the `Stream`, a JAX pytree whose molar flows,
temperature, and pressure are differentiable leaves (component names are static
metadata). Because the underlying EOS phase equilibrium is differentiable, unit
operations are too: you can take a gradient of any downstream quantity (a product
flow, a recovery, a purity) with respect to feed conditions or operating
variables, which is the basis for gradient-based flowsheet optimisation.

## Stream properties

Any `Stream` has a two-phase-aware enthalpy and entropy (via the
`fugacio.thermo` energy core), so unit operations close *energy* balances, not
just material balances: `molar_enthalpy`, `molar_entropy`, `enthalpy_flow`,
`entropy_flow`, `mass_flow`, `molar_mass`.

## Unit operations (rigorous material + energy balances)

- `flash_drum`: isothermal-isobaric vapour/liquid separator.
- `heater`: heater/cooler on a temperature **or** a duty specification.
- `valve`: isenthalpic (Joule-Thomson) pressure letdown.
- `pump`: incompressible-liquid pump with an efficiency.
- `compressor` / `turbine`: isentropic machines with an efficiency.
- `mix`: adiabatic, energy-balanced mixer (exact material balance).
- `splitter` / `component_separator`: flow split and idealised component split.
- `bubble_pressure` / `antoine_psat`: lightweight modified-Raoult helpers.

## Flowsheets with recycle

`tear_solve` closes a recycle by solving the tear fixed point
`tear = g(tear, theta)` with a Wegstein-accelerated iteration, and
differentiates the *converged* flowsheet by the implicit function theorem: a
gradient through the recycle costs one adjoint solve regardless of iteration
count. `Flowsheet` is a small declarative builder on top of it.

## Distillation

- **Shortcut** (Fenske-Underwood-Gilliland): `fenske_min_stages`,
  `underwood_min_reflux`, `gilliland_stages`, `kirkbride_feed_stage`, and the
  `shortcut_column` wrapper.
- **Rigorous** `solve_column`: a multistage equilibrium-stage column (Wang-Henke
  bubble-point, constant molar overflow) with EOS K-values on every stage,
  differentiable through the fixed-point iteration.

## Non-ideal separations & diagrams

Built on the `fugacio.thermo` γ–φ property system (via the `eos_model_for`,
`nrtl_model_for`, `uniquac_model_for`, `unifac_model_for` bridges):

- `flash_vle`, `decanter`, `three_phase_flash`: activity-based VLE / LLE / VLLE
  drums for real, non-ideal mixtures.
- `pxy_diagram`, `txy_diagram`, `azeotrope_pressure`, `azeotrope_temperature`:
  binary phase diagrams and azeotrope finders.
- `residue_curve`, `residue_curve_map`: ternary open-evaporation trajectories for
  laying out distillation boundaries.

## Reactors

Energy-balanced reactor unit operations over one or more `fugacio.thermo`
`Reaction`s, each runnable isothermally (reporting the heat `duty`) or
adiabatically (solving the outlet temperature) and returning a `ReactorResult`:
`equilibrium_reactor` (chemical equilibrium), `stoichiometric_reactor` (specified
extent or conversion), and kinetic `cstr`, `pfr`, and `batch_reactor` sized by
volume (and time). `conversion` is a small helper on the inlet/outlet streams.

## Reactive separations

Reaction coupled to phase separation, both differentiable through the joint solve:
`reactive_flash` (simultaneous chemical + vapour-liquid equilibrium in a drum) and
`reactive_distillation` (a rate-based column with per-stage reaction source terms).

## Example: differentiate a flash drum

```python
import jax
import jax.numpy as jnp
from fugacio.sim import Stream, flash_drum

feed = Stream.from_fractions(
    ("methane", "propane", "n-pentane"),
    jnp.array([0.5, 0.3, 0.2]),
    flow=100.0, t=320.0, p=20e5,
)
vapor, liquid = flash_drum(feed, 320.0, 20e5)
vapor.total, liquid.total  # ~74.7 and ~25.3 mol/s

# Sensitivity of vapour product flow to drum temperature:
d_vapor_dT = jax.grad(lambda T: flash_drum(feed, T, 20e5)[0].total)
d_vapor_dT(320.0)
```

## Example: a recycle, differentiated end-to-end

```python
import jax.numpy as jnp
from fugacio.sim import Stream, flash_drum, mix, splitter, tear_solve

components = ("methane", "propane", "n-pentane")
fresh = Stream.from_fractions(components, jnp.array([0.5, 0.3, 0.2]), 100.0, 320.0, 20e5)

def one_pass(recycle, theta):
    mixed = mix([fresh, recycle], t=320.0)
    _vapor, liquid = flash_drum(mixed, theta["T"], theta["P"])
    recycled, _purge = splitter(liquid, jnp.array([theta["r"], 1.0 - theta["r"]]))
    return recycled

guess = Stream.from_fractions(components, jnp.array([0.1, 0.3, 0.6]), 30.0, 320.0, 20e5)
recycle = tear_solve(one_pass, guess, {"T": 320.0, "P": 20e5, "r": 0.5})
```

## Example: a rigorous distillation column

```python
import jax
import jax.numpy as jnp
from fugacio.sim import Stream, solve_column

feed = Stream.from_fractions(("propane", "n-butane"), jnp.array([0.5, 0.5]), 100.0, 320.0, 10e5)
col = solve_column(feed, n_stages=12, feed_stage=6, reflux=2.0, distillate_rate=50.0)
col.distillate.z  # ~[0.97, 0.03] propane overhead

# Exact gradient of distillate purity w.r.t. reflux ratio:
jax.grad(
    lambda r: solve_column(feed, 12, 6, r, 50.0).distillate.z[0]
)(2.0)
```

Part of the `fugacio` namespace; installs independently:
`pip install fugacio-sim`.
