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

## Property packages

Every energy-balanced unit, the rigorous column, the two-sided heat exchanger,
and the equation-oriented engine take a `model=` argument: a
`fugacio.thermo.PropertyPackage` built with `package_for(components, method)`
for any method in `METHODS` (`"pr"`, `"srk"`, `"rk"`, `"vdw"`, `"nrtl"`,
`"uniquac"`, `"unifac"`, `"dortmund"`, `"pcsaft"`, `"iapws"`). Omitting it keeps
the Peng-Robinson default.

## Unit operations (rigorous material + energy balances)

- `flash_drum`: isothermal-isobaric vapour/liquid separator.
- `heater`: heater/cooler on a temperature **or** a duty specification.
- `valve`: isenthalpic (Joule-Thomson) pressure letdown.
- `pump`: incompressible-liquid pump with an efficiency.
- `compressor` / `turbine`: isentropic machines with an efficiency.
- `mix`: adiabatic, energy-balanced mixer (exact material balance).
- `splitter` / `component_separator`: flow split and idealised component split.
- `heat_exchanger`: two-sided countercurrent (or parallel) exchanger with
  rigorous T-Q curves on both sides, zone-wise LMTD, and one closing spec
  (`duty`, `t_hot_out`, `t_cold_out`, `min_approach`, or `ua`); each side may
  use its own property package.
- `bubble_pressure` / `antoine_psat`: lightweight modified-Raoult helpers.

## Flowsheets with recycle

`tear_solve` closes a recycle by solving the tear fixed point
`tear = g(tear, theta)` by Wegstein acceleration, Broyden, or full Newton
(`method=`), and differentiates the *converged* flowsheet by the implicit
function theorem: a gradient through the recycle costs one adjoint solve
regardless of iteration count. `Flowsheet` is the declarative builder on top of
it: register feeds and units in any order and `partition` finds the strongly
connected blocks (Tarjan), orders them, and selects the tear streams; `solve`
converges every loop and returns all named streams, differentiable in `theta`.

## Equation-oriented flowsheeting

`fugacio.sim.eo` solves a whole flowsheet as one system of equations instead of
unit by unit. `EOFlowsheet` assembles every block's residual equations, the
stream connectivity, the recycles, and any design specs into a single residual
system and solves it simultaneously by Newton's method, with the Jacobian
supplied exactly by JAX autodiff. A recycle needs no tear stream and no ordering,
the converged plant is differentiable by the implicit function theorem,
`degrees_of_freedom` checks the unknown/equation balance, and
`optimize_flowsheet_eo` runs nested or full-space simultaneous optimization. The
blocks mirror the sequential-modular units (`Mixer`, `Splitter`, `Heater`,
`Valve`, `Pump`, `Compressor`, `Turbine`, `Flash`, `ComponentSeparator`), plus
`HeatExchanger`, `StoichiometricReactor`, and `Column` (an embedded rigorous
MESH column), so the two engines agree on any flowsheet both can express.

```python
import jax.numpy as jnp
from fugacio.sim import Stream
from fugacio.sim.eo import EOFlowsheet, Mixer, Flash, Splitter

fresh = Stream.from_fractions(
    ("methane", "propane", "n-pentane"), jnp.array([0.5, 0.3, 0.2]), 100.0, 320.0, 20e5
)

fs = (
    EOFlowsheet()
    .feed("fresh", fresh)
    .add(Mixer(inlets=("fresh", "recycle"), outlets=("mixed",), t=320.0))
    .add(Flash(inlets=("mixed",), outlets=("vapor", "liquid"), t="T", p="P"))
    .add(Splitter(inlets=("liquid",), outlets=("recycle", "purge"), fractions="r"))
)
sol = fs.solve({"T": 320.0, "P": 20e5, "r": jnp.array([0.5, 0.5])})  # recycle closed, no tear
sol["vapor"].total
```

## Distillation

- **Shortcut** (Fenske-Underwood-Gilliland): `fenske_min_stages`,
  `underwood_min_reflux`, `gilliland_stages`, `kirkbride_feed_stage`, and the
  `shortcut_column` wrapper.
- **Rigorous MESH** `rigorous_column`: simultaneous-correction
  (Naphtali-Sandholm) column with full stage energy balances on any property
  package; multiple feeds (`ColumnFeed`), side draws, stage duties, a pressure
  profile, Murphree efficiency, total/partial/no condenser, kettle/no reboiler,
  and design specs as equations (`reflux_ratio`, `distillate_rate`,
  `bottoms_rate`, `boilup_ratio`, `condenser_duty`, `purity`, `recovery`,
  `component_flow`, `stage_temperature`). `absorber` and `stripper` wrap it.
- **Constant molar overflow** `solve_column`: the lighter Wang-Henke
  bubble-point column, kept for quick estimates.

## Non-ideal separations & diagrams

Built on the `fugacio.thermo` property system (via the `eos_model_for`,
`nrtl_model_for`, `uniquac_model_for`, `unifac_model_for`, and `saft_model_for`
bridges, the last building a molecular PC-SAFT model from component names):

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
from fugacio.sim import ColumnFeed, Stream, distillate_rate, purity, reflux_ratio, rigorous_column

feed = Stream.from_fractions(("propane", "n-butane"), jnp.array([0.5, 0.5]), 100.0, 320.0, 10e5)
col = rigorous_column([ColumnFeed(feed, 6)], 12, p=10e5,
                      specs=[reflux_ratio(2.0), distillate_rate(50.0)])
col.distillate.z                     # propane overhead
col.reboiler_duty, col.t             # duty from the stage energy balances; T profile

# Impose the purity instead and ask for the reflux it takes, and its energy cost:
def reboiler_duty(x_target):
    res = rigorous_column([ColumnFeed(feed, 6)], 12, p=10e5,
                          specs=[distillate_rate(50.0), purity("distillate", 0, x_target)])
    return res.reboiler_duty

jax.grad(reboiler_duty)(0.97)        # W per unit mole fraction, exact
```

Part of the `fugacio` namespace; installs independently:
`pip install fugacio-sim`.
