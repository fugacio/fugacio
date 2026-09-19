# Reactions & reactors

Fugacio models chemical reactions end to end: standard-state thermochemistry and
the equilibrium constant `K(T)`, chemical-equilibrium composition, reaction
kinetics, reactor unit operations, and reactive separations. Like the rest
of the stack everything is written in JAX, so conversions, yields, and duties are
differentiable with respect to temperature, pressure, feed, *and* the underlying
thermochemical / kinetic parameters.

For reaction sets, reactive MESH columns, saved cases, and checked design
studies, start with [reactive process workflows](reactive-workflows.md). This
page covers the thermochemistry, the gas-phase equilibrium solver, rate laws,
and the reactor unit operations.

## Stoichiometry & thermochemistry

A `Reaction` is a stoichiometric vector over an ordered component list. Build one
from an equation string or from reactant/product maps:

```python
from fugacio.thermo import Reaction, reaction_properties

components = ("nitrogen", "hydrogen", "ammonia")
rxn = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", components)

props = reaction_properties(rxn, 298.15)
props.delta_h   # standard enthalpy of reaction DH_rxn(T)     (J/mol)
props.delta_g   # standard Gibbs energy of reaction DG_rxn(T) (J/mol)
props.k         # equilibrium constant K(T) = exp(-DG_rxn / R T)
```

`DH_rxn`, `DS_rxn`, `DG_rxn`, and `K(T)` follow from each component's ideal-gas
standard formation properties (`hform_ig`, `gform_ig`) corrected to temperature
with Kirchhoff's law (integrating the ideal-gas `Cp` correlations). The standard
state is the ideal gas at `P_REF` (1 bar), matching the tabulated formation data.
The component-level entry points are `delta_h_rxn`, `delta_s_rxn`, `delta_g_rxn`,
and `equilibrium_constant`.

## Chemical-reaction equilibrium

`equilibrium` solves for the extents of reaction that make every reaction's
activity quotient equal to its `K(T)`. A single reaction is solved by a checked
bracketed root; several simultaneous reactions by a damped Newton system. Both
differentiate the converged composition with respect to `T`, `P`, and the feed by
the implicit function theorem. The result's `report` records convergence. A
failed solve returns NaN fields, and a composition with a negative amount is
reported as `INFEASIBLE`.

```python
import jax
import jax.numpy as jnp
from fugacio.thermo import Reaction
from fugacio.thermo.reaction_equilibrium import equilibrium

components = ("nitrogen", "hydrogen", "ammonia")
rxn = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", components)
feed = jnp.array([1.0, 3.0, 0.0])

res = equilibrium(rxn, feed, 700.0, 100e5)
res.y          # equilibrium mole fractions
res.extent     # extent of each reaction
res.report     # solve report; the fields are NaN if it failed

# Le Chatelier, exactly: ammonia yield rises with pressure (Delta_n = -2).
jax.grad(lambda p: equilibrium(rxn, feed, 700.0, p).y[2])(100e5)  # > 0
```

For a real gas pass `basis="phi"` with `tc`, `pc`, `omega` (and optional `kij`) to
use cubic-EOS fugacity coefficients in the activities instead of the ideal-gas
`a_i = y_i P / P_ref`.

## Kinetics

Rate laws are differentiable pytrees, so their parameters are gradient leaves
(handy for fitting): `PowerLaw` (Arrhenius pre-exponential, activation energy, and
per-component orders), `MassActionReversible`, and `LHHW` (Langmuir-Hinshelwood).
The temperature dependence is the `Arrhenius` form (`arrhenius`, `arrhenius_ref`).

```python
import jax.numpy as jnp
from fugacio.thermo import PowerLaw

# First-order in A: rate = A exp(-Ea/RT) * c_A
law = PowerLaw(a=jnp.asarray(1.0e7), ea=jnp.asarray(75_000.0), orders=jnp.array([1.0, 0.0]))
```

## Reactors

The `fugacio.sim` reactors turn reactions into unit operations on a
differentiable `Stream`. Their energy balances use the property package's
enthalpy plus the formation enthalpies, so the heat of reaction is carried
automatically. `model=` selects the package; the default is Peng-Robinson over
the feed's components. Each reactor takes one energy specification: `t_out` for
an isothermal outlet (the result reports the `duty` that holds it) or `duty` for
a specified heat input, where `duty=0.0` is adiabatic and the outlet
temperature is solved.

| Unit | Model | Result |
| --- | --- | --- |
| `equilibrium_reactor` | Outlet at chemical equilibrium: `K(T)` from formation data, with activities from the package's fugacities | `ReactionResult` |
| `stoichiometric_reactor` | Specified `extent` or fractional `conversion` | `StoichiometricResult` |
| `cstr` | Continuous stirred tank, kinetics balanced over a reacting `volume` | `ReactionResult` |
| `pfr` | Plug flow, RK4 along the volume with step-doubling error control | `ReactionResult` |
| `batch_reactor` | Closed, constant-volume ideal-gas batch over a reaction `time` | `BatchResult` |

```python
import jax.numpy as jnp
from fugacio.sim import (
    ReactionSet, ReferenceRate, Stream, conversion, cstr, equilibrium_reactor,
)
from fugacio.thermo import Reaction

feed = Stream.from_fractions(
    ("nitrogen", "hydrogen", "ammonia"),
    jnp.array([0.25, 0.75, 0.0]),
    flow=100.0, t=700.0, p=100e5,
)
rxn = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", feed.components)

# Isothermal equilibrium reactor: outlet composition and the heat to hold 700 K.
eq = equilibrium_reactor(feed, rxn, t_out=700.0)
eq.outlet.z, eq.duty             # about [0.198, 0.595, 0.206] and -915 kW

# Adiabatic (duty=0): the outlet temperature is solved with the equilibrium.
hot = equilibrium_reactor(feed, rxn, duty=0.0)
hot.outlet.t                     # about 820 K

# A CSTR needs kinetics consistent with K(T). Detailed balance derives the
# reverse rate from the equilibrium constant, on fugacity activities.
law = ReferenceRate(
    k_forward=jnp.asarray(1e-7), ea_forward=jnp.asarray(80_000.0),
    forward_orders=jnp.array([1.0, 3.0, 0.0]),
    k_reverse=jnp.asarray(0.0), ea_reverse=jnp.asarray(0.0),
    reverse_orders=jnp.array([0.0, 0.0, 2.0]),
    reference_temperature=jnp.asarray(700.0), detailed_balance=True,
)
system = ReactionSet.from_reactions(rxn, [law], phase="vapor", rate_basis="activity")
out = cstr(feed, system, 10.0, t_out=700.0)    # 10 m^3 of reacting vapor
conversion(feed, out.outlet, 0)                # about 0.215 fractional N2 conversion
```

The CSTR stays below the 700 K equilibrium conversion (about 0.34) because its
volume limits the reaction. The kinetic coefficients here are illustrative.

An eager reactor solve that fails raises `ConvergenceError`. A traced one can't
raise; it records the failure in its `report` and has nonfinite derivatives
(see the [reliability guide](reliability.md#the-failure-contract)). An
irreversible `PowerLaw` has no reverse term, so it can't respect `K(T)` for
this equilibrium-limited reaction, and an adiabatic CSTR on one stalls. It
raises `ConvergenceError` rather than returning negative flows:

```python
from fugacio.thermo import PowerLaw

irreversible = PowerLaw(
    a=jnp.asarray(5.0e3), ea=jnp.asarray(40_000.0), orders=jnp.array([1.0, 1.0, 0.0])
)
cstr(feed, rxn, 10.0, irreversible, duty=0.0)   # ConvergenceError: reactor failed: stalled ...
```

With `check=False`, the result keeps the failed report and its best iterate for
diagnosis, with nonfinite derivatives. See
[reactive process workflows](reactive-workflows.md#common-package-reactor-api)
for the full `ReactionResult`, including balances and axial profiles.

`stoichiometric_reactor` applies a specified `extent` or key-reactant
`conversion` exactly and raises `ValueError` for an extent that consumes more of
a reactant than the feed holds. `batch_reactor` treats `feed.n` as initial moles
in a rigid, closed vessel and returns `BatchResult(contents, heat, extent)`.
With `adiabatic=True` it conserves internal energy, not enthalpy; otherwise it
holds the temperature and returns the heat `Q = Delta U`. The final pressure of
`contents` is the ideal-gas `N R T / V`.

## Reactive separations

When reaction and phase separation happen together, use the `fugacio.sim`
reactive units. Both take a property package:

- `reactive_flash`: simultaneous chemical *and* vapor-liquid equilibrium in a
  single drum, returning vapor and liquid products, the vapor fraction `beta`,
  the extents, and the duty. The reaction quotient uses the liquid fugacity
  when a liquid is present, otherwise the vapor fugacity.
- `reactive_column`: material- and energy-balanced MESH with volumetric kinetics.
  It delegates to `rigorous_column(..., reactions=system, reaction_volumes=...)`,
  retaining the structured column solver and differentiable reaction parameters.

These make classic reaction-separation processes (for example, esterification
with in-situ water removal) tractable while staying differentiable through the
coupled solve.

## Validation

Reaction thermochemistry and equilibrium are cross-checked against
[Cantera](https://github.com/Cantera/cantera) in the opt-in oracle suite
(`just oracles`). The oracle builds a Cantera ideal-gas phase from Fugacio's *own*
formation and `Cp` data, so `DG_rxn`, `K(T)`, and the equilibrium composition agree
to (near) machine precision and any discrepancy isolates the temperature
integration or the equilibrium solver rather than a difference in reference data.
