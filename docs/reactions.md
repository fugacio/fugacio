# Reactions & reactors

Fugacio models chemical reactions end to end: standard-state thermochemistry and
the equilibrium constant `K(T)`, chemical-equilibrium composition, reaction
kinetics, ideal reactor unit operations, and reactive separations. Like the rest
of the stack everything is written in JAX, so conversions, yields, and duties are
differentiable with respect to temperature, pressure, feed, *and* the underlying
thermochemical / kinetic parameters.

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
activity quotient equal to its `K(T)`. A single reaction is solved by a robust
bracketed root; several simultaneous reactions by a damped Newton system. Both
differentiate the converged composition with respect to `T`, `P`, and the feed by
the implicit function theorem.

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

The `fugacio.sim` layer turns reactions into energy-balanced unit operations on a
differentiable `Stream`. Every reactor accepts one or more reactions and runs
either *isothermal* (reporting the heat `duty` to hold `t_out`) or *adiabatic*
(`adiabatic=True`, solving for the outlet temperature). All return a
`ReactorResult` with `outlet`, `duty`, and `extent`.

| Unit | Model |
| --- | --- |
| `equilibrium_reactor` | Outlet at chemical equilibrium (Gibbs / `K(T)`). |
| `stoichiometric_reactor` | Specified `extent` **or** fractional `conversion`. |
| `cstr` | Continuous stirred tank, kinetics balanced over a `volume`. |
| `pfr` | Plug-flow tubular reactor (RK4 integration along the volume). |
| `batch_reactor` | Constant-volume batch over a reaction `time`. |

```python
import jax.numpy as jnp
from fugacio.sim import Stream, equilibrium_reactor, cstr, conversion
from fugacio.thermo import PowerLaw, Reaction

feed = Stream.from_fractions(
    ("nitrogen", "hydrogen", "ammonia"),
    jnp.array([0.25, 0.75, 0.0]),
    flow=100.0, t=700.0, p=100e5,
)
rxn = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", feed.components)

# Isothermal equilibrium reactor: outlet composition + the cooling duty.
eq = equilibrium_reactor(feed, rxn, t_out=700.0)
eq.outlet.z, eq.duty

# Adiabatic CSTR sized by volume, with a power-law rate:
law = PowerLaw(a=jnp.asarray(5.0e3), ea=jnp.asarray(40_000.0), orders=jnp.array([1.0, 1.0, 0.0]))
out = cstr(feed, rxn, law, volume=10.0, adiabatic=True)
conversion(feed, out.outlet, 0)   # fractional N2 conversion
```

## Reactive separations

When reaction and phase separation happen together, use the `fugacio.sim`
reactive units, which couple kinetics / chemical equilibrium to a `GammaPhiModel`:

- `reactive_flash`: simultaneous chemical *and* vapour-liquid equilibrium in a
  single drum (liquid-activity reaction quotient), returning vapour/liquid
  products, the vapour fraction `beta`, and the extents.
- `reactive_distillation`: a rate-based column that adds per-stage reaction source
  terms (kinetics × molar holdup) to the Wang-Henke mass balances, returning the
  stage profiles, products, and the net `generation` on every stage.

These make classic reaction-separation processes (e.g. esterification with in-situ
water removal) tractable while staying differentiable through the coupled solve.

## Validation

Reaction thermochemistry and equilibrium are cross-checked against
[Cantera](https://github.com/Cantera/cantera) in the opt-in oracle suite
(`just oracles`). The oracle builds a Cantera ideal-gas phase from Fugacio's *own*
formation and `Cp` data, so `DG_rxn`, `K(T)`, and the equilibrium composition agree
to (near) machine precision and any discrepancy isolates the temperature
integration or the equilibrium solver rather than a difference in reference data.
