# Rigorous distillation

`fugacio.sim.rigorous_column` is a simultaneous-correction (Naphtali-Sandholm)
equilibrium-stage column. It solves the full MESH equations of every stage
(component **M**aterial balances, phase **E**quilibrium, mole-fraction
**S**ummation, stage entha**H**py balance) together with the condenser and
reboiler and any design specifications as one Newton system, with the Jacobian
from JAX autodiff and the converged column differentiable in every input by the
implicit function theorem.

That replaces the earlier constant-molar-overflow column (`solve_column`, which
stays available) with the model a process simulator actually needs:

* full stage energy balances, so vapor and liquid traffic vary down the column
  and the reboiler and condenser duties are consistent with the property
  package rather than with an assumed latent heat;
* any [property package](property-packages.md): a cubic EOS for hydrocarbons,
  NRTL or UNIFAC for azeotropic systems, PC-SAFT for associating mixtures;
* multiple feeds and side draws at arbitrary stages, intermediate heaters and
  coolers, a pressure profile, and a Murphree efficiency;
* total or partial condensers, a kettle reboiler, or neither (absorbers and
  strippers);
* design specifications imposed *directly* as equations: reflux ratio,
  distillate or bottoms rate, a product purity or recovery, a component flow, a
  stage temperature, or a duty, in any combination of two (one per free degree
  of freedom).

## A benzene/toluene column

```python
import jax.numpy as jnp
from fugacio.sim import ColumnFeed, Stream, reflux_ratio, rigorous_column, recovery

feed = Stream.from_fractions(
    ("benzene", "toluene"), jnp.array([0.5, 0.5]), flow=100.0, t=365.0, p=1.013e5
)
col = rigorous_column(
    [ColumnFeed(feed, stage=6)],   # stages count from the top; stage 1 is the condenser
    n_stages=12,
    p=1.013e5,
    specs=[reflux_ratio(2.5), recovery(0, "distillate", 0.99)],
)

col.distillate.z, col.bottoms.z     # product compositions
col.t                               # stage temperature profile (K)
col.liquid_flow, col.vapor_flow     # internal traffic (mol/s)
col.condenser_duty, col.reboiler_duty
col.residual_norm                   # scaled max residual at the solution (~1e-10)
```

Stages are numbered from the top: with `condenser="total"` (the default) stage
1 is the condenser and stage `n_stages` is the reboiler. A `"partial"`
condenser returns a vapor distillate in equilibrium with the reflux; `None`
removes the condenser (or reboiler) altogether so the top (or bottom) stage is
an ordinary equilibrium stage with an external feed.

## Specifications

Each condenser and reboiler contributes one degree of freedom; a column with
both needs exactly two `specs`. Build them with the helper constructors, all of
which are exported from `fugacio.sim`:

| Constructor | Fixes |
| --- | --- |
| `reflux_ratio(r)`, `reflux_rate(L)` | reflux ratio \(L_1/D\) or reflux flow |
| `distillate_rate(D)`, `bottoms_rate(B)` | a product molar flow |
| `boilup_ratio(v)` | \(V_{N-1}/B\) |
| `condenser_duty(q)` | the condenser duty |
| `purity(product, i, x)` | mole fraction of component *i* in a product |
| `recovery(i, product, f)` | fraction of fed component *i* leaving in a product |
| `component_flow(product, i, n)` | molar flow of component *i* in a product |
| `stage_temperature(j, t)` | the temperature of stage *j* |

Specifications are equations of the global system, not an outer iteration
around a reflux-and-boilup column. A purity target therefore converges in the
same Newton solve as the balances and costs nothing extra, and the resulting
reflux ratio and duties are reported in the result. Because the spec values may
be JAX tracers, they're valid parameters to differentiate with respect to:

```python
import jax

def reboiler_duty(purity_target):
    res = rigorous_column(
        [ColumnFeed(feed, 6)], 12, p=1.013e5,
        specs=[distillate_rate(50.0), purity("distillate", 0, purity_target)],
    )
    return res.reboiler_duty

jax.grad(reboiler_duty)(0.95)     # marginal energy cost of one more mole-percent purity
```

## Feeds, side draws, stage duties, efficiency

```python
from fugacio.sim import SideDraw, StageDuty

col = rigorous_column(
    [ColumnFeed(light_feed, 4), ColumnFeed(heavy_feed, 9)],
    n_stages=16,
    p_top=1.0e5, p_bottom=1.2e5,                # linear pressure profile
    side_draws=[SideDraw(stage=7, phase="liquid", fraction=0.1)],
    stage_duties=[StageDuty(stage=11, duty=+50e3)],   # intercondenser/interreboiler (W)
    efficiency=0.8,                             # Murphree vapor efficiency on every stage
    specs=[reflux_ratio(3.0), bottoms_rate(60.0)],
    model=package_for(comps, "srk"),
)
col.side_draws[0]                               # the drawn stream
```

A `SideDraw` removes a `fraction` of the stage's liquid or vapor leaving flow.
`StageDuty` adds heat (positive) or removes it (negative) on an interior stage.

## Absorbers and strippers

`absorber` and `stripper` wrap `rigorous_column` with no condenser or reboiler,
the gas fed to the bottom stage and the liquid to the top, and no
specifications (there are no free degrees of freedom):

```python
from fugacio.sim import absorber

gas = Stream.from_fractions(("methane", "propane", "n-decane"), jnp.array([0.9, 0.1, 0.0]), 100.0, 300.0, 20e5)
oil = Stream.from_fractions(("methane", "propane", "n-decane"), jnp.array([0.0, 0.0, 1.0]), 60.0, 300.0, 20e5)
res = absorber(gas, oil, n_stages=6, p=20e5)
res.distillate      # treated gas leaving the top
res.bottoms         # rich solvent leaving the bottom
```

A component that's absent from every feed (here none, but a stabilizer bottoms
fed to a benzene/toluene column has no hydrogen or methane) is carried through
at a trace far below any tolerance so the log-flow unknowns stay finite; the
balances of the components that are present are unaffected.

## How it's solved

The unknowns are the log component liquid and vapor flows on every stage and
the stage temperatures (plus one auxiliary per free spec). Using log flows keeps
the flows positive and lets a component that's essentially absent from a
section span many orders of magnitude without ill-conditioning. The residuals
are scaled so a flow error of one part in the feed, a mole-fraction summation
error, and an enthalpy error of \(10^4\) J/mol each register as order one.

The initial guess comes from a bubble-point sweep over a linear composition
profile with `k_seed` (Wilson for cubics, modified Raoult for gamma-phi) to
start the K-values, then the Newton iteration with a step limiter runs to a
scaled residual of \(10^{-10}\). The whole thing is wrapped in
`newton_system` from `fugacio.thermo.implicit`, so the gradient of anything
computed from the result (a duty, a purity, a reflux ratio) with respect to
anything fed in (the feed state, a spec value, a `kij`, a stage pressure) is one
adjoint solve against the converged Jacobian, never a differentiation *through*
the iterations.

The `Column` block in [`fugacio.sim.eo`](equation-oriented.md) embeds a
converged `rigorous_column` inside an equation-oriented flowsheet, so a column
can sit in the middle of a globally solved plant with recycles around it.
