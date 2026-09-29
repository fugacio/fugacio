# Flowsheeting: partitioning, tears, and exchangers

A flowsheet is a set of units connected by named streams. When a stream fed to
a unit depends on that unit's own output, the loop has to be *torn*: one stream
is guessed, the loop is evaluated, and the guess is corrected until the stream
reproduces itself. `fugacio.sim.Flowsheet` does the bookkeeping for you: it
finds the loops, chooses where to tear them, orders the calculation, converges
every loop with a choice of accelerated fixed-point or quasi-Newton methods,
and keeps the converged plant differentiable in its parameters.

## Declare units, not an order

Register feeds and units; each unit is a plain function of its input streams
and the shared parameter pytree `theta`. It returns one stream, a tuple of
streams matched to `outputs`, or a unit result that exposes `outlets` (such as
`HeaterResult` or `RigorousColumnResult`). The flowsheet keeps the heat, work,
and report of every result it receives. The order you register units in
doesn't matter.

```python
import jax.numpy as jnp
from fugacio.sim import Flowsheet, Stream, flash_drum, heater, mix, splitter

comps = ("methane", "ethane", "propane")
feed = Stream.from_fractions(comps, jnp.array([0.6, 0.3, 0.1]), 100.0, 300.0, 20e5)

fs = Flowsheet()
fs.feed("feed", feed)
fs.unit("mixer",  lambda f, r, th: mix([f, r], t=300.0),          inputs=("feed", "recycle"), outputs=("mixed",))
fs.unit("cooler", lambda s, th: heater(s, t_out=th["T_flash"]),    inputs=("mixed",),  outputs=("cold",))
fs.unit("flash",  lambda s, th: flash_drum(s, th["T_flash"], 20e5), inputs=("cold",),   outputs=("vapor", "liquid"))
fs.unit("split",  lambda s, th: splitter(s, jnp.array([th["purge"], 1 - th["purge"]])),
        inputs=("vapor",), outputs=("purge", "recycle"))

streams = fs.solve({"T_flash": 250.0, "purge": 0.1})
streams["recycle"].total
```

`partition()` exposes the calculation order the solver derived:

```python
for block in fs.partition():
    print(block.units, block.cyclic, block.tears)
# ('mixer', 'cooler', 'flash', 'split')  True  ('recycle',)
```

Units are grouped into strongly connected components (Tarjan's algorithm), the
blocks are ordered so every block's inputs are known before it runs, and inside
each cyclic block tear streams are chosen greedily to break every cycle with as
few tears as possible, preferring streams that flow *backwards* in the order
the units were declared (which is almost always the recycle you had in mind).
The units within a block are then ordered topologically with the tears treated
as known. Acyclic blocks run once; cyclic blocks are converged.

You can still tear by hand. `fs.tear("recycle", guess)` is honored by
`partition` (and supplemented if it leaves a cycle unbroken), and it's also the
way to give a starting guess for a tear whose component list differs from the
feeds', which the automatic seed (one pass of the loop with an empty recycle)
can't infer. A recycle block with no upstream stream to seed it raises
`ValueError` and asks for `Flowsheet.tear`.

## Tear methods

`Flowsheet.solve` forwards its keyword arguments to `tear_solve`, which
accepts three `method`s (see `fugacio.sim.TEAR_METHODS`):

| Method | What it does | Use when |
| --- | --- | --- |
| `"broyden"` (default) | quasi-Newton with rank-one inverse-Jacobian updates and a residual-decreasing line search | most loops, including strongly coupled or slowly converging ones (heat integration around a column) |
| `"wegstein"` | bounded secant acceleration of direct substitution, per component | loops that are already contracting; cheap per iteration |
| `"newton"` | full Newton with the autodiff Jacobian of the loop map | small tears with a badly conditioned loop, or when quadratic convergence is worth a Jacobian per step |

Both execution strategies differentiate the converged connection equations.
Local unit Jacobians assemble into a sparse process matrix; forward and reverse
sensitivities use checked sparse solves. The standalone `tear_solve` API still
supports arbitrary floating pytrees with a small dense fixed-point derivative.

Pass `strategy="simultaneous"` to solve the same registered graph with sparse
Newton steps. Set `linear_solver="dense"` for a small reference comparison.
Concrete orchestration preserves compiled unit boundaries. See the
[shared process runtime](process-runtime.md) for state coordinates,
compilation behavior, and sparse-solver limits.

## Reports, unit records, and audits

`solve` returns the named streams after checking every recycle, unit, and
stream. `solve_with_info` keeps the evidence instead of raising:

```python
from fugacio.sim import package_for

fs.model = package_for(comps)      # audit every stream on this package
result = fs.solve_with_info({"T_flash": 250.0, "purge": 0.1})
result.converged                   # every recycle, unit, and stream check passed
result.accepted                    # converged, and every audited stream accepted
result.units["cooler"].heat        # heat, work, and report a unit returned (W)
result.audits["vapor"].to_dict()   # physical acceptance of one stream
result.check()                     # raise, naming the failed block or stream
```

`FlowsheetResult.reports` holds solve reports keyed `recycle:...`, `unit:...`,
and `stream:...`. `units` holds a `UnitRecord(heat, work, report)` for every
unit, taken from its returned result. With `Flowsheet(model=pkg)`, or
`fs.model` set, every solved stream is audited for equilibrium, stability
(including a second liquid), and parameter applicability after the recycles
converge, outside the iterations. `solve()` raises `PhysicalAcceptanceError` for
a stream that fails its audit; pass `audit=False` to `solve_with_info` to skip
the audits. A failed unit raises `ConvergenceError` in an eager solve, and its
message names the unit. Inside `jax.jit` nothing can raise: `solve()` returns
NaN streams for a result that isn't accepted, and `solve_with_info` keeps the
best iterates, so inspect `result.converged` and `result.accepted` there.

`fugacio.sim.acceptance.audit_flowsheet` adds explicit
`BalanceBoundary` checks on top: each boundary names its input and output
streams, and its heat and work are its explicit values plus those the result
retained for the `units` it lists.

## Two-sided heat exchanger

`fugacio.sim.heat_exchanger` couples a hot and a cold stream through a
countercurrent (or parallel) exchanger with rigorous enthalpy curves on both
sides. Each side may use its own property package, so steam or a refrigerant
from a reference-fluid package can heat a hydrocarbon on a cubic EOS.

```python
from fugacio.sim import heat_exchanger, package_for

steam = Stream.from_fractions(("water",), jnp.array([1.0]), 5.0, 430.0, 3e5)
res = heat_exchanger(
    steam, process,
    t_cold_out=380.0,                    # exactly one closing spec
    zones=6,                             # discretise the T-Q curves for LMTD
    model_hot=package_for(("water",), "iapws"),
)
res.duty, res.ua, res.lmtd             # W, W/K, K
res.hot_out, res.cold_out              # both outlet streams (PH-flashed)
res.approach_hot_end, res.approach_cold_end, res.min_approach
res.hot_curve, res.cold_curve          # temperature at the zone boundaries
```

The closing specification is any one of `duty`, `t_hot_out`, `t_cold_out`,
`min_approach` (the exchanger sized to a pinch), or `ua` (a rating calculation:
the duty that a given \(UA\) delivers, found as the root of the zone-integrated
\(UA\) requirement with a bracketed solve). Both outlets are obtained by PH
flashes, so a condensing or boiling side shows its saturation plateau on the
T-Q curve. A temperature spec that would violate the second law (cold outlet
above the hot inlet) is capped at the feasible duty rather than producing a
temperature cross. Every output is differentiable in the inlet states, the
spec, and the package parameters, including the duty with respect to `ua`.

## A plant with a recycle, a column, and an economizer

The pieces compose. This is the depropanizer from the test suite: an
economizer preheats the feed against the column bottoms, which closes a
heat-integration loop *around* a rigorous column.

```python
from fugacio.sim import ColumnFeed, purity, recovery, rigorous_column

comps = ("propane", "n-butane", "n-pentane")
feed = Stream.from_fractions(comps, jnp.array([0.40, 0.35, 0.25]), 100.0, 300.0, 16e5)
specs = [recovery(0, "distillate", 0.98), purity("distillate", 0, 0.95)]

def economiser(cold, hot, th):
    hx = heat_exchanger(hot, cold, min_approach=th["dt"])
    return hx.cold_out, hx.hot_out

def column(s, th):
    # The result's outlets are (distillate, bottoms); the flowsheet keeps its
    # condenser and reboiler duties as the unit's heat.
    return rigorous_column([ColumnFeed(s, 8)], 16, p=16e5, specs=specs)

fs = Flowsheet()
fs.feed("feed", feed)
fs.unit("economiser", economiser, inputs=("feed", "bottoms"), outputs=("preheated", "bottoms_cooled"))
fs.unit("column", column, inputs=("preheated",), outputs=("distillate", "bottoms"))

s = fs.solve({"dt": 15.0}, tol=1e-8)
s["preheated"].t              # heat recovered from the bottoms
s["distillate"].z[0]          # 0.95, the purity spec
```

The `Flowsheet` finds the single cyclic block, tears `bottoms`, seeds it by one
pass with an empty recycle (an empty hot side exchanges nothing), and converges
the loop with the default Broyden method. Because the column is itself an implicit solve, the loop
map is smooth in the tear and differentiable in `dt` and the spec values, so
`jax.grad` of the reboiler duty with respect to the approach temperature passes
through the exchanger, the column, and the closed loop together.

The [test suite](https://github.com/fugacio/fugacio/tree/main/packages/fugacio-sim/tests)
carries two more plants under the `plant` marker: an HDA-lite toluene
hydrodealkylation loop (mixer, furnace, fixed-conversion reactor, cooler,
flash, purge, automatically partitioned, followed by a stabilizer and a
benzene/toluene column) and an ethanol/water train on NRTL whose bottoms heat
is recovered into the feed.

### Compile time

Every unit, column, and exchanger is compiled once per *structure* (stage
count, component list, specification kinds, package structure) and then
reused, so a unit evaluated repeatedly inside a recycle iteration or an
optimization pays the compilation once. The flowsheet also caches each recycle
block's map, so solving again at a new `theta` reuses the compiled recycle
iteration; registering a unit or a tear clears that cache. The first call is
where the time goes, and a gamma-phi package (whose K-values and liquid
enthalpies carry an EOS saturation-pressure solve per component) compiles
noticeably more slowly than a cubic one. To keep compiled programs across
processes, see [compile-once kernels](performance.md#compile-once-kernels).
Run the quick suite with `just test-fast` to skip the `plant` case studies.

## Equation-oriented alternative

For flowsheets where every unit can be written as residual equations, the
[equation-oriented engine](equation-oriented.md) solves the plant as one Newton
system with no tears at all. It now includes `HeatExchanger`,
`StoichiometricReactor`, and `Column` blocks and takes the same property
packages through `EOFlowsheet(model=...)`. The two engines share the same
physics and agree on any flowsheet both can express; the sequential-modular
`Flowsheet` remains the more forgiving of the two for large recycles with poor
initial guesses.
