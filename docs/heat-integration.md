# Heat integration & pinch analysis

Most of the energy bill of a chemical plant is set before a single exchanger is
drawn: it's fixed by *how much* heat the hot streams must reject and the cold
streams must absorb, and by how cleverly the two are matched. **Pinch analysis**
answers the first question (the thermodynamic minimum hot- and cold-utility
duties for a chosen minimum approach temperature `dt_min`, and the **pinch** that
divides the problem) before any network exists. `fugacio.sim.integration` builds
the whole pinch-technology workflow (targets, composite curves, area/cost
supertargeting, and network synthesis) and keeps it **end-to-end
differentiable**, so a heat-recovery target is just another node in the graph:
you can take its gradient with respect to stream temperatures, duties, or
`dt_min`, and compose it with the flowsheet and the optimizers.

## Heat streams

A `HeatStream` is the heat-integration view of a process stream: a supply and
target temperature, a constant heat-capacity flowrate `CP = m·cp` (W/K), and a
film coefficient `h` for area targeting. A stream is **hot** (a source, to be
cooled) when its supply is hotter than its target, and **cold** otherwise; the
classification is structural, while every numeric leaf is a differentiable JAX
array. Build one directly with `make_stream`, or extract it from a live
`fugacio.sim.Stream` with `heat_stream`, which uses the real, two-phase-aware
`enthalpy_flow` so the duty (and hence `CP`) carries the flowsheet's actual
thermodynamics.

```python
import jax.numpy as jnp
from fugacio.sim import make_stream, heat_stream, Stream

# Directly, from CP (W/K):
hot = make_stream(t_supply=170.0, t_target=60.0, cp=3.0, h=1.0, name="H1")

# Or from a process stream and a target temperature (CP from the real enthalpy):
feed = Stream.from_fractions(("water",), jnp.array([1.0]), flow=10.0, t=380.0, p=2e5)
cold = heat_stream(feed, t_target=300.0, name="condensate")
```

## Energy targets & the pinch

`pinch_analysis` runs the **problem table algorithm** (Linnhoff & Flower): it
shifts the streams onto a common temperature scale (hot down by `dt_min/2`, cold
up by `dt_min/2`), nets the heat-capacity flowrates in each temperature interval,
and cascades the surplus heat downward. The most negative point of the
zero-input cascade is the minimum hot utility; adding it back makes the cascade
non-negative, and the temperature where it touches zero is the pinch. Everything
is a smooth-a.e. function of the inputs, so the targets are differentiable.

```python
from fugacio.sim import make_stream, pinch_analysis

streams = [
    make_stream(20.0, 135.0, cp=2.0, name="C1"),
    make_stream(170.0, 60.0, cp=3.0, name="H1"),
    make_stream(80.0, 140.0, cp=4.0, name="C2"),
    make_stream(150.0, 30.0, cp=1.5, name="H2"),
]

res = pinch_analysis(streams, dt_min=10.0)
res.hot_utility, res.cold_utility      # -> 20.0 W, 60.0 W (the MER targets)
res.heat_recovery                      # -> 450.0 W recovered process-to-process
res.hot_pinch_temperature              # -> 90.0 K  (cold side at 80.0 K)
res.has_pinch                          # -> True (a threshold problem returns False)
```

Because the target is differentiable, sensitivities are exact and free. The
slope of the hot-utility target with respect to the approach temperature, for
instance, is one line:

```python
import jax
from fugacio.sim import minimum_utilities

# d(Q_h,min) / d(dt_min): how fast buying a tighter approach buys back utility.
jax.grad(lambda dt: minimum_utilities(streams, dt)[0])(10.0)   # -> 0.5 W/K
```

`heat_cascade` exposes the full interval table behind the targets, and
`minimum_utilities` is the bare `(Q_h,min, Q_c,min)` convenience.

## Composite & grand composite curves

The composite curves are the canonical temperature–enthalpy picture of the
targets: all hot streams combined into one hot composite, all cold streams into
one cold composite, slid together until they're `dt_min` apart at the pinch. The
**grand composite curve** plots net heat flow against shifted temperature, the
shape that drives utility selection and multiple-utility placement.

```python
from fugacio.sim import composite_curves, grand_composite_curve

cc = composite_curves(streams, dt_min=10.0)
cc.hot_t, cc.hot_h          # hot composite (T, H) polyline
cc.cold_t, cc.cold_h        # cold composite (T, H) polyline
cc.min_approach             # -> 10.0 K (the closest vertical approach == dt_min)

gcc = grand_composite_curve(streams, dt_min=10.0)
gcc.shifted_temperature, gcc.net_heat_flow   # the GCC (zero at the pinch)
```

## Area, units & cost targets

Targets aren't just about energy. With each stream's film coefficient, the
**Bath formula** integrates the area of a vertical-heat-transfer network from the
balanced composite curves, accounting for the individual film resistances and the
local LMTD. `units_target` gives the minimum number of exchanger units from
Euler's network relation (respecting the pinch division), and
`capital_cost_target` turns area and units into installed capital through a smooth
cost law. `total_annual_cost_target` then closes the loop, pricing the utilities
through `fugacio.sim.economics` and annualising the capital into a single TAC.

```python
from fugacio.sim import area_target, units_target, total_annual_cost_target

area_target(streams, dt_min=10.0)        # -> ~0.0999 m^2 (these CPs are in W/K)
units_target(streams, dt_min=10.0).units # -> 7 units (MER, pinch-divided)

tac = total_annual_cost_target(streams, dt_min=10.0,
                               hot_utility="hp_steam", cold_utility="cooling_water")
tac.area, tac.capital, tac.utility_cost, tac.total_annual_cost
```

## Supertargeting: the optimal `dt_min`

A small `dt_min` recovers more heat (less utility, lower operating cost) but needs
ever more area as the composites pinch together (higher capital); a large one does
the reverse. The total annual cost is therefore U-shaped in `dt_min`, and the
minimum is the cost-optimal approach, **supertargeting**, found *before* any
network is designed. The TAC is differentiable between kinks, but the integer
unit-count steps make it only piecewise-smooth, so `optimal_dt_min` locates the
basin with a vectorised grid scan (`jax.vmap` over the target) and polishes it with
a golden-section search; `supertarget` returns the whole cost-vs-`dt_min` curve
for plotting.

```python
import jax.numpy as jnp
from fugacio.sim import optimal_dt_min, supertarget

opt = optimal_dt_min(streams, bounds=(1.0, 40.0))
opt.dt_min, opt.total_annual_cost        # -> ~5.55 K, the minimum-cost design point

curve = supertarget(streams, jnp.linspace(2.0, 40.0, 60))   # arrays for the TAC plot
```

## Heat-exchanger-network synthesis

Targets say what's achievable; `synthesize_network` builds a network that hits
them, by the **pinch design method**. It splits the problem at the pinch, designs
each side inward-out by *tick-off* matching subject to the CP-feasibility rule
(`CP_hot ≤ CP_cold` above the pinch, the reverse below), and tops up the
unfinished duties with utility heaters and coolers. Threshold problems (no pinch)
are anchored at the closed end instead. Every returned network is run through
`verify_network`, which independently checks that each exchanger respects
`dt_min`, that every stream's energy balance closes, and whether the design meets
the minimum-utility (MER) targets.

```python
from fugacio.sim import synthesize_network

net = synthesize_network(streams, dt_min=10.0)
net.feasible, net.achieves_mer    # -> True, True  (respects dt_min and hits MER)
net.hot_utility, net.cold_utility # -> 20.0 W, 60.0 W (the targets, realised)
net.n_units, net.total_area, net.min_approach

for e in net.exchangers:          # process matches, heaters, coolers
    print(e.kind, e.hot, e.cold, round(e.duty, 1), round(e.area, 4))
```

## The AI copilot, integrated

`fugacio.copilot` exposes the layer to an LLM design agent as deterministic,
JSON-in/JSON-out tools: `heat_integration_targets` (utilities, pinch, recovery,
area, units, and cost for a stream set), `composite_curves` (the T–H and grand
composite data for plotting), `optimum_dt_min` (the capital-energy trade-off /
supertargeting), and `heat_exchanger_network` (synthesise and verify a network).
Each stream is given as `t_supply`/`t_target` plus either a `cp` or a `duty`.
`summarize_heat_integration` renders the targets as a Markdown table.

```python
from fugacio.copilot import call_tool, summarize_heat_integration

out = call_tool("heat_integration_targets", {
    "streams": [
        {"name": "C1", "t_supply": 20.0, "t_target": 135.0, "cp": 2.0},
        {"name": "H1", "t_supply": 170.0, "t_target": 60.0, "cp": 3.0},
        {"name": "C2", "t_supply": 80.0, "t_target": 140.0, "cp": 4.0},
        {"name": "H2", "t_supply": 150.0, "t_target": 30.0, "cp": 1.5},
    ],
    "dt_min": 10.0,
})
print(summarize_heat_integration(out))     # targets + pinch, as Markdown
```
