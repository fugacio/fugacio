# Optimization, design specs & economics

The `fugacio.sim` layer adds gradient-based **optimization**, **design
specifications / controllers**, and **process economics** on top of the
differentiable flowsheet engine, and `fugacio.copilot` exposes all of it to an
LLM design agent through a JSON tool registry. Like the rest of the stack, every
solver carries implicit-function-theorem gradient rules, so you can differentiate
*through* an optimum, a met spec, or an annual-cost estimate, with respect to
prices, feed conditions, or model parameters.

## Differentiable optimization

`minimize` solves `min_x f(x, theta)` over an arbitrary decision pytree, with
optional box bounds and equality / inequality constraints. The unconstrained
inner method is BFGS (also `"gradient-descent"` and `"newton"`); bounds switch to
spectral projected gradient, and constraints to an augmented-Lagrangian outer
loop. `least_squares` wraps a Levenberg–Marquardt solver for residual problems.

```python
import jax.numpy as jnp
from fugacio.sim import minimize

# Rosenbrock, unconstrained:
def rosen(x, _):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

res = minimize(rosen, jnp.array([-1.2, 1.0]))
res.x            # -> [1, 1]
res.converged    # True
```

The headline is `argmin`: it returns *only* the optimal `x*(theta)`, but with a
custom VJP that differentiates the solution through the optimality (KKT)
conditions by the implicit function theorem, exact and cheap, with no
backprop-through-iterations. That makes an optimizer just another differentiable
layer you can nest inside a larger gradient.

```python
import jax
import jax.numpy as jnp
from fugacio.sim import argmin

# Ridge fit: x*(theta) = (A^T A + theta I)^{-1} A^T b. Differentiate the
# *solution* with respect to the regularization strength theta.
A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
b = jnp.array([1.0, 2.0, 3.0])

def loss(x, theta):
    return jnp.sum((A @ x - b) ** 2) + theta * jnp.sum(x**2)

dx_dtheta = jax.jacobian(lambda th: argmin(loss, jnp.zeros(2), th))(0.5)
```

Bounds and constraints carry gradients too: for an active bound the sensitivity
collapses to the constraint, for an interior solution it follows the reduced
Hessian.

```python
from fugacio.sim import minimize

# min (x-3)^2 + (y-3)^2  s.t.  x + y <= 4,  0 <= x,y
res = minimize(
    lambda v, _: (v[0] - 3) ** 2 + (v[1] - 3) ** 2,
    jnp.array([0.0, 0.0]),
    bounds=(0.0, None),
    ineq_constraints=lambda v, _: jnp.atleast_1d(v[0] + v[1] - 4.0),
)
res.x  # -> [2, 2] on the active constraint
```

## Design specifications & controllers

A **design spec** adjusts a manipulated variable until a controlled variable hits
a target, the bread-and-butter of flowsheeting. `meet_spec` is the single-variable
solver (bracketed bisection when given `[lo, hi]`, else damped Newton);
`controller` builds a `DesignSpec` that reads like control language; and
`solve_design` satisfies several (generally coupled) specs at once with a Newton
system, re-running the flowsheet (recycles and all) at each step. The converged
manipulated values and the streams computed from them stay differentiable with
respect to the *unmanipulated* parameters.

```python
import jax.numpy as jnp
from fugacio.sim import Stream, flash_drum, controller, solve_design

feed = Stream.from_fractions(
    ("propane", "n-butane", "n-pentane"),
    jnp.array([0.4, 0.35, 0.25]), flow=100.0, t=330.0, p=8e5,
)

def simulate(theta):                       # one flash drum at (T, P)
    vapor, liquid = flash_drum(feed, theta["T"], theta["P"])
    return {"vapor": vapor, "liquid": liquid}

# "Move the drum temperature in [300, 360] K to vaporise 40 mol/s."
spec = controller(
    simulate, manipulated="T",
    controlled=lambda s: s["vapor"].total, set_point=40.0,
    lo=300.0, hi=360.0,
)
out = solve_design(simulate, {"T": 330.0, "P": 8e5}, [spec])
out.theta["T"], out.converged              # the temperature that hits 40 mol/s
```

## Flowsheet optimization

`optimize_flowsheet` ties the pieces together: choose `design_vars` out of the
parameter mapping, pass an economic (or any) `objective(streams, theta)`, and it
optimizes end to end, differentiating straight through the converged flowsheet.

```python
import jax.numpy as jnp
from fugacio.sim import optimize_flowsheet

def objective(streams, theta):             # e.g. a total annual cost
    return cost(streams, theta)

res = optimize_flowsheet(
    simulate, objective,
    theta0={"T": 330.0, "P": 8e5}, design_vars=["T"],
    bounds={"T": (300.0, 360.0)},
)
res.theta["T"], res.objective, res.converged
```

## Process economics

`fugacio.sim` includes differentiable equipment **sizing**, **Turton bare-module
costing**, **utility** costing, and the usual **financial** metrics, so an
objective can be a real screening economics number, and its gradient with respect
to a design variable is exact.

| Step | Functions |
| --- | --- |
| Sizing | `lmtd`, `heat_exchanger_area`, `column_diameter`, `column_height`, `vessel_volume` |
| Capital | `purchased_cost`, `pressure_factor`, `bare_module_cost` (Turton, CEPCI-escalated) |
| Utilities | `utility_cost` (cooling water, steam levels, refrigeration, electricity, …) |
| Finance | `capital_recovery_factor`, `annualized_capital`, `total_annual_cost`, `npv`, `discounted_payback` |

```python
import jax
from fugacio.sim import heat_exchanger_area, bare_module_cost, total_annual_cost, utility_cost

area = heat_exchanger_area(duty=1.0e6, u=500.0, dt_hot=60.0, dt_cold=40.0)  # m^2 via LMTD
capex = bare_module_cost("heat_exchanger", area).bare_module               # installed $
opex = utility_cost(1.0e6, "cooling_water")                                # $/yr

tac = total_annual_cost(capex, opex, rate=0.1, years=10.0)                 # TAC $/yr
d_tac_d_area = jax.grad(
    lambda a: total_annual_cost(bare_module_cost("heat_exchanger", a).bare_module, opex)
)(area)                                                                    # exact
```

## The AI design copilot

`fugacio.copilot` wraps the whole engine in a registry of deterministic,
JSON-in/JSON-out **tools** (properties, flash/units, distillation, reactors,
optimization, sizing, costing, sensitivities) and drives them with an LLM. The
provider layer is vendor-neutral (a small `LLMProvider` protocol with `OpenAI`,
`Anthropic`, and `Mock` implementations), so the agent core never imports a
specific SDK.

```python
from fugacio.copilot import default_registry, tool_schemas, call_tool

reg = default_registry()
tool_schemas(reg)                          # JSON schemas to hand an LLM
call_tool("heat_exchanger_cost",
          {"duty": 1e6, "u": 500.0, "dt_hot": 60.0, "dt_cold": 40.0}, reg)
```

`run_llm_agent` runs the multi-turn tool-calling loop: the model plans, calls
tools, sees their results (and any errors, with schema validation), and returns a
final answer plus a full transcript. Swap in a real provider to go live.

```python
from fugacio.copilot import run_llm_agent
from fugacio.copilot.llm import OpenAIProvider   # needs the `openai` extra

result = run_llm_agent(
    "Size and cost a cooler that removes 1 MW with cooling water.",
    OpenAIProvider(model="gpt-4o-mini"),
)
result.answer        # natural-language summary
result.transcript    # ordered tool calls + results
```

Finally, `fugacio.copilot.report` turns results into Markdown an engineer expects:
`stream_table`, `summarize_optimization`, `summarize_economics`, and
`summarize_transcript`.
