# Equation-oriented flowsheeting

The equation-oriented (EO) engine assembles every unit's equations, the stream
connectivity, the recycles, and any design specs into one residual system and
solves it simultaneously by Newton's method, with the Jacobian supplied exactly
by JAX autodiff and the converged solution differentiable by the implicit
function theorem.

See the [equation-oriented flowsheeting guide](../../equation-oriented.md) for
worked examples. Import the public surface from the subpackage root:

```python
from fugacio.sim.eo import EOFlowsheet, Flash, Heater, Mixer, optimize_flowsheet_eo
```

## Flowsheet, solution & DOF analysis

::: fugacio.sim.eo.flowsheet

## Unit blocks

::: fugacio.sim.eo.blocks

## Optimization

::: fugacio.sim.eo.optimize
