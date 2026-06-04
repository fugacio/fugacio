# fugacio-sim

Differentiable process-simulation layer for the
[Fugacio](https://github.com/owenthcarey/fugacio) stack: flowsheet and
unit-operation models built on top of `fugacio.thermo`.

```python
from fugacio.sim import bubble_pressure

comp1 = (8.07131, 1730.63, 233.426)
comp2 = (7.43155, 1554.68, 240.337)
pressure, y1 = bubble_pressure(0.4, 80.0, comp1, comp2, a12=0.5, a21=0.8)
```

Part of the `fugacio` namespace; installs independently:
`pip install fugacio-sim`.
