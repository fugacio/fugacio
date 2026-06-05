# fugacio-sim

Differentiable process-simulation layer for the
[Fugacio](https://github.com/owenthcarey/fugacio) stack: flowsheet and
unit-operation models built on top of `fugacio.thermo`.

The core abstraction is the `Stream` — a JAX pytree whose molar flows,
temperature, and pressure are differentiable leaves (component names are static
metadata). Because the underlying EOS phase equilibrium is differentiable, unit
operations are too: you can take a gradient of any downstream quantity (a product
flow, a recovery, a purity) with respect to feed conditions or operating
variables, which is the basis for gradient-based flowsheet optimisation.

## Unit operations

- `flash_drum` — rigorous isothermal-isobaric flash, returning vapour and liquid
  product `Stream`s.
- `mix` — exact component material-balance mixer.
- `bubble_pressure` / `antoine_psat` — lightweight modified-Raoult helpers.

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

Part of the `fugacio` namespace; installs independently:
`pip install fugacio-sim`.
