# API reference

The reference is generated directly from the package docstrings with
[mkdocstrings](https://mkdocstrings.github.io/), so it never drifts from the
code. It's split per package and then per topic, so each page stays scannable.

Fugacio is three layered packages that share the `fugacio`
[PEP 420 namespace](https://peps.python.org/pep-0420/), with a strict dependency
direction (**`thermo` < `sim` < `copilot`**, enforced in CI):

| Package | Import | What it covers |
| --- | --- | --- |
| [`fugacio.thermo`](thermo/index.md) | `from fugacio.thermo import ...` | Differentiable properties and phase equilibrium: the component database, equations of state, activity models, reference (Helmholtz) fluids, transport, reactions, and parameter regression. |
| [`fugacio.sim`](sim/index.md) | `from fugacio.sim import ...` | The flowsheet engine: streams, energy-balanced unit operations, the recycle/tear solver, columns, reactors, optimization, economics, dynamics and control, advanced control (MPC), and heat integration. |
| [`fugacio.copilot`](copilot/index.md) | `from fugacio.copilot import ...` | The LLM design agent: a JSON tool registry over the engine, a vendor-neutral provider layer, the agent loops, and human-readable reports. |

## Conventions

- Everything numeric is written against [JAX](https://github.com/jax-ml/jax), so
  inputs accept `jax.Array` (and most accept plain Python floats), and outputs
  are `jax.Array`. The whole stack is differentiable, including through phase
  equilibrium and converged recycle loops.
- SI units throughout: kelvin, pascal, mol/s, watts, dollars.
- Names with a leading underscore are internal and omitted from this reference.

## How the reference relates to the guides

The [guides](../phase-equilibrium.md) are the narrative, worked-example tour of
each subsystem; this reference is the exhaustive, symbol-by-symbol companion.
Most pages link back to the guide that motivates them.
