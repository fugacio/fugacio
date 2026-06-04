# Fugacio

Open, differentiable thermodynamics and process simulation, with an AI design
copilot. See the [project README](https://github.com/owenthcarey/fugacio) for an
overview.

Fugacio is built as three layered packages:

- **`fugacio.thermo`** — differentiable properties + phase equilibrium.
- **`fugacio.sim`** — flowsheet / unit-operation engine (depends on `thermo`).
- **`fugacio.copilot`** — LLM design agent (depends on `sim`).
