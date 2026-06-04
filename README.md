# Fugacio

**Open, differentiable thermodynamics and process simulation — with an AI design copilot.**

Fugacio is an open-source successor to closed, expensive process simulators. The
numerical core is written in [JAX](https://github.com/jax-ml/jax), so an entire
flowsheet is *end-to-end differentiable*: you can take gradients through phase
equilibrium and recycle loops for optimization, parameter estimation, and tight
ML integration.

It ships as three layered packages in a single [`uv`](https://docs.astral.sh/uv/)
workspace:

| Package | Import | Responsibility |
| --- | --- | --- |
| `fugacio-thermo` | `fugacio.thermo` | Differentiable properties + phase equilibrium (the foundation). |
| `fugacio-sim` | `fugacio.sim` | Flowsheet / unit-operation engine and solvers (depends on `thermo`). |
| `fugacio-copilot` | `fugacio.copilot` | LLM design agent: flowsheets, sizing, TEA/LCA (depends on `sim`). |

The dependency direction is strict — **`thermo` < `sim` < `copilot`** — and is
enforced in CI by [import-linter](https://github.com/seddonym/import-linter).
All three distributions share the `fugacio`
[PEP 420 namespace](https://peps.python.org/pep-0420/), so they publish to PyPI
independently yet import under one roof.

## Quickstart

```bash
# Install uv: https://docs.astral.sh/uv/
uv sync --all-packages   # venv + lockfile; install all three packages (editable)
uv run pytest            # run the test suite
```

```python
from fugacio.sim import bubble_pressure

# Antoine constants (log10, mmHg, deg C), illustrative binary mixture
comp1 = (8.07131, 1730.63, 233.426)
comp2 = (7.43155, 1554.68, 240.337)

pressure, y1 = bubble_pressure(
    x1=0.4, temperature=80.0, antoine1=comp1, antoine2=comp2, a12=0.5, a21=0.8
)
```

## Development

```bash
just         # list available tasks
just fmt     # ruff format + autofix
just check   # lint + types + import boundaries + tests (exactly what CI runs)
```

| Task | Command |
| --- | --- |
| Format / autofix | `just fmt` |
| Lint | `just lint` |
| Types | `just types` |
| Import boundaries | `just imports` |
| Tests | `just test` |

## Layout

```text
fugacio/
├── packages/
│   ├── fugacio-thermo/    # fugacio.thermo  (no internal deps)
│   ├── fugacio-sim/       # fugacio.sim     (-> thermo)
│   └── fugacio-copilot/   # fugacio.copilot (-> sim)
├── pyproject.toml         # uv workspace root + shared ruff/mypy/pytest config
├── .importlinter          # enforced layer boundaries
└── .github/workflows/     # CI + trusted-publishing release
```

## License

[Apache-2.0](LICENSE).
