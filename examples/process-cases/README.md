# Portable process examples

These JSON files are exported by `fugacio.sim.cases.examples.example_case` and
can be edited without importing Python functions. The measured heater's fit
and holdout IDs are included in its case. Utility prices and capital inputs are
illustrative screening assumptions.

From the repository root:

```bash
uv run fugacio run examples/process-cases/heater.json --report heater.md
uv run fugacio optimize examples/process-cases/heater.json examples/process-cases/heater-optimization.json
uv run fugacio run examples/process-cases/depropanizer.json --report depropanizer.md
uv run fugacio sweep examples/process-cases/depropanizer.json examples/process-cases/depropanizer-sweep.json
uv run fugacio sensitivities examples/process-cases/depropanizer.json examples/process-cases/depropanizer-sensitivities.json
uv run fugacio optimize examples/process-cases/depropanizer.json examples/process-cases/depropanizer-optimization.json
uv run fugacio run examples/process-cases/measured-heater.json --report measured-heater.md
uv run fugacio profile examples/process-cases/depropanizer.json examples/process-cases/depropanizer-profile.json
uv run fugacio sensitivities examples/process-cases/ethanol-train.json examples/process-cases/ethanol-train-sensitivities.json
uv run fugacio sensitivities examples/process-cases/heater-bank.json examples/process-cases/heater-bank-sensitivities.json
```

The first rigorous plant solve and gradient compilation can take tens of
minutes and require substantial memory, particularly on Intel Macs. Studies
reuse one linearization per point and select forward or reverse directions
from the input/output counts. They reuse compiled recycle maps across optimizer
points. CI runs plant solves, derivatives, and optimization in separate processes to bound
peak memory.
The [performance guide](../../docs/performance.md) covers structured columns,
explicit solver options, profiling, and isolated benchmarks with resource limits.

To reuse compilation work across CLI invocations, enable JAX's persistent cache
in your checkout before running the commands:

```bash
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
```

Results go to `.fugacio-cases` unless `--workspace` selects another directory.
The printed `artifact_id` identifies a saved run or study. Use `fugacio inspect`
to reopen it, and `fugacio compare` to compare compatible baseline and candidate
runs. Failed points remain visible in study manifests.

The recycle example declares both split fractions. If those defaults are
edited, update both so they sum to one. The temperature parameter can be swept
independently.

See the [process cases guide](../../docs/process-cases.md) for the format,
acceptance criteria, backend differences, and accountable copilot workflow.

The `reactive-recycle` and `reactive-separation` examples share butane
isomerization thermochemistry. Their phase-specific kinetics, reacting volumes,
and reverse-rate convention are explicit. Kinetic coefficients and equipment
sizes are illustrative assumptions, without measured catalyst qualification.
The recycle uses a vapor CSTR; the separation uses liquid reaction volumes in
energy-balanced MESH stages.

```bash
uv run fugacio run examples/process-cases/reactive-recycle.json --report reactive-recycle.md
uv run fugacio sensitivities examples/process-cases/reactive-recycle.json examples/process-cases/reactive-recycle-sensitivities.json
uv run fugacio optimize examples/process-cases/reactive-recycle.json examples/process-cases/reactive-recycle-optimization.json
uv run fugacio run examples/process-cases/reactive-separation.json --report reactive-separation.md
uv run fugacio sensitivities examples/process-cases/reactive-separation.json examples/process-cases/reactive-separation-sensitivities.json
uv run fugacio optimize examples/process-cases/reactive-separation.json examples/process-cases/reactive-separation-optimization.json
```

See [reactive workflows](../../docs/reactive-workflows.md) for rate units,
thermal references, phase restrictions, numerical acceptance, and API migration.
