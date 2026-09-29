# Flowsheet & recycle

The `Flowsheet` container and the recycle/tear solver. `Flowsheet.partition`
groups units into strongly connected blocks (Tarjan), orders them, and selects
tear streams automatically. The standalone `tear_solve` converges a loop by
Broyden (the default), Wegstein, or Newton and differentiates its fixed point.
`Flowsheet` assembles local unit derivatives into the shared process graph's
implicit system. `Flowsheet.solve_with_info` keeps every block's
report, each unit's heat and work, and, with a `model`, a physical audit of
every stream.

See the [flowsheeting guide](../../flowsheeting.md) for worked examples.

::: fugacio.sim.flowsheet

## Process graph

See the [shared runtime guide](../../process-runtime.md).

::: fugacio.sim.graph

::: fugacio.sim.numerics

## Continuation

Adaptive continuation reuses accepted states while moving between operating
conditions. See the [reliability guide](../../reliability.md) for examples and
the distinction between a converged endpoint and a partially completed path.

::: fugacio.sim.continuation
