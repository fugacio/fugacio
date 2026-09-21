# Flowsheet & recycle

The `Flowsheet` container and the recycle/tear solver. `Flowsheet.partition`
groups units into strongly connected blocks (Tarjan), orders them, and selects
tear streams automatically; `tear_solve` converges each loop by Broyden (the
default), Wegstein, or Newton and differentiates the fixed point by the
implicit function theorem, so gradients flow through the *converged* flowsheet
rather than the iteration. `Flowsheet.solve_with_info` keeps every block's
report, each unit's heat and work, and, with a `model`, a physical audit of
every stream.

See the [flowsheeting guide](../../flowsheeting.md) for worked examples.

::: fugacio.sim.flowsheet

## Continuation

Adaptive continuation reuses accepted states while moving between operating
conditions. See the [reliability guide](../../reliability.md) for examples and
the distinction between a converged endpoint and a partially completed path.

::: fugacio.sim.continuation
