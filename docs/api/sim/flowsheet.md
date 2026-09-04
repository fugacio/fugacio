# Flowsheet & recycle

The `Flowsheet` container and the recycle/tear solver. `Flowsheet.partition`
groups units into strongly connected blocks (Tarjan), orders them, and selects
tear streams automatically; `tear_solve` converges each loop by Wegstein,
Broyden, or Newton and differentiates the fixed point by the implicit function
theorem, so gradients flow through the *converged* flowsheet rather than the
iteration.

See the [flowsheeting guide](../../flowsheeting.md) for worked examples.

::: fugacio.sim.flowsheet
