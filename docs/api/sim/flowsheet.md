# Flowsheet & recycle

The recycle/tear solver and the `Flowsheet` container. Recycle loops are
converged to a fixed point and differentiated by the implicit function theorem,
so gradients flow through the *converged* flowsheet rather than the iteration.

::: fugacio.sim.flowsheet
