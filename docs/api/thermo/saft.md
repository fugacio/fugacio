# PC-SAFT (molecular SAFT)

The perturbed-chain statistical associating fluid theory (PC-SAFT) of Gross and
Sadowski, implemented as one scalar reduced residual Helmholtz energy whose every
property, the density and saturation solvers, and the Wertheim association
site-fraction solve are exact `jax.grad` derivatives, differentiable in both state
and the molecular parameters themselves. `SAFTModel` exposes the same
`flash_pt`, bubble, and dew calls as a cubic `EOSModel` or a `GammaPhiModel`, so a
flowsheet adopts a molecular EOS by swapping one object.

See the [molecular SAFT guide](../../molecular-saft.md) for worked examples.

::: fugacio.thermo.saft
