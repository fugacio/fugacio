# Reference fluids (Helmholtz)

Multiparameter Helmholtz-energy equations of state of the REFPROP/CoolProp class
(IAPWS-95 water/steam, Span-Wagner CO2, and more), implemented as one scalar
Helmholtz energy whose every property and solver is an exact `jax.grad`
derivative. Includes differentiable saturation lines, steam-table state
functions, and the IAPWS transport formulations for water.

See the [reference-fluids & steam-tables guide](../../reference-fluids.md) for
worked examples.

::: fugacio.thermo.helmholtz
