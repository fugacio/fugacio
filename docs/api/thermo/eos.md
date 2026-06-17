# Equations of state

Cubic equations of state (van der Waals, Redlich-Kwong, Soave-Redlich-Kwong,
and Peng-Robinson) with their fugacity-coefficient and molar-volume routines.
Each `CubicEOS` is a static descriptor, so switching models is a one-object
swap that keeps a flowsheet differentiable.

::: fugacio.thermo.eos
