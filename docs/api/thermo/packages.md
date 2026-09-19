# Property packages

One object that owns phase equilibrium *and* energy for a set of components:
the `PropertyPackage` protocol and its cubic, gamma-phi, PC-SAFT, and
reference-fluid implementations, plus the generic two-phase properties,
stability search, saturation points, and energy-specified flashes built on top
of them. Every solve method has a checked `_with_info` form, and its value-only
form returns NaN when the solve fails.

See the [property-packages guide](../../property-packages.md) for the concepts
and worked examples.

::: fugacio.thermo.package

## Consistency checks

Data-free thermodynamic identities, including the Gibbs-Helmholtz and
fugacity-enthalpy checks that tie a package's energy side to its equilibrium
side.

::: fugacio.thermo.consistency
