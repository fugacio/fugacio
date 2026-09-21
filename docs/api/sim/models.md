# Thermodynamic models

`package_for` turns component names into a ready
[property package](../thermo/packages.md) for any of the methods in `METHODS`
(cubic, NRTL, UNIQUAC, UNIFAC, Dortmund, PC-SAFT, or a reference fluid), with
explicit parameter provenance. Every `model=` argument in `fugacio.sim` takes
its result. The page also covers the lightweight modified-Raoult helpers for
quick ideal-ish estimates.

## Model builders

::: fugacio.sim.models

## Modified-Raoult helpers

::: fugacio.sim.vle
