# Phase equilibrium

Vapor-liquid equilibrium by both routes: an equation of state for both phases
(phi-phi) and an activity-coefficient liquid with an EOS or ideal vapor
(gamma-phi). Each calculation has a checked `_with_info` form that returns a
`SolveReport`, and a value-only form that returns NaN when the report fails.
Every [property package](packages.md) exposes the same calls through one
interface, so the rest of the stack switches thermodynamic models without
changing call sites.

See the [non-ideal phase equilibrium guide](../../phase-equilibrium.md) for
worked examples.

## EOS (phi-phi) equilibrium

::: fugacio.thermo.equilibrium

## Gamma-phi equilibrium

::: fugacio.thermo.gammaphi

## Tangent-plane stability

::: fugacio.thermo.stability
