# Unit operations

Energy-balanced unit operations (flash drum, adiabatic flash, heater, valve,
pump, compressor, turbine, mixer, splitter, component separator) and the
non-ideal separation units (decanter, three-phase flash). Every unit is a
compiled kernel with the property package and its operating values as dynamic
arguments. An eager call raises on failure, and a traced call returns NaN
outlets; see the [reliability guide](../../reliability.md#the-failure-contract).

## Core unit operations

::: fugacio.sim.units

## Operating limits

The predicates that every unit and the saved-case audit enforce.

::: fugacio.sim.unit_limits

## Persistent compilation cache

::: fugacio.sim.compilation

## Non-ideal separations

::: fugacio.sim.separations
