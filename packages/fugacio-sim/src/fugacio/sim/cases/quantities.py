"""Explicit engineering quantities at the process-case boundary.

Numerical kernels consume SI values. Case files retain their declared units;
references are checked dimensionally before a solver is compiled. This small,
closed vocabulary deliberately rejects spelling guesses and ambiguous units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# mass, length, time, amount, temperature, currency
Dimension = tuple[int, int, int, int, int, int]
DIMENSIONLESS: Dimension = (0, 0, 0, 0, 0, 0)
TEMPERATURE: Dimension = (0, 0, 0, 0, 1, 0)
PRESSURE: Dimension = (1, -1, -2, 0, 0, 0)
FLOW: Dimension = (0, 0, -1, 1, 0, 0)
MASS_FLOW: Dimension = (1, 0, -1, 0, 0, 0)
POWER: Dimension = (1, 2, -3, 0, 0, 0)
ENERGY: Dimension = (1, 2, -2, 0, 0, 0)
MOLAR_ENERGY: Dimension = (1, 2, -2, -1, 0, 0)
LENGTH: Dimension = (0, 1, 0, 0, 0, 0)
AREA: Dimension = (0, 2, 0, 0, 0, 0)
VOLUME: Dimension = (0, 3, 0, 0, 0, 0)
CONCENTRATION: Dimension = (0, -3, 0, 1, 0, 0)
REACTION_RATE: Dimension = (0, -3, -1, 1, 0, 0)
TIME: Dimension = (0, 0, 1, 0, 0, 0)
MONEY: Dimension = (0, 0, 0, 0, 0, 1)
CONDUCTANCE: Dimension = (1, 2, -3, 0, -1, 0)
FILM: Dimension = (1, 0, -3, 0, -1, 0)
ENERGY_PRICE: Dimension = (-1, -2, 2, 0, 0, 1)
ANNUAL_COST: Dimension = (0, 0, -1, 0, 0, 1)
YEAR_SECONDS = 365.25 * 86400.0


@dataclass(frozen=True)
class Unit:
    """An affine conversion ``SI = value * factor + offset``."""

    dimension: Dimension
    factor: float = 1.0
    offset: float = 0.0


UNITS: dict[str, Unit] = {
    "1": Unit(DIMENSIONLESS),
    "%": Unit(DIMENSIONLESS, 0.01),
    "K": Unit(TEMPERATURE),
    "degC": Unit(TEMPERATURE, 1.0, 273.15),
    "degF": Unit(TEMPERATURE, 5.0 / 9.0, 255.3722222222222),
    "delta_K": Unit(TEMPERATURE),
    "delta_degC": Unit(TEMPERATURE),
    "delta_degF": Unit(TEMPERATURE, 5.0 / 9.0),
    "Pa": Unit(PRESSURE),
    "kPa": Unit(PRESSURE, 1e3),
    "MPa": Unit(PRESSURE, 1e6),
    "bar": Unit(PRESSURE, 1e5),
    "atm": Unit(PRESSURE, 101325.0),
    "psi": Unit(PRESSURE, 6894.757293168),
    "mol/s": Unit(FLOW),
    "mol/h": Unit(FLOW, 1.0 / 3600),
    "kmol/s": Unit(FLOW, 1e3),
    "kmol/h": Unit(FLOW, 1e3 / 3600),
    "kg/s": Unit(MASS_FLOW),
    "kg/h": Unit(MASS_FLOW, 1.0 / 3600),
    "W": Unit(POWER),
    "kW": Unit(POWER, 1e3),
    "MW": Unit(POWER, 1e6),
    "J": Unit(ENERGY),
    "kJ": Unit(ENERGY, 1e3),
    "MJ": Unit(ENERGY, 1e6),
    "GJ": Unit(ENERGY, 1e9),
    "kWh": Unit(ENERGY, 3.6e6),
    "J/mol": Unit(MOLAR_ENERGY),
    "kJ/mol": Unit(MOLAR_ENERGY, 1e3),
    "m": Unit(LENGTH),
    "ft": Unit(LENGTH, 0.3048),
    "m2": Unit(AREA),
    "ft2": Unit(AREA, 0.09290304),
    "m3": Unit(VOLUME),
    "L": Unit(VOLUME, 1e-3),
    "mol/m3": Unit(CONCENTRATION),
    "kmol/m3": Unit(CONCENTRATION, 1e3),
    "mol/L": Unit(CONCENTRATION, 1e3),
    "mol/(m3 s)": Unit(REACTION_RATE),
    "kmol/(m3 h)": Unit(REACTION_RATE, 1e3 / 3600),
    "s": Unit(TIME),
    "min": Unit(TIME, 60.0),
    "h": Unit(TIME, 3600.0),
    "USD": Unit(MONEY),
    "USD/yr": Unit(ANNUAL_COST, 1.0 / YEAR_SECONDS),
    "USD/J": Unit(ENERGY_PRICE),
    "USD/GJ": Unit(ENERGY_PRICE, 1e-9),
    "USD/kWh": Unit(ENERGY_PRICE, 1.0 / 3.6e6),
    "W/K": Unit(CONDUCTANCE),
    "kW/K": Unit(CONDUCTANCE, 1e3),
    "W/(m2 K)": Unit(FILM),
}


class CaseValidationError(ValueError):
    """Invalid case data, with a path identifying the responsible field."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


def number(value: Any, path: str) -> float:
    """Read a finite JSON number, rejecting booleans and implicit string casts."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CaseValidationError(path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CaseValidationError(path, "expected a finite number")
    return result


def integer(value: Any, path: str, minimum: int = 0, maximum: int = 10000) -> int:
    """Read a bounded JSON integer without truncating floats."""
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise CaseValidationError(path, f"expected an integer in [{minimum}, {maximum}]")
    return value


def object_fields(
    value: Any, path: str, *, allowed: set[str], required: set[str] | None = None
) -> dict[str, Any]:
    """Reject unknown or missing fields, retaining a useful nested error path."""
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise CaseValidationError(path, "expected an object")
    unknown = set(value) - allowed
    missing = (required or set()) - set(value)
    if unknown:
        raise CaseValidationError(path, f"unknown fields: {sorted(unknown)}")
    if missing:
        raise CaseValidationError(path, f"missing fields: {sorted(missing)}")
    return value


def unit_for(symbol: str, path: str = "unit") -> Unit:
    """Resolve a unit exactly; a misspelling never becomes a dimensionless value."""
    if not isinstance(symbol, str) or symbol not in UNITS:
        raise CaseValidationError(path, f"unknown unit {symbol!r}; choose from {sorted(UNITS)}")
    return UNITS[symbol]


def quantity(
    value: Any,
    dimension: Dimension,
    path: str,
    *,
    difference: bool = False,
) -> float:
    """Convert an explicit quantity, or a dimensionless bare number, into SI.

    Dimensional literals require ``{"value": ..., "unit": ...}``. Temperature
    differences reject affine absolute-temperature units.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        if dimension != DIMENSIONLESS:
            raise CaseValidationError(path, "dimensional values require explicit value and unit")
        return number(value, path)
    data = object_fields(value, path, allowed={"value", "unit"}, required={"value", "unit"})
    unit = unit_for(data["unit"], path + ".unit")
    if unit.dimension != dimension:
        raise CaseValidationError(path, f"unit {data['unit']!r} has the wrong dimension")
    if difference and unit.offset:
        raise CaseValidationError(path, "use delta_K, delta_degC, or delta_degF for a difference")
    if not difference and dimension == TEMPERATURE and str(data["unit"]).startswith("delta_"):
        raise CaseValidationError(path, "a temperature requires an absolute-temperature unit")
    return number(
        number(data["value"], path + ".value") * unit.factor + unit.offset, path + ".si_value"
    )


def display_value(value: float, symbol: str) -> float:
    """Convert an SI result to a requested display unit."""
    unit = unit_for(symbol)
    return (value - unit.offset) / unit.factor


def multiply_dimensions(left: Dimension, right: Dimension, sign: int = 1) -> Dimension:
    """Combine dimensions for multiplication or division."""
    return tuple(a + sign * b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]
