"""Explicit utility prices and screening equipment economics for process cases."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from fugacio.sim.cases.expressions import evaluate_expression, expression_dimension
from fugacio.sim.cases.quantities import (
    AREA,
    DIMENSIONLESS,
    ENERGY_PRICE,
    MONEY,
    POWER,
    PRESSURE,
    TIME,
    VOLUME,
    YEAR_SECONDS,
    CaseValidationError,
    object_fields,
)
from fugacio.sim.cases.registry import UnitDefinition
from fugacio.sim.cases.schema import Parameter, identifier, resolve_value, sequence, value_spec

_SIZE_DIMENSIONS = {
    "heat_exchanger": AREA,
    "pump": POWER,
    "compressor": POWER,
    "vessel": VOLUME,
    "tray": AREA,
    "fired_heater": POWER,
}


def _no_cost_reference(value: Any, document: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return True
    if "metric" in value:
        return _no_cost_reference(document["metrics"][value["metric"]]["expression"], document)
    if value.get("plant") in (
        "annual_cost",
        "annual_utility_cost",
        "annualized_capital",
        "installed_capital",
    ):
        return False
    return all(_no_cost_reference(v, document) for v in value.get("args", []))


def parse_economics(
    document: dict[str, Any], parameters: dict[str, Parameter]
) -> dict[str, Any] | None:
    """Compile declared utility prices, operating duration, and annualization inputs."""
    if "economics" not in document:
        return None
    d = object_fields(
        document["economics"],
        "economics",
        allowed={
            "operating_time",
            "heating_price",
            "cooling_price",
            "electricity_price",
            "fixed_capital",
            "interest_rate",
            "years",
            "equipment",
        },
        required={"operating_time", "heating_price", "cooling_price", "electricity_price"},
    )
    dimensions = {
        "operating_time": TIME,
        "heating_price": ENERGY_PRICE,
        "cooling_price": ENERGY_PRICE,
        "electricity_price": ENERGY_PRICE,
        "fixed_capital": MONEY,
        "interest_rate": DIMENSIONLESS,
        "years": DIMENSIONLESS,
    }
    result = {
        k: value_spec(v, dimensions[k], parameters, "economics." + k)
        for k, v in d.items()
        if k != "equipment"
    }
    result.setdefault("fixed_capital", 0.0)
    result.setdefault("interest_rate", 0.1)
    result.setdefault("years", 10.0)
    result["equipment"] = d.get("equipment", [])
    return result


def validate_economic_values(compiled: dict[str, Any] | None, parameters: dict[str, float]) -> None:
    """Reject negative utility prices, invalid annualization, and impossible operating time."""
    if compiled is None:
        return
    values = {
        k: parameters[v] if isinstance(v, str) else float(v)
        for k, v in compiled.items()
        if k != "equipment"
    }
    if not 0 < values["operating_time"] <= YEAR_SECONDS:
        raise CaseValidationError(
            "economics.operating_time", "must be positive and no longer than one year"
        )
    if values["years"] <= 0 or any(
        values[k] < 0
        for k in (
            "heating_price",
            "cooling_price",
            "electricity_price",
            "fixed_capital",
            "interest_rate",
        )
    ):
        raise CaseValidationError(
            "economics", "prices, capital, and interest must be nonnegative; years must be positive"
        )


def validate_economics(
    document: dict[str, Any], units: tuple[UnitDefinition, ...], parameters: dict[str, Parameter]
) -> None:
    """Validate the explicit economics declaration and dimensioned equipment sizes."""
    compiled = parse_economics(document, parameters)
    if compiled is None:
        return
    validate_economic_values(compiled, {k: p.value for k, p in parameters.items()})
    names = set()
    for i, raw in enumerate(sequence(compiled["equipment"], "economics.equipment", maximum=1000)):
        path = f"economics.equipment[{i}]"
        d = object_fields(
            raw,
            path,
            allowed={"name", "kind", "size", "pressure", "material", "cepci"},
            required={"name", "kind", "size", "cepci"},
        )
        name = identifier(d["name"], path + ".name")
        if name in names:
            raise CaseValidationError(path, "equipment names must be unique")
        names.add(name)
        if not isinstance(d["kind"], str) or d["kind"] not in _SIZE_DIMENSIONS:
            raise CaseValidationError(path + ".kind", f"choose from {sorted(_SIZE_DIMENSIONS)}")
        if (
            expression_dimension(d["size"], document, units, parameters, path + ".size")
            != _SIZE_DIMENSIONS[d["kind"]]
        ):
            raise CaseValidationError(path + ".size", "equipment size has the wrong dimension")
        if (
            "pressure" in d
            and expression_dimension(d["pressure"], document, units, parameters, path + ".pressure")
            != PRESSURE
        ):
            raise CaseValidationError(path + ".pressure", "expected an absolute pressure")
        if not _no_cost_reference(d["size"], document) or not _no_cost_reference(
            d.get("pressure"), document
        ):
            raise CaseValidationError(path, "equipment sizing cannot depend on computed costs")
        if d.get("material", "CS") not in ("CS", "SS", "Ni", "Ti"):
            raise CaseValidationError(path + ".material", "choose CS, SS, Ni, or Ti")
        cepci = value_spec(d["cepci"], DIMENSIONLESS, parameters, path + ".cepci")
        if (parameters[cepci].value if isinstance(cepci, str) else cepci) <= 0:
            raise CaseValidationError(path + ".cepci", "CEPCI must be positive")


def evaluate_economics(
    evaluation: Any,
    document: dict[str, Any],
    package: Any,
    parameters: dict[str, Parameter],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Calculate utility and annualized screening capital costs from retained unit results.

    SI cost rates are USD/s; display with ``USD/yr``. Recovered shaft work is
    reported separately and receives no automatic electricity-sale credit.
    Correlation range failures remain visible even though the legacy cost
    function clips its size argument internally.
    """
    from fugacio.sim.economics import _TURTON, bare_module_cost

    compiled = parse_economics(document, parameters)
    if compiled is None:
        return {}, {}
    options = {
        k: resolve_value(v, evaluation.parameters) for k, v in compiled.items() if k != "equipment"
    }
    seconds = options["operating_time"]
    utility = (
        sum(
            evaluation.plant[k] * options[k + "_price"]
            for k in ("heating", "cooling", "electricity")
        )
        * seconds
        / YEAR_SECONDS
    )
    capital = options["fixed_capital"]
    items = {}
    valid = jnp.asarray(True)
    for raw in compiled["equipment"]:
        kind = raw["kind"]
        size = evaluate_expression(raw["size"], evaluation, document, package)
        basis = size / 1000.0 if _SIZE_DIMENSIONS[kind] == POWER else size
        pressure = (
            evaluate_expression(raw["pressure"], evaluation, document, package)
            if "pressure" in raw
            else jnp.asarray(101325.0)
        )
        cepci = resolve_value(
            value_spec(raw["cepci"], DIMENSIONLESS, parameters, "cepci"), evaluation.parameters
        )
        cost = bare_module_cost(
            kind,
            basis,
            pressure_barg=jnp.maximum((pressure - 101325.0) / 1e5, 0),
            material=raw.get("material", "CS"),
            cepci=cepci,
        )
        bounds = _TURTON[kind]
        accepted = (
            jnp.isfinite(cost.bare_module)
            & (basis >= bounds.size_min)
            & (basis <= bounds.size_max)
            & (pressure > 0)
            & (cepci > 0)
        )
        valid = valid & accepted
        capital = capital + cost.bare_module
        items[raw["name"]] = {
            "size_si": size,
            "purchased_usd": cost.purchased,
            "installed_usd": cost.bare_module,
            "within_size_range": accepted,
            "cepci": cepci,
        }
    rate, years = options["interest_rate"], options["years"]
    # Stable zero-interest limit; avoid differentiating an unused 0/0 branch.
    safe_rate = jnp.where(jnp.abs(rate) < 1e-7, 1e-7, rate)
    growth = jnp.expm1(years * jnp.log1p(safe_rate))
    crf = jnp.where(
        jnp.abs(rate) < 1e-7,
        1 / years + rate * (years + 1) / (2 * years),
        safe_rate * (1 + growth) / growth,
    )
    annual_capital = capital * crf / YEAR_SECONDS
    return {
        "installed_capital": capital,
        "annual_utility_cost": utility,
        "annualized_capital": annual_capital,
        "annual_cost": utility + annual_capital,
        "cost_valid": valid & jnp.isfinite(utility + annual_capital),
    }, items
