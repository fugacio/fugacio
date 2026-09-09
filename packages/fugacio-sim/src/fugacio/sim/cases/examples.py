"""Executable, exportable process examples using the portable case vocabulary."""

from __future__ import annotations

from typing import Any

from fugacio.sim.cases.schema import ProcessCase

EXAMPLES = (
    "heater",
    "recycle",
    "depropanizer",
    "reaction",
    "measured-heater",
    "ethanol-train",
    "heater-bank",
)


def _q(value: float, unit: str) -> dict[str, Any]:
    return {"value": value, "unit": unit}


def _p(name: str) -> dict[str, str]:
    return {"parameter": name}


def _metric(expression: Any, unit: str) -> dict[str, Any]:
    return {"expression": expression, "unit": unit}


def example_case(name: str = "heater") -> ProcessCase:
    """Construct a built-in case that can be exported and reopened as plain JSON.

    Economic prices and CEPCI are illustrative declared assumptions, not live
    market data. The depropanizer reproduces the existing rigorous plant test's
    95% propane purity and 98% recovery with feed/bottoms heat recovery.
    """
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; choose from {EXAMPLES}")
    if name == "ethanol-train":
        return ethanol_train_case()
    if name == "heater-bank":
        return heater_bank_case()
    document: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "components": ["methane", "ethane"],
        "property_package": {"method": "pr"},
        "parameters": {"temperature": {"value": 350, "unit": "K", "lower": 310, "upper": 400}},
        "feeds": {
            "feed": {
                "flow": _q(1, "mol/s"),
                "z": [0.8, 0.2],
                "temperature": _q(300, "K"),
                "pressure": _q(1, "bar"),
            }
        },
        "units": [
            {
                "name": "heater",
                "kind": "heater",
                "inlets": ["feed"],
                "outlets": ["product"],
                "settings": {"t_out": _p("temperature")},
            }
        ],
        "metrics": {
            "duty": _metric({"unit": "heater", "property": "heat"}, "kW"),
            "product_temperature": _metric({"stream": "product", "property": "temperature"}, "K"),
            "annual_cost": _metric({"plant": "annual_cost"}, "USD/yr"),
        },
        "economics": {
            "operating_time": _q(8000, "h"),
            "heating_price": _q(8, "USD/GJ"),
            "cooling_price": _q(0.4, "USD/GJ"),
            "electricity_price": _q(0.12, "USD/kWh"),
        },
    }
    if name == "recycle":
        document["parameters"]["recycle_fraction"] = {
            "value": 0.4,
            "unit": "1",
            "lower": 0.1,
            "upper": 0.8,
        }
        document["parameters"]["purge_fraction"] = {
            "value": 0.6,
            "unit": "1",
            "lower": 0.2,
            "upper": 0.9,
        }
        # The two independent fractions must sum to one. A single-variable
        # recycle study should revise both together, retaining explicit closure.
        document["units"] = [
            {
                "name": "mixer",
                "kind": "mixer",
                "inlets": ["feed", "recycle"],
                "outlets": ["mixed"],
                "settings": {"t": _q(300, "K")},
            },
            {
                "name": "heater",
                "kind": "heater",
                "inlets": ["mixed"],
                "outlets": ["heated"],
                "settings": {"t_out": _p("temperature")},
            },
            {
                "name": "split",
                "kind": "splitter",
                "inlets": ["heated"],
                "outlets": ["recycle", "product"],
                "settings": {"fractions": [_p("recycle_fraction"), _p("purge_fraction")]},
            },
        ]
    if name == "depropanizer":
        document.update(
            description=(
                "Rigorous 16-stage propane/butane/pentane separation with a "
                "feed/bottoms economizer. Cost inputs are illustrative screening "
                "assumptions."
            ),
            components=["propane", "n-butane", "n-pentane"],
            parameters={
                "pressure": {"value": 16, "unit": "bar", "lower": 14, "upper": 18},
                "approach": {"value": 15, "unit": "delta_K", "lower": 10, "upper": 25},
            },
            feeds={
                "feed": {
                    "flow": _q(100, "mol/s"),
                    "z": [0.4, 0.35, 0.25],
                    "temperature": _q(300, "K"),
                    "pressure": _p("pressure"),
                }
            },
            units=[
                {
                    "name": "economizer",
                    "kind": "heat_exchanger",
                    "inlets": ["bottoms", "feed"],
                    "outlets": ["bottoms_cooled", "preheated"],
                    "settings": {"min_approach": _p("approach"), "u": _q(600, "W/(m2 K)")},
                },
                {
                    "name": "column",
                    "kind": "column",
                    "inlets": ["preheated"],
                    "outlets": ["distillate", "bottoms"],
                    "settings": {
                        "n_stages": 16,
                        "feed_stages": [8],
                        "p": _p("pressure"),
                        "specs": [
                            {
                                "kind": "recovery",
                                "component": "propane",
                                "product": "distillate",
                                "value": 0.98,
                            },
                            {
                                "kind": "mole_fraction",
                                "component": "propane",
                                "product": "distillate",
                                "value": 0.95,
                            },
                        ],
                    },
                },
            ],
            metrics={
                "purity": _metric(
                    {"stream": "distillate", "property": "mole_fraction", "component": "propane"},
                    "%",
                ),
                "recovery": _metric(
                    {
                        "op": "divide",
                        "args": [
                            {
                                "stream": "distillate",
                                "property": "component_flow",
                                "component": "propane",
                            },
                            {
                                "stream": "feed",
                                "property": "component_flow",
                                "component": "propane",
                            },
                        ],
                    },
                    "%",
                ),
                "reboiler_duty": _metric({"unit": "column", "property": "reboiler_duty"}, "MW"),
                "recovered_heat": _metric({"unit": "economizer", "property": "duty"}, "MW"),
                "exchanger_area": _metric({"unit": "economizer", "property": "area"}, "m2"),
                "annual_cost": _metric({"plant": "annual_cost"}, "USD/yr"),
            },
        )
        document["economics"].update(
            fixed_capital=_q(1e6, "USD"),
            interest_rate=0.1,
            years=10,
            equipment=[
                {
                    "name": "economizer",
                    "kind": "heat_exchanger",
                    "size": {"unit": "economizer", "property": "area"},
                    "pressure": _p("pressure"),
                    "cepci": 800,
                    "material": "CS",
                }
            ],
        )
    if name == "reaction":
        document.update(
            description=(
                "Isothermal toluene hydrodealkylation with explicit component "
                "generation and standard formation enthalpies."
            ),
            components=["hydrogen", "toluene", "benzene", "methane"],
            parameters={"conversion": {"value": 0.6, "unit": "1", "lower": 0.1, "upper": 0.9}},
            feeds={
                "feed": {
                    "flow": _q(10, "mol/s"),
                    "z": [0.7, 0.3, 0, 0],
                    "temperature": _q(850, "K"),
                    "pressure": _q(20, "bar"),
                }
            },
            units=[
                {
                    "name": "reactor",
                    "kind": "stoichiometric_reactor",
                    "inlets": ["feed"],
                    "outlets": ["product"],
                    "settings": {
                        "nu": [-1, -1, 1, 1],
                        "key": "toluene",
                        "conversion": _p("conversion"),
                        "t_out": _q(850, "K"),
                    },
                }
            ],
            metrics={
                "benzene_production": _metric(
                    {"stream": "product", "property": "component_flow", "component": "benzene"},
                    "mol/s",
                ),
                "duty": _metric({"unit": "reactor", "property": "heat"}, "kW"),
                "annual_cost": _metric({"plant": "annual_cost"}, "USD/yr"),
            },
        )
    if name == "measured-heater":
        from importlib.resources import files

        from fugacio.sim.cases.jsonio import loads

        evidence = loads(
            files("fugacio.sim.cases").joinpath("data", "ethanol-water-evidence.json").read_text()
        )
        document.update(
            description=(
                "Ethanol-water heater with a measured NRTL fit and an "
                "independently reevaluated publication holdout."
            ),
            components=["ethanol", "water"],
            property_package={
                "method": "nrtl",
                "measured_fit": evidence["measured_fit"],
                "qualification": {"holdout_ids": evidence["holdout_ids"]},
            },
            parameters={"temperature": {"value": 335, "unit": "K", "lower": 331, "upper": 335.1}},
            feeds={
                "feed": {
                    "flow": _q(10, "mol/s"),
                    "z": [0.5, 0.5],
                    "temperature": _q(330, "K"),
                    "pressure": _q(20000, "Pa"),
                }
            },
        )
    return ProcessCase.from_dict(document)


def ethanol_train_case(n_stages: int = 12) -> ProcessCase:
    """An NRTL ethanol-water column with feed/bottoms heat recovery.

    Uses the curated parameter case from the existing numerical plant test.
    This performance example makes no new measured qualification claim.
    """
    if isinstance(n_stages, bool) or not isinstance(n_stages, int) or not 6 <= n_stages <= 100:
        raise ValueError("ethanol train needs from six to 100 stages")
    document = example_case("depropanizer").to_dict()
    document.update(
        name="ethanol-train",
        description="Curated NRTL ethanol-water heat-recovery train; numerical benchmark scope.",
        components=["ethanol", "water"],
        property_package={"method": "nrtl"},
        parameters={
            "pressure": {"value": 1.013, "unit": "bar", "lower": 1.0, "upper": 1.1},
            "approach": {"value": 10, "unit": "delta_K", "lower": 8, "upper": 20},
            "reflux": {"value": 3, "unit": "1", "lower": 2.5, "upper": 5},
        },
        feeds={
            "feed": {
                "flow": _q(100, "mol/s"),
                "z": [0.1, 0.9],
                "temperature": _q(300, "K"),
                "pressure": _q(1.5, "bar"),
            }
        },
    )
    document["units"].insert(
        1,
        {
            "name": "letdown",
            "kind": "valve",
            "inlets": ["preheated"],
            "outlets": ["column_feed"],
            "settings": {"p_out": _p("pressure")},
        },
    )
    column = document["units"][-1]
    column["inlets"] = ["column_feed"]
    column["settings"].update(
        n_stages=n_stages,
        feed_stages=[n_stages // 2],
        specs=[
            {"kind": "reflux_ratio", "value": _p("reflux")},
            {"kind": "distillate_rate", "value": _q(11.5, "mol/s")},
        ],
    )
    document["metrics"]["purity"]["expression"]["component"] = "ethanol"
    for expression in document["metrics"]["recovery"]["expression"]["args"]:
        expression["component"] = "ethanol"
    return ProcessCase.from_dict(document)


def heater_bank_case(count: int = 24) -> ProcessCase:
    """Independent energy-balanced heaters for many-variable derivative studies.

    Each heater has its own fresh feed, outlet, and bounded temperature. Total
    heat and annual cost depend on every variable, so a reverse derivative
    measures the cost of increasing the parameter count without adding an
    artificial optimization objective.
    """
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 80:
        raise ValueError("heater bank needs from two to 80 units")
    document = example_case("heater").to_dict()
    feed = document["feeds"]["feed"]
    document.update(name="heater-bank", parameters={}, feeds={}, units=[], metrics={})
    terms: list[Any] = []
    for i in range(count):
        name = f"heater_{i:02d}"
        parameter = f"temperature_{i:02d}"
        document["parameters"][parameter] = {
            "value": 350 + i / count,
            "unit": "K",
            "lower": 320,
            "upper": 400,
        }
        document["feeds"][f"feed_{i:02d}"] = feed
        document["units"].append(
            {
                "name": name,
                "kind": "heater",
                "inlets": [f"feed_{i:02d}"],
                "outlets": [f"product_{i:02d}"],
                "settings": {"t_out": _p(parameter)},
            }
        )
        terms.append({"unit": name, "property": "heat"})
    # Balanced expression tree stays within the portable expression depth cap.
    while len(terms) > 1:
        terms = [
            {"op": "add", "args": terms[i : i + 2]} if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    document["metrics"] = {
        "duty": _metric(terms[0], "kW"),
        "annual_cost": _metric({"plant": "annual_cost"}, "USD/yr"),
    }
    return ProcessCase.from_dict(document)
