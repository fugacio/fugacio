"""NIST ThermoML reader: parsing, unit handling, and a regression round-trip."""

import math

import jax.numpy as jnp
import pytest

from fugacio.thermo import (
    component_arrays,
    fit_nrtl_binary,
    list_samples,
    load_sample,
    read_thermoml,
    sample_path,
)
from fugacio.thermo import thermoml as tml

ETHANOL_WATER = "ethanol_water_vle_323K"
WATER_PSAT = "water_vapor_pressure"


def test_list_samples_includes_bundled() -> None:
    samples = list_samples()
    assert ETHANOL_WATER in samples
    assert WATER_PSAT in samples


def test_unknown_sample_raises() -> None:
    with pytest.raises(FileNotFoundError):
        sample_path("does_not_exist")


def test_ethanol_water_compounds_parsed() -> None:
    data = load_sample(ETHANOL_WATER)
    assert {c.name for c in data.compounds} == {"ethanol", "water"}
    ethanol = data.compound(1)
    assert ethanol.cas == "64-17-5"
    assert ethanol.formula == "C2H6O"
    assert ethanol.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_ethanol_water_dataset_shape() -> None:
    data = load_sample(ETHANOL_WATER)
    assert len(data.datasets) == 1
    ds = data.datasets[0]
    assert ds.components == (1, 2)
    assert ds.phase == "Liquid"
    assert len(ds) == 11
    assert data.component_names(ds) == ["ethanol", "water"]


def test_temperature_and_composition_columns() -> None:
    ds = load_sample(ETHANOL_WATER).datasets[0]
    assert all(t == 323.15 for t in ds.temperature())
    x1 = ds.mole_fraction(1)
    assert x1[0] == pytest.approx(0.05)
    assert x1[-1] == pytest.approx(0.95)


def test_pressure_unit_conversion_to_pa() -> None:
    ds = load_sample(ETHANOL_WATER).datasets[0]
    p_pa = ds.pressure()  # default Pa
    p_kpa = ds.pressure(unit="kPa")
    assert p_pa[0] == pytest.approx(24879.1, rel=1e-6)
    assert p_kpa[0] == pytest.approx(24.8791, rel=1e-6)
    # Every cell scaled by exactly 1000.
    assert all(a == pytest.approx(b * 1000.0) for a, b in zip(p_pa, p_kpa, strict=True))


def test_to_dict_is_aligned() -> None:
    ds = load_sample(ETHANOL_WATER).datasets[0]
    d = ds.to_dict()
    assert set(d) == {"Temperature, K", "Mole fraction", "Pressure, kPa"}
    assert all(len(v) == len(ds) for v in d.values())


def test_pure_water_vapor_pressure_sample() -> None:
    data = load_sample(WATER_PSAT)
    assert len(data.compounds) == 1
    ds = data.datasets[0]
    assert ds.components == (1,)
    assert len(ds) == 4
    p_pa = ds.pressure()
    # The 373.15 K row is one standard atmosphere.
    assert p_pa[-1] == pytest.approx(101325.0, rel=1e-6)


def test_loads_from_string_matches_path() -> None:
    text = sample_path(ETHANOL_WATER).read_text()
    from_str = tml.loads(text)
    from_path = read_thermoml(sample_path(ETHANOL_WATER))
    assert len(from_str.datasets[0]) == len(from_path.datasets[0])
    assert from_str.compounds == from_path.compounds


def test_regression_roundtrip_recovers_nrtl_from_thermoml() -> None:
    """Fit NRTL straight to the parsed ThermoML table; residual should be tiny."""
    ds = load_sample(ETHANOL_WATER).datasets[0]
    t = jnp.array(ds.temperature())
    x1 = jnp.array(ds.mole_fraction(1))
    x = jnp.stack([x1, 1.0 - x1], axis=1)
    p = jnp.array(ds.pressure())  # Pa

    arr = component_arrays(["ethanol", "water"])
    model, cost = fit_nrtl_binary(t, x, p, arr["tc"], arr["pc"], arr["omega"], alpha=0.3)

    # Data were generated from b12=330, b21=600 (K); the fit should recover them.
    assert float(model.b[0, 1]) == pytest.approx(330.0, rel=0.05)
    assert float(model.b[1, 0]) == pytest.approx(600.0, rel=0.05)
    m = x.shape[0]
    rmse_scaled = math.sqrt(2.0 * float(cost) / m)
    assert rmse_scaled < 1e-3
