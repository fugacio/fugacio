"""Equipment sizing, Turton costing, utilities, and financial metrics.

Checks the correlations against known values and limits, that costs behave
sensibly (monotone in size, installed > purchased, pressure factor >= 1), and
that the money objectives are differentiable -- the whole point of a cost model
in a gradient-based design tool.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import economics as ec


def test_lmtd_known_and_equal_limit() -> None:
    # Classic example: approaches 100 and 50 -> LMTD ~ 72.13.
    assert float(ec.lmtd(100.0, 50.0)) == pytest.approx(72.134752, abs=1e-3)
    # Equal approaches -> the common value, with a finite gradient.
    assert float(ec.lmtd(30.0, 30.0)) == pytest.approx(30.0, abs=1e-6)
    g = jax.grad(lambda d: ec.lmtd(d, 30.0))(30.0)
    assert jnp.isfinite(g)


def test_heat_exchanger_area() -> None:
    # Q = U A LMTD  =>  A = Q/(U LMTD).
    area = ec.heat_exchanger_area(1.0e6, 500.0, 60.0, 40.0)
    expected = 1.0e6 / (500.0 * float(ec.lmtd(60.0, 40.0)))
    assert float(area) == pytest.approx(expected, rel=1e-6)


def test_purchased_cost_monotone_and_positive() -> None:
    small = float(ec.purchased_cost("heat_exchanger", 20.0))
    large = float(ec.purchased_cost("heat_exchanger", 200.0))
    assert 0.0 < small < large


def test_bare_module_exceeds_purchased_and_pressure_factor() -> None:
    item = ec.bare_module_cost("heat_exchanger", 100.0, pressure_barg=50.0, material="SS")
    assert float(item.bare_module) > float(item.purchased) > 0.0
    assert float(ec.pressure_factor("heat_exchanger", 50.0)) >= 1.0
    # Stainless costs more than carbon steel.
    cs = ec.bare_module_cost("heat_exchanger", 100.0, material="CS")
    ss = ec.bare_module_cost("heat_exchanger", 100.0, material="SS")
    assert float(ss.bare_module) > float(cs.bare_module)


def test_unknown_kinds_raise() -> None:
    with pytest.raises(KeyError):
        ec.purchased_cost("warp_drive", 10.0)
    with pytest.raises(KeyError):
        ec.utility_cost(1e6, "antimatter")


def test_utility_cost_formula() -> None:
    # 1 MW of cooling water for 8000 h: energy = 1e6 W * 8000 * 3600 s = 2.88e13 J = 2.88e4 GJ.
    cost = ec.utility_cost(1.0e6, "cooling_water")
    expected = 1.0e6 * 8000.0 * 3600.0 / 1e9 * ec.UTILITIES["cooling_water"].price_per_gj
    assert float(cost) == pytest.approx(expected, rel=1e-9)
    # Steam (heating) is far pricier than cooling water for the same duty.
    assert float(ec.utility_cost(1e6, "hp_steam")) > float(ec.utility_cost(1e6, "cooling_water"))


def test_capital_recovery_and_tac() -> None:
    # CRF(10%, 10 yr) ~ 0.16275.
    crf = float(ec.capital_recovery_factor(0.1, 10.0))
    assert crf == pytest.approx(0.162745, abs=1e-5)
    tac = ec.total_annual_cost(1.0e6, 2.0e5, rate=0.1, years=10.0)
    assert float(tac) == pytest.approx(crf * 1.0e6 + 2.0e5, rel=1e-6)


def test_npv_and_payback_consistency() -> None:
    # Uniform 250k/yr for 10 yr against 1M up-front, 10% discount.
    cash = jnp.concatenate([jnp.array([-1.0e6]), jnp.full((10,), 2.5e5)])
    value = ec.npv(cash, rate=0.1)
    assert float(value) > 0.0  # the project clears the hurdle rate
    payback = ec.discounted_payback(1.0e6, 2.5e5, rate=0.1)
    assert 4.0 < float(payback) < 6.0


def test_sizing_helpers_sane() -> None:
    d = ec.column_diameter(50.0, 5.0, 600.0, molar_mass=0.06)
    assert float(d) > 0.0
    assert float(ec.column_height(20.0)) == pytest.approx(20.0 * 0.6 + 4.0, abs=1e-6)
    assert float(ec.vessel_volume(0.01, residence_time=300.0, fill=0.5)) == pytest.approx(
        6.0, rel=1e-6
    )


def test_tac_is_differentiable_in_size() -> None:
    def tac_of_area(area: jax.Array) -> jax.Array:
        capex = ec.bare_module_cost("heat_exchanger", area, material="CS").bare_module
        opex = ec.utility_cost(5.0e5, "cooling_water")
        return ec.total_annual_cost(capex, opex)

    g = float(jax.grad(tac_of_area)(80.0))
    assert g > 0.0  # a bigger exchanger costs more
    assert jnp.isfinite(g)


def test_installed_capital_sums_items() -> None:
    items = [
        ec.bare_module_cost("pump", 20.0),
        ec.bare_module_cost("heat_exchanger", 100.0),
    ]
    total = ec.installed_capital(items)
    assert float(total) == pytest.approx(
        float(items[0].bare_module) + float(items[1].bare_module), rel=1e-9
    )
