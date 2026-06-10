"""Steam and cooling-water utilities on IAPWS-95: physics and differentiability.

Anchor values are textbook steam-table numbers (latent heat at 1 atm,
saturation temperatures of standard headers), so the tests are hermetic; the
oracle-grade validation of the underlying EOS lives in ``fugacio-thermo``.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    STEAM_LEVELS,
    condensate_flash_fraction,
    cooling_water,
    saturated_steam_temperature,
    steam_enthalpy,
    steam_heating,
    steam_quality_after_letdown,
    steam_turbine,
)

M_WATER = 0.018015268


def test_steam_heating_latent_heat_at_atmospheric() -> None:
    result = steam_heating(2256.5e3, pressure=101325.0)
    # 2256.5 kJ/kg latent heat at 1 atm -> almost exactly 1 kg/s.
    assert float(result.mass_flow) == pytest.approx(1.0, rel=1e-3)
    assert float(result.t_steam) == pytest.approx(373.124, abs=5e-3)
    assert float(result.dh_specific) == pytest.approx(2256.5e3, rel=1e-3)


def test_steam_headers_have_textbook_saturation_temperatures() -> None:
    # 5 bar -> 151.8 C; 11 bar -> 184.1 C; 42 bar -> 253.2 C.
    assert float(saturated_steam_temperature(STEAM_LEVELS["lp"])) == pytest.approx(425.0, abs=0.3)
    assert float(saturated_steam_temperature(STEAM_LEVELS["mp"])) == pytest.approx(457.2, abs=0.3)
    assert float(saturated_steam_temperature(STEAM_LEVELS["hp"])) == pytest.approx(526.4, abs=0.3)


def test_higher_pressure_steam_carries_less_latent_heat() -> None:
    lp = steam_heating(1e6, pressure=STEAM_LEVELS["lp"])
    hp = steam_heating(1e6, pressure=STEAM_LEVELS["hp"])
    assert float(lp.dh_specific) > float(hp.dh_specific)
    assert float(hp.mass_flow) > float(lp.mass_flow)


def test_superheat_and_subcooling_reduce_steam_demand() -> None:
    base = steam_heating(1e6, pressure=5e5)
    superheated = steam_heating(1e6, pressure=5e5, superheat=50.0)
    subcooled = steam_heating(1e6, pressure=5e5, condensate_subcooling=20.0)
    assert float(superheated.mass_flow) < float(base.mass_flow)
    assert float(subcooled.mass_flow) < float(base.mass_flow)
    assert float(superheated.t_steam) == pytest.approx(float(base.t_steam) + 50.0)


def test_steam_heating_uses_duty_magnitude() -> None:
    condenser_duty = -3.5e6  # sign convention of heater(): negative = cooling
    result = steam_heating(condenser_duty, pressure=5e5)
    assert float(result.mass_flow) > 0.0
    assert float(result.duty) == pytest.approx(3.5e6)


def test_cooling_water_flow_matches_cp_estimate() -> None:
    result = cooling_water(1e6, t_supply=303.15, t_return=318.15)
    # ~ cp 4.18 kJ/kg/K * 15 K = 62.7 kJ/kg -> 15.9 kg/s; IAPWS within 1 %.
    assert float(result.mass_flow) == pytest.approx(1e6 / (4180.0 * 15.0), rel=1e-2)
    assert float(result.molar_flow) * M_WATER == pytest.approx(float(result.mass_flow), rel=1e-12)


def test_steam_turbine_against_steam_table_case() -> None:
    """Classic Rankine expansion: 40 bar / 450 C to 1 bar at eta = 1 and 0.75."""
    ideal = steam_turbine(1.0, p_in=40e5, t_in=723.15, p_out=1e5, isentropic_efficiency=1.0)
    # Steam tables: h_in ~ 3330.3 kJ/kg and s_in ~ 6.9363 kJ/kg/K; isentropic
    # outlet at 1 bar is wet (q ~ 0.930, h_s ~ 2517.6 kJ/kg), so the ideal
    # specific work is ~ 812.7 kJ/kg. With 1 kg/s the power equals it in W.
    assert float(ideal.power) == pytest.approx(812.7e3, rel=1e-2)
    assert bool(ideal.two_phase)
    assert float(ideal.q_out) == pytest.approx(0.930, abs=0.01)

    real = steam_turbine(1.0, p_in=40e5, t_in=723.15, p_out=1e5, isentropic_efficiency=0.75)
    assert float(real.power) == pytest.approx(0.75 * float(ideal.power), rel=1e-12)
    assert float(real.h_out) > float(ideal.h_out)
    assert float(real.t_out) >= float(ideal.t_out) - 1e-6


def test_steam_turbine_power_scales_with_mass_flow() -> None:
    one = steam_turbine(1.0, p_in=40e5, t_in=723.15, p_out=5e5)
    ten = steam_turbine(10.0, p_in=40e5, t_in=723.15, p_out=5e5)
    assert float(ten.power) == pytest.approx(10.0 * float(one.power), rel=1e-12)


def test_letdown_of_saturated_steam_stays_wet_or_superheats() -> None:
    # Saturated HP steam let down to LP: slightly superheated (q is nan).
    q = steam_quality_after_letdown(STEAM_LEVELS["hp"], STEAM_LEVELS["lp"])
    assert bool(jnp.isnan(q))
    # With superheat it must remain single-phase too.
    q_sh = steam_quality_after_letdown(STEAM_LEVELS["hp"], STEAM_LEVELS["lp"], superheat=30.0)
    assert bool(jnp.isnan(q_sh))


def test_condensate_flash_fraction_textbook_value() -> None:
    # HP condensate (42 bar) flashed to LP (5 bar): from steam tables
    # q = (h_f(42 bar) - h_f(5 bar)) / h_fg(5 bar) ~ (1101.6 - 640.2)/2108.5 ~ 0.219.
    fraction = float(condensate_flash_fraction(STEAM_LEVELS["hp"], STEAM_LEVELS["lp"]))
    assert fraction == pytest.approx(0.219, abs=0.005)
    # Letting down to the same pressure produces no flash steam.
    assert float(condensate_flash_fraction(5e5, 5e5)) == pytest.approx(0.0, abs=1e-9)


def test_steam_enthalpy_specs() -> None:
    h_sat_vap = steam_enthalpy(101325.0, quality=1.0)
    h_sat_liq = steam_enthalpy(101325.0, quality=0.0)
    assert float(h_sat_vap - h_sat_liq) / M_WATER == pytest.approx(2256.5e3, rel=1e-3)
    h_superheated = steam_enthalpy(101325.0, temperature=473.15)
    assert float(h_superheated) > float(h_sat_vap)
    with pytest.raises(ValueError, match="exactly one"):
        steam_enthalpy(101325.0)
    with pytest.raises(ValueError, match="exactly one"):
        steam_enthalpy(101325.0, quality=0.5, temperature=400.0)


def test_steam_consumption_is_differentiable_in_header_pressure() -> None:
    """d(mass flow)/d(P_header): the knob a utility optimizer turns."""

    def flow(p: jnp.ndarray) -> jnp.ndarray:
        return steam_heating(1e6, pressure=p).mass_flow

    p0 = jnp.asarray(8e5)
    ad = float(jax.grad(flow)(p0))
    eps = 1e3
    fd = float((flow(p0 + eps) - flow(p0 - eps)) / (2 * eps))
    assert ad == pytest.approx(fd, rel=1e-5)
    assert ad > 0.0  # higher header pressure -> less latent heat -> more steam


def test_turbine_power_is_differentiable_in_backpressure() -> None:
    def power(p_out: jnp.ndarray) -> jnp.ndarray:
        return steam_turbine(1.0, p_in=40e5, t_in=723.15, p_out=p_out).power

    p0 = jnp.asarray(2e5)
    ad = float(jax.grad(power)(p0))
    eps = 1e2
    fd = float((power(p0 + eps) - power(p0 - eps)) / (2 * eps))
    assert ad == pytest.approx(fd, rel=1e-4)
    assert ad < 0.0  # raising backpressure costs power
