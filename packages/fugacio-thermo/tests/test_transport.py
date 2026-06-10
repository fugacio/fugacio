"""Unit tests for transport properties (viscosity, conductivity, tension, diffusion).

Spot values are experimental (CRC Handbook / Poling et al. appendices) with
tolerances matching the documented accuracy of each route: tight for curated
DIPPR-style fits, loose for corresponding-states estimates. Mixture rules are
checked for their exact pure-component limits and for landing between the pure
values on well-behaved pairs. Everything must be differentiable and jittable.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo.components import get
from fugacio.thermo.transport.diffusivity import (
    diffusion_volume,
    fuller_diffusivity,
    gas_diffusivity,
    liquid_diffusivity,
    wilke_chang_diffusivity,
)
from fugacio.thermo.transport.surface_tension import (
    brock_bird_surface_tension,
    mixture_surface_tension,
    surface_tensions,
    winterfeld_scriven_davis,
)
from fugacio.thermo.transport.thermal_conductivity import (
    dippr9h_mixture,
    gas_mixture_thermal_conductivity,
    gas_thermal_conductivities,
    liquid_mixture_thermal_conductivity,
    liquid_thermal_conductivities,
    sato_riedel_thermal_conductivity,
    wassiljewa_mixture,
)
from fugacio.thermo.transport.viscosity import (
    chung_viscosity_gas,
    gas_mixture_viscosity,
    gas_viscosities,
    grunberg_nissan_viscosity,
    letsou_stiel_viscosity,
    liquid_mixture_viscosity,
    liquid_viscosities,
    neufeld_collision_integral,
    wilke_mixture_viscosity,
)


class TestViscosity:
    def test_neufeld_collision_integral_table_values(self) -> None:
        # Lennard-Jones Omega_v from the standard tabulation (Poling, app. B).
        assert float(neufeld_collision_integral(1.0)) == pytest.approx(1.593, rel=0.005)
        assert float(neufeld_collision_integral(2.0)) == pytest.approx(1.175, rel=0.005)
        assert float(neufeld_collision_integral(10.0)) == pytest.approx(0.8242, rel=0.005)

    def test_chung_gas_viscosity_methane(self) -> None:
        # Methane at 300 K: 11.2 micropoise*... = 1.12e-5 Pa*s experimental.
        comp = get("methane")
        assert comp.vc is not None
        got = float(chung_viscosity_gas(300.0, comp.tc, comp.vc, comp.omega, comp.mw))
        assert got == pytest.approx(1.12e-5, rel=0.05)

    def test_letsou_stiel_magnitude(self) -> None:
        # n-decane at 0.8*Tc = 494.2 K: reference value 1.46e-4 Pa*s (CoolProp);
        # Letsou-Stiel is a ~10-15% method inside its 0.76 < Tr < 0.98 window.
        comp = get("n-decane")
        got = float(letsou_stiel_viscosity(0.8 * comp.tc, comp.tc, comp.pc, comp.omega, comp.mw))
        assert got == pytest.approx(1.46e-4, rel=0.20)

    @pytest.mark.parametrize(
        ("name", "t", "want", "rel"),
        [
            ("water", 298.15, 8.90e-4, 0.05),  # CRC
            ("n-hexane", 298.15, 3.00e-4, 0.10),
            ("ethanol", 298.15, 1.074e-3, 0.10),
        ],
    )
    def test_liquid_viscosity_spot_values(
        self, name: str, t: float, want: float, rel: float
    ) -> None:
        assert float(liquid_viscosities([name], t)[0]) == pytest.approx(want, rel=rel)

    @pytest.mark.parametrize(
        ("name", "t", "want", "rel"),
        [
            ("nitrogen", 300.0, 1.79e-5, 0.05),
            ("oxygen", 300.0, 2.07e-5, 0.05),
            ("methane", 300.0, 1.12e-5, 0.05),
        ],
    )
    def test_gas_viscosity_spot_values(self, name: str, t: float, want: float, rel: float) -> None:
        assert float(gas_viscosities([name], t)[0]) == pytest.approx(want, rel=rel)

    def test_wilke_pure_limit(self) -> None:
        mu = jnp.array([1.1e-5, 1.9e-5])
        mw = jnp.array([16.0, 28.0])
        got = float(wilke_mixture_viscosity(jnp.array([1.0, 0.0]), mu, mw))
        assert got == pytest.approx(1.1e-5, rel=1e-12)

    def test_air_mixture_viscosity(self) -> None:
        # Air at 300 K: 1.846e-5 Pa*s.
        got = float(
            gas_mixture_viscosity(
                ["nitrogen", "oxygen", "argon"], 300.0, jnp.array([0.781, 0.209, 0.01])
            )
        )
        assert got == pytest.approx(1.846e-5, rel=0.05)

    def test_grunberg_nissan_pure_limit_and_interaction(self) -> None:
        mu = jnp.array([5.0e-4, 1.0e-3])
        x = jnp.array([1.0, 0.0])
        assert float(grunberg_nissan_viscosity(x, mu)) == pytest.approx(5.0e-4, rel=1e-12)
        # A positive interaction parameter raises the mixture viscosity.
        x = jnp.array([0.5, 0.5])
        g = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        assert float(grunberg_nissan_viscosity(x, mu, g)) > float(grunberg_nissan_viscosity(x, mu))

    def test_liquid_mixture_between_pure_values(self) -> None:
        comps = ["benzene", "toluene"]
        mu = liquid_viscosities(comps, 298.15)
        mid = float(liquid_mixture_viscosity(comps, 298.15, jnp.array([0.5, 0.5])))
        assert float(jnp.min(mu)) < mid < float(jnp.max(mu))


class TestThermalConductivity:
    @pytest.mark.parametrize(
        ("name", "t", "want", "rel"),
        [
            ("nitrogen", 300.0, 0.0259, 0.05),
            ("carbon dioxide", 300.0, 0.0167, 0.07),
        ],
    )
    def test_gas_conductivity_spot_values(
        self, name: str, t: float, want: float, rel: float
    ) -> None:
        assert float(gas_thermal_conductivities([name], t)[0]) == pytest.approx(want, rel=rel)

    @pytest.mark.parametrize(
        ("name", "t", "want", "rel"),
        [
            ("water", 298.15, 0.607, 0.05),
            ("benzene", 298.15, 0.141, 0.10),
        ],
    )
    def test_liquid_conductivity_spot_values(
        self, name: str, t: float, want: float, rel: float
    ) -> None:
        assert float(liquid_thermal_conductivities([name], t)[0]) == pytest.approx(want, rel=rel)

    def test_sato_riedel_magnitude(self) -> None:
        # Toluene at 298 K: 0.131 W/m/K experimental; Sato-Riedel is a ~15-25% method.
        comp = get("toluene")
        assert comp.tb is not None
        got = float(sato_riedel_thermal_conductivity(298.15, comp.tc, comp.tb, comp.mw))
        assert got == pytest.approx(0.131, rel=0.25)

    def test_wassiljewa_pure_limit(self) -> None:
        k = jnp.array([0.026, 0.017])
        mu = jnp.array([1.79e-5, 1.49e-5])
        mw = jnp.array([28.0, 44.0])
        got = float(wassiljewa_mixture(jnp.array([0.0, 1.0]), k, mu, mw))
        assert got == pytest.approx(0.017, rel=1e-12)

    def test_dippr9h_pure_limit_and_bounds(self) -> None:
        k = jnp.array([0.59, 0.16])
        assert float(dippr9h_mixture(jnp.array([1.0, 0.0]), k)) == pytest.approx(0.59, rel=1e-12)
        mid = float(dippr9h_mixture(jnp.array([0.5, 0.5]), k))
        assert 0.16 < mid < 0.59

    def test_gas_mixture_between_pure_values(self) -> None:
        comps = ["nitrogen", "carbon dioxide"]
        k = gas_thermal_conductivities(comps, 300.0)
        mid = float(gas_mixture_thermal_conductivity(comps, 300.0, jnp.array([0.5, 0.5])))
        assert float(jnp.min(k)) < mid < float(jnp.max(k))

    def test_liquid_mixture_between_pure_values(self) -> None:
        comps = ["water", "ethanol"]
        k = liquid_thermal_conductivities(comps, 298.15)
        mid = float(liquid_mixture_thermal_conductivity(comps, 298.15, jnp.array([0.5, 0.5])))
        assert float(jnp.min(k)) < mid < float(jnp.max(k))


class TestSurfaceTension:
    @pytest.mark.parametrize(
        ("name", "t", "want", "rel"),
        [
            ("water", 298.15, 0.0720, 0.03),
            ("benzene", 298.15, 0.0282, 0.04),
            ("ethanol", 298.15, 0.0220, 0.05),
        ],
    )
    def test_spot_values(self, name: str, t: float, want: float, rel: float) -> None:
        assert float(surface_tensions([name], t)[0]) == pytest.approx(want, rel=rel)

    def test_brock_bird_benzene(self) -> None:
        comp = get("benzene")
        assert comp.tb is not None
        got = float(brock_bird_surface_tension(298.15, comp.tb, comp.tc, comp.pc))
        assert got == pytest.approx(0.0282, rel=0.10)

    def test_zero_at_critical(self) -> None:
        comp = get("n-hexane")
        assert float(surface_tensions(["n-hexane"], comp.tc)[0]) == pytest.approx(0.0, abs=1e-6)

    def test_wsd_pure_limit(self) -> None:
        sigma = jnp.array([0.022, 0.072])
        v = jnp.array([5.87e-5, 1.81e-5])
        got = float(winterfeld_scriven_davis(jnp.array([0.0, 1.0]), sigma, v))
        assert got == pytest.approx(0.072, rel=1e-12)

    def test_mixture_between_pure_values(self) -> None:
        comps = ["ethanol", "water"]
        sigma = surface_tensions(comps, 298.15)
        mid = float(mixture_surface_tension(comps, 298.15, jnp.array([0.5, 0.5])))
        assert float(jnp.min(sigma)) < mid < float(jnp.max(sigma))


class TestDiffusivity:
    def test_diffusion_volume_special_molecules(self) -> None:
        assert diffusion_volume("nitrogen") == 18.5
        assert diffusion_volume("water") == 13.1

    def test_diffusion_volume_atomic_sum(self) -> None:
        # n-hexane C6H14: 6*15.9 + 14*2.31 = 127.74.
        assert diffusion_volume("n-hexane") == pytest.approx(127.74)

    def test_fuller_n2_o2(self) -> None:
        # N2-O2 at 293.15 K, 1 atm: D = 0.22 cm^2/s experimental (Poling, tab. 11-2
        # quotes Fuller at ~2% for this pair).
        got = float(gas_diffusivity("nitrogen", "oxygen", 293.15, 101325.0))
        assert got == pytest.approx(2.2e-5, rel=0.10)

    def test_fuller_pressure_scaling(self) -> None:
        d1 = float(fuller_diffusivity(300.0, 1.0e5, 28.0, 32.0, 18.5, 16.3))
        d2 = float(fuller_diffusivity(300.0, 2.0e5, 28.0, 32.0, 18.5, 16.3))
        assert d1 == pytest.approx(2.0 * d2, rel=1e-12)

    def test_wilke_chang_ethanol_in_water(self) -> None:
        # Experimental D0 = 1.24e-9 m^2/s at 298 K; Wilke-Chang is a 10-20% method.
        got = float(liquid_diffusivity("ethanol", "water", 298.15))
        assert got == pytest.approx(1.24e-9, rel=0.25)

    def test_wilke_chang_formula(self) -> None:
        # Hand evaluation in the original cm/cP units.
        t, mu, mw, vb, phi = 298.15, 8.9e-4, 18.015, 6.25e-5, 2.6
        want_cm2 = 7.4e-8 * math.sqrt(phi * mw) * t / ((mu * 1e3) * (vb * 1e6) ** 0.6)
        got = float(wilke_chang_diffusivity(t, mu, mw, vb, phi))
        assert got == pytest.approx(want_cm2 * 1e-4, rel=1e-12)


class TestDifferentiability:
    def test_liquid_viscosity_gradient(self) -> None:
        g = float(jax.grad(lambda t: liquid_viscosities(["water"], t)[0])(298.15))
        assert math.isfinite(g)
        assert g < 0.0  # liquid viscosity falls with temperature

    def test_gas_viscosity_gradient(self) -> None:
        g = float(jax.grad(lambda t: gas_viscosities(["nitrogen"], t)[0])(300.0))
        assert math.isfinite(g)
        assert g > 0.0  # gas viscosity rises with temperature

    def test_mixture_rules_jit_and_composition_grad(self) -> None:
        comps = ["nitrogen", "oxygen"]

        def f(y: jnp.ndarray) -> jnp.ndarray:
            return gas_mixture_viscosity(comps, 300.0, y)

        y = jnp.array([0.79, 0.21])
        assert float(jax.jit(f)(y)) == pytest.approx(float(f(y)), rel=1e-12)
        grad = jax.grad(lambda y: f(y))(y)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_surface_tension_gradient(self) -> None:
        g = float(jax.grad(lambda t: surface_tensions(["water"], t)[0])(298.15))
        assert math.isfinite(g)
        assert g < 0.0  # tension falls with temperature

    def test_diffusivity_gradient(self) -> None:
        g = float(jax.grad(lambda t: gas_diffusivity("nitrogen", "oxygen", t, 1.0e5))(300.0))
        assert math.isfinite(g)
        assert g > 0.0  # D ~ T^1.75
