"""Unit tests for liquid/vapour volumetric properties (Rackett, COSTALD, Peneloux).

Spot values are experimental saturated-liquid densities (CRC Handbook) with
tolerances reflecting each route's documented accuracy; structural tests cover
the pure-component limits of the mixture rules, the ideal-gas limit of the
vapour density, and that the Peneloux translation actually improves the raw
EOS liquid volume. Differentiability and jit are part of the contract.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo._property_data import COSTALD_VOLUME, RHO_LIQUID_DIPPR105
from fugacio.thermo.components import get
from fugacio.thermo.constants import R
from fugacio.thermo.eos import PR, molar_volume
from fugacio.thermo.volumetric import (
    costald_volume,
    liquid_density,
    liquid_molar_volumes,
    mixture_liquid_volume,
    peneloux_shift,
    rackett_volume,
    translated_liquid_volume_for,
    tyn_calus_vb,
    vapor_density,
    zra_estimate,
)

# Experimental saturated-liquid densities at 298.15 K (CRC Handbook), kg/m^3.
RHO_298 = {
    "water": 997.0,
    "ethanol": 785.0,
    "benzene": 873.7,
    "n-hexane": 654.8,
    "toluene": 862.3,
    "acetone": 784.5,
}


class TestPureVolumes:
    @pytest.mark.parametrize(("name", "rho_ref"), sorted(RHO_298.items()))
    def test_liquid_density_at_298(self, name: str, rho_ref: float) -> None:
        got = float(liquid_density([name], 298.15, jnp.array([1.0])))
        assert got == pytest.approx(rho_ref, rel=0.025)

    def test_zra_estimate_yamada_gunn(self) -> None:
        assert float(zra_estimate(0.0)) == pytest.approx(0.29056)
        assert float(zra_estimate(0.3)) == pytest.approx(0.29056 - 0.08775 * 0.3)

    def test_rackett_volume_hexane(self) -> None:
        # n-hexane: V_L(298.15) = 131.6 cm^3/mol experimental.
        comp = get("n-hexane")
        assert comp.zra is not None
        v = float(rackett_volume(298.15, comp.tc, comp.pc, comp.zra))
        assert v == pytest.approx(131.6e-6, rel=0.03)

    def test_costald_volume_propane(self) -> None:
        # Saturated propane at 300 K: rho = 489.3 kg/m^3 -> v = 90.1 cm^3/mol.
        vchar, omega_srk = COSTALD_VOLUME["propane"]
        comp = get("propane")
        v = float(costald_volume(300.0, comp.tc, vchar, omega_srk))
        assert v == pytest.approx(90.1e-6, rel=0.03)

    def test_dispatch_covers_whole_database(self) -> None:
        # Every component must produce a finite positive liquid volume at 0.7*Tc.
        from fugacio.thermo.components import DATABASE

        for name, comp in sorted(DATABASE.items()):
            v = float(liquid_molar_volumes([name], 0.7 * comp.tc)[0])
            assert math.isfinite(v) and v > 0.0, name

    def test_tyn_calus_estimate(self) -> None:
        # Vb ~ 0.285 * Vc^1.048 (volumes in cm^3/mol inside the correlation).
        vc = 200.0e-6
        want_cm3 = 0.285 * 200.0**1.048
        assert float(tyn_calus_vb(vc)) == pytest.approx(want_cm3 * 1e-6, rel=1e-10)


class TestMixtures:
    def test_pure_limit_matches_pure_volume(self) -> None:
        v_mix = float(mixture_liquid_volume(["benzene"], 298.15, jnp.array([1.0])))
        v_pure = float(liquid_molar_volumes(["benzene"], 298.15)[0])
        assert v_mix == pytest.approx(v_pure, rel=1e-12)

    def test_amagat_is_mole_fraction_average(self) -> None:
        comps = ["benzene", "toluene"]
        x = jnp.array([0.4, 0.6])
        v = liquid_molar_volumes(comps, 298.15)
        want = float(jnp.sum(x * v))
        got = float(mixture_liquid_volume(comps, 298.15, x, method="amagat"))
        assert got == pytest.approx(want, rel=1e-12)

    def test_costald_method_close_to_amagat_for_alkanes(self) -> None:
        # Near-ideal alkane pair: HBT mixing rules and Amagat agree to ~2%.
        comps = ["n-pentane", "n-hexane"]
        x = jnp.array([0.5, 0.5])
        v_hbt = float(mixture_liquid_volume(comps, 298.15, x, method="costald"))
        v_am = float(mixture_liquid_volume(comps, 298.15, x, method="amagat"))
        assert v_hbt == pytest.approx(v_am, rel=0.02)

    def test_costald_method_requires_coverage(self) -> None:
        missing = next(n for n in sorted(RHO_LIQUID_DIPPR105) if n not in COSTALD_VOLUME)
        with pytest.raises(KeyError, match="characteristic volume"):
            mixture_liquid_volume(
                ["n-hexane", missing], 300.0, jnp.array([0.5, 0.5]), method="costald"
            )

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            mixture_liquid_volume(["water"], 300.0, jnp.array([1.0]), method="bogus")


class TestEosVolumes:
    def test_vapor_density_ideal_limit(self) -> None:
        # At 10 kPa and 400 K steam is nearly ideal: rho = p*MW/(R*T).
        comp = get("water")
        got = float(vapor_density(["water"], 400.0, 1.0e4, jnp.array([1.0])))
        want = 1.0e4 * comp.mw * 1.0e-3 / (R * 400.0)
        assert got == pytest.approx(want, rel=0.01)

    def test_peneloux_shift_sign(self) -> None:
        # Z_RA below the PR threshold (k2 = 0.25969) gives a positive
        # (volume-reducing) shift; above it, negative.
        assert float(peneloux_shift(PR, 500.0, 3.0e6, 0.24)) > 0.0
        assert float(peneloux_shift(PR, 500.0, 3.0e6, 0.28)) < 0.0

    def test_translation_improves_pr_liquid_volume(self) -> None:
        # Water at 298.15 K, 1 bar: V_L = 18.07 cm^3/mol experimental. Raw PR
        # overshoots by ~17%; the Peneloux shift must land within 2%.
        comp = get("water")
        args = (
            298.15,
            1.0e5,
            jnp.array([1.0]),
            jnp.array([comp.tc]),
            jnp.array([comp.pc]),
            jnp.array([comp.omega]),
        )
        v_raw = float(molar_volume(PR, *args, phase="liquid"))
        v_trans = float(translated_liquid_volume_for(["water"], 298.15, 1.0e5, jnp.array([1.0])))
        v_exp = 18.07e-6
        assert abs(v_trans - v_exp) < abs(v_raw - v_exp)
        assert v_trans == pytest.approx(v_exp, rel=0.02)


class TestDifferentiability:
    def test_density_gradient_negative_in_t(self) -> None:
        g = float(jax.grad(lambda t: liquid_density(["n-hexane"], t, jnp.array([1.0])))(298.15))
        assert math.isfinite(g)
        assert g < 0.0  # liquids expand on heating

    def test_mixture_volume_jit_and_composition_grad(self) -> None:
        comps = ["benzene", "toluene"]

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return mixture_liquid_volume(comps, 320.0, x)

        x = jnp.array([0.3, 0.7])
        assert float(jax.jit(f)(x)) == pytest.approx(float(f(x)), rel=1e-12)
        grad = jax.grad(lambda x: f(x))(x)
        assert bool(jnp.all(jnp.isfinite(grad)))
