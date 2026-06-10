"""Unit tests for the pure-component property correlation kernels and dispatchers.

Kernels are checked against hand-evaluated values of their published forms;
dispatchers against well-established experimental spot values (CRC Handbook /
Poling et al.) with tolerances matching each correlation's documented accuracy.
Differentiability and jit-compilability are part of the contract, so both are
exercised here. The opt-in oracle suite (``test_property_oracles.py``) does the
broad-coverage grading against CoolProp.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo.components import get
from fugacio.thermo.correlations import (
    dippr100,
    dippr101,
    dippr102,
    dippr105,
    dippr106,
    heat_of_vaporization,
    liquid_heat_capacity,
    mulero_cachadina,
    pitzer_hvap,
    rowlinson_bondi_cp,
    watson_hvap,
)


class TestKernels:
    """Each DIPPR-form kernel reproduces a hand-computed value of its formula."""

    def test_dippr100_is_a_polynomial(self) -> None:
        got = float(dippr100(300.0, 1.0, 2.0, 3.0, 4.0, 5.0))
        want = 1.0 + 2.0 * 300.0 + 3.0 * 300.0**2 + 4.0 * 300.0**3 + 5.0 * 300.0**4
        assert got == pytest.approx(want, rel=1e-12)

    def test_dippr101_exponential_form(self) -> None:
        c = (-52.843, 3703.6, 5.866, -5.879e-29, 10.0)  # water liquid viscosity row
        got = float(dippr101(298.15, *c))
        want = math.exp(c[0] + c[1] / 298.15 + c[2] * math.log(298.15) + c[3] * 298.15 ** c[4])
        assert got == pytest.approx(want, rel=1e-12)
        assert want == pytest.approx(8.9e-4, rel=0.05)  # and it is water's ~0.89 mPa*s

    def test_dippr102_rational_form(self) -> None:
        c = (1.7096e-8, 1.1146, 0.0, 0.0)  # water dilute-gas viscosity row
        got = float(dippr102(373.15, *c))
        want = c[0] * 373.15 ** c[1] / (1.0 + c[2] / 373.15 + c[3] / 373.15**2)
        assert got == pytest.approx(want, rel=1e-12)
        assert want == pytest.approx(1.2e-5, rel=0.10)  # steam at 100 C

    def test_dippr105_rackett_shape(self) -> None:
        c1, c2, c3, c4 = 1234.6, 0.27216, 425.0, 0.28707  # 1,3-butadiene density row
        got = float(dippr105(300.0, c1, c2, c3, c4))
        want = c1 / c2 ** (1.0 + (1.0 - 300.0 / c3) ** c4)
        assert got == pytest.approx(want, rel=1e-12)

    def test_dippr106_vanishes_at_critical(self) -> None:
        # Tr is clipped at 1 - 1e-12 to keep autodiff finite, so the value at Tc
        # is not exactly zero -- just negligible against the J/mol scale of c1.
        tc = 647.096
        assert float(dippr106(tc, tc, 5.2053e4, 0.3199, -0.212, 0.25795)) == pytest.approx(
            0.0, abs=5.0
        )

    def test_dippr106_against_formula(self) -> None:
        tc, c1, c2, c3, c4 = 647.096, 5.2053e4, 0.3199, -0.212, 0.25795
        tr = 350.0 / tc
        want = c1 * (1.0 - tr) ** (c2 + c3 * tr + c4 * tr**2)
        assert float(dippr106(350.0, tc, c1, c2, c3, c4)) == pytest.approx(want, rel=1e-12)

    def test_mulero_cachadina_single_term_guggenheim(self) -> None:
        # One term with n = 1.26 is the van der Waals-Guggenheim form.
        tc, s0 = 562.05, 0.07298
        got = float(mulero_cachadina(300.0, tc, s0, 1.26))
        assert got == pytest.approx(s0 * (1.0 - 300.0 / tc) ** 1.26, rel=1e-12)

    def test_mulero_cachadina_zero_above_critical(self) -> None:
        assert float(mulero_cachadina(700.0, 647.096, 0.2358, 1.256)) == 0.0


class TestEstimators:
    """Corresponding-states estimators hit literature values for normal fluids."""

    def test_pitzer_hvap_benzene(self) -> None:
        # Benzene at 298.15 K: Hvap = 33.83 kJ/mol (CRC). Pitzer is good to a few %.
        comp = get("benzene")
        got = float(pitzer_hvap(298.15, comp.tc, comp.omega))
        assert got == pytest.approx(33830.0, rel=0.05)

    def test_watson_rescaling_water(self) -> None:
        # Rescale water's Hvap at the normal boiling point (40.65 kJ/mol) to
        # 298.15 K; experimental value there is 43.99 kJ/mol.
        got = float(watson_hvap(40650.0, 373.15, 298.15, 647.096))
        assert got == pytest.approx(43990.0, rel=0.03)

    def test_watson_identity_at_reference(self) -> None:
        assert float(watson_hvap(35000.0, 320.0, 320.0, 600.0)) == pytest.approx(35000.0)

    def test_rowlinson_bondi_hexane(self) -> None:
        # n-hexane liquid Cp at 298.15 K: 195.6 J/mol/K (Poling, app. A).
        comp = get("n-hexane")
        cp = comp.cp_ig
        assert cp is not None
        from fugacio.thermo.ideal import cp_ig

        cp_id = cp_ig(298.15, cp.a, cp.b, cp.c, cp.d, cp.e)
        got = float(rowlinson_bondi_cp(298.15, comp.tc, comp.omega, cp_id))
        assert got == pytest.approx(195.6, rel=0.05)


class TestDispatchers:
    def test_hvap_water_at_298(self) -> None:
        # 43.99 kJ/mol experimental.
        got = float(heat_of_vaporization(["water"], 298.15)[0])
        assert got == pytest.approx(43990.0, rel=0.02)

    def test_hvap_zero_at_critical(self) -> None:
        # The curated fit's critical temperature can differ from the component
        # record by a few hundredths of a kelvin, so evaluate at the fit's Tc.
        from fugacio.thermo._property_data import HVAP_DIPPR106

        tc_fit = HVAP_DIPPR106["ethanol"][0]
        assert float(heat_of_vaporization(["ethanol"], tc_fit)[0]) == pytest.approx(0.0, abs=50.0)

    def test_hvap_estimator_fallback(self) -> None:
        # A component without a curated DIPPR-106 row still gets a value (Pitzer).
        from fugacio.thermo._property_data import HVAP_DIPPR106
        from fugacio.thermo.components import DATABASE

        name = next(n for n in sorted(DATABASE) if n not in HVAP_DIPPR106)
        comp = get(name)
        got = float(heat_of_vaporization([name], 0.7 * comp.tc)[0])
        assert 0.0 < got < 2.0e5

    def test_hvap_vector_shape(self) -> None:
        vals = heat_of_vaporization(["water", "ethanol", "benzene"], 300.0)
        assert vals.shape == (3,)
        assert bool(jnp.all(vals > 0.0))

    def test_liquid_cp_water_magnitude(self) -> None:
        # Rowlinson-Bondi is documented as rough for water; just bracket it.
        got = float(liquid_heat_capacity(["water"], 298.15)[0])
        assert 60.0 < got < 110.0  # experimental 75.3 J/mol/K

    def test_liquid_cp_missing_cp_ig_raises(self) -> None:
        import dataclasses

        broken = dataclasses.replace(get("water"), cp_ig=None)
        with pytest.raises(ValueError, match="missing ideal-gas Cp"):
            liquid_heat_capacity([broken], 300.0)

    def test_hvap_differentiable_and_jittable(self) -> None:
        f = lambda t: heat_of_vaporization(["water", "benzene"], t).sum()  # noqa: E731
        g = float(jax.grad(f)(350.0))
        assert math.isfinite(g)
        assert g < 0.0  # Hvap decreases with temperature
        jitted = jax.jit(f)
        assert float(jitted(350.0)) == pytest.approx(float(f(350.0)), rel=1e-12)
