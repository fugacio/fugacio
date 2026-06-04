import pytest

from fugacio.sim import antoine_psat, bubble_pressure

COMP1 = (8.07131, 1730.63, 233.426)
COMP2 = (7.43155, 1554.68, 240.337)


def test_ideal_reduces_to_raoult() -> None:
    t, x1 = 80.0, 0.4
    psat1 = float(antoine_psat(t, *COMP1))
    psat2 = float(antoine_psat(t, *COMP2))
    pressure, y1 = bubble_pressure(x1, t, COMP1, COMP2, a12=0.0, a21=0.0)
    expected_p = x1 * psat1 + (1.0 - x1) * psat2
    assert float(pressure) == pytest.approx(expected_p, rel=1e-6)
    assert float(y1) == pytest.approx(x1 * psat1 / expected_p, rel=1e-6)


def test_non_ideal_pressure_is_positive_and_bounded() -> None:
    pressure, y1 = bubble_pressure(0.5, 80.0, COMP1, COMP2, a12=0.5, a21=0.8)
    assert float(pressure) > 0.0
    assert 0.0 < float(y1) < 1.0
