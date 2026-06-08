"""Pure-liquid reference fugacity: Raoult limit, Poynting, Henry's law."""

import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import reference as ref
from fugacio.thermo.eos import PR


def test_reference_reduces_to_psat_without_corrections() -> None:
    arr = comp.component_arrays(["benzene", "toluene"])
    fref, psat = ref.liquid_reference_fugacity(
        PR,
        370.0,
        1.5e5,
        arr["tc"],
        arr["pc"],
        arr["omega"],
        poynting=False,
        phi_saturation=False,
    )
    assert float(jnp.max(jnp.abs(fref - psat))) == pytest.approx(0.0, abs=1.0)


def test_poynting_factor_is_unity_at_saturation() -> None:
    arr = comp.component_arrays(["benzene"])
    psat = ref.saturation_pressures(PR, 370.0, arr["tc"], arr["pc"], arr["omega"])
    v_l = ref.pure_liquid_volumes(PR, 370.0, psat, arr["tc"], arr["pc"], arr["omega"])
    poy = ref.poynting_factor(v_l, psat, psat, 370.0)
    assert float(poy[0]) == pytest.approx(1.0, abs=1e-5)


def test_poynting_factor_increases_with_pressure() -> None:
    arr = comp.component_arrays(["benzene"])
    psat = ref.saturation_pressures(PR, 350.0, arr["tc"], arr["pc"], arr["omega"])
    v_l = ref.pure_liquid_volumes(PR, 350.0, psat, arr["tc"], arr["pc"], arr["omega"])
    poy = ref.poynting_factor(v_l, 10e5, psat, 350.0)
    assert float(poy[0]) > 1.0


def test_corrections_keep_reference_positive_and_close_to_psat() -> None:
    arr = comp.component_arrays(["benzene", "toluene"])
    fref, psat = ref.liquid_reference_fugacity(
        PR,
        360.0,
        2e5,
        arr["tc"],
        arr["pc"],
        arr["omega"],
        poynting=True,
        phi_saturation=True,
    )
    assert bool(jnp.all(fref > 0.0))
    # Corrections are modest at low pressure: within ~10% of the plain Psat.
    assert float(jnp.max(jnp.abs(fref / psat - 1.0))) < 0.1


def test_henry_constant_is_positive() -> None:
    h = ref.henry_constant(298.15, 100.0, -5000.0, -12.0, 0.0)
    assert float(h) > 0.0
