"""ThermoML batch regression driver and the bundled binary-parameter bank.

The bundled samples are synthetic (generated from documented NRTL parameters by
``scripts/gen_thermoml_samples.py``), so the batch driver must recover those
exact parameters and the shipped bank must contain them. One end-to-end fit runs
here (the cheap near-ideal pair); the recovery of every pair is exercised when
the bank is regenerated.
"""

import jax.numpy as jnp
import pytest

from fugacio.thermo import (
    FittedBinary,
    ParameterBank,
    bubble_pressure_gamma,
    component_arrays,
    fit_vle_dataset,
    load_sample,
)

#: Generating parameters of the bundled synthetic samples (see
#: scripts/gen_thermoml_samples.py); the fitted bank must reproduce them.
GENERATING = {
    frozenset(("ethanol", "water")): ("ethanol", 330.0, 600.0),
    frozenset(("methanol", "water")): ("methanol", 80.0, 220.0),
    frozenset(("acetone", "chloroform")): ("acetone", -150.0, -130.0),
    frozenset(("benzene", "toluene")): ("benzene", 20.0, 15.0),
    frozenset(("2-propanol", "water")): ("2-propanol", 170.0, 680.0),
}


@pytest.fixture(scope="module")
def bank() -> ParameterBank:
    return ParameterBank.load_bundled()


class TestBundledBank:
    def test_contains_all_sample_pairs(self, bank: ParameterBank) -> None:
        assert len(bank) == len(GENERATING)
        for pair in GENERATING:
            a, b = sorted(pair)
            assert (a, b) in bank

    def test_recovers_generating_parameters(self, bank: ParameterBank) -> None:
        for pair, (first, b12, b21) in GENERATING.items():
            other = next(iter(pair - {first}))
            entry = bank.get(first, other)
            assert entry is not None
            assert entry.b12 == pytest.approx(b12, abs=0.5)
            assert entry.b21 == pytest.approx(b21, abs=0.5)
            assert entry.rmse < 1e-4
            assert entry.n_points == 11

    def test_orientation_swap(self, bank: ParameterBank) -> None:
        fwd = bank.get("ethanol", "water")
        rev = bank.get("water", "ethanol")
        assert fwd is not None and rev is not None
        assert fwd.b12 == rev.b21 and fwd.b21 == rev.b12
        assert rev.components == ("water", "ethanol")

    def test_nrtl_models_mirror_under_swap(self, bank: ParameterBank) -> None:
        t = 323.15
        x = jnp.array([0.3, 0.7])
        g_fwd = bank.nrtl("ethanol", "water").ln_gamma(x, t)
        g_rev = bank.nrtl("water", "ethanol").ln_gamma(x[::-1], t)
        assert float(g_fwd[0]) == pytest.approx(float(g_rev[1]), rel=1e-12)
        assert float(g_fwd[1]) == pytest.approx(float(g_rev[0]), rel=1e-12)

    def test_missing_pair(self, bank: ParameterBank) -> None:
        assert bank.get("benzene", "water") is None
        with pytest.raises(KeyError, match="no fitted parameters"):
            bank.nrtl("benzene", "water")

    def test_json_roundtrip(self, bank: ParameterBank) -> None:
        clone = ParameterBank.from_json(bank.to_json())
        assert clone.to_json() == bank.to_json()
        assert [e.components for e in clone.entries] == [e.components for e in bank.entries]

    def test_bank_model_reproduces_sample_pressures(self, bank: ParameterBank) -> None:
        # Predict the acetone/chloroform sample table straight from the bank.
        data = load_sample("acetone_chloroform_vle_318K")
        ds = data.datasets[0]
        model = bank.nrtl("acetone", "chloroform")
        arr = component_arrays(["acetone", "chloroform"])
        for t, x1, p_exp in zip(
            ds.temperature(),
            ds.mole_fraction(1),
            ds.pressure(),
            strict=True,
        ):
            x = jnp.array([x1, 1.0 - x1])
            p, _ = bubble_pressure_gamma(model, t, x, arr["tc"], arr["pc"], arr["omega"])
            assert float(p) == pytest.approx(p_exp, rel=1e-3)


class TestBatchDriver:
    def test_fit_recovers_near_ideal_pair(self) -> None:
        # The cheapest sample end-to-end: parse, resolve by CAS, fit, grade.
        data = load_sample("benzene_toluene_vle_363K")
        entry = fit_vle_dataset(data, data.datasets[0], source="benzene_toluene_vle_363K")
        assert isinstance(entry, FittedBinary)
        assert entry.components == ("benzene", "toluene")
        assert entry.b12 == pytest.approx(20.0, abs=0.5)
        assert entry.b21 == pytest.approx(15.0, abs=0.5)
        assert entry.rmse < 1e-4
        assert entry.t_min == entry.t_max == 363.15

    def test_rejects_non_binary_dataset(self) -> None:
        data = load_sample("water_vapor_pressure")
        with pytest.raises(ValueError, match="binary"):
            fit_vle_dataset(data, data.datasets[0])
