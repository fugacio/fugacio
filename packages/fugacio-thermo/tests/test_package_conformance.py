"""Property-package conformance: one protocol, the same identities, every package.

Each package (cubic, gamma-phi, PC-SAFT, a reference fluid, and a custom
`_PackageBase` subclass that supplies only the four single-phase primitives)
must return checked reports from every ``*_with_info`` method, close material
and fugacity balances, invert its own saturation and energy calculations, and
keep its enthalpy consistent with its Gibbs energy and fugacity coefficients.
"""

import gc
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import (
    component_arrays,
    cubic_package,
    gamma_phi_package,
    get,
    helmholtz_package,
    ideal_gas_coeffs,
    nrtl_from_database,
    reference_fluid,
    saft_package,
    saft_parameters_for,
)
from fugacio.thermo.consistency import fugacity_enthalpy_residual, gibbs_helmholtz_residual
from fugacio.thermo.constants import R
from fugacio.thermo.diagnostics import SolveReport
from fugacio.thermo.ideal import enthalpy_ig_mixture, entropy_ig_mixture
from fugacio.thermo.package import PropertyPackage, _PackageBase


@dataclass(frozen=True)
class IdealSolutionPackage(_PackageBase):
    """Raoult's law on Wilson vapor pressures: only the four primitives are defined."""

    tc: jax.Array
    pc: jax.Array
    omega: jax.Array
    cp: tuple[jax.Array, ...]

    @property
    def n_components(self) -> int:
        return int(self.tc.shape[0])

    def _ln_psat(self, t: Any) -> jax.Array:
        return jnp.log(self.pc) + 5.373 * (1.0 + self.omega) * (1.0 - self.tc / t)

    def ln_phi(self, t: Any, p: Any, x: jax.Array, *, phase: str) -> jax.Array:
        if phase == "vapor":
            return jnp.zeros_like(jnp.asarray(x, dtype=float))
        return self._ln_psat(t) - jnp.log(p)

    def enthalpy(self, t: Any, p: Any, x: jax.Array, *, phase: str) -> jax.Array:
        h_ig = enthalpy_ig_mixture(t, x, *self.cp)
        if phase == "vapor":
            return h_ig
        slope = jax.jacfwd(self._ln_psat)(jnp.asarray(t, dtype=float))
        return h_ig - R * jnp.asarray(t) ** 2 * jnp.sum(x * slope)

    def entropy(self, t: Any, p: Any, x: jax.Array, *, phase: str) -> jax.Array:
        s_ig = entropy_ig_mixture(t, p, x, *self.cp)
        if phase == "vapor":
            return s_ig
        g_residual = R * t * jnp.sum(x * self.ln_phi(t, p, x, phase="liquid"))
        h_residual = self.enthalpy(t, p, x, phase="liquid") - enthalpy_ig_mixture(t, x, *self.cp)
        return s_ig + (h_residual - g_residual) / t

    def volume(self, t: Any, p: Any, x: jax.Array, *, phase: str) -> jax.Array:
        return R * jnp.asarray(t) / p if phase == "vapor" else jnp.asarray(1e-4)


jax.tree_util.register_dataclass(
    IdealSolutionPackage, data_fields=["tc", "pc", "omega", "cp"], meta_fields=[]
)


@pytest.fixture(autouse=True)
def _release_compiled_programs():
    """Keep this suite's resident memory bounded.

    Every case compiles its own solver programs, and JAX keeps each compiled
    executable alive for the life of the process. Holding all of them at once
    takes more memory than a 16 GB runner has, so each test releases them
    afterwards; the persistent compilation cache keeps the next test cheap.
    """
    yield
    jax.clear_caches()
    gc.collect()


def _arrays(names):
    a = component_arrays(list(names))
    return a["tc"], a["pc"], a["omega"], ideal_gas_coeffs([get(c) for c in names])


def _gamma_phi(**options):
    names = ("ethanol", "water")
    return gamma_phi_package(nrtl_from_database(list(names)), *_arrays(names), **options)


# Each case: a builder, a two-phase state (T, P, z) for mixtures, and single-phase
# vapor and liquid states (T, P).
CASES = {
    "cubic": (
        lambda: cubic_package(*_arrays(("benzene", "toluene"))),
        (370.0, 1.013e5, [0.4, 0.6]),
        (420.0, 1e5),
        (300.0, 1e5),
    ),
    "gamma_phi": (
        lambda: _gamma_phi(poynting=True, phi_saturation=True),
        (356.0, 1.013e5, [0.4, 0.6]),
        (400.0, 1e5),
        (300.0, 1e5),
    ),
    "pcsaft": (
        lambda: saft_package(
            saft_parameters_for(["propane", "n-butane"]), *_arrays(("propane", "n-butane"))
        ),
        (320.0, 8e5, [0.5, 0.5]),
        (350.0, 1e5),
        (300.0, 30e5),
    ),
    "reference_fluid": (
        lambda: helmholtz_package(reference_fluid("water")),
        None,
        (400.0, 1e5),
        (300.0, 1e5),
    ),
    "custom": (
        lambda: IdealSolutionPackage(*_arrays(("benzene", "toluene"))),
        (370.0, 1.013e5, [0.4, 0.6]),
        (420.0, 1e5),
        (300.0, 1e5),
    ),
}

_BUILT: dict[str, Any] = {}


def _case(name: str):
    build, two_phase, vapor, liquid = CASES[name]
    if name not in _BUILT:
        _BUILT[name] = build()
    return _BUILT[name], two_phase, vapor, liquid


def _composition(pkg, two_phase):
    return jnp.asarray(two_phase[2]) if two_phase else jnp.ones(pkg.n_components)


@pytest.mark.parametrize("name", list(CASES))
def test_every_with_info_method_returns_a_report(name) -> None:
    pkg, two_phase, vapor, _ = _case(name)
    assert isinstance(pkg, PropertyPackage)
    z = _composition(pkg, two_phase)
    t, p = two_phase[:2] if two_phase else (373.0, 1e5)
    h = pkg.mixture_enthalpy(*vapor, z)
    s = pkg.mixture_entropy(*vapor, z)
    results = [
        pkg.flash_pt_with_info(t, p, z),
        pkg.bubble_pressure_with_info(t, z),
        pkg.dew_pressure_with_info(t, z),
        pkg.bubble_temperature_with_info(p, z),
        pkg.dew_temperature_with_info(p, z),
        pkg.flash_ph_with_info(vapor[1], h, z),
        pkg.flash_ps_with_info(vapor[1], s, z),
    ]
    for result in results:
        assert isinstance(result.report, SolveReport)
        assert bool(result.report.converged)


@pytest.mark.parametrize("name", [n for n in CASES if CASES[n][1]])
def test_two_phase_flash_closes_material_and_fugacity_balances(name) -> None:
    pkg, (t, p, z), _, _ = _case(name)
    z = jnp.asarray(z)
    solved = pkg.flash_pt_with_info(t, p, z)
    res = solved.value
    assert bool(solved.report.converged)
    assert 0.0 < float(res.beta) < 1.0
    closure = res.beta * res.y + (1.0 - res.beta) * res.x - z
    assert float(jnp.max(jnp.abs(closure))) < 1e-10
    ln_f_liquid = jnp.log(res.x) + pkg.ln_phi(t, p, res.x, phase="liquid")
    ln_f_vapor = jnp.log(res.y) + pkg.ln_phi(t, p, res.y, phase="vapor")
    assert float(jnp.max(jnp.abs(ln_f_liquid - ln_f_vapor))) < 1e-8
    # The value-only call returns the same state.
    assert float(pkg.flash_pt(t, p, z).beta) == pytest.approx(float(res.beta), abs=1e-12)
    # The split is stability-consistent: the feed is unstable as one phase.
    assert not bool(pkg.stability(t, p, z).stable)


@pytest.mark.parametrize("name", list(CASES))
def test_single_phase_states_are_stable(name) -> None:
    pkg, two_phase, vapor, liquid = _case(name)
    z = _composition(pkg, two_phase)
    for t, p in (vapor, liquid):
        assert bool(pkg.stability(t, p, z).stable)
    assert float(pkg.flash_pt(*vapor, z).beta) == 1.0
    assert float(pkg.flash_pt(*liquid, z).beta) == 0.0


@pytest.mark.parametrize("name", list(CASES))
def test_saturation_temperatures_invert_saturation_pressures(name) -> None:
    pkg, two_phase, _, _ = _case(name)
    z = _composition(pkg, two_phase)
    t = two_phase[0] if two_phase else 373.15
    p_bubble = pkg.bubble_pressure(t, z).value
    p_dew = pkg.dew_pressure(t, z).value
    assert float(p_bubble) >= float(p_dew) * (1.0 - 1e-9)
    assert float(pkg.bubble_temperature(p_bubble, z).value) == pytest.approx(t, abs=1e-6)
    assert float(pkg.dew_temperature(p_dew, z).value) == pytest.approx(t, abs=1e-6)


@pytest.mark.parametrize("name", list(CASES))
def test_energy_flashes_invert_the_bulk_properties(name) -> None:
    pkg, two_phase, vapor, liquid = _case(name)
    z = _composition(pkg, two_phase)
    for t, p in (vapor, liquid):
        h = pkg.mixture_enthalpy(t, p, z)
        s = pkg.mixture_entropy(t, p, z)
        assert float(pkg.flash_ph(p, h, z, t_init=t).t) == pytest.approx(t, abs=1e-6)
        assert float(pkg.flash_ps(p, s, z, t_init=t).t) == pytest.approx(t, abs=1e-6)


@pytest.mark.parametrize(
    ("name", "tolerance"),
    [("cubic", 1e-8), ("pcsaft", 1e-8), ("reference_fluid", 1e-8), ("custom", 1e-8),
     ("gamma_phi", 1e-4)],
)  # fmt: skip
def test_enthalpy_is_consistent_with_gibbs_energy_and_fugacity(name, tolerance) -> None:
    pkg, two_phase, vapor, liquid = _case(name)
    z = _composition(pkg, two_phase)
    for (t, p), phase in ((vapor, "vapor"), (liquid, "liquid")):
        assert float(gibbs_helmholtz_residual(pkg, t, p, z, phase=phase)) < tolerance
    t, p = (two_phase[0], two_phase[1]) if two_phase else (373.15, 1.013e5)
    assert float(fugacity_enthalpy_residual(pkg, t, p, z)) < tolerance


def test_default_gamma_phi_liquid_reference_deviation_is_bounded() -> None:
    # Without the Poynting and saturation-fugacity corrections, the gamma-phi
    # liquid fugacity and the liquid enthalpy come from slightly different
    # reference states. The documented deviation is about one percent.
    pkg = _gamma_phi()
    error = float(fugacity_enthalpy_residual(pkg, 356.0, 1.013e5, jnp.array([0.4, 0.6])))
    assert 1e-3 < error < 5e-2


@pytest.mark.parametrize("name", [n for n in CASES if CASES[n][1]])
def test_flash_is_differentiable_through_the_package(name) -> None:
    pkg, (t, p, z), _, _ = _case(name)
    z = jnp.asarray(z)

    def beta(tt):
        return pkg.flash_pt(tt, p, z).beta

    step = 1e-3
    exact = float(jax.grad(beta)(jnp.asarray(t)))
    central = float((beta(t + step) - beta(t - step)) / (2 * step))
    assert exact == pytest.approx(central, rel=1e-5)


def test_invalid_state_is_nan_not_an_answer() -> None:
    pkg, (t, p, z), _, _ = _case("cubic")
    solved = pkg.flash_pt_with_info(-t, p, jnp.asarray(z))
    assert not bool(solved.report.converged)
    assert bool(jnp.isnan(pkg.flash_pt(-t, p, jnp.asarray(z)).beta))
