"""Property packages: one object that owns phase equilibrium *and* energy.

A process simulator needs two things from its thermodynamics: "what splits?"
(fugacities, K-values, flashes) and "how much energy?" (enthalpy, entropy,
volume). The equilibrium models in `fugacio.thermo.phase` answer the first
question for the cubic, gamma-phi, and PC-SAFT routes, but the energy side of the
engine was, until now, wired to the cubic equation of state alone: a heater, a
mixer, an isentropic compressor, or a column energy balance could only be
evaluated on Peng-Robinson or SRK. A `PropertyPackage` closes that gap. It is the
single interface every energy-balanced unit operation consumes, and it is
implemented for all four method classes Fugacio carries:

* `CubicPackage`: phi-phi on a cubic EOS (PR, SRK, RK, vdW), the default;
* `GammaPhiPackage`: an activity-coefficient liquid (NRTL, UNIQUAC, Wilson,
  UNIFAC, ...) with an ideal or EOS vapour. Its liquid enthalpy is the pure-liquid
  enthalpies plus the *excess enthalpy* obtained by automatic differentiation of
  the excess Gibbs energy (Gibbs-Helmholtz), so heat of mixing comes for free and
  stays consistent with the activity coefficients themselves;
* `SAFTPackage`: PC-SAFT with Wertheim association, residual properties from the
  temperature derivatives of the residual Helmholtz energy;
* `HelmholtzPackage`: a pure reference fluid (IAPWS-95 water, Span-Wagner CO2,
  ...) wrapped as a one-component package, so steam and refrigerant loops can be
  simulated with reference-grade properties inside the same flowsheet engine.

Every package exposes the same calls: single-phase ``ln_phi`` / ``enthalpy`` /
``entropy`` / ``volume`` at ``(T, P, x)``, the isothermal ``flash_pt`` and the four
saturation calculations, and (implemented once, generically, on top of those) the
two-phase-aware ``mixture_enthalpy`` / ``mixture_entropy`` / ``mixture_volume``
and the energy-specified ``flash_ph`` / ``flash_ps`` / ``flash_tv``. Packages are
registered JAX pytrees whose model parameters are differentiable leaves, so a
flowsheet built on any of them is differentiable with respect to the
thermodynamic parameters as well as the operating conditions.

Every package also satisfies the `fugacio.thermo.phase.EquilibriumModel`
protocol, so it can be passed anywhere an equilibrium model is accepted.

Enthalpy and entropy are relative to the ideal-gas reference at ``T_REF`` /
``P_REF`` (the reference state of `fugacio.thermo.properties`), except for
`HelmholtzPackage`, whose reference is the one built into the published
formulation. Only differences are physical, and any consistent reference cancels
in a balance, so do not mix packages with different references across one energy
balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array, lax

from fugacio.thermo.activity.models import ActivityModel
from fugacio.thermo.constants import R
from fugacio.thermo.departure import residual_properties
from fugacio.thermo.diagnostics import SolveReport, residual_report
from fugacio.thermo.energy import EnergyFlashResult, _implicit_temperature
from fugacio.thermo.eos import PR, CubicEOS, ln_phi_mixture, molar_volume
from fugacio.thermo.equilibrium import (
    FlashResult,
    StabilityResult,
    bubble_pressure_eos,
    dew_pressure_eos,
    flash_pt,
    stability_analysis,
    wilson_k,
)
from fugacio.thermo.gammaphi import (
    bubble_pressure_gamma,
    bubble_temperature_gamma,
    dew_pressure_gamma,
    dew_temperature_gamma,
    flash_pt_gamma,
)
from fugacio.thermo.helmholtz.fluids import HelmholtzFluid
from fugacio.thermo.helmholtz.props import ln_fugacity_coefficient as _hf_ln_phi
from fugacio.thermo.helmholtz.saturation import (
    T_SAT_MAX_FRACTION,
    saturation_pressure,
    saturation_temperature,
)
from fugacio.thermo.helmholtz.states import state_ph, state_ps, state_tp
from fugacio.thermo.ideal import (
    enthalpy_ig,
    enthalpy_ig_mixture,
    entropy_ig,
    entropy_ig_mixture,
)
from fugacio.thermo.implicit import bracketed_root, newton_system_with_info
from fugacio.thermo.reference import (
    liquid_reference_fugacity,
    pure_liquid_volumes,
    saturation_pressures,
)
from fugacio.thermo.saft.equilibrium import (
    bubble_pressure_saft,
    dew_pressure_saft,
    flash_pt_saft,
    stability_saft,
)
from fugacio.thermo.saft.parameters import SaftParameters
from fugacio.thermo.saft.properties import ln_fugacity_coefficients as _saft_ln_phi
from fugacio.thermo.saft.properties import molar_density as _saft_density
from fugacio.thermo.saft.properties import residual_properties as _saft_residual

ArrayLike = Array | float
CpCoeffs = tuple[Array, Array, Array, Array, Array]

#: Vapour fractions closer than this to 0 or 1 are treated as single-phase when a
#: bulk property is differentiated (so the absent phase is never differentiated).
_SINGLE_PHASE_EPS = 1.0e-9


class EnergySolveResult(NamedTuple):
    """A PH/PS flash state and its independently verified solve report."""

    value: EnergyFlashResult
    report: SolveReport


@runtime_checkable
class PropertyPackage(Protocol):
    """Structural type of a property package (equilibrium + energy + volume).

    Implementations bundle their component constants and model parameters, so
    callers pass only the state ``(T, P, composition)``. ``phase`` is
    ``"liquid"`` or ``"vapor"`` and selects the phase branch to evaluate.
    """

    @property
    def n_components(self) -> int:
        """Number of components the package describes."""
        ...

    def ln_phi(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Log fugacity coefficients ``ln phi_i`` of the phase at ``(T, P, x)``."""
        ...

    def enthalpy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Single-phase molar enthalpy (J/mol)."""
        ...

    def entropy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Single-phase molar entropy (J/mol/K)."""
        ...

    def volume(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Single-phase molar volume (m^3/mol)."""
        ...

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Isothermal-isobaric two-phase flash."""
        ...

    def bubble_pressure(self, t: ArrayLike, x: Array) -> tuple[Array, Array]:
        """Bubble pressure and incipient vapour ``(P, y)`` at fixed ``T``, ``x``."""
        ...

    def dew_pressure(self, t: ArrayLike, y: Array) -> tuple[Array, Array]:
        """Dew pressure and incipient liquid ``(P, x)`` at fixed ``T``, ``y``."""
        ...

    def bubble_temperature(self, p: ArrayLike, x: Array) -> tuple[Array, Array]:
        """Bubble temperature and incipient vapour ``(T, y)`` at fixed ``P``, ``x``."""
        ...

    def dew_temperature(self, p: ArrayLike, y: Array) -> tuple[Array, Array]:
        """Dew temperature and incipient liquid ``(T, x)`` at fixed ``P``, ``y``."""
        ...

    def k_values(self, t: ArrayLike, p: ArrayLike, x: Array, y: Array) -> Array:
        """Equilibrium ratios ``K_i = phi_i^L(x) / phi_i^V(y)``."""
        ...

    def k_seed(self, t: ArrayLike, p: ArrayLike, x: Array) -> Array:
        """Composition-light K-value estimate used to initialise staged solvers."""
        ...

    def mixture_enthalpy(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Two-phase-aware molar enthalpy of a feed ``z`` at ``(T, P)`` (J/mol)."""
        ...

    def mixture_entropy(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Two-phase-aware molar entropy of a feed ``z`` at ``(T, P)`` (J/mol/K)."""
        ...

    def mixture_volume(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Two-phase-aware molar volume of a feed ``z`` at ``(T, P)`` (m^3/mol)."""
        ...

    def flash_ph(
        self,
        p: ArrayLike,
        h: ArrayLike,
        z: Array,
        *,
        t_init: ArrayLike = 300.0,
        t_min: float = 50.0,
        t_max: float = 1500.0,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> EnergyFlashResult:
        """Isenthalpic flash: the temperature (and split) at which ``H = h``."""
        ...

    def flash_ps(
        self,
        p: ArrayLike,
        s: ArrayLike,
        z: Array,
        *,
        t_init: ArrayLike = 300.0,
        t_min: float = 50.0,
        t_max: float = 1500.0,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> EnergyFlashResult:
        """Isentropic flash: the temperature (and split) at which ``S = s``."""
        ...

    def signature(self) -> tuple[Any, ...]:
        """Hashable description of the package *structure* (not its values)."""
        ...


class _PackageBase:
    """Generic machinery shared by every concrete package.

    Subclasses provide the five single-phase primitives (``ln_phi``,
    ``enthalpy``, ``entropy``, ``volume``, ``flash_pt``) and the two
    fixed-temperature saturation solves; everything else here is derived from
    them, so a new thermodynamic method plugs into the whole flowsheet engine by
    implementing that small core.
    """

    # -- primitives every subclass must supply ----------------------------- #
    def ln_phi(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Log fugacity coefficients of the phase at ``(T, P, x)``."""
        raise NotImplementedError

    def enthalpy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Single-phase molar enthalpy (J/mol)."""
        raise NotImplementedError

    def entropy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Single-phase molar entropy (J/mol/K)."""
        raise NotImplementedError

    def volume(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Single-phase molar volume (m^3/mol)."""
        raise NotImplementedError

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Isothermal-isobaric two-phase flash."""
        raise NotImplementedError

    def bubble_pressure(self, t: ArrayLike, x: Array) -> tuple[Array, Array]:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``."""
        raise NotImplementedError

    def dew_pressure(self, t: ArrayLike, y: Array) -> tuple[Array, Array]:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``."""
        raise NotImplementedError

    @property
    def n_components(self) -> int:
        """Number of components."""
        raise NotImplementedError

    def signature(self) -> tuple[Any, ...]:
        """Hashable structural description (class name plus static metadata)."""
        return (type(self).__name__, self.n_components)

    # -- derived: K-values, Gibbs energy, heat capacity -------------------- #
    def k_values(self, t: ArrayLike, p: ArrayLike, x: Array, y: Array) -> Array:
        """Equilibrium ratios ``K_i = phi_i^L(x) / phi_i^V(y)`` at ``(T, P)``."""
        ln_l = self.ln_phi(t, p, x, phase="liquid")
        ln_v = self.ln_phi(t, p, y, phase="vapor")
        return jnp.exp(ln_l - ln_v)

    def k_seed(self, t: ArrayLike, p: ArrayLike, x: Array) -> Array:
        """Wilson-correlation K-values, the standard seed for staged separations.

        A phi-phi model evaluated with ``x == y`` returns ``K = 1`` wherever the
        equation of state has a single density root, which is useless for
        initialising a column; the Wilson estimate uses only the critical
        constants and acentric factors and is always well defined.
        """
        return wilson_k(t, p, self.tc, self.pc, self.omega)  # type: ignore[attr-defined]

    def gibbs(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Single-phase molar Gibbs energy ``G = H - T S`` (J/mol)."""
        t_arr = jnp.asarray(t, dtype=float)
        return self.enthalpy(t, p, x, phase=phase) - t_arr * self.entropy(t, p, x, phase=phase)

    def heat_capacity(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Single-phase isobaric heat capacity ``(dH/dT)_P`` by autodiff (J/mol/K)."""
        return jax.grad(lambda tt: self.enthalpy(tt, p, x, phase=phase))(
            jnp.asarray(t, dtype=float)
        )

    # -- derived: saturation temperatures by inversion --------------------- #
    def bubble_temperature(
        self, p: ArrayLike, x: Array, *, t_min: float = 150.0, t_max: float = 700.0
    ) -> tuple[Array, Array]:
        """Bubble temperature and incipient vapour at fixed ``P``, ``x``.

        Found by inverting `bubble_pressure` (saturation pressure rises
        monotonically with temperature) with the bracketed, implicitly
        differentiated root finder.
        """

        def residual(t: Array, params: tuple[Array, Array]) -> Array:
            p_, x_ = params
            return jnp.log(self.bubble_pressure(t, x_)[0]) - jnp.log(p_)

        t_star = bracketed_root(
            residual,
            (jnp.asarray(p, dtype=float), jnp.asarray(x)),
            jnp.asarray(t_min),
            jnp.asarray(t_max),
            1e-9,
            200,
        )
        _, y = self.bubble_pressure(t_star, x)
        return t_star, y

    def dew_temperature(
        self, p: ArrayLike, y: Array, *, t_min: float = 150.0, t_max: float = 700.0
    ) -> tuple[Array, Array]:
        """Dew temperature and incipient liquid at fixed ``P``, ``y``."""

        def residual(t: Array, params: tuple[Array, Array]) -> Array:
            p_, y_ = params
            return jnp.log(self.dew_pressure(t, y_)[0]) - jnp.log(p_)

        t_star = bracketed_root(
            residual,
            (jnp.asarray(p, dtype=float), jnp.asarray(y)),
            jnp.asarray(t_min),
            jnp.asarray(t_max),
            1e-9,
            200,
        )
        _, x = self.dew_pressure(t_star, y)
        return t_star, x

    # -- derived: two-phase-aware bulk properties -------------------------- #
    def _blend(self, t: ArrayLike, p: ArrayLike, z: Array, prop: str) -> Array:
        """Bulk molar property of feed ``z``: ``(1 - beta) M^L(x) + beta M^V(y)``.

        The blend is evaluated with a `jax.lax.switch` on the phase regime so a
        single-phase feed differentiates *only* the phase that exists. Naively
        multiplying an absent phase by zero would still propagate the ``NaN``
        gradient of a cubic root that does not exist in that region.
        """
        beta = lax.stop_gradient(self.flash_pt(t, p, z).beta)
        fn = getattr(self, prop)

        def liquid(_: None) -> Array:
            return fn(t, p, z, phase="liquid")

        def vapor(_: None) -> Array:
            return fn(t, p, z, phase="vapor")

        def both(_: None) -> Array:
            # Differentiate the equilibrium split only when both phases exist.
            # A single-phase K iteration can have a singular adjoint even though
            # its bulk property is a regular single-phase EOS evaluation.
            r = self.flash_pt(t, p, z)
            return (1.0 - r.beta) * fn(t, p, r.x, phase="liquid") + r.beta * fn(
                t, p, r.y, phase="vapor"
            )

        idx = jnp.where(
            beta <= _SINGLE_PHASE_EPS, 0, jnp.where(beta >= 1.0 - _SINGLE_PHASE_EPS, 2, 1)
        ).astype(jnp.int32)
        return lax.switch(idx, [liquid, both, vapor], None)

    def mixture_enthalpy(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Molar enthalpy of an equilibrium feed ``z`` at ``(T, P)`` (J/mol of feed).

        Runs the isothermal flash and blends the phase enthalpies by vapour
        fraction, so the latent heat is included automatically; in a single-phase
        region it reduces to that phase's enthalpy.
        """
        return self._blend(t, p, z, "enthalpy")

    def mixture_entropy(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Molar entropy of an equilibrium feed ``z`` at ``(T, P)`` (J/mol/K of feed)."""
        return self._blend(t, p, z, "entropy")

    def mixture_volume(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Molar volume of an equilibrium feed ``z`` at ``(T, P)`` (m^3/mol of feed)."""
        return self._blend(t, p, z, "volume")

    # -- derived: energy-specified flashes --------------------------------- #
    def _energy_flash(
        self,
        p: ArrayLike,
        target: ArrayLike,
        z: Array,
        prop: str,
        t_init: ArrayLike,
        t_min: float,
        t_max: float,
        tol: float,
        max_iter: int,
    ) -> EnergyFlashResult:
        """Resolve temperature and, for pure fluids, saturation quality."""
        params = (
            self,
            jnp.asarray(p, dtype=float),
            jnp.asarray(target, dtype=float),
            jnp.asarray(z),
        )

        def residual(t: Array, th: Any) -> Array:
            pkg, pressure, specified, composition = th
            return getattr(pkg, "mixture_" + prop)(t, pressure, composition) - specified

        # For a pure fluid this first solve only locates the phase regime. Its
        # temperature-only residual jumps inside the saturation dome, so its
        # failed implicit derivative must never enter the saturation branch,
        # even with a zero reverse-mode cotangent.
        locator_params = lax.stop_gradient(params) if self.n_components == 1 else params
        locator_init = lax.stop_gradient(t_init) if self.n_components == 1 else t_init
        t_star = _implicit_temperature(
            residual, locator_params, locator_init, t_min, t_max, tol, max_iter
        )

        def pt(_: None) -> EnergyFlashResult:
            temperature = t_star
            if self.n_components == 1:
                temperature = _implicit_temperature(
                    residual, params, t_star, t_min, t_max, tol, max_iter
                )
            classified = lax.stop_gradient(self.flash_pt(temperature, p, z))

            def two_phase(_: None) -> EnergyFlashResult:
                r = self.flash_pt(temperature, p, z)
                return EnergyFlashResult(t=temperature, beta=r.beta, x=r.x, y=r.y, k=r.k)

            def single_phase(_: None) -> EnergyFlashResult:
                # Present-phase composition is the feed composition. An absent
                # phase's trial K iteration must not supply its derivative to
                # an otherwise regular single-phase energy state.
                return EnergyFlashResult(
                    t=temperature,
                    beta=classified.beta,
                    x=jnp.where(classified.beta <= 0.0, z, classified.x),
                    y=jnp.where(classified.beta >= 1.0, z, classified.y),
                    k=classified.k,
                )

            return lax.cond(
                (classified.beta > 0.0) & (classified.beta < 1.0), two_phase, single_phase, None
            )

        if self.n_components != 1:
            return pt(None)

        # A temperature-only PH/PS solve encounters a jump at pure-fluid
        # saturation. Its temperature locates the jump, but its PT flash cannot
        # determine the phase amounts. Close fugacity and energy together there.
        seed_t = lax.stop_gradient(t_star)
        fn = getattr(self, prop)
        ml = fn(seed_t, p, z, phase="liquid")
        mv = fn(seed_t, p, z, phase="vapor")
        gap = mv - ml
        beta = (target - ml) / jnp.where(jnp.abs(gap) > 1e-12, gap, 1.0)
        inside = (gap > 1e-6) & (beta > 1e-8) & (beta < 1.0 - 1e-8)

        def saturated(_: None) -> EnergyFlashResult:
            def equations(u: Array, th: Any) -> Array:
                pkg, pressure, specified, composition = th
                temperature, quality = u
                phase_property = getattr(pkg, prop)
                liquid = phase_property(temperature, pressure, composition, phase="liquid")
                vapor = phase_property(temperature, pressure, composition, phase="vapor")
                fugacity = pkg.ln_phi(
                    temperature, pressure, composition, phase="liquid"
                ) - pkg.ln_phi(temperature, pressure, composition, phase="vapor")
                energy = ((1 - quality) * liquid + quality * vapor - specified) / jnp.maximum(
                    jnp.abs(specified), 1.0
                )
                return jnp.array([fugacity[0], energy])

            result = newton_system_with_info(
                equations,
                jnp.array([seed_t, beta]),
                params,
                tol=1e-10,
                max_iter=max_iter,
                scale=jnp.array([300.0, 1.0]),
                lower=jnp.array([t_min, 0.0]),
                upper=jnp.array([t_max, 1.0]),
            )
            state = jnp.where(result.report.converged, result.value, jnp.nan)
            return EnergyFlashResult(t=state[0], beta=state[1], x=z, y=z, k=jnp.ones_like(z))

        return lax.cond(lax.stop_gradient(inside), saturated, pt, None)

    def flash_ph(
        self,
        p: ArrayLike,
        h: ArrayLike,
        z: Array,
        *,
        t_init: ArrayLike = 300.0,
        t_min: float = 50.0,
        t_max: float = 1500.0,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> EnergyFlashResult:
        """Isenthalpic flash: find ``T`` so the feed enthalpy equals ``h``.

        The temperature is a safeguarded Newton/bisection root of the monotone
        enthalpy residual on ``[t_min, t_max]`` and is differentiated by the
        implicit function theorem with respect to ``p``, ``h``, ``z`` and the
        package's own parameters. Returns the temperature together with the
        equilibrium split there.
        """
        return self._energy_flash(p, h, z, "enthalpy", t_init, t_min, t_max, tol, max_iter)

    def flash_ps(
        self,
        p: ArrayLike,
        s: ArrayLike,
        z: Array,
        *,
        t_init: ArrayLike = 300.0,
        t_min: float = 50.0,
        t_max: float = 1500.0,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> EnergyFlashResult:
        """Isentropic flash: find ``T`` so the feed entropy equals ``s``.

        The backbone of isentropic compressor and turbine models; same solver and
        differentiability as `flash_ph`.
        """
        return self._energy_flash(p, s, z, "entropy", t_init, t_min, t_max, tol, max_iter)

    def flash_tv(
        self,
        t: ArrayLike,
        v: ArrayLike,
        z: Array,
        *,
        p_min: float = 1.0e3,
        p_max: float = 1.0e9,
        tol: float = 1e-10,
        max_iter: int = 200,
    ) -> tuple[Array, FlashResult]:
        """Isothermal-isochoric flash: the pressure at which the feed volume equals ``v``.

        Solves ``V(T, P, z) = v`` for ``ln P`` on ``[p_min, p_max]`` (the bulk
        molar volume decreases monotonically with pressure), the specification a
        fixed-volume vessel or a receiver imposes. Returns ``(P, flash)``.
        """
        params = (self, jnp.asarray(t, dtype=float), jnp.asarray(v, dtype=float), jnp.asarray(z))

        def residual(ln_p: Array, params: Any) -> Array:
            pkg, t_, v_, z_ = params
            return jnp.log(pkg.mixture_volume(t_, jnp.exp(ln_p), z_)) - jnp.log(v_)

        ln_p = bracketed_root(
            residual,
            params,
            jnp.log(jnp.asarray(p_min)),
            jnp.log(jnp.asarray(p_max)),
            tol,
            max_iter,
        )
        p_star = jnp.exp(ln_p)
        return p_star, self.flash_pt(t, p_star, z)


# --------------------------------------------------------------------------- #
# Cubic equation of state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CubicPackage(_PackageBase):
    """Phi-phi property package on a cubic equation of state.

    Attributes:
        tc: Critical temperatures (K).
        pc: Critical pressures (Pa).
        omega: Acentric factors.
        cp: Ideal-gas heat-capacity coefficient arrays ``(a, b, c, d, e)``.
        kij: Binary interaction matrix (``None`` means zeros).
        eos: Cubic equation of state (static; default Peng-Robinson).
    """

    tc: Array
    pc: Array
    omega: Array
    cp: CpCoeffs
    kij: Array | None = None
    eos: CubicEOS = PR

    component_names: tuple[str, ...] = ()

    @property
    def n_components(self) -> int:
        """Number of components."""
        return int(self.tc.shape[0])

    def signature(self) -> tuple[Any, ...]:
        """Class, component count, cubic, and whether a ``kij`` matrix is present."""
        return (
            "CubicPackage",
            self.n_components,
            repr(self.eos),
            self.kij is not None,
            self.component_names,
        )

    def ln_phi(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Log fugacity coefficients from the cubic EOS."""
        ln_phi, _ = ln_phi_mixture(
            self.eos, t, p, x, self.tc, self.pc, self.omega, phase=phase, kij=self.kij
        )
        return ln_phi

    def enthalpy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Ideal-gas enthalpy plus the cubic residual enthalpy (J/mol)."""
        a, b, c, d, e = self.cp
        h_ig = enthalpy_ig_mixture(t, x, a, b, c, d, e)
        res = residual_properties(
            self.eos, t, p, x, self.tc, self.pc, self.omega, phase=phase, kij=self.kij
        )
        return h_ig + res.enthalpy

    def entropy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Ideal-gas entropy (with mixing) plus the cubic residual entropy (J/mol/K)."""
        a, b, c, d, e = self.cp
        s_ig = entropy_ig_mixture(t, p, x, a, b, c, d, e)
        res = residual_properties(
            self.eos, t, p, x, self.tc, self.pc, self.omega, phase=phase, kij=self.kij
        )
        return s_ig + res.entropy

    def volume(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Molar volume ``Z R T / P`` on the requested cubic root (m^3/mol)."""
        return molar_volume(
            self.eos, t, p, x, self.tc, self.pc, self.omega, phase=phase, kij=self.kij
        )

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Isothermal-isobaric two-phase flash via the cubic EOS."""
        return flash_pt(self.eos, t, p, z, self.tc, self.pc, self.omega, kij=self.kij)

    def bubble_pressure(self, t: ArrayLike, x: Array) -> tuple[Array, Array]:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``."""
        return bubble_pressure_eos(self.eos, t, x, self.tc, self.pc, self.omega, kij=self.kij)

    def dew_pressure(self, t: ArrayLike, y: Array) -> tuple[Array, Array]:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``."""
        return dew_pressure_eos(self.eos, t, y, self.tc, self.pc, self.omega, kij=self.kij)

    def stability(self, t: ArrayLike, p: ArrayLike, z: Array) -> StabilityResult:
        """Michelsen tangent-plane stability of feed ``z`` at ``(T, P)``."""
        return stability_analysis(self.eos, t, p, z, self.tc, self.pc, self.omega, kij=self.kij)


jax.tree_util.register_dataclass(
    CubicPackage,
    data_fields=["tc", "pc", "omega", "cp", "kij"],
    meta_fields=["eos", "component_names"],
)


# --------------------------------------------------------------------------- #
# Gamma-phi (activity-coefficient liquid)
# --------------------------------------------------------------------------- #


def _excess_gibbs_over_rt(activity: ActivityModel, x: Array, t: Array) -> Array:
    """``g^E / (R T) = sum_i x_i ln gamma_i``."""
    return jnp.sum(x * activity.ln_gamma(x, t))


def excess_enthalpy(activity: ActivityModel, x: Array, t: ArrayLike) -> Array:
    """Excess (mixing) enthalpy ``h^E = -R T^2 d(g^E/RT)/dT`` (J/mol) by autodiff.

    The Gibbs-Helmholtz relation turns the temperature derivative of the excess
    Gibbs energy into the heat of mixing. Taking that derivative with
    automatic differentiation keeps ``h^E`` exactly consistent with the
    activity coefficients (and their temperature dependence), with no separate
    correlation to maintain.

    Args:
        activity: Liquid activity-coefficient model.
        x: Liquid mole fractions.
        t: Temperature (K).

    Returns:
        The molar excess enthalpy of the liquid mixture.
    """
    t_arr = jnp.asarray(t, dtype=float)
    x = jnp.asarray(x)
    dg_dt = jax.grad(lambda tt: _excess_gibbs_over_rt(activity, x, tt))(t_arr)
    return -R * t_arr**2 * dg_dt


def excess_entropy(activity: ActivityModel, x: Array, t: ArrayLike) -> Array:
    """Excess entropy ``s^E = (h^E - g^E) / T`` (J/mol/K)."""
    t_arr = jnp.asarray(t, dtype=float)
    x = jnp.asarray(x)
    g_e = R * t_arr * _excess_gibbs_over_rt(activity, x, t_arr)
    return (excess_enthalpy(activity, x, t_arr) - g_e) / t_arr


def _pure_saturated_liquid_residuals(
    eos: CubicEOS, t: Array, psat: Array, tc: Array, pc: Array, omega: Array
) -> tuple[Array, Array]:
    """Per-component residual ``(h, s)`` of each *pure* saturated liquid from the EOS."""

    def one(ps: Array, a: Array, b: Array, c: Array) -> tuple[Array, Array]:
        res = residual_properties(
            eos, t, ps, jnp.ones(1), a[None], b[None], c[None], phase="liquid"
        )
        return res.enthalpy, res.entropy

    return jax.vmap(one)(psat, tc, pc, omega)


@dataclass(frozen=True)
class GammaPhiPackage(_PackageBase):
    """Gamma-phi property package: activity-coefficient liquid, ideal or EOS vapour.

    The liquid fugacity is ``x_i gamma_i f_i^{0,L}(T, P)`` (see
    `fugacio.thermo.reference`), expressed here as an effective liquid fugacity
    coefficient ``phi_i^L = gamma_i f_i^{0,L} / P`` so the package presents the
    same ``ln_phi`` interface as the phi-phi routes. The liquid enthalpy is

        h^L(T, P, x) = sum_i x_i h_i^{L,pure}(T, P) + h^E(T, x)

    with each pure-liquid enthalpy taken from the cubic EOS at that component's
    saturation point (plus the small ``v_i^L (P - Psat_i)`` pressure term that
    partners the Poynting factor) and the excess enthalpy from `excess_enthalpy`
    (autodiff Gibbs-Helmholtz on the activity model). The liquid entropy follows
    the same construction with the ideal entropy of mixing and ``s^E``.

    Components must be subcritical at the conditions of interest (the reference
    fugacity is saturation-based, the standard gamma-phi limitation).

    Attributes:
        activity: Liquid activity-coefficient model (a differentiable pytree).
        tc: Critical temperatures (K).
        pc: Critical pressures (Pa).
        omega: Acentric factors.
        cp: Ideal-gas heat-capacity coefficient arrays ``(a, b, c, d, e)``.
        kij: Binary interaction matrix for an EOS vapour (``None`` means zeros).
        eos: Cubic EOS used for the saturation reference and, if selected, the vapour.
        vapor: ``"ideal"`` (phi^V = 1) or ``"eos"`` (static).
        poynting: Include the Poynting correction in the reference (static).
        phi_saturation: Include the saturation fugacity coefficient (static).
    """

    activity: ActivityModel
    tc: Array
    pc: Array
    omega: Array
    cp: CpCoeffs
    kij: Array | None = None
    eos: CubicEOS = PR
    vapor: str = "ideal"
    poynting: bool = False
    phi_saturation: bool = False

    component_names: tuple[str, ...] = ()

    @property
    def n_components(self) -> int:
        """Number of components."""
        return int(self.tc.shape[0])

    def signature(self) -> tuple[Any, ...]:
        """Class, component count, activity-model class, and the static flags."""
        return (
            "GammaPhiPackage",
            self.component_names,
            self.n_components,
            type(self.activity).__name__,
            repr(self.eos),
            self.vapor,
            self.poynting,
            self.phi_saturation,
            self.kij is not None,
        )

    def _psat(self, t: ArrayLike) -> Array:
        return saturation_pressures(self.eos, t, self.tc, self.pc, self.omega)

    def ln_phi(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Effective ``ln phi``: ``ln gamma + ln f^{0,L} - ln P`` (liquid) or the vapour model."""
        x = jnp.asarray(x)
        if phase == "liquid":
            f_ref, _ = liquid_reference_fugacity(
                self.eos,
                t,
                p,
                self.tc,
                self.pc,
                self.omega,
                poynting=self.poynting,
                phi_saturation=self.phi_saturation,
            )
            return self.activity.ln_gamma(x, t) + jnp.log(f_ref) - jnp.log(jnp.asarray(p))
        if self.vapor == "ideal":
            return jnp.zeros_like(x)
        ln_phi, _ = ln_phi_mixture(
            self.eos, t, p, x, self.tc, self.pc, self.omega, phase="vapor", kij=self.kij
        )
        return ln_phi

    def enthalpy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Liquid: pure saturated-liquid enthalpies plus ``h^E``; vapour: ideal gas (+ EOS)."""
        a, b, c, d, e = self.cp
        t_arr = jnp.asarray(t, dtype=float)
        x = jnp.asarray(x)
        if phase == "vapor":
            h = enthalpy_ig_mixture(t_arr, x, a, b, c, d, e)
            if self.vapor == "eos":
                h = (
                    h
                    + residual_properties(
                        self.eos,
                        t_arr,
                        p,
                        x,
                        self.tc,
                        self.pc,
                        self.omega,
                        phase="vapor",
                        kij=self.kij,
                    ).enthalpy
                )
            return h
        psat = self._psat(t_arr)
        h_res, _ = _pure_saturated_liquid_residuals(
            self.eos, t_arr, psat, self.tc, self.pc, self.omega
        )
        v_l = pure_liquid_volumes(self.eos, t_arr, psat, self.tc, self.pc, self.omega)
        h_pure = enthalpy_ig(t_arr, a, b, c, d, e) + h_res + v_l * (jnp.asarray(p) - psat)
        return jnp.sum(x * h_pure) + excess_enthalpy(self.activity, x, t_arr)

    def entropy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Liquid: saturated-liquid entropies, ideal mixing, ``s^E``; vapour: ideal gas (+ EOS)."""
        a, b, c, d, e = self.cp
        t_arr = jnp.asarray(t, dtype=float)
        x = jnp.asarray(x)
        if phase == "vapor":
            s = entropy_ig_mixture(t_arr, p, x, a, b, c, d, e)
            if self.vapor == "eos":
                s = (
                    s
                    + residual_properties(
                        self.eos,
                        t_arr,
                        p,
                        x,
                        self.tc,
                        self.pc,
                        self.omega,
                        phase="vapor",
                        kij=self.kij,
                    ).entropy
                )
            return s
        psat = self._psat(t_arr)
        _, s_res = _pure_saturated_liquid_residuals(
            self.eos, t_arr, psat, self.tc, self.pc, self.omega
        )
        s_pure = entropy_ig(t_arr, psat, a, b, c, d, e) + s_res
        x_ln_x = jnp.where(x > 0.0, x * jnp.log(jnp.where(x > 0.0, x, 1.0)), 0.0)
        return jnp.sum(x * s_pure) - R * jnp.sum(x_ln_x) + excess_entropy(self.activity, x, t_arr)

    def volume(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Liquid: mole-fraction sum of pure saturated-liquid volumes; vapour: ideal gas or EOS."""
        x = jnp.asarray(x)
        t_arr = jnp.asarray(t, dtype=float)
        if phase == "liquid":
            psat = self._psat(t_arr)
            v_l = pure_liquid_volumes(self.eos, t_arr, psat, self.tc, self.pc, self.omega)
            return jnp.sum(x * v_l)
        if self.vapor == "ideal":
            return R * t_arr / jnp.asarray(p, dtype=float)
        return molar_volume(
            self.eos, t_arr, p, x, self.tc, self.pc, self.omega, phase="vapor", kij=self.kij
        )

    def _kw(self) -> dict[str, Any]:
        return {
            "eos": self.eos,
            "kij": self.kij,
            "vapor": self.vapor,
            "poynting": self.poynting,
            "phi_saturation": self.phi_saturation,
        }

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Isothermal-isobaric gamma-phi flash."""
        return flash_pt_gamma(self.activity, t, p, z, self.tc, self.pc, self.omega, **self._kw())

    def k_seed(self, t: ArrayLike, p: ArrayLike, x: Array) -> Array:
        """``gamma_i(x) Psat_i / P``: the modified Raoult K-values (ideal vapour)."""
        gamma = jnp.exp(self.activity.ln_gamma(x, t))
        psat = saturation_pressures(self.eos, t, self.tc, self.pc, self.omega)
        return gamma * psat / jnp.asarray(p, dtype=float)

    def bubble_pressure(self, t: ArrayLike, x: Array) -> tuple[Array, Array]:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``."""
        return bubble_pressure_gamma(
            self.activity, t, x, self.tc, self.pc, self.omega, **self._kw()
        )

    def dew_pressure(self, t: ArrayLike, y: Array) -> tuple[Array, Array]:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``."""
        return dew_pressure_gamma(self.activity, t, y, self.tc, self.pc, self.omega, **self._kw())

    def bubble_temperature(
        self, p: ArrayLike, x: Array, *, t_min: float = 150.0, t_max: float = 700.0
    ) -> tuple[Array, Array]:
        """Bubble temperature and incipient vapour at fixed ``P``, ``x`` (native gamma-phi)."""
        return bubble_temperature_gamma(
            self.activity,
            p,
            x,
            self.tc,
            self.pc,
            self.omega,
            t_min=t_min,
            t_max=t_max,
            **self._kw(),
        )

    def dew_temperature(
        self, p: ArrayLike, y: Array, *, t_min: float = 150.0, t_max: float = 700.0
    ) -> tuple[Array, Array]:
        """Dew temperature and incipient liquid at fixed ``P``, ``y`` (native gamma-phi solve)."""
        return dew_temperature_gamma(
            self.activity,
            p,
            y,
            self.tc,
            self.pc,
            self.omega,
            t_min=t_min,
            t_max=t_max,
            **self._kw(),
        )


jax.tree_util.register_dataclass(
    GammaPhiPackage,
    data_fields=["activity", "tc", "pc", "omega", "cp", "kij"],
    meta_fields=["eos", "vapor", "poynting", "phi_saturation", "component_names"],
)


# --------------------------------------------------------------------------- #
# PC-SAFT
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SAFTPackage(_PackageBase):
    """PC-SAFT property package (phi-phi on the molecular equation of state).

    Residual enthalpy and entropy come from the temperature derivatives of the
    reduced residual Helmholtz energy (`fugacio.thermo.saft.properties`), so
    association effects on the heat of mixing and vaporisation are captured.

    Attributes:
        params: PC-SAFT parameter set (a differentiable pytree).
        tc: Critical temperatures (K), used to seed the flash K-values.
        pc: Critical pressures (Pa), used to seed the flash K-values.
        omega: Acentric factors, used to seed the flash K-values.
        cp: Ideal-gas heat-capacity coefficient arrays ``(a, b, c, d, e)``.
    """

    params: SaftParameters
    tc: Array
    pc: Array
    omega: Array
    cp: CpCoeffs

    component_names: tuple[str, ...] = ()

    @property
    def n_components(self) -> int:
        """Number of components."""
        return int(self.tc.shape[0])

    def signature(self) -> tuple[Any, ...]:
        """Class and component count."""
        return ("SAFTPackage", self.n_components, self.component_names)

    def ln_phi(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Log fugacity coefficients on the PC-SAFT density branch ``phase``."""
        return _saft_ln_phi(self.params, t, p, x, phase=phase)

    def enthalpy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Ideal-gas enthalpy plus the PC-SAFT residual enthalpy (J/mol)."""
        a, b, c, d, e = self.cp
        h_ig = enthalpy_ig_mixture(t, x, a, b, c, d, e)
        return h_ig + _saft_residual(self.params, t, p, x, phase=phase).enthalpy

    def entropy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Ideal-gas entropy (with mixing) plus the PC-SAFT residual entropy (J/mol/K)."""
        a, b, c, d, e = self.cp
        s_ig = entropy_ig_mixture(t, p, x, a, b, c, d, e)
        return s_ig + _saft_residual(self.params, t, p, x, phase=phase).entropy

    def volume(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Molar volume ``1 / rho`` on the requested density branch (m^3/mol)."""
        return 1.0 / _saft_density(self.params, t, p, x, phase=phase)

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Isothermal-isobaric two-phase flash via PC-SAFT."""
        return flash_pt_saft(self.params, t, p, z, self.tc, self.pc, self.omega)

    def bubble_pressure(self, t: ArrayLike, x: Array) -> tuple[Array, Array]:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``."""
        return bubble_pressure_saft(self.params, t, x, self.tc, self.pc, self.omega)

    def dew_pressure(self, t: ArrayLike, y: Array) -> tuple[Array, Array]:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``."""
        return dew_pressure_saft(self.params, t, y, self.tc, self.pc, self.omega)

    def stability(self, t: ArrayLike, p: ArrayLike, z: Array) -> StabilityResult:
        """Michelsen tangent-plane stability of feed ``z`` at ``(T, P)``."""
        return stability_saft(self.params, t, p, z, self.tc, self.pc, self.omega)


jax.tree_util.register_dataclass(
    SAFTPackage, data_fields=["params", "tc", "pc", "omega", "cp"], meta_fields=["component_names"]
)


# --------------------------------------------------------------------------- #
# Pure reference fluid (multiparameter Helmholtz EOS)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HelmholtzPackage(_PackageBase):
    """One-component package on a reference multiparameter Helmholtz EOS.

    Wraps a `fugacio.thermo.helmholtz.HelmholtzFluid` (IAPWS-95 water, Span-Wagner
    CO2, ...) so a pure utility or working fluid can flow through the ordinary
    unit operations with reference-grade properties. Compositions are the
    one-element vector ``[1.0]``; the "flash" is the saturation-line test of a
    pure substance (vapour fraction 0 or 1 away from the dome, the quality on it),
    and the energy flashes resolve directly to the steam-table state functions
    `state_ph` / `state_ps`.

    Enthalpy and entropy carry the reference state of the published formulation
    (not the ideal-gas ``T_REF`` reference of the mixture packages).

    Attributes:
        fluid: The reference fluid.
    """

    fluid: HelmholtzFluid

    component_names: tuple[str, ...] = ()

    @property
    def n_components(self) -> int:
        """Always one."""
        return 1

    def signature(self) -> tuple[Any, ...]:
        """Class and fluid name."""
        return ("HelmholtzPackage", self.fluid.name, self.component_names)

    def _branch(self, phase: str) -> str:
        return "liquid" if phase == "liquid" else "vapor"

    def ln_phi(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Pure-fluid ``ln phi`` on the requested density branch, as a length-1 vector."""
        st = state_tp(self.fluid, t, p, phase=self._branch(phase))
        return jnp.reshape(_hf_ln_phi(self.fluid, st.rho, st.t), (1,))

    def enthalpy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Molar enthalpy on the requested branch (J/mol)."""
        return state_tp(self.fluid, t, p, phase=self._branch(phase)).h

    def entropy(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Molar entropy on the requested branch (J/mol/K)."""
        return state_tp(self.fluid, t, p, phase=self._branch(phase)).s

    def volume(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Molar volume ``1 / rho`` on the requested branch (m^3/mol)."""
        return 1.0 / state_tp(self.fluid, t, p, phase=self._branch(phase)).rho

    def _psat(self, t: Array) -> Array:
        t_sat = jnp.clip(t, self.fluid.t_triple, T_SAT_MAX_FRACTION * self.fluid.t_critical)
        return saturation_pressure(self.fluid, t_sat)

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Pure-fluid phase test: all vapour below ``Psat(T)`` (or above ``Tc``), else liquid."""
        t_arr = jnp.asarray(t, dtype=float)
        p_arr = jnp.asarray(p, dtype=float)
        psat = self._psat(t_arr)
        vapor = (t_arr >= self.fluid.t_critical) | (p_arr < psat)
        beta = jnp.where(vapor, 1.0, 0.0)
        one = jnp.ones(1)
        return FlashResult(beta=beta, x=one, y=one, k=jnp.reshape(psat / p_arr, (1,)))

    def k_seed(self, t: ArrayLike, p: ArrayLike, x: Array) -> Array:
        """``Psat(T) / P`` as a length-1 vector."""
        return jnp.reshape(
            self._psat(jnp.asarray(t, dtype=float)) / jnp.asarray(p, dtype=float), (1,)
        )

    def mixture_enthalpy(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Stable-branch molar enthalpy at ``(T, P)`` (J/mol)."""
        return state_tp(self.fluid, t, p, phase="auto").h

    def mixture_entropy(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Stable-branch molar entropy at ``(T, P)`` (J/mol/K)."""
        return state_tp(self.fluid, t, p, phase="auto").s

    def mixture_volume(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Stable-branch molar volume at ``(T, P)`` (m^3/mol)."""
        return 1.0 / state_tp(self.fluid, t, p, phase="auto").rho

    def _energy_result(self, st: Any) -> EnergyFlashResult:
        vapor_like = st.rho < self.fluid.rho_critical
        beta = jnp.where(st.two_phase, st.q, jnp.where(vapor_like, 1.0, 0.0))
        one = jnp.ones(1)
        return EnergyFlashResult(t=st.t, beta=beta, x=one, y=one, k=one)

    def flash_ph(self, p: ArrayLike, h: ArrayLike, z: Array, **kwargs: Any) -> EnergyFlashResult:
        """Steam-table ``(P, h)`` state: saturation temperature and quality inside the dome."""
        return self._energy_result(state_ph(self.fluid, p, h))

    def flash_ps(self, p: ArrayLike, s: ArrayLike, z: Array, **kwargs: Any) -> EnergyFlashResult:
        """Steam-table ``(P, s)`` state: saturation temperature and quality inside the dome."""
        return self._energy_result(state_ps(self.fluid, p, s))

    def bubble_pressure(self, t: ArrayLike, x: Array) -> tuple[Array, Array]:
        """Saturation pressure at ``T`` (pure fluid)."""
        return self._psat(jnp.asarray(t, dtype=float)), jnp.ones(1)

    def dew_pressure(self, t: ArrayLike, y: Array) -> tuple[Array, Array]:
        """Saturation pressure at ``T`` (pure fluid)."""
        return self._psat(jnp.asarray(t, dtype=float)), jnp.ones(1)

    def bubble_temperature(
        self, p: ArrayLike, x: Array, *, t_min: float = 150.0, t_max: float = 700.0
    ) -> tuple[Array, Array]:
        """Saturation temperature at ``P`` (pure fluid)."""
        return saturation_temperature(self.fluid, p), jnp.ones(1)

    def dew_temperature(
        self, p: ArrayLike, y: Array, *, t_min: float = 150.0, t_max: float = 700.0
    ) -> tuple[Array, Array]:
        """Saturation temperature at ``P`` (pure fluid)."""
        return saturation_temperature(self.fluid, p), jnp.ones(1)


jax.tree_util.register_dataclass(
    HelmholtzPackage, data_fields=["fluid"], meta_fields=["component_names"]
)


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #


def cubic_package(
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    *,
    kij: Array | None = None,
    eos: CubicEOS = PR,
) -> CubicPackage:
    """Construct a `CubicPackage` from component constants and ``Cp`` coefficients."""
    return CubicPackage(
        tc=jnp.asarray(tc, dtype=float),
        pc=jnp.asarray(pc, dtype=float),
        omega=jnp.asarray(omega, dtype=float),
        cp=tuple(jnp.asarray(c, dtype=float) for c in cp),  # type: ignore[arg-type]
        kij=None if kij is None else jnp.asarray(kij, dtype=float),
        eos=eos,
    )


def gamma_phi_package(
    activity: ActivityModel,
    tc: Array,
    pc: Array,
    omega: Array,
    cp: CpCoeffs,
    *,
    kij: Array | None = None,
    eos: CubicEOS = PR,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
) -> GammaPhiPackage:
    """Construct a `GammaPhiPackage` from an activity model and component constants."""
    if vapor not in ("ideal", "eos"):
        raise ValueError(f"unknown vapor model {vapor!r}; use 'ideal' or 'eos'")
    return GammaPhiPackage(
        activity=activity,
        tc=jnp.asarray(tc, dtype=float),
        pc=jnp.asarray(pc, dtype=float),
        omega=jnp.asarray(omega, dtype=float),
        cp=tuple(jnp.asarray(c, dtype=float) for c in cp),  # type: ignore[arg-type]
        kij=None if kij is None else jnp.asarray(kij, dtype=float),
        eos=eos,
        vapor=vapor,
        poynting=poynting,
        phi_saturation=phi_saturation,
    )


def saft_package(
    params: SaftParameters, tc: Array, pc: Array, omega: Array, cp: CpCoeffs
) -> SAFTPackage:
    """Construct a `SAFTPackage` from PC-SAFT parameters, seeding constants, and ``Cp``."""
    return SAFTPackage(
        params=params,
        tc=jnp.asarray(tc, dtype=float),
        pc=jnp.asarray(pc, dtype=float),
        omega=jnp.asarray(omega, dtype=float),
        cp=tuple(jnp.asarray(c, dtype=float) for c in cp),  # type: ignore[arg-type]
    )


def helmholtz_package(fluid: HelmholtzFluid) -> HelmholtzPackage:
    """Wrap a reference fluid as a one-component `HelmholtzPackage`."""
    return HelmholtzPackage(fluid=fluid)


__all__ = [
    "CubicPackage",
    "EnergySolveResult",
    "GammaPhiPackage",
    "HelmholtzPackage",
    "PropertyPackage",
    "SAFTPackage",
    "cubic_package",
    "energy_flash_report",
    "excess_enthalpy",
    "excess_entropy",
    "flash_ph_with_info",
    "flash_ps_with_info",
    "gamma_phi_package",
    "helmholtz_package",
    "saft_package",
]


def energy_flash_report(
    pkg: PropertyPackage,
    result: EnergyFlashResult,
    p: ArrayLike,
    target: ArrayLike,
    z: Array,
    *,
    prop: str = "enthalpy",
    tol: float = 1e-7,
) -> SolveReport:
    """Verify an energy flash independently from its temperature iteration.

    The residual checks component closure, composition normalization, phase
    fractions, and the specified molar property. The iteration count is zero
    because this is a verification of the returned state, not its iteration log.
    """
    fn = getattr(pkg, prop)
    beta = result.beta

    def liquid(_: None) -> Array:
        return fn(result.t, p, result.x, phase="liquid")

    def vapor(_: None) -> Array:
        return fn(result.t, p, result.y, phase="vapor")

    index = jnp.where(beta <= 0, 0, jnp.where(beta >= 1, 2, 1)).astype(jnp.int32)
    value = lax.switch(
        index, [liquid, lambda _: (1 - beta) * liquid(None) + beta * vapor(None), vapor], None
    )
    scale = jnp.maximum(jnp.abs(target), 1e4 if prop == "enthalpy" else 10.0)
    errors = jnp.concatenate(
        [
            (1 - beta) * result.x + beta * result.y - z,
            jnp.array(
                [
                    (value - target) / scale,
                    jnp.sum(z) - 1.0,
                    jnp.maximum(-beta, 0.0) + jnp.maximum(beta - 1.0, 0.0),
                ]
            ),
        ]
    )
    return residual_report(errors, tol)


def flash_ph_with_info(
    pkg: PropertyPackage,
    p: ArrayLike,
    h: ArrayLike,
    z: Array,
    **options: Any,
) -> EnergySolveResult:
    """PH flash and an independent material/enthalpy verification report."""
    result = pkg.flash_ph(p, h, z, **options)
    return EnergySolveResult(result, energy_flash_report(pkg, result, p, h, z))


def flash_ps_with_info(
    pkg: PropertyPackage,
    p: ArrayLike,
    s: ArrayLike,
    z: Array,
    **options: Any,
) -> EnergySolveResult:
    """PS flash and an independent material/entropy verification report."""
    result = pkg.flash_ps(p, s, z, **options)
    return EnergySolveResult(result, energy_flash_report(pkg, result, p, s, z, prop="entropy"))
