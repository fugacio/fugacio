"""Property packages: one object that owns phase equilibrium *and* energy.

A process simulator needs two things from its thermodynamics: "what splits?"
(fugacities, K-values, flashes) and "how much energy?" (enthalpy, entropy,
volume). A `PropertyPackage` answers both through one interface that every
energy-balanced unit operation, the rigorous column, and the equation-oriented
engine consume. It is implemented for all four method classes Fugacio carries:

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

**One contract for every package.** Each iterative calculation has a checked
``*_with_info`` method returning the best state and a
`fugacio.thermo.diagnostics.SolveReport`: ``flash_pt_with_info``,
``bubble_pressure_with_info``, ``dew_pressure_with_info``,
``bubble_temperature_with_info``, ``dew_temperature_with_info``,
``flash_ph_with_info``, and ``flash_ps_with_info``. The matching value-only
method returns NaN whenever that report fails, so a failed or out-of-domain
solve never returns a finite number. ``stability`` runs the shared tangent-plane
search (`fugacio.thermo.stability`) on the package's own fugacity branches.

A new thermodynamic method plugs into the whole engine by subclassing
`_PackageBase` and implementing the single-phase primitives (``ln_phi``,
``enthalpy``, ``entropy``, ``volume``); every other method, including a generic
flash, saturation solves, stability, and energy flashes, is derived from them
and may be overridden with a faster specialized solver. Packages are registered
JAX pytrees whose model parameters are differentiable leaves, so a flowsheet
built on any of them is differentiable with respect to the thermodynamic
parameters as well as the operating conditions.

Enthalpy and entropy are relative to the ideal-gas reference at ``T_REF`` /
``P_REF``, except for `HelmholtzPackage`, whose reference is the one built into
the published formulation. Only differences are physical, and any consistent
reference cancels in a balance, so do not mix packages with different
references across one energy balance.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from fugacio.thermo.activity.models import ActivityModel
from fugacio.thermo.constants import R
from fugacio.thermo.departure import residual_properties
from fugacio.thermo.diagnostics import (
    SolveReport,
    SolveStatus,
    nan_unless_converged,
    residual_report,
    with_status,
)
from fugacio.thermo.energy import EnergyFlashResult, _implicit_temperature
from fugacio.thermo.eos import PR, CubicEOS, ln_phi_mixture, molar_volume
from fugacio.thermo.equilibrium import (
    FlashResult,
    FlashSolveResult,
    SaturationResult,
    SaturationSolveResult,
    bubble_pressure_eos_with_info,
    dew_pressure_eos_with_info,
    flash_pt_with_info,
    input_report,
    phase_compositions,
    rachford_rice,
    wilson_k,
)
from fugacio.thermo.gammaphi import (
    bubble_pressure_gamma_with_info,
    bubble_temperature_gamma_with_info,
    default_temperature_bracket,
    dew_pressure_gamma_with_info,
    dew_temperature_gamma_with_info,
    flash_pt_gamma_with_info,
    reference_domain,
)
from fugacio.thermo.helmholtz.fluids import HelmholtzFluid
from fugacio.thermo.helmholtz.props import ln_fugacity_coefficient as _hf_ln_phi
from fugacio.thermo.helmholtz.saturation import (
    T_SAT_MAX_FRACTION,
    saturation_pressure_with_info,
    saturation_temperature_with_info,
)
from fugacio.thermo.helmholtz.states import state_ph, state_ps, state_tp
from fugacio.thermo.ideal import (
    enthalpy_ig,
    enthalpy_ig_mixture,
    entropy_ig,
    entropy_ig_mixture,
)
from fugacio.thermo.implicit import (
    bracketed_root_with_info,
    fixed_point_with_info,
    gate_tree,
    newton_system_with_info,
    scanned_root_with_info,
)
from fugacio.thermo.provenance import PackageEvidence
from fugacio.thermo.reference import (
    liquid_reference_fugacity,
    pure_liquid_volumes,
    saturation_pressures_with_info,
)
from fugacio.thermo.saft.equilibrium import (
    bubble_pressure_saft_with_info,
    dew_pressure_saft_with_info,
    flash_pt_saft_with_info,
)
from fugacio.thermo.saft.parameters import SaftParameters
from fugacio.thermo.saft.properties import ln_fugacity_coefficients as _saft_ln_phi
from fugacio.thermo.saft.properties import molar_density as _saft_density
from fugacio.thermo.saft.properties import residual_properties as _saft_residual
from fugacio.thermo.stability import (
    StabilityResult,
    enrichment_starts,
    feed_potentials,
    tpd_search,
)

ArrayLike = Array | float
CpCoeffs = tuple[Array, Array, Array, Array, Array]
LnPhiFn = Callable[[Array], Array]

#: Vapour fractions closer than this to 0 or 1 are treated as single-phase when a
#: bulk property is differentiated (so the absent phase is never differentiated).
_SINGLE_PHASE_EPS = 1.0e-9

#: Solve methods that run through a compiled kernel (see `_compiled`). The
#: value-only wrappers (``flash_pt``, ``bubble_pressure``, ...) aren't listed:
#: they call these kernels and mask a failure, and compiling them separately
#: would compile each nested solve a second time.
_COMPILED_METHODS = (
    "flash_pt_with_info",
    "stability",
    "bubble_pressure_with_info",
    "dew_pressure_with_info",
    "bubble_temperature_with_info",
    "dew_temperature_with_info",
    "mixture_enthalpy",
    "mixture_entropy",
    "mixture_volume",
    "flash_ph_with_info",
    "flash_ps_with_info",
    "flash_tv",
)

_KERNELS: dict[tuple[Any, ...], Any] = {}


def _strip_weak_type(value: Any) -> Any:
    return jnp.asarray(value, dtype=jnp.asarray(value).dtype)


def _compiled(method: Callable[..., Any]) -> Callable[..., Any]:
    """Evaluate a package method through a compiled kernel cached per method and options.

    The package and the state arguments are dynamic, so every package with the
    same structure (class, component count, static settings) shares one
    compilation, and a new state or new parameter values reuse it. Array
    keyword options stay dynamic; other options (tolerances, iteration caps)
    are part of the cache key. A package that isn't a registered pytree runs
    eagerly, as does a call with an unhashable option.
    """

    @functools.wraps(method)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_pytree(self):
            return method(self, *args, **kwargs)
        dynamic = {
            k: v for k, v in kwargs.items() if isinstance(v, Array | np.ndarray | jax.core.Tracer)
        }
        static = tuple(sorted((k, v) for k, v in kwargs.items() if k not in dynamic))
        key = (method, static)
        try:
            kernel = _KERNELS.get(key)
        except TypeError:
            return method(self, *args, **kwargs)
        if kernel is None:
            fixed = dict(static)

            def run(pkg: Any, positional: tuple[Any, ...], named: dict[str, Any]) -> Any:
                return method(pkg, *positional, **named, **fixed)

            kernel = _KERNELS.setdefault(key, jax.jit(run))
        positional = tuple(jnp.asarray(a, dtype=float) for a in args)
        pkg, named = jax.tree_util.tree_map(_strip_weak_type, (self, dynamic))
        return kernel(pkg, positional, named)

    return call


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
    Subclassing `_PackageBase` supplies every derived method.
    """

    @property
    def component_names(self) -> tuple[str, ...]:
        """Canonical component names, in the package's composition order."""
        ...

    @property
    def evidence(self) -> PackageEvidence:
        """Parameter provenance and the observed ranges behind the package."""
        ...

    @property
    def n_components(self) -> int:
        """Number of components the package describes."""
        ...

    def signature(self) -> tuple[Any, ...]:
        """Hashable description of the package *structure* (not its values)."""
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

    def ln_phi_function(self, t: ArrayLike, p: ArrayLike, *, phase: str) -> LnPhiFn:
        """A composition-only ``ln phi`` at fixed ``(T, P)`` for repeated trial evaluations."""
        ...

    def in_domain(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Whether the model is defined for the components present at ``(T, P)``."""
        ...

    def k_values(self, t: ArrayLike, p: ArrayLike, x: Array, y: Array) -> Array:
        """Equilibrium ratios ``K_i = phi_i^L(x) / phi_i^V(y)``."""
        ...

    def k_seed(self, t: ArrayLike, p: ArrayLike, x: Array) -> Array:
        """Composition-light K-value estimate used to initialise solvers."""
        ...

    def flash_pt_with_info(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashSolveResult:
        """Isothermal-isobaric vapour-liquid flash and its report."""
        ...

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Isothermal-isobaric flash; NaN on failure."""
        ...

    def stability(self, t: ArrayLike, p: ArrayLike, z: Array) -> StabilityResult:
        """Tangent-plane stability of feed ``z`` at ``(T, P)``."""
        ...

    def bubble_pressure_with_info(self, t: ArrayLike, x: Array) -> SaturationSolveResult:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``, with a report."""
        ...

    def bubble_pressure(self, t: ArrayLike, x: Array) -> SaturationResult:
        """Bubble pressure and incipient vapour ``(P, y)``; NaN on failure."""
        ...

    def dew_pressure_with_info(self, t: ArrayLike, y: Array) -> SaturationSolveResult:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``, with a report."""
        ...

    def dew_pressure(self, t: ArrayLike, y: Array) -> SaturationResult:
        """Dew pressure and incipient liquid ``(P, x)``; NaN on failure."""
        ...

    def bubble_temperature_with_info(
        self,
        p: ArrayLike,
        x: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationSolveResult:
        """Bubble temperature and incipient vapour at fixed ``P``, ``x``, with a report."""
        ...

    def bubble_temperature(
        self,
        p: ArrayLike,
        x: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationResult:
        """Bubble temperature and incipient vapour ``(T, y)``; NaN on failure."""
        ...

    def dew_temperature_with_info(
        self,
        p: ArrayLike,
        y: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationSolveResult:
        """Dew temperature and incipient liquid at fixed ``P``, ``y``, with a report."""
        ...

    def dew_temperature(
        self,
        p: ArrayLike,
        y: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationResult:
        """Dew temperature and incipient liquid ``(T, x)``; NaN on failure."""
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

    def flash_ph_with_info(
        self,
        p: ArrayLike,
        h: ArrayLike,
        z: Array,
        *,
        t_init: ArrayLike = ...,
        t_min: float = ...,
        t_max: float = ...,
        tol: float = ...,
        max_iter: int = ...,
    ) -> EnergySolveResult:
        """Isenthalpic flash with an independent energy and equilibrium report."""
        ...

    def flash_ph(self, p: ArrayLike, h: ArrayLike, z: Array, **options: Any) -> EnergyFlashResult:
        """Isenthalpic flash; NaN on failure."""
        ...

    def flash_ps_with_info(
        self,
        p: ArrayLike,
        s: ArrayLike,
        z: Array,
        *,
        t_init: ArrayLike = ...,
        t_min: float = ...,
        t_max: float = ...,
        tol: float = ...,
        max_iter: int = ...,
    ) -> EnergySolveResult:
        """Isentropic flash with an independent entropy and equilibrium report."""
        ...

    def flash_ps(self, p: ArrayLike, s: ArrayLike, z: Array, **options: Any) -> EnergyFlashResult:
        """Isentropic flash; NaN on failure."""
        ...


def _phase_classification(pkg: Any, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
    """Evaluate a detached phase locator without tracing an unused flash derivative.

    Stopping only the output is too late for eager linearization: the flash's
    implicit rule can already have assembled its Jacobian. Detach input array
    leaves, including registered package parameters, before calling the flash.
    The output gate also supports custom packages held as opaque Python objects.
    """
    package, temperature, pressure, composition = jax.tree_util.tree_map(
        lambda value: (
            lax.stop_gradient(value) if isinstance(value, Array | jax.core.Tracer) else value
        ),
        (pkg, t, p, z),
    )
    return lax.stop_gradient(package.flash_pt(temperature, pressure, composition))


def _is_pytree(pkg: Any) -> bool:
    """Whether a package is a registered pytree of array leaves.

    Such a package can cross a compilation boundary and its parameters are
    differentiable. A package holding an opaque leaf (an unregistered activity
    model, say) is evaluated eagerly instead.
    """
    leaves = jax.tree_util.tree_leaves(pkg)
    if len(leaves) == 1 and leaves[0] is pkg:
        return False
    return all(
        isinstance(leaf, Array | np.ndarray | np.generic | float | int | jax.core.Tracer)
        for leaf in leaves
    )


def _distinct_phases(pkg: Any, t: ArrayLike, p: Array, liquid: Array, vapor: Array) -> Array:
    """Whether a converged saturation point has two genuinely different phases."""
    v_l = pkg.volume(t, p, liquid, phase="liquid")
    v_v = pkg.volume(t, p, vapor, phase="vapor")
    same_density = jnp.abs(v_v - v_l) <= 1e-6 * jnp.maximum(jnp.abs(v_v), 1e-30)
    same_composition = jnp.max(jnp.abs(vapor - liquid)) <= 1e-6
    return lax.stop_gradient(~(same_density & same_composition))


class _PackageBase:
    """Generic machinery shared by every concrete package.

    Subclasses provide the four single-phase primitives (``ln_phi``,
    ``enthalpy``, ``entropy``, ``volume``) and ``n_components``. Everything
    else is derived here from those primitives: a successive-substitution PT
    flash, phi-phi saturation pressures, saturation temperatures on a scanned
    bracket, tangent-plane stability, two-phase-aware bulk properties, and
    energy-specified flashes. Concrete packages override the calculations for
    which they have faster specialized solvers.
    """

    component_names: tuple[str, ...] = ()
    evidence: PackageEvidence = PackageEvidence()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Compile the solve methods a subclass defines (see `_compiled`)."""
        super().__init_subclass__(**kwargs)
        for name in _COMPILED_METHODS:
            method = cls.__dict__.get(name)
            if method is not None:
                setattr(cls, name, _compiled(method))

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

    @property
    def n_components(self) -> int:
        """Number of components."""
        raise NotImplementedError

    def signature(self) -> tuple[Any, ...]:
        """Hashable structural description (class name plus static metadata)."""
        return (type(self).__name__, self.n_components, self.component_names)

    # -- hooks with generic defaults ---------------------------------------- #
    def ln_phi_function(self, t: ArrayLike, p: ArrayLike, *, phase: str) -> LnPhiFn:
        """``ln phi`` as a function of composition alone, at fixed ``(T, P)``.

        Stability searches evaluate hundreds of trial compositions; a package
        with composition-independent parts (a gamma-phi liquid reference) caches
        them here.
        """
        return lambda w: self.ln_phi(t, p, w, phase=phase)

    def in_domain(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Whether the model is defined for the components present at ``(T, P)``."""
        return jnp.asarray(True)

    def k_values(self, t: ArrayLike, p: ArrayLike, x: Array, y: Array) -> Array:
        """Equilibrium ratios ``K_i = phi_i^L(x) / phi_i^V(y)`` at ``(T, P)``."""
        ln_l = self.ln_phi(t, p, x, phase="liquid")
        ln_v = self.ln_phi(t, p, y, phase="vapor")
        return jnp.exp(ln_l - ln_v)

    def k_seed(self, t: ArrayLike, p: ArrayLike, x: Array) -> Array:
        """Composition-light K-values for initializing flashes and staged solvers.

        Packages with critical constants use the Wilson correlation (a phi-phi
        model evaluated with ``x == y`` returns ``K = 1`` wherever the equation of
        state has a single density root, which is useless for initialization).
        Others use the fugacity-coefficient ratio of the two branches at ``x``.
        """
        tc, pc, omega = (getattr(self, name, None) for name in ("tc", "pc", "omega"))
        if tc is not None and pc is not None and omega is not None:
            return wilson_k(t, p, tc, pc, omega)
        return self.k_values(t, p, x, x)

    def saturation_bracket(self, composition: Array) -> tuple[Array, Array]:
        """Default temperature bracket for saturation solves at ``composition``."""
        tc = getattr(self, "tc", None)
        if tc is not None:
            return default_temperature_bracket(tc, composition)
        return jnp.asarray(150.0), jnp.asarray(700.0)

    def gibbs(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Single-phase molar Gibbs energy ``G = H - T S`` (J/mol)."""
        t_arr = jnp.asarray(t, dtype=float)
        return self.enthalpy(t, p, x, phase=phase) - t_arr * self.entropy(t, p, x, phase=phase)

    def heat_capacity(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Single-phase isobaric heat capacity ``(dH/dT)_P`` by autodiff (J/mol/K)."""
        return jax.grad(lambda tt: self.enthalpy(tt, p, x, phase=phase))(
            jnp.asarray(t, dtype=float)
        )

    # -- PT flash ------------------------------------------------------------ #
    def flash_pt_with_info(
        self, t: ArrayLike, p: ArrayLike, z: Array, *, tol: float = 1e-12, max_iter: int = 300
    ) -> FlashSolveResult:
        """Isothermal-isobaric vapour-liquid flash by successive substitution on ``ln_phi``.

        Seeded from `k_seed`. A single-phase feed collapses onto the trivial
        solution ``K = 1``; its phase is the branch with the lower Gibbs energy.
        """
        z = jnp.asarray(z, dtype=float)
        t_arr, p_arr = jnp.asarray(t, dtype=float), jnp.asarray(p, dtype=float)
        k0 = jnp.clip(lax.stop_gradient(self.k_seed(t_arr, p_arr, z)), 1e-10, 1e10)
        differentiable = _is_pytree(self)

        def g(ln_k: Array, theta: Any) -> Array:
            pkg, t_, p_, z_ = theta if differentiable else (self, *theta)
            k = jnp.exp(ln_k)
            x, y = phase_compositions(z_, k, rachford_rice(z_, k))
            y = y / jnp.sum(y)
            x = x / jnp.sum(x)
            return pkg.ln_phi(t_, p_, x, phase="liquid") - pkg.ln_phi(t_, p_, y, phase="vapor")

        theta = (self, t_arr, p_arr, z) if differentiable else (t_arr, p_arr, z)
        solved = fixed_point_with_info(g, jnp.log(k0), theta, tol, max_iter)
        k = jnp.exp(solved.value)
        beta = rachford_rice(z, k)
        support = z > 0
        trivial = jnp.max(jnp.where(support, jnp.abs(solved.value), 0.0)) < 1e-6
        ln_z = jnp.log(jnp.where(support, z, 1.0))
        g_l = jnp.sum(jnp.where(support, z * (ln_z + self.ln_phi(t, p, z, phase="liquid")), 0.0))
        g_v = jnp.sum(jnp.where(support, z * (ln_z + self.ln_phi(t, p, z, phase="vapor")), 0.0))
        beta = jnp.where(trivial, jnp.where(lax.stop_gradient(g_l <= g_v), 0.0, 1.0), beta)
        x, y = phase_compositions(z, k, beta)
        report = with_status(solved.report, ~input_report(t, p, z), SolveStatus.INVALID_INPUT)
        report = with_status(report, ~self.in_domain(t, p, z), SolveStatus.OUT_OF_DOMAIN)
        return FlashSolveResult(FlashResult(beta=beta, x=x, y=y, k=k), report)

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Isothermal-isobaric flash; NaN when `flash_pt_with_info` reports failure."""
        solved = self.flash_pt_with_info(t, p, z)
        return nan_unless_converged(solved.value, solved.report)

    # -- phase stability ----------------------------------------------------- #
    def stability(
        self,
        t: ArrayLike,
        p: ArrayLike,
        z: Array,
        *,
        iterations: int = 160,
        tol: float = 1e-7,
    ) -> StabilityResult:
        """Tangent-plane stability of feed ``z`` at ``(T, P)`` on both phase branches.

        The feed is referred to its lower-Gibbs single phase. Trial phases start
        from the feed, the Wilson-like vapour and liquid estimates, and every
        pure-component enrichment, on both the liquid and the vapour branch, so a
        liquid-liquid split is found as reliably as a vapour-liquid one. See
        `fugacio.thermo.stability.tpd_search`.
        """
        z = jnp.asarray(z, dtype=float)
        support = z > 0
        branches = (
            self.ln_phi_function(t, p, phase="liquid"),
            self.ln_phi_function(t, p, phase="vapor"),
        )
        d = feed_potentials(branches, z, support)
        k = jnp.clip(lax.stop_gradient(self.k_seed(t, p, z)), 1e-10, 1e10)
        starts = jnp.concatenate((enrichment_starts(z), (z * k)[None, :], (z / k)[None, :]))
        return tpd_search(branches, d, support, starts, iterations=iterations, tol=tol)

    # -- saturation pressures (generic phi-phi) ------------------------------ #
    def _saturation_pressure(
        self, t: ArrayLike, fixed: Array, *, bubble: bool, tol: float = 1e-12, max_iter: int = 300
    ) -> SaturationSolveResult:
        fixed = jnp.asarray(fixed, dtype=float)
        t_arr = jnp.asarray(t, dtype=float)
        p_ref = jnp.asarray(1e5)
        k = jnp.clip(lax.stop_gradient(self.k_seed(t_arr, p_ref, fixed)), 1e-10, 1e10)
        p0 = p_ref * (jnp.sum(fixed * k) if bubble else 1.0 / jnp.sum(fixed / k))
        other0 = fixed * k if bubble else fixed / k
        state0 = jnp.concatenate([jnp.log(p0)[None], other0 / jnp.sum(other0)])
        differentiable = _is_pytree(self)

        def g(state: Array, theta: Any) -> Array:
            pkg, t_, fixed_ = theta if differentiable else (self, *theta)
            pressure = jnp.exp(state[0])
            other = state[1:]
            liquid, vapor = (fixed_, other) if bubble else (other, fixed_)
            ratio = jnp.exp(
                pkg.ln_phi(t_, pressure, liquid, phase="liquid")
                - pkg.ln_phi(t_, pressure, vapor, phase="vapor")
            )
            unnormalized = fixed_ * ratio if bubble else fixed_ / ratio
            s = jnp.sum(unnormalized)
            shift = jnp.log(s) if bubble else -jnp.log(s)
            return jnp.concatenate([(state[0] + shift)[None], unnormalized / s])

        theta = (self, t_arr, fixed) if differentiable else (t_arr, fixed)
        solved = fixed_point_with_info(g, state0, theta, tol, max_iter)
        pressure, other = jnp.exp(solved.value[0]), solved.value[1:]
        liquid, vapor = (fixed, other) if bubble else (other, fixed)
        report = with_status(
            solved.report,
            ~_distinct_phases(self, t_arr, pressure, liquid, vapor),
            SolveStatus.TRIVIAL,
        )
        report = with_status(report, ~input_report(t, 1.0, fixed), SolveStatus.INVALID_INPUT)
        report = with_status(report, ~self.in_domain(t, pressure, fixed), SolveStatus.OUT_OF_DOMAIN)
        value = SaturationResult(pressure, other)
        return SaturationSolveResult(gate_tree(value, report.converged), report)

    def bubble_pressure_with_info(self, t: ArrayLike, x: Array) -> SaturationSolveResult:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``, with a report.

        A coupled fixed point in ``(ln P, y)`` on the package's fugacity
        coefficients; a collapse onto one phase is ``TRIVIAL``.
        """
        return self._saturation_pressure(t, x, bubble=True)

    def bubble_pressure(self, t: ArrayLike, x: Array) -> SaturationResult:
        """Bubble pressure and incipient vapour ``(P, y)``; NaN on failure."""
        solved = self.bubble_pressure_with_info(t, x)
        return nan_unless_converged(solved.value, solved.report)

    def dew_pressure_with_info(self, t: ArrayLike, y: Array) -> SaturationSolveResult:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``, with a report."""
        return self._saturation_pressure(t, y, bubble=False)

    def dew_pressure(self, t: ArrayLike, y: Array) -> SaturationResult:
        """Dew pressure and incipient liquid ``(P, x)``; NaN on failure."""
        solved = self.dew_pressure_with_info(t, y)
        return nan_unless_converged(solved.value, solved.report)

    # -- saturation temperatures (pressure inversion on a scanned bracket) --- #
    def _saturation_temperature(
        self,
        p: ArrayLike,
        fixed: Array,
        t_min: ArrayLike | None,
        t_max: ArrayLike | None,
        *,
        bubble: bool,
    ) -> SaturationSolveResult:
        fixed = jnp.asarray(fixed, dtype=float)
        lo, hi = self.saturation_bracket(fixed)
        lo = lo if t_min is None else jnp.asarray(t_min, dtype=float)
        hi = hi if t_max is None else jnp.asarray(t_max, dtype=float)
        differentiable = _is_pytree(self)

        def residual(t: Array, params: Any) -> Array:
            pkg, p_, fixed_ = params if differentiable else (self, *params)
            point = pkg.bubble_pressure(t, fixed_) if bubble else pkg.dew_pressure(t, fixed_)
            # NaN wherever no saturation point exists; the bracket scan skips it.
            return jnp.log(point.value) - jnp.log(p_)

        p_arr = jnp.asarray(p, dtype=float)
        params = (self, p_arr, fixed) if differentiable else (p_arr, fixed)
        solved = scanned_root_with_info(residual, params, lo, hi, 1e-9, 200)
        at_root = (
            self.bubble_pressure_with_info(solved.value, fixed)
            if bubble
            else self.dew_pressure_with_info(solved.value, fixed)
        )
        report = with_status(solved.report, ~at_root.report.converged, at_root.report.status)
        value = SaturationResult(solved.value, at_root.value.composition)
        return SaturationSolveResult(gate_tree(value, report.converged), report)

    def bubble_temperature_with_info(
        self,
        p: ArrayLike,
        x: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationSolveResult:
        """Bubble temperature and incipient vapour at fixed ``P``, ``x``, with a report.

        Inverts `bubble_pressure` on ``[t_min, t_max]`` (by default
        `saturation_bracket`), scanning for the first finite sign change, so a
        bracket that only partly admits a bubble point still works and one that
        admits none is reported rather than returning an endpoint.
        """
        return self._saturation_temperature(p, x, t_min, t_max, bubble=True)

    def bubble_temperature(
        self,
        p: ArrayLike,
        x: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationResult:
        """Bubble temperature and incipient vapour ``(T, y)``; NaN on failure."""
        solved = self.bubble_temperature_with_info(p, x, t_min=t_min, t_max=t_max)
        return nan_unless_converged(solved.value, solved.report)

    def dew_temperature_with_info(
        self,
        p: ArrayLike,
        y: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationSolveResult:
        """Dew temperature and incipient liquid at fixed ``P``, ``y``, with a report."""
        return self._saturation_temperature(p, y, t_min, t_max, bubble=False)

    def dew_temperature(
        self,
        p: ArrayLike,
        y: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationResult:
        """Dew temperature and incipient liquid ``(T, x)``; NaN on failure."""
        solved = self.dew_temperature_with_info(p, y, t_min=t_min, t_max=t_max)
        return nan_unless_converged(solved.value, solved.report)

    # -- derived: two-phase-aware bulk properties -------------------------- #
    def _blend(self, t: ArrayLike, p: ArrayLike, z: Array, prop: str) -> Array:
        """Bulk molar property of feed ``z``: ``(1 - beta) M^L(x) + beta M^V(y)``.

        The blend is evaluated with a `jax.lax.switch` on the phase regime so a
        single-phase feed differentiates *only* the phase that exists. Naively
        multiplying an absent phase by zero would still propagate the ``NaN``
        gradient of a cubic root that does not exist in that region. A failed
        flash (NaN phase fraction) gives a NaN property.
        """
        beta = _phase_classification(self, t, p, z).beta
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
            classified = _phase_classification(self, temperature, p, z)

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

    def flash_ph_with_info(
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
    ) -> EnergySolveResult:
        """Isenthalpic flash: find ``T`` so the feed enthalpy equals ``h``.

        The temperature is a safeguarded Newton/bisection root of the monotone
        enthalpy residual on ``[t_min, t_max]`` and is differentiated by the
        implicit function theorem with respect to ``p``, ``h``, ``z`` and the
        package's own parameters. The report independently verifies energy,
        material, and equilibrium closure of the returned state.
        """
        result = self._energy_flash(p, h, z, "enthalpy", t_init, t_min, t_max, tol, max_iter)
        return EnergySolveResult(result, energy_flash_report(self, result, p, h, z))

    def flash_ph(self, p: ArrayLike, h: ArrayLike, z: Array, **options: Any) -> EnergyFlashResult:
        """Isenthalpic flash; NaN when `flash_ph_with_info` reports failure."""
        solved = self.flash_ph_with_info(p, h, z, **options)
        return nan_unless_converged(solved.value, solved.report)

    def flash_ps_with_info(
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
    ) -> EnergySolveResult:
        """Isentropic flash: find ``T`` so the feed entropy equals ``s``.

        The backbone of isentropic compressor and turbine models; same solver,
        differentiability, and independent verification as `flash_ph_with_info`.
        """
        result = self._energy_flash(p, s, z, "entropy", t_init, t_min, t_max, tol, max_iter)
        return EnergySolveResult(result, energy_flash_report(self, result, p, s, z, prop="entropy"))

    def flash_ps(self, p: ArrayLike, s: ArrayLike, z: Array, **options: Any) -> EnergyFlashResult:
        """Isentropic flash; NaN when `flash_ps_with_info` reports failure."""
        solved = self.flash_ps_with_info(p, s, z, **options)
        return nan_unless_converged(solved.value, solved.report)

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
        fixed-volume vessel or a receiver imposes. Returns ``(P, flash)``, NaN
        when no pressure in the bracket reproduces ``v``.
        """
        params = (self, jnp.asarray(t, dtype=float), jnp.asarray(v, dtype=float), jnp.asarray(z))

        def residual(ln_p: Array, params: Any) -> Array:
            pkg, t_, v_, z_ = params
            return jnp.log(pkg.mixture_volume(t_, jnp.exp(ln_p), z_)) - jnp.log(v_)

        solved = bracketed_root_with_info(
            residual,
            params,
            jnp.log(jnp.asarray(p_min)),
            jnp.log(jnp.asarray(p_max)),
            tol,
            max_iter,
        )
        p_star = nan_unless_converged(jnp.exp(solved.value), solved.report)
        return p_star, self.flash_pt(t, p_star, z)


for _name in _COMPILED_METHODS:
    setattr(_PackageBase, _name, _compiled(_PackageBase.__dict__[_name]))


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
    evidence: PackageEvidence = field(default_factory=PackageEvidence)

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

    def flash_pt_with_info(
        self, t: ArrayLike, p: ArrayLike, z: Array, *, tol: float = 1e-12, max_iter: int = 300
    ) -> FlashSolveResult:
        """Isothermal-isobaric vapour-liquid flash via the cubic EOS."""
        return flash_pt_with_info(
            self.eos,
            t,
            p,
            z,
            self.tc,
            self.pc,
            self.omega,
            kij=self.kij,
            tol=tol,
            max_iter=max_iter,
        )

    def bubble_pressure_with_info(self, t: ArrayLike, x: Array) -> SaturationSolveResult:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``, with a report."""
        return bubble_pressure_eos_with_info(
            self.eos, t, x, self.tc, self.pc, self.omega, kij=self.kij
        )

    def dew_pressure_with_info(self, t: ArrayLike, y: Array) -> SaturationSolveResult:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``, with a report."""
        return dew_pressure_eos_with_info(
            self.eos, t, y, self.tc, self.pc, self.omega, kij=self.kij
        )


jax.tree_util.register_dataclass(
    CubicPackage,
    data_fields=["tc", "pc", "omega", "cp", "kij"],
    meta_fields=["eos", "component_names", "evidence"],
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

    The saturation-based reference exists only below each component's critical
    temperature: a state with a present supercritical component is out of the
    package's domain (every checked calculation reports ``OUT_OF_DOMAIN``).

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
    evidence: PackageEvidence = field(default_factory=PackageEvidence)

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
        """Finite saturation pressures (extrapolated for supercritical components)."""
        return saturation_pressures_with_info(self.eos, t, self.tc, self.pc, self.omega)[0]

    def _liquid_reference(self, t: ArrayLike, p: ArrayLike) -> Array:
        """Composition-independent ``ln f^{0,L} - ln P`` of the liquid branch."""
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
        return jnp.log(f_ref) - jnp.log(jnp.asarray(p))

    def in_domain(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Every present component must have a saturation reference at ``T``."""
        return reference_domain(t, z, self.tc, self.eos, self.pc, self.omega)

    def ln_phi(self, t: ArrayLike, p: ArrayLike, x: Array, *, phase: str) -> Array:
        """Effective ``ln phi``: ``ln gamma + ln f^{0,L} - ln P`` (liquid) or the vapour model."""
        x = jnp.asarray(x)
        if phase == "liquid":
            return self.activity.ln_gamma(x, t) + self._liquid_reference(t, p)
        if self.vapor == "ideal":
            return jnp.zeros_like(x)
        ln_phi, _ = ln_phi_mixture(
            self.eos, t, p, x, self.tc, self.pc, self.omega, phase="vapor", kij=self.kij
        )
        return ln_phi

    def ln_phi_function(self, t: ArrayLike, p: ArrayLike, *, phase: str) -> LnPhiFn:
        """Liquid trials reuse one reference fugacity; only ``ln gamma`` varies."""
        if phase != "liquid":
            return super().ln_phi_function(t, p, phase=phase)
        reference = self._liquid_reference(t, p)
        return lambda w: self.activity.ln_gamma(w, t) + reference

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

    def flash_pt_with_info(
        self, t: ArrayLike, p: ArrayLike, z: Array, *, tol: float = 1e-12, max_iter: int = 300
    ) -> FlashSolveResult:
        """Isothermal-isobaric gamma-phi flash."""
        return flash_pt_gamma_with_info(
            self.activity,
            t,
            p,
            z,
            self.tc,
            self.pc,
            self.omega,
            tol=tol,
            max_iter=max_iter,
            **self._kw(),
        )

    def k_seed(self, t: ArrayLike, p: ArrayLike, x: Array) -> Array:
        """``gamma_i(x) Psat_i / P``: the modified Raoult K-values (ideal vapour)."""
        gamma = jnp.exp(self.activity.ln_gamma(x, t))
        return gamma * self._psat(t) / jnp.asarray(p, dtype=float)

    def bubble_pressure_with_info(self, t: ArrayLike, x: Array) -> SaturationSolveResult:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``, with a report."""
        return bubble_pressure_gamma_with_info(
            self.activity, t, x, self.tc, self.pc, self.omega, **self._kw()
        )

    def dew_pressure_with_info(self, t: ArrayLike, y: Array) -> SaturationSolveResult:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``, with a report."""
        return dew_pressure_gamma_with_info(
            self.activity, t, y, self.tc, self.pc, self.omega, **self._kw()
        )

    def bubble_temperature_with_info(
        self,
        p: ArrayLike,
        x: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationSolveResult:
        """Bubble temperature and incipient vapour (native gamma-phi solve)."""
        return bubble_temperature_gamma_with_info(
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

    def dew_temperature_with_info(
        self,
        p: ArrayLike,
        y: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationSolveResult:
        """Dew temperature and incipient liquid (native gamma-phi solve)."""
        return dew_temperature_gamma_with_info(
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
    meta_fields=["eos", "vapor", "poynting", "phi_saturation", "component_names", "evidence"],
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
    evidence: PackageEvidence = field(default_factory=PackageEvidence)

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

    def flash_pt_with_info(
        self, t: ArrayLike, p: ArrayLike, z: Array, *, tol: float = 1e-12, max_iter: int = 300
    ) -> FlashSolveResult:
        """Isothermal-isobaric vapour-liquid flash via PC-SAFT."""
        return flash_pt_saft_with_info(
            self.params, t, p, z, self.tc, self.pc, self.omega, tol=tol, max_iter=max_iter
        )

    def bubble_pressure_with_info(self, t: ArrayLike, x: Array) -> SaturationSolveResult:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``, with a report."""
        return bubble_pressure_saft_with_info(self.params, t, x, self.tc, self.pc, self.omega)

    def dew_pressure_with_info(self, t: ArrayLike, y: Array) -> SaturationSolveResult:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``, with a report."""
        return dew_pressure_saft_with_info(self.params, t, y, self.tc, self.pc, self.omega)


jax.tree_util.register_dataclass(
    SAFTPackage,
    data_fields=["params", "tc", "pc", "omega", "cp"],
    meta_fields=["component_names", "evidence"],
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
    pure substance (vapour fraction 0 or 1 away from the dome), and the energy
    flashes resolve directly to the steam-table state functions `state_ph` /
    `state_ps`, including the quality inside the dome.

    Enthalpy and entropy carry the reference state of the published formulation
    (not the ideal-gas ``T_REF`` reference of the mixture packages).

    Attributes:
        fluid: The reference fluid.
    """

    fluid: HelmholtzFluid

    component_names: tuple[str, ...] = ()
    evidence: PackageEvidence = field(default_factory=PackageEvidence)

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

    def _psat(self, t: Array) -> tuple[Array, SolveReport]:
        t_sat = jnp.clip(t, self.fluid.t_triple, T_SAT_MAX_FRACTION * self.fluid.t_critical)
        solved = saturation_pressure_with_info(self.fluid, t_sat)
        return solved.value, solved.report

    def in_domain(self, t: ArrayLike, p: ArrayLike, z: Array) -> Array:
        """Within the published range of the formulation."""
        t_arr = jnp.asarray(t, dtype=float)
        return (t_arr >= self.fluid.t_triple) & (t_arr <= self.fluid.t_max)

    def flash_pt_with_info(
        self, t: ArrayLike, p: ArrayLike, z: Array, *, tol: float = 1e-12, max_iter: int = 300
    ) -> FlashSolveResult:
        """Pure-fluid phase test: all vapour below ``Psat(T)`` (or above ``Tc``), else liquid.

        Exactly on the saturation line a PT specification can't fix the phase
        amounts; use a PH/PS specification (or a stream's retained inventory).
        """
        t_arr = jnp.asarray(t, dtype=float)
        p_arr = jnp.asarray(p, dtype=float)
        psat, psat_report = self._psat(t_arr)
        supercritical = t_arr >= self.fluid.t_critical
        vapor = supercritical | (p_arr < psat)
        beta = jnp.where(vapor, 1.0, 0.0)
        one = jnp.ones(1)
        report = residual_report(jnp.zeros(1))
        report = with_status(report, ~supercritical & ~psat_report.converged, psat_report.status)
        report = with_status(report, ~input_report(t, p, one), SolveStatus.INVALID_INPUT)
        report = with_status(report, ~self.in_domain(t, p, one), SolveStatus.OUT_OF_DOMAIN)
        k = jnp.reshape(psat / p_arr, (1,))
        return FlashSolveResult(FlashResult(beta=beta, x=one, y=one, k=k), report)

    def stability(
        self, t: ArrayLike, p: ArrayLike, z: Array, *, iterations: int = 160, tol: float = 1e-7
    ) -> StabilityResult:
        """A pure fluid is stable on its lower-Gibbs branch; the test compares the branches."""
        g_l = self.ln_phi(t, p, jnp.ones(1), phase="liquid")[0]
        g_v = self.ln_phi(t, p, jnp.ones(1), phase="vapor")[0]
        return StabilityResult(
            stable=jnp.asarray(True),
            tpd=jnp.abs(g_l - g_v),
            trial=jnp.ones(1),
            branch=jnp.where(g_l <= g_v, 0, 1),
            converged=jnp.isfinite(g_l) & jnp.isfinite(g_v),
        )

    def k_seed(self, t: ArrayLike, p: ArrayLike, x: Array) -> Array:
        """``Psat(T) / P`` as a length-1 vector."""
        psat, _ = self._psat(jnp.asarray(t, dtype=float))
        return jnp.reshape(psat / jnp.asarray(p, dtype=float), (1,))

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

    def flash_ph_with_info(
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
    ) -> EnergySolveResult:
        """Steam-table ``(P, h)`` state, with an independent verification report.

        The reference-fluid state functions initialize themselves, so the
        iteration options of the mixture packages are accepted but unused.
        """
        result = self._energy_result(state_ph(self.fluid, p, h))
        return EnergySolveResult(result, energy_flash_report(self, result, p, h, z))

    def flash_ps_with_info(
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
    ) -> EnergySolveResult:
        """Steam-table ``(P, s)`` state, with an independent verification report."""
        result = self._energy_result(state_ps(self.fluid, p, s))
        return EnergySolveResult(result, energy_flash_report(self, result, p, s, z, prop="entropy"))

    def bubble_pressure_with_info(self, t: ArrayLike, x: Array) -> SaturationSolveResult:
        """Saturation pressure at ``T`` (pure fluid), with a report."""
        t_arr = jnp.asarray(t, dtype=float)
        solved = saturation_pressure_with_info(self.fluid, t_arr)
        domain = (t_arr >= self.fluid.t_triple) & (
            t_arr < T_SAT_MAX_FRACTION * self.fluid.t_critical
        )
        report = with_status(solved.report, ~domain, SolveStatus.OUT_OF_DOMAIN)
        value = SaturationResult(solved.value, jnp.ones(1))
        return SaturationSolveResult(gate_tree(value, report.converged), report)

    def dew_pressure_with_info(self, t: ArrayLike, y: Array) -> SaturationSolveResult:
        """Saturation pressure at ``T`` (pure fluid), with a report."""
        return self.bubble_pressure_with_info(t, y)

    def bubble_temperature_with_info(
        self,
        p: ArrayLike,
        x: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationSolveResult:
        """Saturation temperature at ``P`` (pure fluid), with a report."""
        p_arr = jnp.asarray(p, dtype=float)
        solved = saturation_temperature_with_info(self.fluid, p_arr)
        domain = (p_arr >= self.fluid.p_triple) & (p_arr < self.fluid.p_critical)
        report = with_status(solved.report, ~domain, SolveStatus.OUT_OF_DOMAIN)
        value = SaturationResult(solved.value, jnp.ones(1))
        return SaturationSolveResult(gate_tree(value, report.converged), report)

    def dew_temperature_with_info(
        self,
        p: ArrayLike,
        y: Array,
        *,
        t_min: ArrayLike | None = None,
        t_max: ArrayLike | None = None,
    ) -> SaturationSolveResult:
        """Saturation temperature at ``P`` (pure fluid), with a report."""
        return self.bubble_temperature_with_info(p, y)


jax.tree_util.register_dataclass(
    HelmholtzPackage, data_fields=["fluid"], meta_fields=["component_names", "evidence"]
)


# --------------------------------------------------------------------------- #
# Constructors and verification
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
    """Construct a `GammaPhiPackage` from an activity model and component constants.

    Raises:
        ValueError: If ``vapor`` isn't ``"ideal"`` or ``"eos"``.
    """
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
    fractions, equifugacity of present phases, and the specified molar
    property. The iteration count is zero because this is a verification of
    the returned state, not its iteration log. A state outside the package's
    domain is ``OUT_OF_DOMAIN``.
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
    from fugacio.thermo.acceptance import equilibrium_residual

    equilibrium = equilibrium_residual(
        pkg,
        result.t,
        p,
        FlashResult(beta, result.x, result.y, result.y / jnp.maximum(result.x, 1e-300)),
        z,
    )
    errors = jnp.concatenate(
        [
            (1 - beta) * result.x + beta * result.y - z,
            equilibrium,
            jnp.array(
                [
                    (value - target) / scale,
                    jnp.sum(z) - 1.0,
                    jnp.maximum(-beta, 0.0) + jnp.maximum(beta - 1.0, 0.0),
                ]
            ),
        ]
    )
    report = residual_report(errors, tol)
    return with_status(report, ~pkg.in_domain(result.t, p, z), SolveStatus.OUT_OF_DOMAIN)


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
    "gamma_phi_package",
    "helmholtz_package",
    "saft_package",
]
