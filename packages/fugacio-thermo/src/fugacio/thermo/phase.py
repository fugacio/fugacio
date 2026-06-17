"""A uniform phase-equilibrium model: one interface over EOS and gamma-phi.

The two routes to vapour-liquid equilibrium -- a cubic equation of state for both
phases (`fugacio.thermo.equilibrium`) and an activity model for the liquid
with an EOS/ideal vapour (`fugacio.thermo.gammaphi`) -- have, until now,
different call signatures. That made the rest of the stack (flashes inside unit
operations, column K-values, the copilot tools) hard-wire the EOS. This module
unifies them behind a single `EquilibriumModel` interface so a flowsheet
can be switched from Peng-Robinson to NRTL by swapping one object.

Both concrete models -- `EOSModel` and `GammaPhiModel` -- bundle the
component constants they need and expose the same five calls: ``flash_pt``,
``bubble_pressure``/``bubble_temperature`` and ``dew_pressure``/``dew_temperature``.
They are registered JAX pytrees whose parameters (critical constants, ``kij``, and,
for the gamma-phi model, the *activity-model parameters*) are differentiable
leaves, while structural choices (which cubic, ideal-vs-EOS vapour) are static
metadata. A flowsheet built on a model therefore stays end-to-end differentiable,
including with respect to the thermodynamic model's own parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.activity.models import ActivityModel
from fugacio.thermo.eos import PR, CubicEOS
from fugacio.thermo.equilibrium import (
    FlashResult,
    StabilityResult,
    bubble_pressure_eos,
    dew_pressure_eos,
    flash_pt,
    stability_analysis,
)
from fugacio.thermo.gammaphi import (
    bubble_pressure_gamma,
    bubble_temperature_gamma,
    dew_pressure_gamma,
    dew_temperature_gamma,
    flash_pt_gamma,
)

ArrayLike = Array | float


@runtime_checkable
class EquilibriumModel(Protocol):
    """Structural type for a vapour-liquid equilibrium model.

    Every model maps operating conditions to a phase split (``flash_pt``) and to
    the four saturation calculations. Implementations bundle their own component
    constants, so callers pass only state (``T``, ``P``, composition).
    """

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


@dataclass(frozen=True)
class EOSModel:
    """Equation-of-state (phi-phi) equilibrium model.

    Attributes:
        tc, pc, omega: Component critical constants and acentric factors.
        kij: Binary interaction matrix (``None`` => zeros).
        eos: Cubic equation of state (static; default Peng-Robinson).
    """

    tc: Array
    pc: Array
    omega: Array
    kij: Array | None
    eos: CubicEOS

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Isothermal-isobaric two-phase flash via the cubic EOS."""
        return flash_pt(self.eos, t, p, z, self.tc, self.pc, self.omega, kij=self.kij)

    def bubble_pressure(self, t: ArrayLike, x: Array) -> tuple[Array, Array]:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``."""
        return bubble_pressure_eos(self.eos, t, x, self.tc, self.pc, self.omega, kij=self.kij)

    def dew_pressure(self, t: ArrayLike, y: Array) -> tuple[Array, Array]:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``."""
        return dew_pressure_eos(self.eos, t, y, self.tc, self.pc, self.omega, kij=self.kij)

    def bubble_temperature(
        self, p: ArrayLike, x: Array, *, t_min: float = 150.0, t_max: float = 700.0
    ) -> tuple[Array, Array]:
        """Bubble temperature and incipient vapour at fixed ``P``, ``x``.

        Found by inverting `bubble_pressure` for the temperature whose bubble
        pressure equals ``P`` (the saturation pressure rises monotonically with
        temperature), then returning the incipient vapour there.
        """
        from fugacio.thermo.implicit import bracketed_root

        def residual(t: Array, params: tuple[Array, Array]) -> Array:
            p_, x_ = params
            return self.bubble_pressure(t, x_)[0] - p_

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
        from fugacio.thermo.implicit import bracketed_root

        def residual(t: Array, params: tuple[Array, Array]) -> Array:
            p_, y_ = params
            return self.dew_pressure(t, y_)[0] - p_

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

    def stability(self, t: ArrayLike, p: ArrayLike, z: Array) -> StabilityResult:
        """Michelsen tangent-plane stability of feed ``z`` at ``(T, P)``."""
        return stability_analysis(self.eos, t, p, z, self.tc, self.pc, self.omega, kij=self.kij)


@dataclass(frozen=True)
class GammaPhiModel:
    """Gamma-phi equilibrium model: activity-coefficient liquid, EOS/ideal vapour.

    Attributes:
        activity: Liquid activity-coefficient model (a differentiable pytree).
        tc, pc, omega: Component critical constants and acentric factors.
        kij: Binary interaction matrix for an EOS vapour (``None`` => zeros).
        eos: Cubic EOS used for the saturation reference and (if selected) vapour.
        vapor: ``"ideal"`` (phi^V = 1) or ``"eos"`` (static).
        poynting: Include the Poynting correction in the reference (static).
        phi_saturation: Include the saturation fugacity coefficient (static).
    """

    activity: ActivityModel
    tc: Array
    pc: Array
    omega: Array
    kij: Array | None
    eos: CubicEOS
    vapor: str
    poynting: bool
    phi_saturation: bool

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Isothermal-isobaric gamma-phi flash."""
        return flash_pt_gamma(
            self.activity,
            t,
            p,
            z,
            self.tc,
            self.pc,
            self.omega,
            eos=self.eos,
            kij=self.kij,
            vapor=self.vapor,
            poynting=self.poynting,
            phi_saturation=self.phi_saturation,
        )

    def bubble_pressure(self, t: ArrayLike, x: Array) -> tuple[Array, Array]:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``."""
        return bubble_pressure_gamma(
            self.activity,
            t,
            x,
            self.tc,
            self.pc,
            self.omega,
            eos=self.eos,
            kij=self.kij,
            vapor=self.vapor,
            poynting=self.poynting,
            phi_saturation=self.phi_saturation,
        )

    def dew_pressure(self, t: ArrayLike, y: Array) -> tuple[Array, Array]:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``."""
        return dew_pressure_gamma(
            self.activity,
            t,
            y,
            self.tc,
            self.pc,
            self.omega,
            eos=self.eos,
            kij=self.kij,
            vapor=self.vapor,
            poynting=self.poynting,
            phi_saturation=self.phi_saturation,
        )

    def bubble_temperature(
        self, p: ArrayLike, x: Array, *, t_min: float = 150.0, t_max: float = 700.0
    ) -> tuple[Array, Array]:
        """Bubble temperature and incipient vapour at fixed ``P``, ``x``."""
        return bubble_temperature_gamma(
            self.activity,
            p,
            x,
            self.tc,
            self.pc,
            self.omega,
            eos=self.eos,
            kij=self.kij,
            vapor=self.vapor,
            poynting=self.poynting,
            phi_saturation=self.phi_saturation,
            t_min=t_min,
            t_max=t_max,
        )

    def dew_temperature(
        self, p: ArrayLike, y: Array, *, t_min: float = 150.0, t_max: float = 700.0
    ) -> tuple[Array, Array]:
        """Dew temperature and incipient liquid at fixed ``P``, ``y``."""
        return dew_temperature_gamma(
            self.activity,
            p,
            y,
            self.tc,
            self.pc,
            self.omega,
            eos=self.eos,
            kij=self.kij,
            vapor=self.vapor,
            poynting=self.poynting,
            phi_saturation=self.phi_saturation,
            t_min=t_min,
            t_max=t_max,
        )


jax.tree_util.register_dataclass(
    EOSModel, data_fields=["tc", "pc", "omega", "kij"], meta_fields=["eos"]
)
jax.tree_util.register_dataclass(
    GammaPhiModel,
    data_fields=["activity", "tc", "pc", "omega", "kij"],
    meta_fields=["eos", "vapor", "poynting", "phi_saturation"],
)


def eos_model(
    tc: Array, pc: Array, omega: Array, *, kij: Array | None = None, eos: CubicEOS = PR
) -> EOSModel:
    """Construct an `EOSModel` from component constants."""
    return EOSModel(
        tc=jnp.asarray(tc), pc=jnp.asarray(pc), omega=jnp.asarray(omega), kij=kij, eos=eos
    )


def gamma_phi_model(
    activity: ActivityModel,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    kij: Array | None = None,
    eos: CubicEOS = PR,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
) -> GammaPhiModel:
    """Construct a `GammaPhiModel` from an activity model and constants."""
    return GammaPhiModel(
        activity=activity,
        tc=jnp.asarray(tc),
        pc=jnp.asarray(pc),
        omega=jnp.asarray(omega),
        kij=kij,
        eos=eos,
        vapor=vapor,
        poynting=poynting,
        phi_saturation=phi_saturation,
    )
