"""`SAFTModel`: PC-SAFT behind the unified `EquilibriumModel` interface.

`SAFTModel` exposes the same five calls as `fugacio.thermo.EOSModel` and
`fugacio.thermo.GammaPhiModel` (``flash_pt``, bubble/dew pressure and
temperature) plus a tangent-plane ``stability`` test, so the rest of the Fugacio
stack switches to a molecular-based equation of state by swapping one object,
exactly as it switches between Peng-Robinson and NRTL. It is a registered JAX
pytree whose PC-SAFT parameters are differentiable leaves, so a flowsheet built on
a `SAFTModel` stays end-to-end differentiable with respect to the thermodynamic
model's own parameters.

The model carries the critical constants ``(tc, pc, omega)`` only to seed the
Wilson K-values that initialise the flashes; the equilibrium itself is entirely
PC-SAFT.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.equilibrium import FlashResult, StabilityResult
from fugacio.thermo.implicit import bracketed_root
from fugacio.thermo.saft.equilibrium import (
    bubble_pressure_saft,
    dew_pressure_saft,
    flash_pt_saft,
    stability_saft,
)
from fugacio.thermo.saft.parameters import SaftParameters

ArrayLike = Array | float


@dataclass(frozen=True)
class SAFTModel:
    """PC-SAFT (phi-phi) equilibrium model.

    Attributes:
        params: PC-SAFT parameter set (a differentiable pytree).
        tc, pc, omega: Component critical constants and acentric factors, used to
            seed the Wilson K-values that initialise the flashes.
    """

    params: SaftParameters
    tc: Array
    pc: Array
    omega: Array

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult:
        """Isothermal-isobaric two-phase flash via PC-SAFT."""
        return flash_pt_saft(self.params, t, p, z, self.tc, self.pc, self.omega)

    def bubble_pressure(self, t: ArrayLike, x: Array) -> tuple[Array, Array]:
        """Bubble pressure and incipient vapour at fixed ``T``, ``x``."""
        return bubble_pressure_saft(self.params, t, x, self.tc, self.pc, self.omega)

    def dew_pressure(self, t: ArrayLike, y: Array) -> tuple[Array, Array]:
        """Dew pressure and incipient liquid at fixed ``T``, ``y``."""
        return dew_pressure_saft(self.params, t, y, self.tc, self.pc, self.omega)

    def bubble_temperature(
        self, p: ArrayLike, x: Array, *, t_min: float = 150.0, t_max: float = 700.0
    ) -> tuple[Array, Array]:
        """Bubble temperature and incipient vapour at fixed ``P``, ``x``.

        Found by inverting `bubble_pressure` for the temperature whose bubble
        pressure equals ``P`` (saturation pressure rises monotonically with
        temperature), then returning the incipient vapour there.
        """

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
        return stability_saft(self.params, t, p, z, self.tc, self.pc, self.omega)


jax.tree_util.register_dataclass(
    SAFTModel, data_fields=["params", "tc", "pc", "omega"], meta_fields=[]
)


def saft_model(params: SaftParameters, tc: Array, pc: Array, omega: Array) -> SAFTModel:
    """Construct a `SAFTModel` from a PC-SAFT parameter set and seeding constants."""
    return SAFTModel(
        params=params,
        tc=jnp.asarray(tc, dtype=float),
        pc=jnp.asarray(pc, dtype=float),
        omega=jnp.asarray(omega, dtype=float),
    )


__all__ = ["SAFTModel", "saft_model"]
