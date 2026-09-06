"""Material streams: the data passed between flowsheet unit operations.

A `Stream` carries per-component molar flows together with temperature and
pressure and an optional resolved vapor inventory. It is registered as a JAX
pytree (both inventories, ``T``, and ``P`` are differentiable leaves while the
component *names* are static metadata) so an
entire flowsheet built from streams remains end-to-end differentiable. You can
take a gradient of any downstream quantity with respect to a feed flow,
temperature, or pressure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.diagnostics import SolveReport, SolveStatus, require_converged, residual_report

if TYPE_CHECKING:
    from fugacio.sim.properties import Model

ArrayLike = Array | float


@dataclass(frozen=True)
class Stream:
    """A process stream of fixed composition basis.

    Attributes:
        n: Per-component molar flow rates (mol/s), 1-D array aligned with ``components``.
        t: Temperature (K).
        p: Pressure (Pa).
        components: Canonical component names (static metadata).
        vapor_n: Per-component vapor flows for an explicitly resolved state.
            A negative sentinel means that a PT flash determines the state.
            This extra inventory preserves saturation quality between units.
    """

    n: Array
    t: Array
    p: Array
    components: tuple[str, ...]
    vapor_n: Array | None = None

    def __post_init__(self) -> None:
        """Give unresolved and resolved streams the same JAX pytree structure."""
        if self.vapor_n is None:
            object.__setattr__(self, "vapor_n", -jnp.ones_like(self.n))

    @property
    def report(self) -> SolveReport:
        """Check finite physical inventories, state, and phase bookkeeping."""
        vapor = jnp.asarray(self.vapor_n)
        finite = (
            jnp.all(jnp.isfinite(self.n))
            & jnp.all(jnp.isfinite(vapor))
            & jnp.all(jnp.isfinite(self.t))
            & jnp.all(jnp.isfinite(self.p))
        )
        phase_valid = jnp.all(vapor == -1.0) | (
            jnp.all(vapor >= -1e-10) & jnp.all(vapor <= self.n + 1e-10)
        )
        valid = jnp.array(
            [jnp.all(self.n >= -1e-10), jnp.all(self.t > 0), jnp.all(self.p > 0), phase_valid]
        )
        return residual_report(
            jnp.where(finite, jnp.where(valid, 0.0, 1.0), jnp.nan),
            failure=SolveStatus.INVALID_INPUT,
        )

    def check(self) -> None:
        """Raise for an invalid concrete stream; compiled callers inspect ``report``."""
        require_converged(
            self.report, "stream", ("component flows", "temperature", "pressure", "phase inventory")
        )

    @property
    def phase_known(self) -> Array:
        """Whether explicit vapor component flows determine the phase split."""
        return jnp.all(jnp.asarray(self.vapor_n) >= 0.0)

    def scaled(self, fraction: ArrayLike) -> Stream:
        """Scale material and phase inventories by the same flow fraction."""
        vapor = jnp.where(self.phase_known, jnp.asarray(self.vapor_n) * fraction, -1.0)
        return Stream(self.n * fraction, self.t, self.p, self.components, vapor)

    def reordered(self, components: tuple[str, ...]) -> Stream:
        """Reorder a stream's material and phase inventories without changing it.

        Raises:
            ValueError: If the requested basis has missing or repeated components.
        """
        if len(set(components)) != len(components) or set(components) != set(self.components):
            raise ValueError("component reordering requires the same unique component names")
        order = jnp.asarray([self.components.index(c) for c in components])
        return Stream(self.n[order], self.t, self.p, components, jnp.asarray(self.vapor_n)[order])

    @property
    def total(self) -> Array:
        """Total molar flow rate (mol/s)."""
        return jnp.sum(self.n)

    @property
    def z(self) -> Array:
        """Mole fractions (the flow normalised to sum to one; all zero for an empty stream)."""
        total = jnp.sum(self.n)
        return self.n / jnp.where(total > 0.0, total, 1.0)

    @classmethod
    def from_fractions(
        cls,
        components: tuple[str, ...],
        z: Array,
        flow: ArrayLike,
        t: ArrayLike,
        p: ArrayLike,
        *,
        phase: str | None = None,
    ) -> Stream:
        """Build a PT stream, optionally selecting a known liquid or vapor branch.

        Use ``phase`` to distinguish saturated liquid and saturated vapor, which
        have the same temperature and pressure. For a partially vaporized state,
        use :meth:`from_ph` or :meth:`from_ps`.

        Raises:
            ValueError: If ``phase`` is not liquid, vapor, or None.
        """
        if phase not in (None, "liquid", "vapor"):
            raise ValueError("phase must be 'liquid', 'vapor', or None")
        z = jnp.asarray(z)
        if (
            z.shape != (len(components),)
            or not components
            or len(set(components)) != len(components)
        ):
            raise ValueError("fractions must match a nonempty unique component basis")
        fractions_report = residual_report(
            jnp.atleast_1d(jnp.sum(z) - 1.0), failure=SolveStatus.INVALID_INPUT
        )
        require_converged(fractions_report, "stream mole fractions")
        n = z * jnp.asarray(flow)
        return cls(
            n=n,
            t=jnp.asarray(t),
            p=jnp.asarray(p),
            components=tuple(components),
            vapor_n=None if phase is None else (n if phase == "vapor" else jnp.zeros_like(n)),
        )

    @classmethod
    def from_ph(
        cls,
        components: tuple[str, ...],
        z: Array,
        flow: ArrayLike,
        p: ArrayLike,
        h: ArrayLike,
        *,
        model: Model = None,
    ) -> Stream:
        """Build a stream from pressure and molar enthalpy (Pa, J/mol).

        ``model`` sets the property package and its enthalpy reference.
        The phase split is retained even on a pure fluid's saturation line.
        """
        from fugacio.sim.properties import resolve_package
        from fugacio.thermo.package import flash_ph_with_info

        pkg = resolve_package(components, model)
        solved = flash_ph_with_info(pkg, p, h, jnp.asarray(z))
        require_converged(solved.report, "PH stream")
        result = jax.tree_util.tree_map(
            lambda x: x * jnp.where(solved.report.converged, 1.0, jnp.nan), solved.value
        )
        return cls(
            jnp.asarray(z) * flow,
            result.t,
            jnp.asarray(p),
            tuple(components),
            result.beta * flow * result.y,
        )

    @classmethod
    def from_ps(
        cls,
        components: tuple[str, ...],
        z: Array,
        flow: ArrayLike,
        p: ArrayLike,
        s: ArrayLike,
        *,
        model: Model = None,
    ) -> Stream:
        """Build a stream from pressure and molar entropy (Pa, J/mol/K)."""
        from fugacio.sim.properties import resolve_package
        from fugacio.thermo.package import flash_ps_with_info

        pkg = resolve_package(components, model)
        solved = flash_ps_with_info(pkg, p, s, jnp.asarray(z))
        require_converged(solved.report, "PS stream")
        result = jax.tree_util.tree_map(
            lambda x: x * jnp.where(solved.report.converged, 1.0, jnp.nan), solved.value
        )
        return cls(
            jnp.asarray(z) * flow,
            result.t,
            jnp.asarray(p),
            tuple(components),
            result.beta * flow * result.y,
        )


jax.tree_util.register_dataclass(
    Stream,
    data_fields=["n", "t", "p", "vapor_n"],
    meta_fields=["components"],
)
