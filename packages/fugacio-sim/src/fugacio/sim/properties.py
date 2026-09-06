"""Stream property bridge: enthalpy, entropy, flows, and transport for a `Stream`.

Unit operations close *material and energy* balances, so they need a stream's
enthalpy and entropy, not just its composition. This module resolves a stream's
(static) component names to the array constants the `fugacio.thermo` kernels
expect (caching that lookup, since names never change during a solve) and
exposes the resulting molar and total-flow properties.

Enthalpy and entropy use the stream's resolved phase inventory when present.
Otherwise, they run the equilibrium flash at its ``(T, P)`` and blend the phase
properties. The same calls handle subcooled liquid, superheated vapor, and
partially vaporized streams. Properties are differentiable with respect to the stream's flows,
temperature, and pressure (the component constants are not differentiated, which
is exactly right: they are reference data, not decision variables).

Which *thermodynamic method* evaluates those properties is set by the ``model``
argument, a `fugacio.thermo.PropertyPackage`. Left unset, the stream's
components resolve to a Peng-Robinson `fugacio.thermo.CubicPackage` (the
historical default, still selectable through the ``eos`` / ``kij`` arguments).
Pass an NRTL/UNIFAC gamma-phi package, a PC-SAFT package, or a reference-fluid
package instead and every unit operation downstream uses it for its energy
balance as well as its phase split. `resolve_package` performs that resolution
and also upgrades a bare equilibrium model (`EOSModel`, `GammaPhiModel`,
`SAFTModel`) to the matching package by attaching the components' ideal-gas
heat capacities.

Sizing-grade physical properties are surfaced too: phase densities and
volumetric flows (`liquid_density`, `vapor_volumetric_flow`),
viscosities, thermal conductivities, and surface tension, all evaluated at the
stream's state through the curated correlations and mixture rules in
`fugacio.thermo`. A stream-aware Souders-Brown helper
(`column_diameter_for`) wires them straight into the equipment-sizing
correlations of `fugacio.sim.economics`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import cache, partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.economics import column_diameter
from fugacio.sim.stream import Stream
from fugacio.thermo import (
    PR,
    CubicEOS,
    CubicPackage,
    EOSModel,
    GammaPhiModel,
    PropertyPackage,
    SAFTModel,
    component_arrays,
    cubic_package,
    gamma_phi_package,
    get,
    ideal_gas_coeffs,
    saft_package,
)
from fugacio.thermo import (
    gas_mixture_thermal_conductivity as _gas_k,
)
from fugacio.thermo import (
    gas_mixture_viscosity as _gas_mu,
)
from fugacio.thermo import (
    liquid_density as _liquid_density,
)
from fugacio.thermo import (
    liquid_mixture_thermal_conductivity as _liquid_k,
)
from fugacio.thermo import (
    liquid_mixture_viscosity as _liquid_mu,
)
from fugacio.thermo import (
    mixture_surface_tension as _surface_tension,
)
from fugacio.thermo import (
    vapor_density as _vapor_density,
)

ArrayLike = Array | float
CpCoeffs = tuple[Array, Array, Array, Array, Array]

#: Anything a unit accepts as its thermodynamic method: a property package, a
#: bare equilibrium model (upgraded on the fly), or ``None`` for the cubic default.
Model = PropertyPackage | EOSModel | GammaPhiModel | SAFTModel | None


@cache
def _resolve(components: tuple[str, ...]) -> tuple[Array, Array, Array, Array, CpCoeffs]:
    """Resolve component names to ``(tc, pc, omega, mw, cp)`` array constants (cached)."""
    # The first lookup may happen while a flowsheet is being traced. Keep
    # cached constants concrete so no tracer escapes into later evaluations.
    with jax.ensure_compile_time_eval():
        arr = component_arrays(list(components))
        cp = ideal_gas_coeffs([get(c) for c in components])
        return arr["tc"], arr["pc"], arr["omega"], arr["mw"], cp


def default_package(
    components: Sequence[str], *, eos: CubicEOS = PR, kij: Array | None = None
) -> CubicPackage:
    """The cubic-EOS package a stream falls back to when no ``model`` is given."""
    from fugacio.thermo.provenance import database_evidence

    tc, pc, omega, _, cp = _resolve(tuple(components))
    return replace(
        cubic_package(tc, pc, omega, cp, kij=kij, eos=eos),
        component_names=tuple(get(c).name for c in components),
        evidence=database_evidence(
            tuple(components),
            {
                "Peng-Robinson": "pr",
                "Soave-Redlich-Kwong": "srk",
                "Redlich-Kwong": "rk",
                "van der Waals": "vdw",
            }.get(eos.name, "custom"),
            explicit_kij=kij is not None,
        ),
    )


def as_package(model: Any, components: Sequence[str]) -> PropertyPackage:
    """Upgrade an equilibrium model to a property package for ``components``.

    A `PropertyPackage` is returned unchanged. An `EOSModel`, `GammaPhiModel`, or
    `SAFTModel` (which know equilibrium but not energy) is completed with the
    components' ideal-gas heat capacities into the matching package, so the
    existing model factories in `fugacio.sim.models` can feed any unit.

    Raises:
        TypeError: if ``model`` is none of the supported kinds.
    """
    if isinstance(model, PropertyPackage):
        return model
    _, _, _, _, cp = _resolve(tuple(components))
    if isinstance(model, EOSModel):
        return replace(
            cubic_package(model.tc, model.pc, model.omega, cp, kij=model.kij, eos=model.eos),
            component_names=tuple(get(c).name for c in components),
        )
    if isinstance(model, GammaPhiModel):
        return replace(
            gamma_phi_package(
                model.activity,
                model.tc,
                model.pc,
                model.omega,
                cp,
                kij=model.kij,
                eos=model.eos,
                vapor=model.vapor,
                poynting=model.poynting,
                phi_saturation=model.phi_saturation,
            ),
            component_names=tuple(get(c).name for c in components),
        )
    if isinstance(model, SAFTModel):
        return replace(
            saft_package(model.params, model.tc, model.pc, model.omega, cp),
            component_names=tuple(get(c).name for c in components),
        )
    raise TypeError(
        f"unsupported thermodynamic model {type(model).__name__}; pass a PropertyPackage "
        "(CubicPackage, GammaPhiPackage, SAFTPackage, HelmholtzPackage) or an "
        "EOSModel / GammaPhiModel / SAFTModel"
    )


def resolve_package(
    components: Sequence[str],
    model: Model = None,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
) -> PropertyPackage:
    """Pick the property package a unit should use for ``components``.

    ``model`` wins when given (upgraded through `as_package` if it is a bare
    equilibrium model); otherwise the cubic default built from ``eos`` / ``kij``.

    Raises:
        ValueError: if the package's component count does not match the stream's.
    """
    pkg = (
        default_package(components, eos=eos, kij=kij)
        if model is None
        else as_package(model, components)
    )
    if pkg.n_components != len(components):
        raise ValueError(
            f"property package describes {pkg.n_components} components but the stream has "
            f"{len(components)} ({', '.join(components)})"
        )
    names = getattr(pkg, "component_names", ())
    if names and tuple(get(c).name for c in components) != names:
        raise ValueError(
            f"property package component order {names} does not match stream {tuple(components)}; "
            "reorder the stream or rebuild the package"
        )
    return pkg


def _composition(n: Array) -> Array:
    """Finite trial composition even for an absent or empty phase."""
    total = jnp.sum(n)
    return jnp.where(total > 0.0, n / jnp.where(total > 0, total, 1.0), jnp.ones_like(n) / n.size)


@partial(jax.jit, static_argnames=("prop",))
def _stream_property(stream: Stream, pkg: PropertyPackage, prop: str) -> Array:
    """Evaluate a preserved phase split or resolve an unspecified PT state."""
    z = _composition(stream.n)

    def resolved(_: None) -> Array:
        nv = jnp.asarray(stream.vapor_n)
        nl = stream.n - nv
        nt = jnp.maximum(stream.total, 1e-300)
        beta = jnp.sum(nv) / nt
        fn = getattr(pkg, prop)

        def liquid(_: None) -> Array:
            return fn(stream.t, stream.p, _composition(nl), phase="liquid")

        def vapor(_: None) -> Array:
            return fn(stream.t, stream.p, _composition(nv), phase="vapor")

        def both(_: None) -> Array:
            return (1.0 - beta) * liquid(None) + beta * vapor(None)

        index = jnp.where(beta <= 0.0, 0, jnp.where(beta >= 1.0, 2, 1)).astype(jnp.int32)
        return jax.lax.switch(index, [liquid, both, vapor], None)

    def unspecified(_: None) -> Array:
        return getattr(pkg, "mixture_" + prop)(stream.t, stream.p, z)

    return jax.lax.cond(stream.phase_known, resolved, unspecified, None)


def molar_enthalpy(
    stream: Stream, *, model: Model = None, eos: CubicEOS = PR, kij: Array | None = None
) -> Array:
    """Molar enthalpy of the stream (J/mol), relative to the package's reference."""
    pkg = resolve_package(stream.components, model, eos=eos, kij=kij)
    return _stream_property(stream, pkg, "enthalpy")


def molar_entropy(
    stream: Stream, *, model: Model = None, eos: CubicEOS = PR, kij: Array | None = None
) -> Array:
    """Molar entropy of the stream (J/mol/K), relative to the package's reference."""
    pkg = resolve_package(stream.components, model, eos=eos, kij=kij)
    return _stream_property(stream, pkg, "entropy")


def molar_volume(
    stream: Stream, *, model: Model = None, eos: CubicEOS = PR, kij: Array | None = None
) -> Array:
    """Two-phase-aware molar volume of the stream (m^3/mol)."""
    pkg = resolve_package(stream.components, model, eos=eos, kij=kij)
    return _stream_property(stream, pkg, "volume")


def vapor_fraction(
    stream: Stream, *, model: Model = None, eos: CubicEOS = PR, kij: Array | None = None
) -> Array:
    """Equilibrium molar vapour fraction of the stream at its ``(T, P)``."""
    pkg = resolve_package(stream.components, model, eos=eos, kij=kij)
    return jax.lax.cond(
        stream.phase_known,
        lambda _: jnp.sum(jnp.asarray(stream.vapor_n)) / jnp.maximum(stream.total, 1e-300),
        lambda _: pkg.flash_pt(stream.t, stream.p, _composition(stream.n)).beta,
        None,
    )


def enthalpy_flow(
    stream: Stream, *, model: Model = None, eos: CubicEOS = PR, kij: Array | None = None
) -> Array:
    """Total enthalpy flow of the stream (W = J/s)."""
    return stream.total * molar_enthalpy(stream, model=model, eos=eos, kij=kij)


def entropy_flow(
    stream: Stream, *, model: Model = None, eos: CubicEOS = PR, kij: Array | None = None
) -> Array:
    """Total entropy flow of the stream (W/K)."""
    return stream.total * molar_entropy(stream, model=model, eos=eos, kij=kij)


def volumetric_flow(
    stream: Stream, *, model: Model = None, eos: CubicEOS = PR, kij: Array | None = None
) -> Array:
    """Actual volumetric flow of the stream at its state (m^3/s)."""
    return stream.total * molar_volume(stream, model=model, eos=eos, kij=kij)


def molar_mass(stream: Stream) -> Array:
    """Mole-fraction-averaged molar mass of the stream (g/mol)."""
    _, _, _, mw, _ = _resolve(stream.components)
    return jnp.sum(_composition(stream.n) * mw)


def mass_flow(stream: Stream) -> Array:
    """Total mass flow of the stream (kg/s)."""
    _, _, _, mw, _ = _resolve(stream.components)
    return jnp.sum(stream.n * mw) * 1.0e-3


# --------------------------------------------------------------------------- #
# Volumetric and transport properties at the stream state
# --------------------------------------------------------------------------- #


def _names(stream: Stream) -> list[str]:
    return list(stream.components)


def liquid_density(stream: Stream) -> Array:
    """Saturated-liquid mass density at the stream's ``T`` and composition (kg/m^3)."""
    return _liquid_density(_names(stream), stream.t, _composition(stream.n))


def vapor_density(stream: Stream, *, eos: CubicEOS = PR) -> Array:
    """Vapour mass density from the EOS at the stream's ``(T, P)`` (kg/m^3)."""
    return _vapor_density(_names(stream), stream.t, stream.p, _composition(stream.n), eos=eos)


def liquid_volumetric_flow(stream: Stream) -> Array:
    """Volumetric flow if the stream is all liquid (m^3/s)."""
    return mass_flow(stream) / liquid_density(stream)


def vapor_volumetric_flow(stream: Stream, *, eos: CubicEOS = PR) -> Array:
    """Volumetric flow if the stream is all vapour (m^3/s)."""
    return mass_flow(stream) / vapor_density(stream, eos=eos)


def liquid_viscosity(stream: Stream) -> Array:
    """Liquid-mixture viscosity at the stream's ``T`` (Pa*s), Grunberg-Nissan."""
    return _liquid_mu(_names(stream), stream.t, _composition(stream.n))


def vapor_viscosity(stream: Stream) -> Array:
    """Dilute-gas mixture viscosity at the stream's ``T`` (Pa*s), Wilke."""
    return _gas_mu(_names(stream), stream.t, _composition(stream.n))


def liquid_thermal_conductivity(stream: Stream) -> Array:
    """Liquid-mixture thermal conductivity at the stream's ``T`` (W/m/K), DIPPR9H."""
    return _liquid_k(_names(stream), stream.t, _composition(stream.n))


def vapor_thermal_conductivity(stream: Stream) -> Array:
    """Gas-mixture thermal conductivity at the stream's ``T`` (W/m/K), Wassiljewa."""
    return _gas_k(_names(stream), stream.t, _composition(stream.n))


def surface_tension(stream: Stream) -> Array:
    """Liquid-mixture surface tension at the stream's ``T`` (N/m)."""
    return _surface_tension(_names(stream), stream.t, _composition(stream.n))


def column_diameter_for(
    vapor: Stream,
    liquid: Stream | None = None,
    *,
    k_drum: ArrayLike = 0.07,
    flooding: ArrayLike = 0.8,
) -> Array:
    """Souders-Brown column/drum diameter sized from the actual stream states (m).

    The vapour density, molar mass, and flow come from ``vapor``; the liquid
    density from ``liquid`` (defaulting to the vapour stream's composition at
    its own temperature, the saturated-liquid view of the same material, a
    sensible drum approximation).
    """
    rho_v = vapor_density(vapor)
    rho_l = liquid_density(liquid if liquid is not None else vapor)
    return column_diameter(
        vapor.total,
        rho_v,
        rho_l,
        molar_mass=molar_mass(vapor) * 1.0e-3,
        k_drum=k_drum,
        flooding=flooding,
    )
