"""Liquid molar volume and density: Rackett, COSTALD, DIPPR fits, Peneloux shifts.

Cubic equations of state are excellent for phase equilibrium but notoriously poor
for saturated-liquid density (Peng-Robinson is typically 5-15% off). This module
supplies the standard remedies:

* **Rackett** (`rackett_volume`): the two-line corresponding-states
  classic, sharpened by the Spencer-Danner ``Z_RA`` parameter when the curated
  tables carry one (`zra_estimate` otherwise);
* **COSTALD** (`costald_volume`, `costald_mixture_volume`): the
  Hankinson-Thomson correlation with its characteristic volumes and SRK acentric
  factors, including the standard mixing rules;
* **DIPPR-105 fits**: per-component saturated-density correlations transcribed
  from open data (the most accurate route where available);
* **Peneloux volume translation** (`peneloux_shift`,
  `translated_molar_volume`): the constant shift ``v = v_eos - sum x_i c_i``
  that repairs cubic-EOS liquid volumes *without changing phase equilibrium*
  (fugacity ratios are invariant to a composition-linear translation).

The name-based dispatchers (`liquid_molar_volumes`,
`mixture_liquid_volume`, `liquid_density`) choose the best available
route per component (DIPPR fit, then COSTALD, then Rackett) so callers get a
sensible answer for every database component, differentiable in ``T`` and
composition throughout.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo._property_data import COSTALD_VOLUME, RHO_LIQUID_DIPPR105
from fugacio.thermo.components import Component, component_arrays, get
from fugacio.thermo.constants import R
from fugacio.thermo.correlations import dippr105
from fugacio.thermo.eos import PR, CubicEOS, molar_volume

ArrayLike = Array | float


# --- Rackett ---------------------------------------------------------------------


def zra_estimate(omega: ArrayLike) -> Array:
    """Estimate the Rackett compressibility ``Z_RA = 0.29056 - 0.08775*omega``.

    The Yamada-Gunn estimate, used when no fitted Spencer-Danner value is
    tabulated for a component.
    """
    return jnp.asarray(0.29056 - 0.08775 * jnp.asarray(omega))


def rackett_volume(t: ArrayLike, tc: ArrayLike, pc: ArrayLike, zra: ArrayLike) -> Array:
    """Rackett saturated-liquid molar volume (m^3/mol).

    ``V = (R*Tc/Pc) * Z_RA^(1 + (1-Tr)^(2/7))`` (Rackett 1970, with the
    Spencer-Danner ``Z_RA``). ``Tr`` is clipped at one so the expression stays
    defined (and differentiable) through the critical point.
    """
    tr = jnp.clip(jnp.asarray(t) / tc, 0.0, 1.0)
    exponent = 1.0 + (1.0 - tr) ** (2.0 / 7.0)
    return R * jnp.asarray(tc) / jnp.asarray(pc) * jnp.asarray(zra) ** exponent


# --- COSTALD ---------------------------------------------------------------------

# Saturated-volume shape polynomials of Hankinson & Thomson (AIChE J., 1979).
_COSTALD_A = (-1.52816, 1.43907, -0.81446, 0.190454)
_COSTALD_B = (-0.296123, 0.386914, -0.0427258, -0.0480645)


def _costald_vr(tr: Array) -> tuple[Array, Array]:
    """COSTALD shape functions ``(V_R^0, V_R^delta)`` of the reduced temperature."""
    tau = jnp.clip(1.0 - tr, 0.0, 1.0)
    a1, a2, a3, a4 = _COSTALD_A
    vr0 = (
        1.0 + a1 * tau ** (1.0 / 3.0) + a2 * tau ** (2.0 / 3.0) + a3 * tau + a4 * tau ** (4.0 / 3.0)
    )
    b1, b2, b3, b4 = _COSTALD_B
    vrd = (b1 + b2 * tr + b3 * tr**2 + b4 * tr**3) / (tr - 1.00001)
    return vr0, vrd


def costald_volume(t: ArrayLike, tc: ArrayLike, vchar: ArrayLike, omega_srk: ArrayLike) -> Array:
    """COSTALD saturated-liquid molar volume of a pure component (m^3/mol).

    ``V = V* . V_R^0(Tr) . (1 - omega_SRK . V_R^delta(Tr))`` with the
    characteristic volume ``V*`` and SRK acentric factor from the curated tables
    (Hankinson & Thomson 1979). Valid for ``0.25 < Tr < 0.95``: the upper
    clipping keeps the correlation finite as ``Tr -> 1``.
    """
    tr = jnp.clip(jnp.asarray(t) / tc, 0.0, 0.999)
    vr0, vrd = _costald_vr(tr)
    return jnp.asarray(vchar) * vr0 * (1.0 - jnp.asarray(omega_srk) * vrd)


def costald_mixture_volume(
    t: ArrayLike, x: Array, tc: Array, vchar: Array, omega_srk: Array
) -> Array:
    """COSTALD saturated molar volume of a liquid mixture (m^3/mol).

    Applies the original Hankinson-Thomson mixing rules::

        V*_m  = (1/4) [ sum x_i V*_i + 3 (sum x_i V*_i^(2/3)) (sum x_i V*_i^(1/3)) ]
        Tc_m  = sum_i sum_j x_i x_j sqrt(V*_i Tc_i V*_j Tc_j) / V*_m
        w_m   = sum x_i w_SRK_i

    then evaluates the pure-component correlation at the mixture parameters.
    """
    x = jnp.asarray(x)
    vchar = jnp.asarray(vchar)
    tc = jnp.asarray(tc)
    vm = 0.25 * (
        jnp.sum(x * vchar)
        + 3.0 * jnp.sum(x * vchar ** (2.0 / 3.0)) * jnp.sum(x * vchar ** (1.0 / 3.0))
    )
    vt = vchar * tc
    tcm = jnp.sum(jnp.sqrt(vt[:, None] * vt[None, :]) * x[:, None] * x[None, :]) / vm
    wm = jnp.sum(x * jnp.asarray(omega_srk))
    return costald_volume(t, tcm, vm, wm)


# --- Peneloux volume translation --------------------------------------------------

# Peneloux et al. (1982) constants for SRK and the Jhaveri-Youngren-style PR
# analog: c = k1 * (k2 - Z_RA) * R * Tc / Pc.
_PENELOUX = {
    "Soave-Redlich-Kwong": (0.40768, 0.29441),
    "Peng-Robinson": (0.50033, 0.25969),
}


def peneloux_shift(eos: CubicEOS, tc: ArrayLike, pc: ArrayLike, zra: ArrayLike) -> Array:
    """Peneloux volume-translation constant ``c_i`` (m^3/mol) for SRK or PR.

    ``c = k1*(k2 - Z_RA)*R*Tc/Pc``, positive for most fluids, so the translated
    liquid volume ``v - c`` shrinks toward the experimental value. Raises for EOS
    families without published constants (VDW, RK).
    """
    if eos.name not in _PENELOUX:
        raise ValueError(f"no Peneloux constants for EOS {eos.name!r}; use SRK or PR")
    k1, k2 = _PENELOUX[eos.name]
    return k1 * (k2 - jnp.asarray(zra)) * R * jnp.asarray(tc) / jnp.asarray(pc)


def translated_molar_volume(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    zra: Array,
    *,
    phase: str = "liquid",
    kij: Array | None = None,
) -> Array:
    """Peneloux-translated EOS molar volume ``v_eos - sum_i x_i c_i`` (m^3/mol).

    The translation is linear in composition, so all fugacity-coefficient
    *ratios* (and therefore every phase-equilibrium result) are unchanged;
    only the volumetric (and caloric-at-constant-V) properties improve.
    """
    x = jnp.asarray(x)
    v = molar_volume(eos, t, p, x, tc, pc, omega, phase=phase, kij=kij)
    c = peneloux_shift(eos, tc, pc, zra)
    return v - jnp.sum(x * c)


# --- Name-based dispatchers -------------------------------------------------------


def _resolve(components: list[str] | list[Component]) -> list[Component]:
    return [get(c) if isinstance(c, str) else c for c in components]


def _zra_of(comp: Component) -> Array:
    if comp.zra is not None:
        return jnp.asarray(comp.zra)
    return zra_estimate(comp.omega)


def liquid_molar_volumes(components: list[str] | list[Component], t: ArrayLike) -> Array:
    """Per-component saturated-liquid molar volumes ``v_i^L(T)`` (m^3/mol).

    Picks the best available correlation for each component: the transcribed
    DIPPR-105 density fit, then COSTALD, then Rackett (with the curated ``Z_RA``
    or its acentric-factor estimate). Defined for every database component.
    """
    values: list[Array] = []
    for comp in _resolve(components):
        rho_fit = RHO_LIQUID_DIPPR105.get(comp.name)
        costald_fit = COSTALD_VOLUME.get(comp.name)
        if rho_fit is not None:
            c1, c2, c3, c4, _tmin, _tmax = rho_fit
            values.append(1.0 / dippr105(t, c1, c2, c3, c4))
        elif costald_fit is not None:
            vchar, omega_srk = costald_fit
            values.append(costald_volume(t, comp.tc, vchar, omega_srk))
        else:
            values.append(rackett_volume(t, comp.tc, comp.pc, _zra_of(comp)))
    return jnp.stack(values)


def mixture_liquid_volume(
    components: list[str] | list[Component],
    t: ArrayLike,
    x: Array,
    *,
    method: str = "auto",
) -> Array:
    """Saturated-liquid molar volume of a mixture (m^3/mol).

    ``method="auto"`` (default) mole-fraction-averages the best available pure
    volumes (`liquid_molar_volumes`), so the pure-component limits match
    the transcribed DIPPR fits exactly. ``method="costald"`` opts into the
    Hankinson-Thomson corresponding-states mixing rules, which capture excess
    volume but require a curated characteristic volume for every component and
    inherit COSTALD's weakness for strongly polar species (e.g. water).
    ``method="amagat"`` is an explicit alias for the default ideal mixing.
    """
    resolved = _resolve(components)
    x = jnp.asarray(x)
    if method == "costald":
        missing = [c.name for c in resolved if c.name not in COSTALD_VOLUME]
        if missing:
            raise KeyError(f"no COSTALD characteristic volume for: {missing}")
        tc = jnp.asarray([c.tc for c in resolved])
        vchar = jnp.asarray([COSTALD_VOLUME[c.name][0] for c in resolved])
        omega_srk = jnp.asarray([COSTALD_VOLUME[c.name][1] for c in resolved])
        return costald_mixture_volume(t, x, tc, vchar, omega_srk)
    if method in ("amagat", "auto"):
        return jnp.sum(x * liquid_molar_volumes(resolved, t))
    raise ValueError(f"unknown method {method!r}; expected 'auto', 'costald' or 'amagat'")


def liquid_density(
    components: list[str] | list[Component],
    t: ArrayLike,
    x: Array,
    *,
    method: str = "auto",
) -> Array:
    """Mass density of a saturated liquid mixture (kg/m^3)."""
    resolved = _resolve(components)
    x = jnp.asarray(x)
    mw = jnp.asarray([c.mw for c in resolved])
    v = mixture_liquid_volume(resolved, t, x, method=method)
    return jnp.sum(x * mw) * 1.0e-3 / v


def vapor_density(
    components: list[str] | list[Component],
    t: ArrayLike,
    p: ArrayLike,
    y: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
) -> Array:
    """Mass density of a vapour mixture from the cubic EOS (kg/m^3)."""
    resolved = _resolve(components)
    y = jnp.asarray(y)
    arr = component_arrays(resolved)
    v = molar_volume(eos, t, p, y, arr["tc"], arr["pc"], arr["omega"], phase="vapor", kij=kij)
    return jnp.sum(y * arr["mw"]) * 1.0e-3 / v


def translated_liquid_volume_for(
    components: list[str] | list[Component],
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
) -> Array:
    """Peneloux-translated EOS liquid molar volume for named components (m^3/mol).

    Convenience wrapper over `translated_molar_volume` that assembles the
    critical constants and ``Z_RA`` values (curated, else estimated) from the
    component database.
    """
    resolved = _resolve(components)
    arr = component_arrays(resolved)
    zra = jnp.stack([_zra_of(c) for c in resolved])
    return translated_molar_volume(
        eos, t, p, jnp.asarray(x), arr["tc"], arr["pc"], arr["omega"], zra, phase="liquid", kij=kij
    )


def tyn_calus_vb(vc: ArrayLike) -> Array:
    """Tyn-Calus estimate of the molar volume at the normal boiling point (m^3/mol).

    ``Vb = 0.285 * Vc^1.048`` with both volumes in cm^3/mol (Poling et al., 5th
    ed., eq. 4-11.2); inputs and outputs here are SI (m^3/mol). Used by the
    Wilke-Chang diffusivity estimator.
    """
    vc_cm3 = jnp.asarray(vc) * 1.0e6
    return 0.285 * vc_cm3**1.048 * 1.0e-6
