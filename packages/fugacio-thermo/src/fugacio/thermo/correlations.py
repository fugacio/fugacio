"""Temperature-dependent pure-component property correlations.

Two layers live here:

* **Correlation kernels** -- the classic DIPPR-numbered functional forms
  (`dippr100`, `dippr101`, `dippr102`, `dippr105`,
  `dippr106`) plus the REFPROP-style `mulero_cachadina` surface-tension
  expansion. These are plain `jax.numpy` functions of temperature and their
  coefficients, so they are differentiable in both (handy when regressing
  coefficients to data).
* **Corresponding-states estimators and dispatchers** -- enthalpy of vaporization
  (curated DIPPR-106 table, else `pitzer_hvap`; rescaled between
  temperatures by `watson_hvap`) and liquid heat capacity
  (`rowlinson_bondi_cp` on top of the ideal-gas ``Cp``). The name-based
  dispatchers (`heat_of_vaporization`, `liquid_heat_capacity`) pull
  per-component coefficients from the generated
  `fugacio.thermo._property_data` tables and fall back to the
  corresponding-states estimate when no curated fit exists, so they work for the
  whole component database.

Coefficient tables are transcribed (or least-squares refitted onto these forms)
from the open ``chemicals`` dataset by ``scripts/gen_property_data.py``; the
companion oracle tests grade both the tables and the estimators against
CoolProp's multiparameter reference fluids.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo._property_data import HVAP_DIPPR106
from fugacio.thermo.components import Component, get
from fugacio.thermo.constants import R
from fugacio.thermo.ideal import cp_ig

ArrayLike = Array | float


# --- DIPPR-form correlation kernels ---------------------------------------------


def dippr100(
    t: ArrayLike,
    c1: ArrayLike,
    c2: ArrayLike = 0.0,
    c3: ArrayLike = 0.0,
    c4: ArrayLike = 0.0,
    c5: ArrayLike = 0.0,
) -> Array:
    """DIPPR equation 100: plain polynomial ``c1 + c2*T + c3*T^2 + c4*T^3 + c5*T^4``."""
    t = jnp.asarray(t)
    return c1 + c2 * t + c3 * t**2 + c4 * t**3 + c5 * t**4


def dippr101(
    t: ArrayLike,
    c1: ArrayLike,
    c2: ArrayLike,
    c3: ArrayLike = 0.0,
    c4: ArrayLike = 0.0,
    c5: ArrayLike = 1.0,
) -> Array:
    """DIPPR equation 101: ``exp(c1 + c2/T + c3*ln(T) + c4*T^c5)``.

    The workhorse form for liquid viscosity (and vapour pressure). ``c5`` is a
    fixed exponent from the source table, not usually a regression variable.
    """
    t = jnp.asarray(t)
    return jnp.exp(c1 + c2 / t + c3 * jnp.log(t) + c4 * t ** jnp.asarray(c5))


def dippr102(
    t: ArrayLike,
    c1: ArrayLike,
    c2: ArrayLike,
    c3: ArrayLike = 0.0,
    c4: ArrayLike = 0.0,
) -> Array:
    """DIPPR equation 102: ``c1*T^c2 / (1 + c3/T + c4/T^2)``.

    The standard form for dilute-gas viscosity and thermal conductivity.
    """
    t = jnp.asarray(t)
    return c1 * t ** jnp.asarray(c2) / (1.0 + c3 / t + c4 / t**2)


def dippr105(
    t: ArrayLike,
    c1: ArrayLike,
    c2: ArrayLike,
    c3: ArrayLike,
    c4: ArrayLike,
) -> Array:
    """DIPPR equation 105: ``c1 / c2^(1 + (1 - T/c3)^c4)``.

    The Rackett-shaped form used for saturated-liquid molar density; the units of
    the result are the units of ``c1`` (mol/m^3 in Fugacio's tables).
    """
    t = jnp.asarray(t)
    return c1 / c2 ** (1.0 + (1.0 - t / c3) ** jnp.asarray(c4))


def dippr106(
    t: ArrayLike,
    tc: ArrayLike,
    c1: ArrayLike,
    c2: ArrayLike,
    c3: ArrayLike = 0.0,
    c4: ArrayLike = 0.0,
    c5: ArrayLike = 0.0,
) -> Array:
    """DIPPR equation 106: ``c1 * (1-Tr)^(c2 + c3*Tr + c4*Tr^2 + c5*Tr^3)``.

    Used for enthalpy of vaporization and surface tension; vanishes at the
    critical point by construction. ``Tr = T/tc`` is clipped just below one so
    the expression (and its temperature derivative) stays finite under autodiff.
    """
    tr = jnp.clip(jnp.asarray(t) / tc, 0.0, 1.0 - 1e-12)
    tau = 1.0 - tr
    return c1 * tau ** (c2 + c3 * tr + c4 * tr**2 + c5 * tr**3)


def mulero_cachadina(
    t: ArrayLike,
    tc: ArrayLike,
    s0: ArrayLike,
    n0: ArrayLike,
    s1: ArrayLike = 0.0,
    n1: ArrayLike = 1.0,
    s2: ArrayLike = 0.0,
    n2: ArrayLike = 1.0,
) -> Array:
    """Mulero-Cachadina surface tension ``sum_k s_k * (1 - T/tc)^n_k`` (N/m).

    The three-term REFPROP fit of Mulero, Cachadina & Parra (J. Phys. Chem. Ref.
    Data, 2012). With a single term it reduces to the van der Waals-Guggenheim
    form ``sigma0 * (1-Tr)^1.26``. Clipped at ``Tr = 1`` so the tension is zero
    (not NaN) above the critical point.
    """
    tau = jnp.clip(1.0 - jnp.asarray(t) / tc, 0.0, 1.0)
    return s0 * tau ** jnp.asarray(n0) + s1 * tau ** jnp.asarray(n1) + s2 * tau ** jnp.asarray(n2)


# --- Corresponding-states estimators --------------------------------------------


def pitzer_hvap(t: ArrayLike, tc: ArrayLike, omega: ArrayLike) -> Array:
    """Pitzer acentric-factor enthalpy of vaporization (J/mol).

    ``Hvap / (R*Tc) = 7.08*(1-Tr)^0.354 + 10.95*omega*(1-Tr)^0.456`` -- the
    three-parameter corresponding-states correlation recommended by Poling et al.
    (5th ed., eq. 7-9.4), good to a few percent for normal fluids over
    ``0.6 < Tr < 1``. Clipped to zero above the critical temperature.
    """
    tr = jnp.clip(jnp.asarray(t) / tc, 0.0, 1.0)
    tau = 1.0 - tr
    return R * jnp.asarray(tc) * (7.08 * tau**0.354 + 10.95 * omega * tau**0.456)


def watson_hvap(
    hvap_ref: ArrayLike, t_ref: ArrayLike, t: ArrayLike, tc: ArrayLike, n: float = 0.38
) -> Array:
    """Watson rescaling of a known enthalpy of vaporization to another temperature.

    ``Hvap(T) = Hvap(T_ref) * ((1-Tr) / (1-Tr_ref))^n`` with the classic
    ``n = 0.38`` exponent. Use when a single measured point (say at the normal
    boiling temperature) is available.
    """
    tc = jnp.asarray(tc)
    tau = jnp.clip(1.0 - jnp.asarray(t) / tc, 0.0, 1.0)
    tau_ref = jnp.clip(1.0 - jnp.asarray(t_ref) / tc, 1e-12, 1.0)
    return jnp.asarray(hvap_ref) * (tau / tau_ref) ** n


def rowlinson_bondi_cp(t: ArrayLike, tc: ArrayLike, omega: ArrayLike, cp_ideal: ArrayLike) -> Array:
    """Rowlinson-Bondi saturated-liquid heat capacity (J/mol/K).

    ``(Cp_L - Cp_ig)/R = 1.45 + 0.45/(1-Tr) + 0.25*omega*(17.11
    + 25.2*(1-Tr)^(1/3)/Tr + 1.742/(1-Tr))`` (Poling et al., 5th ed.,
    eq. 6-6.1). Diverges at the critical point, as the physical ``Cp`` does;
    ``Tr`` is clipped just below one to keep autodiff finite.
    """
    tr = jnp.clip(jnp.asarray(t) / tc, 1e-6, 1.0 - 1e-9)
    tau = 1.0 - tr
    bracket = 17.11 + 25.2 * tau ** (1.0 / 3.0) / tr + 1.742 / tau
    return jnp.asarray(cp_ideal) + R * (1.45 + 0.45 / tau + 0.25 * omega * bracket)


# --- Name-based dispatchers over the curated tables ------------------------------


def _resolve(components: list[str] | list[Component]) -> list[Component]:
    return [get(c) if isinstance(c, str) else c for c in components]


def heat_of_vaporization(components: list[str] | list[Component], t: ArrayLike) -> Array:
    """Per-component enthalpy of vaporization ``Hvap_i(T)`` (J/mol).

    Uses the curated DIPPR-106 fit where one exists and the Pitzer
    corresponding-states correlation (`pitzer_hvap`) otherwise, so the
    result is defined for every database component. Values go to zero at each
    component's critical temperature.
    """
    values: list[Array] = []
    for comp in _resolve(components):
        row = HVAP_DIPPR106.get(comp.name)
        if row is not None:
            tc_fit, c1, c2, c3, c4, _tmin, _tmax = row
            values.append(dippr106(t, tc_fit, c1, c2, c3, c4))
        else:
            values.append(pitzer_hvap(t, comp.tc, comp.omega))
    return jnp.stack(values)


def liquid_heat_capacity(components: list[str] | list[Component], t: ArrayLike) -> Array:
    """Per-component saturated-liquid heat capacity ``Cp_L_i(T)`` (J/mol/K).

    Rowlinson-Bondi corresponding states on top of each component's ideal-gas
    ``Cp`` correlation.

    Raises:
        ValueError: if any component lacks ideal-gas heat-capacity data.
    """
    resolved = _resolve(components)
    missing = [c.name for c in resolved if c.cp_ig is None]
    if missing:
        raise ValueError(f"missing ideal-gas Cp data for: {missing}")
    values: list[Array] = []
    for comp in resolved:
        cp = comp.cp_ig
        assert cp is not None
        cp_id = cp_ig(t, cp.a, cp.b, cp.c, cp.d, cp.e)
        values.append(rowlinson_bondi_cp(t, comp.tc, comp.omega, cp_id))
    return jnp.stack(values)
