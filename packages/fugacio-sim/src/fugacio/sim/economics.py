"""Differentiable equipment sizing, capital/operating costing, and economics.

To *optimize* a process you need an objective with units of money, and that means
sizing equipment and costing it. This module supplies smooth, differentiable
correlations for both, so a total-annual-cost or net-present-value objective
plugs straight into the gradient-based optimizers in
`fugacio.sim.optimize`: you can take the derivative of installed cost with
respect to a reflux ratio or a heat-exchanger approach temperature.

Sizing
------
Physically-grounded sizing from process quantities: heat-exchanger area from duty
and a log-mean temperature difference, column diameter from the Souders-Brown
flooding velocity, vessel volume from a residence time, and so on.

Costing
-------
The **Turton** correlations (Turton et al., *Analysis, Synthesis and Design of
Chemical Processes*) for purchased and bare-module cost:

* purchased cost ``log10 Cp0 = K1 + K2 log10 A + K3 (log10 A)^2`` in the size
  attribute ``A``;
* a pressure factor ``log10 F_P = C1 + C2 log10 P + C3 (log10 P)^2``;
* the bare-module cost ``C_BM = Cp0 (B1 + B2 F_M F_P)`` with a material factor
  ``F_M``;
* escalation from the correlation basis (CEPCI 397, year 2001) to a target CEPCI.

These are smooth in the size and pressure, hence differentiable. The bundled
coefficients are representative textbook values for screening-level estimates,
not a substitute for a vendor quote.

Operating cost & finance
------------------------
Utility pricing on an energy basis, the capital-recovery factor, total annual
cost, net present value, and discounted payback, all differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import R

ArrayLike = Array | float

#: Reference Chemical Engineering Plant Cost Index of the Turton correlations
#: (year 2001). Escalate to a target year by the ratio of indices.
CEPCI_REF = 397.0
#: A recent CEPCI value (2018) used as the default target.
CEPCI_DEFAULT = 603.1
#: Default operating hours per year (~91 % stream factor).
HOURS_PER_YEAR = 8000.0


# --------------------------------------------------------------------------- #
# Turton cost-correlation coefficients (representative screening values)
# --------------------------------------------------------------------------- #
class _Turton(NamedTuple):
    """Purchased-cost / bare-module / pressure-factor coefficients for one equipment type."""

    k1: float
    k2: float
    k3: float
    b1: float
    b2: float
    c1: float
    c2: float
    c3: float
    size_kind: str
    size_min: float
    size_max: float


#: Representative Turton coefficients keyed by equipment kind. ``size_kind`` names
#: the capacity attribute the correlation expects (area m^2, power kW, volume m^3).
_TURTON: dict[str, _Turton] = {
    # Shell-and-tube heat exchanger (floating head); A = area [m^2].
    "heat_exchanger": _Turton(
        4.8306, -0.8509, 0.3187, 1.63, 1.66, -0.00164, -0.00627, 0.0123, "area_m2", 10.0, 1000.0
    ),
    # Centrifugal pump; A = shaft power [kW].
    "pump": _Turton(
        3.3892, 0.0536, 0.1538, 1.89, 1.35, -0.3935, 0.3957, -0.00226, "power_kw", 1.0, 300.0
    ),
    # Centrifugal compressor; A = fluid power [kW]. Costed without a pressure
    # factor (B1=1, B2=F_BM); F_BM ~ 2.7 for a carbon-steel centrifugal machine.
    "compressor": _Turton(
        2.2897, 1.3604, -0.1027, 1.0, 2.7, 0.0, 0.0, 0.0, "power_kw", 450.0, 3000.0
    ),
    # Vertical process vessel / column shell; A = volume [m^3].
    "vessel": _Turton(3.4974, 0.4485, 0.1074, 2.25, 1.82, -0.0, 0.0, 0.0, "volume_m3", 0.3, 520.0),
    # Sieve tray (per tray); A = tray area [m^2]. Added on top of the shell.
    "tray": _Turton(2.9949, 0.4465, 0.3961, 1.0, 1.0, 0.0, 0.0, 0.0, "area_m2", 0.07, 12.3),
    # Fired heater (process, non-reactive); A = duty [kW].
    "fired_heater": _Turton(
        3.0684, 0.6606, 0.0599, 1.0, 2.2, 0.1347, -0.2368, 0.1021, "duty_kw", 1000.0, 100000.0
    ),
}

#: Material factors ``F_M`` (multiplies the pressure-dependent part of ``F_BM``).
_MATERIAL_FACTOR: dict[str, float] = {
    "CS": 1.0,  # carbon steel
    "SS": 2.9,  # stainless steel (shell-and-tube, representative)
    "Ni": 5.4,  # nickel alloy
    "Ti": 7.0,  # titanium
}


class EquipmentCost(NamedTuple):
    """Costed equipment item.

    Attributes:
        kind: Equipment kind (key into the Turton table).
        size: The sizing attribute used (area m^2, power kW, or volume m^3).
        purchased: Purchased cost ``Cp0`` escalated to the target CEPCI ($).
        bare_module: Installed bare-module cost ``C_BM`` ($).
    """

    kind: str
    size: Array
    purchased: Array
    bare_module: Array


# --------------------------------------------------------------------------- #
# Equipment sizing
# --------------------------------------------------------------------------- #
def lmtd(dt1: ArrayLike, dt2: ArrayLike) -> Array:
    """Log-mean temperature difference of the two terminal approaches.

    Smooth everywhere for positive approaches: as ``dt1 -> dt2`` the value tends
    to their common limit, with a series expansion used near equality so the
    gradient stays finite.
    """
    d1 = jnp.asarray(dt1, dtype=float)
    d2 = jnp.asarray(dt2, dtype=float)
    ratio = d1 / d2
    log_r = jnp.log(jnp.where(jnp.abs(ratio - 1.0) < 1e-6, 1.0, ratio))
    safe = d2 * (ratio - 1.0) / jnp.where(jnp.abs(log_r) < 1e-12, 1.0, log_r)
    # Near-equal terminals: LMTD ~ mean * (1 - (dt1-dt2)^2/(12 mean^2)).
    mean = 0.5 * (d1 + d2)
    return jnp.where(jnp.abs(ratio - 1.0) < 1e-6, mean, safe)


def heat_exchanger_area(
    duty: ArrayLike, u: ArrayLike, dt_hot: ArrayLike, dt_cold: ArrayLike
) -> Array:
    """Required heat-transfer area ``A = |Q| / (U * LMTD)`` (m^2).

    Args:
        duty: Heat duty ``Q`` (W); the magnitude is used.
        u: Overall heat-transfer coefficient (W/m^2/K).
        dt_hot: Terminal temperature approach at the hot end (K).
        dt_cold: Terminal temperature approach at the cold end (K).
    """
    return jnp.abs(jnp.asarray(duty)) / (jnp.asarray(u) * lmtd(dt_hot, dt_cold))


def vapor_molar_volume_ideal(t: ArrayLike, p: ArrayLike) -> Array:
    """Ideal-gas molar volume ``V = R T / P`` (m^3/mol), a sizing convenience."""
    return R * jnp.asarray(t) / jnp.asarray(p)


def column_diameter(
    vapor_molar_flow: ArrayLike,
    vapor_density: ArrayLike,
    liquid_density: ArrayLike,
    *,
    molar_mass: ArrayLike = 0.05,
    k_drum: ArrayLike = 0.07,
    flooding: ArrayLike = 0.8,
) -> Array:
    """Column (or knock-out drum) diameter from the Souders-Brown flooding velocity.

    The maximum vapour velocity is ``u_max = k sqrt((rho_L - rho_V)/rho_V)``; the
    diameter follows from the volumetric vapour load at the design fraction of
    flooding.

    Args:
        vapor_molar_flow: Vapour molar flow (mol/s).
        vapor_density: Vapour mass density (kg/m^3).
        liquid_density: Liquid mass density (kg/m^3).
        molar_mass: Vapour molar mass (kg/mol) to convert molar to mass flow.
        k_drum: Souders-Brown coefficient (m/s).
        flooding: Design fraction of the flooding velocity.

    Returns:
        Diameter (m).
    """
    rho_v = jnp.asarray(vapor_density)
    rho_l = jnp.asarray(liquid_density)
    u_max = jnp.asarray(k_drum) * jnp.sqrt(jnp.maximum(rho_l - rho_v, 1e-6) / rho_v)
    u_design = jnp.asarray(flooding) * u_max
    mass_flow = jnp.asarray(vapor_molar_flow) * jnp.asarray(molar_mass)  # kg/s
    volumetric = mass_flow / rho_v  # m^3/s
    area = volumetric / u_design
    return jnp.sqrt(4.0 * area / jnp.pi)


def column_height(
    n_stages: ArrayLike, *, tray_spacing: ArrayLike = 0.6, extra: ArrayLike = 4.0
) -> Array:
    """Tangent-to-tangent column height from the stage count and tray spacing (m)."""
    return jnp.asarray(n_stages) * jnp.asarray(tray_spacing) + jnp.asarray(extra)


def vessel_volume(
    volumetric_flow: ArrayLike, *, residence_time: ArrayLike = 300.0, fill: ArrayLike = 0.5
) -> Array:
    """Vessel volume from a volumetric flow and liquid residence time (m^3)."""
    return jnp.asarray(volumetric_flow) * jnp.asarray(residence_time) / jnp.asarray(fill)


def cylinder_volume(diameter: ArrayLike, height: ArrayLike) -> Array:
    """Volume of a vertical cylinder (m^3), a column shell from its size."""
    d = jnp.asarray(diameter)
    return jnp.pi * d**2 / 4.0 * jnp.asarray(height)


# --------------------------------------------------------------------------- #
# Turton costing
# --------------------------------------------------------------------------- #
def purchased_cost(kind: str, size: ArrayLike) -> Array:
    """Purchased equipment cost ``Cp0`` at the correlation basis (CEPCI 397), in $.

    Uses ``log10 Cp0 = K1 + K2 log10 A + K3 (log10 A)^2`` with ``A`` the
    equipment's capacity attribute (see `_TURTON`). The size is clipped to
    the correlation's validity range before the (smooth) evaluation.
    """
    c = _coeffs(kind)
    a = jnp.clip(jnp.asarray(size, dtype=float), c.size_min, c.size_max)
    log_a = jnp.log10(a)
    return 10.0 ** (c.k1 + c.k2 * log_a + c.k3 * log_a**2)


def pressure_factor(kind: str, pressure_barg: ArrayLike) -> Array:
    """Turton pressure factor ``F_P`` (>= 1) for the equipment at gauge pressure (barg)."""
    c = _coeffs(kind)
    if c.c1 == 0.0 and c.c2 == 0.0 and c.c3 == 0.0:
        return jnp.ones_like(jnp.asarray(pressure_barg, dtype=float))
    p = jnp.maximum(jnp.asarray(pressure_barg, dtype=float), 0.0)
    log_p = jnp.log10(jnp.maximum(p, 1.0))
    fp = 10.0 ** (c.c1 + c.c2 * log_p + c.c3 * log_p**2)
    return jnp.maximum(fp, 1.0)


def bare_module_cost(
    kind: str,
    size: ArrayLike,
    *,
    pressure_barg: ArrayLike = 0.0,
    material: str = "CS",
    cepci: ArrayLike = CEPCI_DEFAULT,
) -> EquipmentCost:
    """Installed (bare-module) cost ``C_BM`` escalated to the target CEPCI.

    ``C_BM = Cp0 (B1 + B2 F_M F_P)`` with a material factor ``F_M`` and pressure
    factor ``F_P``, all escalated by ``cepci / CEPCI_REF``.

    Args:
        kind: Equipment kind (key in the Turton table).
        size: Capacity attribute (area m^2, power kW, or volume m^3).
        pressure_barg: Design gauge pressure (barg) for the pressure factor.
        material: Material-of-construction key (``"CS"``, ``"SS"``, ...).
        cepci: Target cost index.

    Returns:
        An `EquipmentCost`.
    """
    c = _coeffs(kind)
    fm = _MATERIAL_FACTOR.get(material, 1.0)
    fp = pressure_factor(kind, pressure_barg)
    cp0 = purchased_cost(kind, size)
    escalation = jnp.asarray(cepci) / CEPCI_REF
    cp0_now = cp0 * escalation
    c_bm = cp0_now * (c.b1 + c.b2 * fm * fp)
    return EquipmentCost(
        kind=kind, size=jnp.asarray(size, dtype=float), purchased=cp0_now, bare_module=c_bm
    )


def _coeffs(kind: str) -> _Turton:
    if kind not in _TURTON:
        raise KeyError(f"unknown equipment kind {kind!r}; known: {sorted(_TURTON)}")
    return _TURTON[kind]


# --------------------------------------------------------------------------- #
# Utilities (operating cost)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Utility:
    """A priced utility on an energy basis.

    Attributes:
        name: Utility name.
        price_per_gj: Price ($/GJ of heating or cooling delivered).
    """

    name: str
    price_per_gj: float


#: Representative utility prices ($/GJ); screening-level values.
UTILITIES: dict[str, Utility] = {
    "cooling_water": Utility("cooling water", 0.35),
    "chilled_water": Utility("chilled water", 4.5),
    "refrigeration": Utility("refrigeration", 7.9),
    "lp_steam": Utility("low-pressure steam", 13.3),
    "mp_steam": Utility("medium-pressure steam", 14.2),
    "hp_steam": Utility("high-pressure steam", 17.7),
    "fired_heat": Utility("fired heat", 11.1),
    "electricity": Utility("electricity", 18.7),
}


def utility_cost(
    duty_w: ArrayLike, kind: str, *, hours_per_year: ArrayLike = HOURS_PER_YEAR
) -> Array:
    """Annual cost of a utility duty ($/yr) from its energy price.

    Args:
        duty_w: Duty (W); the magnitude is used (heating or cooling alike).
        kind: Utility key in `UTILITIES`.
        hours_per_year: Operating hours per year.
    """
    if kind not in UTILITIES:
        raise KeyError(f"unknown utility {kind!r}; known: {sorted(UTILITIES)}")
    energy_gj = jnp.abs(jnp.asarray(duty_w)) * jnp.asarray(hours_per_year) * 3600.0 / 1e9
    return energy_gj * UTILITIES[kind].price_per_gj


# --------------------------------------------------------------------------- #
# Financial metrics
# --------------------------------------------------------------------------- #
def capital_recovery_factor(rate: ArrayLike, years: ArrayLike) -> Array:
    """Capital-recovery factor ``i(1+i)^n / ((1+i)^n - 1)`` for interest ``i`` over ``n`` years."""
    i = jnp.asarray(rate, dtype=float)
    n = jnp.asarray(years, dtype=float)
    growth = (1.0 + i) ** n
    return i * growth / (growth - 1.0)


def annualized_capital(
    capex: ArrayLike, *, rate: ArrayLike = 0.1, years: ArrayLike = 10.0
) -> Array:
    """Annualized capital charge ``CRF * CAPEX`` ($/yr)."""
    return capital_recovery_factor(rate, years) * jnp.asarray(capex)


def total_annual_cost(
    capex: ArrayLike, opex: ArrayLike, *, rate: ArrayLike = 0.1, years: ArrayLike = 10.0
) -> Array:
    """Total annual cost ``TAC = CRF * CAPEX + OPEX`` ($/yr), the screening objective."""
    return annualized_capital(capex, rate=rate, years=years) + jnp.asarray(opex)


def npv(cash_flows: Array, *, rate: ArrayLike = 0.1) -> Array:
    """Net present value of a cash-flow stream (index 0 is now), discounted at ``rate``."""
    cf = jnp.asarray(cash_flows, dtype=float)
    periods = jnp.arange(cf.shape[0])
    return jnp.sum(cf / (1.0 + jnp.asarray(rate)) ** periods)


def discounted_payback(capex: ArrayLike, annual_cash: ArrayLike, *, rate: ArrayLike = 0.1) -> Array:
    """Discounted payback period (years) for a uniform annual cash inflow.

    Solves ``CAPEX = annual_cash * (1 - (1+i)^-n) / i`` for ``n`` in closed form;
    smooth and differentiable in all arguments.
    """
    i = jnp.asarray(rate, dtype=float)
    a = jnp.asarray(annual_cash, dtype=float)
    c = jnp.asarray(capex, dtype=float)
    return -jnp.log(1.0 - c * i / a) / jnp.log(1.0 + i)


def installed_capital(items: list[EquipmentCost], *, lang_factor: ArrayLike = 1.0) -> Array:
    """Total installed capital from a list of costed items (optionally Lang-scaled)."""
    total = sum((it.bare_module for it in items), jnp.asarray(0.0))
    return jnp.asarray(lang_factor) * total
