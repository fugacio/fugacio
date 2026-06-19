"""A tool registry exposing the Fugacio engine to an LLM design agent.

The copilot layer turns natural-language design goals into engineering
calculations. The bridge between a language model and the differentiable engine
is a set of *tools*: deterministic, JSON-in/JSON-out functions with machine-
readable schemas (the same shape OpenAI / Anthropic function-calling expects).

This module defines that registry over `fugacio.sim` and
`fugacio.thermo`. The tools are plain Python (floats and lists, not JAX
arrays) so they are trivial to serialise; the LLM-planning loop that selects and
sequences them lives behind the optional ``llm`` extra (`fugacio.copilot.agent`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from fugacio.copilot.dynamics_tools import dynamics_tool_specs
from fugacio.copilot.integration_tools import integration_tool_specs
from fugacio.copilot.mpc_tools import mpc_tool_specs
from fugacio.copilot.saft_tools import saft_tool_specs
from fugacio.sim import (
    STEAM_LEVELS,
    Stream,
    annualized_capital,
    azeotrope_pressure,
    azeotrope_temperature,
    bare_module_cost,
    capital_recovery_factor,
    column_diameter,
    column_height,
    compressor,
    cooling_water,
    cstr,
    cylinder_volume,
    equilibrium_reactor,
    flash_drum,
    heat_exchanger_area,
    heater,
    lmtd,
    npv,
    nrtl_model_for,
    pfr,
    pump,
    pxy_diagram,
    reactive_flash,
    relative_volatility,
    residue_curve_map,
    shortcut_column,
    solve_column,
    steam_heating,
    steam_turbine,
    stoichiometric_reactor,
    total_annual_cost,
    turbine,
    txy_diagram,
    unifac_model_for,
    uniquac_model_for,
    utility_cost,
    valve,
    vapor_molar_volume_ideal,
)
from fugacio.thermo import (
    PR,
    PowerLaw,
    Reaction,
    bubble_pressure_eos,
    component_arrays,
    fit_nrtl_binary,
    flash_lle,
    flash_pt,
    flash_vlle,
    gas_diffusivity,
    gas_mixture_thermal_conductivity,
    gas_mixture_viscosity,
    get,
    heat_of_vaporization,
    liquid_density,
    liquid_diffusivity,
    liquid_heat_capacity,
    liquid_mixture_thermal_conductivity,
    liquid_mixture_viscosity,
    liquid_stability,
    mixture_surface_tension,
    names,
    psat_eos,
    reaction_properties,
    vapor_density,
)
from fugacio.thermo import helmholtz as _helmholtz
from fugacio.thermo.reaction_equilibrium import equilibrium as _reaction_equilibrium_solve

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """A callable tool with an LLM-facing JSON schema.

    Attributes:
        name: Unique tool name.
        description: One-line description for the planner.
        parameters: JSON-schema object describing the arguments.
        run: The implementation, taking keyword arguments and returning a dict.
    """

    name: str
    description: str
    parameters: JsonDict
    run: Callable[..., JsonDict]


def _list_components() -> JsonDict:
    return {"components": names()}


def _component_properties(component: str) -> JsonDict:
    c = get(component)
    return {
        "name": c.name,
        "formula": c.formula,
        "molar_mass_g_mol": c.mw,
        "critical_temperature_k": c.tc,
        "critical_pressure_pa": c.pc,
        "critical_volume_m3_mol": c.vc,
        "acentric_factor": c.omega,
        "normal_boiling_point_k": c.tb,
        "dipole_moment_debye": c.dipole,
    }


def _saturation_pressure(component: str, temperature: float) -> JsonDict:
    c = get(component)
    psat = float(psat_eos(PR, temperature, c.tc, c.pc, c.omega))
    return {"component": c.name, "temperature_k": temperature, "psat_pa": psat}


def _physical_properties(
    components: list[str],
    x: list[float],
    temperature: float,
    pressure: float = 101325.0,
) -> JsonDict:
    """Sizing-grade physical and transport properties of a mixture at ``T`` (and ``P``).

    Liquid properties are at saturation (composition-averaged through the curated
    correlations and mixture rules); the vapour density comes from the EOS at the
    given pressure. Heat of vaporization and liquid heat capacity are returned
    per component (J/mol basis).
    """
    t = float(temperature)
    p = float(pressure)
    x_arr = jnp.asarray(x, dtype=float)
    hvap = heat_of_vaporization(components, t)
    cp_l = liquid_heat_capacity(components, t)
    return {
        "components": list(components),
        "x": [float(v) for v in x],
        "temperature_k": t,
        "pressure_pa": p,
        "liquid_density_kg_m3": float(liquid_density(components, t, x_arr)),
        "vapor_density_kg_m3": float(vapor_density(components, t, p, x_arr)),
        "liquid_viscosity_pa_s": float(liquid_mixture_viscosity(components, t, x_arr)),
        "vapor_viscosity_pa_s": float(gas_mixture_viscosity(components, t, x_arr)),
        "liquid_thermal_conductivity_w_m_k": float(
            liquid_mixture_thermal_conductivity(components, t, x_arr)
        ),
        "vapor_thermal_conductivity_w_m_k": float(
            gas_mixture_thermal_conductivity(components, t, x_arr)
        ),
        "surface_tension_n_m": float(mixture_surface_tension(components, t, x_arr)),
        "heat_of_vaporization_j_mol": [float(v) for v in hvap],
        "liquid_heat_capacity_j_mol_k": [float(v) for v in cp_l],
    }


def _binary_diffusivity(
    solute: str,
    solvent: str,
    temperature: float,
    pressure: float = 101325.0,
) -> JsonDict:
    """Binary diffusion coefficients: Fuller (gas) and Wilke-Chang (liquid)."""
    t = float(temperature)
    p = float(pressure)
    return {
        "solute": solute,
        "solvent": solvent,
        "temperature_k": t,
        "pressure_pa": p,
        "gas_diffusivity_m2_s": float(gas_diffusivity(solute, solvent, t, p)),
        "liquid_diffusivity_m2_s": float(liquid_diffusivity(solute, solvent, t)),
    }


def _bubble_pressure(components: list[str], x: list[float], temperature: float) -> JsonDict:
    arr = component_arrays(components)
    p, y = bubble_pressure_eos(PR, temperature, jnp.asarray(x), arr["tc"], arr["pc"], arr["omega"])
    return {
        "temperature_k": temperature,
        "bubble_pressure_pa": float(p),
        "vapor_composition": [float(v) for v in y],
    }


def _flash(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
) -> JsonDict:
    feed = Stream.from_fractions(tuple(components), jnp.asarray(z), flow, temperature, pressure)
    vapor, liquid = flash_drum(feed, temperature, pressure)
    return {
        "vapor_fraction": float(vapor.total / feed.total),
        "vapor": {"flow_mol_s": float(vapor.total), "composition": [float(v) for v in vapor.z]},
        "liquid": {"flow_mol_s": float(liquid.total), "composition": [float(v) for v in liquid.z]},
    }


def _stream(components: list[str], z: list[float], flow: float, t: float, p: float) -> Stream:
    return Stream.from_fractions(tuple(components), jnp.asarray(z), flow, t, p)


def _heat_exchanger(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
    t_out: float | None = None,
    duty: float | None = None,
) -> JsonDict:
    feed = _stream(components, z, flow, temperature, pressure)
    res = heater(feed, t_out=t_out, duty=duty)
    return {
        "outlet_temperature_k": float(res.outlet.t),
        "duty_w": float(res.duty),
        "outlet_pressure_pa": float(res.outlet.p),
    }


def _compressor(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
    pressure_out: float,
    efficiency: float = 0.75,
    machine: str = "compressor",
) -> JsonDict:
    feed = _stream(components, z, flow, temperature, pressure)
    model = turbine if machine == "turbine" else compressor
    res = model(feed, pressure_out, efficiency=efficiency)
    return {
        "outlet_temperature_k": float(res.outlet.t),
        "outlet_pressure_pa": float(res.outlet.p),
        "work_w": float(res.work),
        "ideal_work_w": float(res.ideal_work),
    }


def _pump(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
    pressure_out: float,
    efficiency: float = 0.75,
) -> JsonDict:
    feed = _stream(components, z, flow, temperature, pressure)
    res = pump(feed, pressure_out, efficiency=efficiency)
    return {
        "outlet_temperature_k": float(res.outlet.t),
        "outlet_pressure_pa": float(res.outlet.p),
        "work_w": float(res.work),
    }


def _valve(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
    pressure_out: float,
) -> JsonDict:
    feed = _stream(components, z, flow, temperature, pressure)
    out = valve(feed, pressure_out)
    return {
        "outlet_temperature_k": float(out.t),
        "outlet_pressure_pa": float(out.p),
        "temperature_drop_k": float(temperature - out.t),
    }


def _distillation_recoveries(
    alpha: list[float], lk: int, hk: int, lk_rec: float, hk_rec: float
) -> list[float]:
    """Per-component recovery to the distillate via the key recoveries and volatility."""
    a_lk, a_hk = alpha[lk], alpha[hk]
    recoveries = []
    for i, a_i in enumerate(alpha):
        if i == lk:
            recoveries.append(lk_rec)
        elif i == hk:
            recoveries.append(hk_rec)
        elif a_i >= a_lk:
            recoveries.append(0.999)
        elif a_i <= a_hk:
            recoveries.append(0.001)
        else:
            frac = (a_i - a_hk) / (a_lk - a_hk)
            recoveries.append(hk_rec + frac * (lk_rec - hk_rec))
    return recoveries


def _shortcut_distillation(
    components: list[str],
    z: list[float],
    feed_flow: float,
    temperature: float,
    pressure: float,
    light_key: int,
    heavy_key: int,
    lk_recovery: float = 0.99,
    hk_recovery: float = 0.01,
    q: float = 1.0,
) -> JsonDict:
    arr = component_arrays(components)
    z_arr = jnp.asarray(z)
    alpha = relative_volatility(
        PR, temperature, pressure, z_arr, arr["tc"], arr["pc"], arr["omega"], ref=heavy_key
    )
    recoveries = _distillation_recoveries(
        [float(a) for a in alpha], light_key, heavy_key, lk_recovery, hk_recovery
    )
    rec = jnp.asarray(recoveries)
    d = rec * feed_flow * z_arr
    b = (1.0 - rec) * feed_flow * z_arr
    res = shortcut_column(z_arr, d, b, alpha, q, light_key, heavy_key)
    return {
        "minimum_stages": float(res.n_min),
        "minimum_reflux": float(res.r_min),
        "reflux_ratio": float(res.r),
        "actual_stages": float(res.n_stages),
        "stages_above_feed": float(res.feed_stage),
        "relative_volatilities": [float(a) for a in alpha],
    }


def _rigorous_distillation(
    components: list[str],
    z: list[float],
    feed_flow: float,
    feed_temperature: float,
    pressure: float,
    n_stages: int,
    feed_stage: int,
    reflux: float,
    distillate_rate: float,
    q: float = 1.0,
) -> JsonDict:
    feed = _stream(components, z, feed_flow, feed_temperature, pressure)
    res = solve_column(feed, n_stages, feed_stage, reflux, distillate_rate, q=q)
    return {
        "distillate": {
            "flow_mol_s": float(res.distillate.total),
            "composition": [float(v) for v in res.distillate.z],
            "temperature_k": float(res.distillate.t),
        },
        "bottoms": {
            "flow_mol_s": float(res.bottoms.total),
            "composition": [float(v) for v in res.bottoms.z],
            "temperature_k": float(res.bottoms.t),
        },
        "condenser_duty_w": float(res.condenser_duty),
        "reboiler_duty_w": float(res.reboiler_duty),
        "top_temperature_k": float(res.t[0]),
        "bottom_temperature_k": float(res.t[-1]),
    }


def _safeguarded_newton(
    f: Callable[[float], Any], lo: float, hi: float, iters: int = 40, tol: float = 1e-7
) -> float:
    """Find an increasing scalar's root in ``[lo, hi]`` by gradient (Newton) + bisection.

    Uses ``jax.value_and_grad`` for the Newton step and falls back to bisection when
    the step leaves the bracket or the slope is unusable: robust and still
    gradient-driven where the function is well behaved.
    """
    lo, hi = float(lo), float(hi)
    x = 0.5 * (lo + hi)
    for _ in range(iters):
        val, grad = jax.value_and_grad(f)(x)
        val, grad = float(val), float(grad)
        if val > 0.0:
            hi = x
        else:
            lo = x
        if abs(val) < tol:
            break
        step_ok = grad != 0.0 and math.isfinite(grad)
        x_newton = x - val / grad if step_ok else math.inf
        x = x_newton if (step_ok and lo < x_newton < hi) else 0.5 * (lo + hi)
    return x


def _optimize_flash_temperature(
    components: list[str],
    z: list[float],
    pressure: float,
    target_vapor_fraction: float,
    t_min: float = 100.0,
    t_max: float = 800.0,
) -> JsonDict:
    arr = component_arrays(components)
    z_arr = jnp.asarray(z)

    def beta_of_t(t: float) -> Any:
        result = flash_pt(PR, t, pressure, z_arr, arr["tc"], arr["pc"], arr["omega"])
        return result.beta

    t_opt = _safeguarded_newton(lambda t: beta_of_t(t) - target_vapor_fraction, t_min, t_max)
    return {
        "temperature_k": t_opt,
        "achieved_vapor_fraction": float(beta_of_t(t_opt)),
        "target_vapor_fraction": target_vapor_fraction,
    }


def _optimize_column_reflux(
    components: list[str],
    z: list[float],
    feed_flow: float,
    feed_temperature: float,
    pressure: float,
    n_stages: int,
    feed_stage: int,
    distillate_rate: float,
    light_key: int,
    target_purity: float,
    q: float = 1.0,
    iters: int = 8,
) -> JsonDict:
    feed = _stream(components, z, feed_flow, feed_temperature, pressure)

    def purity_of_reflux(reflux: float) -> Any:
        res = solve_column(feed, n_stages, feed_stage, reflux, distillate_rate, q=q)
        return res.distillate.z[light_key]

    reflux_opt = _safeguarded_newton(
        lambda r: purity_of_reflux(r) - target_purity, 0.2, 25.0, iters=iters
    )
    return {
        "reflux_ratio": reflux_opt,
        "achieved_purity": float(purity_of_reflux(reflux_opt)),
        "target_purity": target_purity,
    }


_ACTIVITY_METHODS = ("nrtl", "uniquac", "unifac", "dortmund")


def _gamma_phi(components: list[str], method: str) -> Any:
    """Build a gamma-phi model for ``components`` by activity ``method`` name."""
    key = method.lower()
    if key == "nrtl":
        return nrtl_model_for(components)
    if key == "uniquac":
        return uniquac_model_for(components)
    if key == "unifac":
        return unifac_model_for(components)
    if key in ("dortmund", "modified_unifac"):
        return unifac_model_for(components, dortmund=True)
    raise ValueError(f"unknown method {method!r}; use one of {_ACTIVITY_METHODS}")


def _ln_gamma_values(components: list[str], x: list[float], temperature: float, method: str) -> Any:
    """Log activity coefficients via the chosen activity method."""
    model = _gamma_phi(components, method)
    return model.activity.ln_gamma(jnp.asarray(x), float(temperature))


def _activity_coefficients(
    components: list[str], x: list[float], temperature: float, method: str = "unifac"
) -> JsonDict:
    ln_g = _ln_gamma_values(components, x, temperature, method)
    return {
        "components": list(components),
        "x": [float(v) for v in x],
        "temperature_k": float(temperature),
        "method": method,
        "activity_coefficients": [float(jnp.exp(v)) for v in ln_g],
        "ln_gamma": [float(v) for v in ln_g],
    }


def _vle_diagram(
    components: list[str],
    method: str = "nrtl",
    kind: str = "Pxy",
    temperature: float | None = None,
    pressure: float | None = None,
    points: int = 21,
    t_min: float = 200.0,
    t_max: float = 600.0,
) -> JsonDict:
    if len(components) != 2:
        raise ValueError("vle_diagram requires exactly two components")
    model = _gamma_phi(components, method)
    kind_l = kind.lower()
    if kind_l == "pxy":
        if temperature is None:
            raise ValueError("a Pxy diagram requires 'temperature'")
        pxy = pxy_diagram(model, float(temperature), n=int(points))
        return {
            "kind": "Pxy",
            "components": list(components),
            "method": method,
            "temperature_k": float(temperature),
            "x1": [float(v) for v in pxy.x1],
            "y1": [float(v) for v in pxy.y1],
            "pressure_pa": [float(v) for v in pxy.p],
        }
    if kind_l == "txy":
        if pressure is None:
            raise ValueError("a Txy diagram requires 'pressure'")
        txy = txy_diagram(model, float(pressure), n=int(points), t_min=t_min, t_max=t_max)
        return {
            "kind": "Txy",
            "components": list(components),
            "method": method,
            "pressure_pa": float(pressure),
            "x1": [float(v) for v in txy.x1],
            "y1": [float(v) for v in txy.y1],
            "temperature_k": [float(v) for v in txy.t],
        }
    raise ValueError("kind must be 'Pxy' or 'Txy'")


def _find_azeotrope(
    components: list[str],
    method: str = "nrtl",
    temperature: float | None = None,
    pressure: float | None = None,
    t_min: float = 200.0,
    t_max: float = 600.0,
) -> JsonDict:
    if len(components) != 2:
        raise ValueError("find_azeotrope requires exactly two components")
    model = _gamma_phi(components, method)
    if temperature is not None:
        az = azeotrope_pressure(model, float(temperature))
    elif pressure is not None:
        az = azeotrope_temperature(model, float(pressure), t_min=t_min, t_max=t_max)
    else:
        raise ValueError("provide exactly one of 'temperature' or 'pressure'")
    x1 = float(az.x1)
    return {
        "components": list(components),
        "method": method,
        "exists": bool(az.exists),
        "x_azeotrope": [x1, 1.0 - x1],
        "temperature_k": float(az.t),
        "pressure_pa": float(az.p),
    }


def _liquid_liquid_split(
    components: list[str], z: list[float], temperature: float, method: str = "unifac"
) -> JsonDict:
    model = _gamma_phi(components, method)
    t = float(temperature)
    z_arr = jnp.asarray(z)
    stab = liquid_stability(model.activity, t, z_arr)
    res = flash_lle(model.activity, t, z_arr)
    return {
        "components": list(components),
        "temperature_k": t,
        "method": method,
        "splits_into_two_liquids": (not bool(stab.stable)),
        "tangent_plane_distance": float(stab.tpd),
        "phase_I": {
            "fraction": float(1.0 - res.psi),
            "composition": [float(v) for v in res.x_i],
        },
        "phase_II": {
            "fraction": float(res.psi),
            "composition": [float(v) for v in res.x_ii],
        },
    }


def _three_phase_flash(
    components: list[str],
    z: list[float],
    temperature: float,
    pressure: float,
    method: str = "nrtl",
) -> JsonDict:
    model = _gamma_phi(components, method)
    res = flash_vlle(
        model.activity,
        float(temperature),
        float(pressure),
        jnp.asarray(z),
        model.tc,
        model.pc,
        model.omega,
        eos=model.eos,
        kij=model.kij,
        vapor=model.vapor,
        poynting=model.poynting,
        phi_saturation=model.phi_saturation,
    )
    return {
        "components": list(components),
        "method": method,
        "three_phase": bool(res.three_phase),
        "vapor": {"fraction": float(res.beta_v), "composition": [float(v) for v in res.y]},
        "liquid_I": {"fraction": float(res.beta_l1), "composition": [float(v) for v in res.x_i]},
        "liquid_II": {"fraction": float(res.beta_l2), "composition": [float(v) for v in res.x_ii]},
    }


def _residue_curve_map(
    components: list[str],
    pressure: float,
    method: str = "unifac",
    starts: list[list[float]] | None = None,
    steps: int = 60,
    t_min: float = 250.0,
    t_max: float = 500.0,
    max_points: int = 41,
) -> JsonDict:
    """Residue-curve map for a ternary at fixed pressure (simple-distillation paths).

    Each curve is the still-pot liquid composition during open distillation; the
    family of curves (and the vertices/azeotropes they emanate from or collapse to)
    is the master diagram for ternary distillation sequencing. Curves are thinned to
    ``max_points`` for a compact payload.
    """
    if len(components) != 3:
        raise ValueError("residue_curve_map requires exactly three components")
    model = _gamma_phi(components, method)
    if starts is None:
        seeds = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.34, 0.33, 0.33],
            [0.45, 0.45, 0.10],
        ]
    else:
        seeds = [list(s) for s in starts]
    curves = residue_curve_map(
        model,
        jnp.asarray(seeds, dtype=float),
        float(pressure),
        steps=int(steps),
        t_min=float(t_min),
        t_max=float(t_max),
    )
    out_curves: list[JsonDict] = []
    for curve in curves:
        n = curve.x.shape[0]
        idx = [round(float(p)) for p in jnp.linspace(0, n - 1, min(int(max_points), n))]
        out_curves.append(
            {
                "x": [[float(v) for v in curve.x[i]] for i in idx],
                "temperature_k": [float(curve.t[i]) for i in idx],
            }
        )
    return {
        "components": list(components),
        "method": method,
        "pressure_pa": float(pressure),
        "curves": out_curves,
    }


def _solvent_screening(
    solute: str, solvents: list[str], temperature: float = 298.15, method: str = "unifac"
) -> JsonDict:
    """Rank candidate solvents by the solute's infinite-dilution activity coefficient.

    A lower ``gamma_inf`` means stronger solute-solvent affinity (higher solubility),
    so the ranking is ascending. Predictive UNIFAC is the sensible default because it
    needs no fitted binary parameters for novel solute/solvent pairs.
    """
    scored: list[tuple[str, float]] = []
    for solvent in solvents:
        ln_g = _ln_gamma_values([solute, solvent], [1e-4, 1.0 - 1e-4], temperature, method)
        scored.append((solvent, float(jnp.exp(ln_g[0]))))
    scored.sort(key=lambda r: r[1])
    return {
        "solute": solute,
        "temperature_k": float(temperature),
        "method": method,
        "ranking": [{"solvent": s, "gamma_inf_solute": g} for s, g in scored],
    }


def _fit_activity_parameters(
    components: list[str],
    x1: list[float],
    temperature: list[float],
    bubble_pressure: list[float],
    alpha: float = 0.3,
    max_iter: int = 80,
) -> JsonDict:
    """Fit binary NRTL ``b`` parameters to isothermal/isobaric bubble-pressure data."""
    if len(components) != 2:
        raise ValueError("fit_activity_parameters supports binary systems")
    arr = component_arrays(components)
    x1_arr = jnp.asarray(x1)
    x = jnp.stack([x1_arr, 1.0 - x1_arr], axis=1)
    t = jnp.asarray(temperature, dtype=float)
    p_exp = jnp.asarray(bubble_pressure, dtype=float)
    model, cost = fit_nrtl_binary(
        t, x, p_exp, arr["tc"], arr["pc"], arr["omega"], alpha=alpha, max_iter=int(max_iter)
    )
    m = x1_arr.shape[0]
    rmse_pa = float((2.0 * float(cost) / m) ** 0.5 * float(jnp.mean(p_exp)))
    return {
        "model": "nrtl",
        "components": list(components),
        "alpha": float(alpha),
        "b12": float(model.b[0, 1]),
        "b21": float(model.b[1, 0]),
        "final_cost": float(cost),
        "rmse_pa": rmse_pa,
    }


def _heat_of_reaction(
    equation: str, components: list[str], temperature: float = 298.15
) -> JsonDict:
    """Standard enthalpy/entropy/Gibbs of reaction and ``K(T)`` for one reaction."""
    rxn = Reaction.parse(equation, components)
    props = reaction_properties(rxn, float(temperature))
    return {
        "equation": equation,
        "components": list(components),
        "temperature_k": float(temperature),
        "delta_h_rxn_j_mol": float(props.delta_h),
        "delta_s_rxn_j_mol_k": float(props.delta_s),
        "delta_g_rxn_j_mol": float(props.delta_g),
        "ln_k": float(props.ln_k),
        "k": float(props.k),
        "exothermic": bool(float(props.delta_h) < 0.0),
    }


def _reaction_equilibrium(
    components: list[str],
    equations: list[str],
    n: list[float],
    temperature: float,
    pressure: float,
    basis: str = "ideal-gas",
) -> JsonDict:
    """Gas-phase chemical-equilibrium composition for one or more reactions."""
    rxns = [Reaction.parse(e, components) for e in equations]
    reactions: Any = rxns[0] if len(rxns) == 1 else rxns
    kwargs: JsonDict = {"basis": basis}
    if basis == "phi":
        arr = component_arrays(components)
        kwargs.update(tc=arr["tc"], pc=arr["pc"], omega=arr["omega"])
    res = _reaction_equilibrium_solve(
        reactions, jnp.asarray(n, dtype=float), float(temperature), float(pressure), **kwargs
    )
    return {
        "components": list(components),
        "equations": list(equations),
        "temperature_k": float(temperature),
        "pressure_pa": float(pressure),
        "basis": basis,
        "extent": [float(v) for v in jnp.atleast_1d(res.extent)],
        "moles": [float(v) for v in res.moles],
        "mole_fractions": [float(v) for v in res.y],
        "k": [float(v) for v in jnp.atleast_1d(res.k)],
    }


def _reaction_conversions(n_in: jnp.ndarray, n_out: jnp.ndarray) -> list[float]:
    """Per-component fractional conversion ``(in - out)/in`` (0 where nothing fed)."""
    out = []
    for a, b in zip(n_in.tolist(), n_out.tolist(), strict=True):
        out.append((a - b) / a if a > 0.0 else 0.0)
    return out


def _reactor(
    components: list[str],
    equation: str,
    n: list[float],
    temperature: float,
    pressure: float,
    mode: str = "equilibrium",
    adiabatic: bool = False,
    t_out: float | None = None,
    conversion: float | None = None,
    extent: list[float] | None = None,
    a: float | None = None,
    ea: float | None = None,
    orders: list[float] | None = None,
    volume: float | None = None,
) -> JsonDict:
    """Single-reaction reactor with material and energy balances.

    ``mode`` selects the model: ``equilibrium`` (chemical-equilibrium outlet),
    ``stoichiometric`` (specified ``conversion`` or ``extent``), or a kinetic
    ``cstr``/``pfr`` (needs power-law ``a``, ``ea``, ``orders`` and a ``volume``).
    Run isothermally at ``t_out`` (default feed T) and get the duty, or set
    ``adiabatic`` and get the outlet temperature.
    """
    rxn = Reaction.parse(equation, components)
    feed = Stream(
        jnp.asarray(n, dtype=float),
        jnp.asarray(temperature),
        jnp.asarray(pressure),
        tuple(components),
    )
    if mode == "equilibrium":
        res = equilibrium_reactor(feed, rxn, t_out=t_out, adiabatic=adiabatic)
    elif mode == "stoichiometric":
        extent_arr = None if extent is None else jnp.asarray(extent, dtype=float)
        res = stoichiometric_reactor(
            feed, rxn, extent=extent_arr, conversion=conversion, t_out=t_out, adiabatic=adiabatic
        )
    elif mode in ("cstr", "pfr"):
        if a is None or ea is None or orders is None or volume is None:
            raise ValueError(f"mode={mode!r} needs power-law 'a', 'ea', 'orders' and 'volume'")
        law = PowerLaw(
            a=jnp.asarray(a), ea=jnp.asarray(ea), orders=jnp.asarray(orders, dtype=float)
        )
        unit = cstr if mode == "cstr" else pfr
        res = unit(feed, rxn, law, float(volume), t_out=t_out, adiabatic=adiabatic)
    else:
        raise ValueError(f"unknown reactor mode {mode!r}")
    return {
        "mode": mode,
        "outlet_flow_mol_s": float(res.outlet.total),
        "outlet_composition": [float(v) for v in res.outlet.z],
        "outlet_temperature_k": float(res.outlet.t),
        "duty_w": float(res.duty),
        "extent": [float(v) for v in jnp.atleast_1d(res.extent)],
        "conversion": _reaction_conversions(feed.n, res.outlet.n),
    }


def _reactive_flash(
    components: list[str],
    equation: str,
    n: list[float],
    temperature: float,
    pressure: float,
    method: str = "nrtl",
) -> JsonDict:
    """Flash with simultaneous chemical and vapour-liquid equilibrium (gamma-phi)."""
    model = _gamma_phi(components, method)
    rxn = Reaction.parse(equation, components)
    feed = Stream(
        jnp.asarray(n, dtype=float),
        jnp.asarray(temperature),
        jnp.asarray(pressure),
        tuple(components),
    )
    res = reactive_flash(feed, rxn, float(temperature), float(pressure), model)
    return {
        "components": list(components),
        "equation": equation,
        "temperature_k": float(temperature),
        "pressure_pa": float(pressure),
        "method": method,
        "vapor_fraction": float(res.beta),
        "extent": [float(v) for v in jnp.atleast_1d(res.extent)],
        "vapor": {
            "flow_mol_s": float(res.vapor.total),
            "composition": [float(v) for v in res.vapor.z],
        },
        "liquid": {
            "flow_mol_s": float(res.liquid.total),
            "composition": [float(v) for v in res.liquid.z],
        },
    }


def _fit_kinetics(temperature: list[float], rate_constant: list[float]) -> JsonDict:
    """Fit Arrhenius ``k = A exp(-Ea/RT)`` to ``(T, k)`` data (log-linear least squares).

    ``ln k = ln A - Ea/(R T)`` is linear in ``(ln A, Ea)``, so the fit is the exact
    closed-form regression; returns ``A``, ``Ea`` and the coefficient of
    determination ``R^2`` of the ``ln k`` fit.
    """
    from fugacio.thermo.constants import R as _R

    t = jnp.asarray(temperature, dtype=float)
    k = jnp.asarray(rate_constant, dtype=float)
    ln_k = jnp.log(k)
    design = jnp.stack([jnp.ones_like(t), -1.0 / (_R * t)], axis=1)
    coeffs, *_ = jnp.linalg.lstsq(design, ln_k, rcond=None)
    ln_a, ea = coeffs
    pred = design @ coeffs
    ss_res = float(jnp.sum((ln_k - pred) ** 2))
    ss_tot = float(jnp.sum((ln_k - jnp.mean(ln_k)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {
        "pre_exponential_a": float(jnp.exp(ln_a)),
        "activation_energy_j_mol": float(ea),
        "r_squared": r_squared,
        "n_points": int(t.shape[0]),
    }


_EQUIPMENT_KINDS = ("heat_exchanger", "pump", "compressor", "vessel", "tray", "fired_heater")


def _flash_sensitivity(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
) -> JsonDict:
    """Exact sensitivities of a flash to temperature and pressure (via autodiff).

    Showcases the differentiable core: the derivatives are computed by
    differentiating straight through the equation-of-state phase equilibrium,
    not by finite differences.
    """
    feed = _stream(components, z, flow, temperature, pressure)
    total = float(feed.total)

    def vapor_fraction(t: float, p: float) -> Any:
        vap, _liq = flash_drum(feed, t, p)
        return vap.total / feed.total

    d_beta_dt = jax.grad(lambda t: vapor_fraction(t, pressure))(temperature)
    d_beta_dp = jax.grad(lambda p: vapor_fraction(temperature, p))(pressure)
    d_flow_dt = jax.grad(lambda t: vapor_fraction(t, pressure) * total)(temperature)
    return {
        "vapor_fraction": float(vapor_fraction(temperature, pressure)),
        "d_vapor_fraction_dT_per_k": float(d_beta_dt),
        "d_vapor_fraction_dP_per_pa": float(d_beta_dp),
        "d_vapor_flow_dT_mol_s_per_k": float(d_flow_dt),
    }


def _equipment_cost(
    kind: str,
    size: float,
    pressure_barg: float = 0.0,
    material: str = "CS",
) -> JsonDict:
    """Turton purchased and installed (bare-module) cost of one equipment item."""
    item = bare_module_cost(kind, size, pressure_barg=pressure_barg, material=material)
    return {
        "kind": kind,
        "size": float(size),
        "material": material,
        "purchased_usd": float(item.purchased),
        "bare_module_usd": float(item.bare_module),
    }


def _heat_exchanger_cost(
    duty: float,
    u: float,
    dt_hot: float,
    dt_cold: float,
    pressure_barg: float = 0.0,
    material: str = "CS",
) -> JsonDict:
    """Size a heat exchanger from its duty and approaches, then cost it (Turton)."""
    area = heat_exchanger_area(duty, u, dt_hot, dt_cold)
    item = bare_module_cost("heat_exchanger", area, pressure_barg=pressure_barg, material=material)
    return {
        "area_m2": float(area),
        "lmtd_k": float(lmtd(dt_hot, dt_cold)),
        "purchased_usd": float(item.purchased),
        "bare_module_usd": float(item.bare_module),
    }


def _utility_cost(duty: float, utility: str, hours_per_year: float = 8000.0) -> JsonDict:
    """Annual cost of a heating/cooling duty for a named utility."""
    annual = utility_cost(duty, utility, hours_per_year=hours_per_year)
    return {
        "utility": utility,
        "duty_w": float(duty),
        "hours_per_year": float(hours_per_year),
        "annual_cost_usd": float(annual),
    }


def _annual_cost(
    capex: float,
    opex: float,
    interest_rate: float = 0.1,
    years: float = 10.0,
) -> JsonDict:
    """Annualized capital and total annual cost (TAC) for a capex/opex split."""
    return {
        "annualized_capital_usd_per_year": float(
            annualized_capital(capex, rate=interest_rate, years=years)
        ),
        "operating_cost_usd_per_year": float(opex),
        "total_annual_cost_usd_per_year": float(
            total_annual_cost(capex, opex, rate=interest_rate, years=years)
        ),
        "capital_recovery_factor": float(capital_recovery_factor(interest_rate, years)),
    }


def _net_present_value(cash_flows: list[float], discount_rate: float = 0.1) -> JsonDict:
    """Net present value of a yearly cash-flow stream (index 0 is now)."""
    return {
        "npv_usd": float(npv(jnp.asarray(cash_flows), rate=discount_rate)),
        "discount_rate": float(discount_rate),
        "periods": len(cash_flows),
    }


def _size_column(
    vapor_molar_flow: float,
    temperature: float,
    pressure: float,
    n_stages: int,
    molar_mass: float = 0.06,
    liquid_density: float = 700.0,
    pressure_barg: float = 0.0,
    material: str = "CS",
) -> JsonDict:
    """Size a distillation column (diameter, height) and cost the shell + trays."""
    vapor_density = float(molar_mass / vapor_molar_volume_ideal(temperature, pressure))
    diameter = column_diameter(
        vapor_molar_flow, vapor_density, liquid_density, molar_mass=molar_mass
    )
    height = column_height(n_stages)
    shell_volume = cylinder_volume(diameter, height)
    shell = bare_module_cost("vessel", shell_volume, pressure_barg=pressure_barg, material=material)
    tray_area = float(jnp.pi * diameter**2 / 4.0)
    tray = bare_module_cost("tray", tray_area, material=material)
    trays_total = float(tray.bare_module) * n_stages
    return {
        "diameter_m": float(diameter),
        "height_m": float(height),
        "shell_volume_m3": float(shell_volume),
        "shell_cost_usd": float(shell.bare_module),
        "trays_cost_usd": trays_total,
        "installed_cost_usd": float(shell.bare_module) + trays_total,
    }


def _finite(value: float) -> float | None:
    """JSON-safe float: ``nan`` (e.g. quality outside the dome) becomes ``None``."""
    number = float(value)
    return None if math.isnan(number) else number


def _fluid_state_dict(fluid: object, state: object) -> JsonDict:
    """Serialize a `fugacio.thermo.FluidState` on a mass basis (steam-table units)."""
    assert isinstance(fluid, _helmholtz.HelmholtzFluid)
    assert isinstance(state, _helmholtz.FluidState)
    m = fluid.molar_mass
    return {
        "temperature_k": float(state.t),
        "pressure_pa": float(state.p),
        "density_kg_m3": float(state.rho) * m,
        "compressibility_factor": float(state.z),
        "enthalpy_kj_kg": float(state.h) / m / 1e3,
        "entropy_kj_kg_k": float(state.s) / m / 1e3,
        "internal_energy_kj_kg": float(state.u) / m / 1e3,
        "cp_kj_kg_k": _finite(float(state.cp) / m / 1e3),
        "cv_kj_kg_k": _finite(float(state.cv) / m / 1e3),
        "speed_of_sound_m_s": _finite(float(state.w)),
        "quality": _finite(float(state.q)),
        "two_phase": bool(state.two_phase),
    }


def _steam_state(
    pressure: float,
    temperature: float | None = None,
    quality: float | None = None,
    enthalpy_kj_kg: float | None = None,
    entropy_kj_kg_k: float | None = None,
) -> JsonDict:
    """IAPWS-95 steam-table lookup at a pressure plus one other specification."""
    water = _helmholtz.reference_fluid("water")
    given = [v is not None for v in (temperature, quality, enthalpy_kj_kg, entropy_kj_kg_k)]
    if sum(given) != 1:
        raise ValueError(
            "specify exactly one of temperature, quality, enthalpy_kj_kg, entropy_kj_kg_k"
        )
    m = water.molar_mass
    if temperature is not None:
        state = _helmholtz.state_tp(water, float(temperature), float(pressure))
    elif quality is not None:
        state = _helmholtz.state_pq(water, float(pressure), float(quality))
    elif enthalpy_kj_kg is not None:
        state = _helmholtz.state_ph(water, float(pressure), float(enthalpy_kj_kg) * 1e3 * m)
    else:
        assert entropy_kj_kg_k is not None
        state = _helmholtz.state_ps(water, float(pressure), float(entropy_kj_kg_k) * 1e3 * m)
    result = _fluid_state_dict(water, state)
    if not bool(state.two_phase):
        result["viscosity_pa_s"] = float(_helmholtz.water_viscosity(state.t, state.rho))
        result["thermal_conductivity_w_m_k"] = float(
            _helmholtz.water_thermal_conductivity(state.t, state.rho)
        )
    return result


def _reference_fluid_state(fluid: str, temperature: float, pressure: float) -> JsonDict:
    """Reference-EOS single-phase state for any vendored fluid."""
    ref = _helmholtz.reference_fluid(fluid)
    state = _helmholtz.state_tp(ref, float(temperature), float(pressure))
    result = _fluid_state_dict(ref, state)
    result["fluid"] = ref.name
    result["equation_of_state"] = ref.bibtex_eos
    if ref.name == "water":
        result["viscosity_pa_s"] = float(_helmholtz.water_viscosity(state.t, state.rho))
        result["thermal_conductivity_w_m_k"] = float(
            _helmholtz.water_thermal_conductivity(state.t, state.rho)
        )
    return result


def _reference_saturation(
    fluid: str, temperature: float | None = None, pressure: float | None = None
) -> JsonDict:
    """Reference-EOS saturation state at a temperature or a pressure."""
    ref = _helmholtz.reference_fluid(fluid)
    if (temperature is None) == (pressure is None):
        raise ValueError("specify exactly one of temperature or pressure")
    if temperature is not None:
        sat = _helmholtz.saturation_state(ref, t=float(temperature))
    else:
        assert pressure is not None
        sat = _helmholtz.saturation_state(ref, p=float(pressure))
    m = ref.molar_mass
    return {
        "fluid": ref.name,
        "temperature_k": float(sat.t),
        "pressure_pa": float(sat.p),
        "liquid_density_kg_m3": float(sat.rho_liquid) * m,
        "vapor_density_kg_m3": float(sat.rho_vapor) * m,
        "heat_of_vaporization_kj_kg": float(sat.h_vaporization) / m / 1e3,
        "heat_of_vaporization_j_mol": float(sat.h_vaporization),
        "surface_tension_n_m": float(_helmholtz.surface_tension(ref, sat.t)),
        "critical_temperature_k": ref.t_critical,
        "critical_pressure_pa": ref.p_critical,
    }


_STEAM_UTILITIES = {"lp_steam": "lp", "mp_steam": "mp", "hp_steam": "hp"}


def _steam_utility_requirements(
    duty: float, utility: str = "lp_steam", hours_per_year: float = 8000.0
) -> JsonDict:
    """Physical utility sizing (IAPWS-95) plus the priced annual cost."""
    if utility not in (*_STEAM_UTILITIES, "cooling_water"):
        raise ValueError(
            f"unknown utility {utility!r}; use one of "
            f"{sorted([*_STEAM_UTILITIES, 'cooling_water'])}"
        )
    result: JsonDict = {
        "utility": utility,
        "duty_w": abs(float(duty)),
        "annual_cost_usd": float(utility_cost(duty, utility, hours_per_year=hours_per_year)),
    }
    if utility in _STEAM_UTILITIES:
        level = STEAM_LEVELS[_STEAM_UTILITIES[utility]]
        steam = steam_heating(duty, pressure=level)
        result |= {
            "pressure_pa": float(steam.p),
            "steam_mass_flow_kg_s": float(steam.mass_flow),
            "steam_temperature_k": float(steam.t_steam),
            "condensate_temperature_k": float(steam.t_condensate),
            "latent_heat_kj_kg": float(steam.dh_specific) / 1e3,
        }
    else:
        water = cooling_water(duty)
        result |= {
            "water_mass_flow_kg_s": float(water.mass_flow),
            "supply_temperature_k": float(water.t_supply),
            "return_temperature_k": float(water.t_return),
        }
    return result


def _steam_turbine_tool(
    mass_flow_kg_s: float,
    inlet_pressure_pa: float,
    inlet_temperature_k: float,
    outlet_pressure_pa: float,
    isentropic_efficiency: float = 0.75,
) -> JsonDict:
    """Steam-turbine expansion through IAPWS-95 with an isentropic efficiency."""
    water = _helmholtz.reference_fluid("water")
    m = water.molar_mass
    result = steam_turbine(
        float(mass_flow_kg_s),
        p_in=float(inlet_pressure_pa),
        t_in=float(inlet_temperature_k),
        p_out=float(outlet_pressure_pa),
        isentropic_efficiency=float(isentropic_efficiency),
    )
    return {
        "power_w": float(result.power),
        "mass_flow_kg_s": float(result.mass_flow),
        "outlet_temperature_k": float(result.t_out),
        "outlet_quality": _finite(float(result.q_out)),
        "outlet_two_phase": bool(result.two_phase),
        "inlet_enthalpy_kj_kg": float(result.h_in) / m / 1e3,
        "outlet_enthalpy_kj_kg": float(result.h_out) / m / 1e3,
        "isentropic_outlet_enthalpy_kj_kg": float(result.h_out_isentropic) / m / 1e3,
    }


def default_registry() -> dict[str, ToolSpec]:
    """Return the built-in tool registry keyed by tool name."""
    specs = [
        ToolSpec(
            name="list_components",
            description="List the components available in the Fugacio database.",
            parameters={"type": "object", "properties": {}, "required": []},
            run=_list_components,
        ),
        ToolSpec(
            name="component_properties",
            description="Look up critical constants and basic properties of one component.",
            parameters={
                "type": "object",
                "properties": {"component": {"type": "string"}},
                "required": ["component"],
            },
            run=_component_properties,
        ),
        ToolSpec(
            name="physical_properties",
            description=(
                "Physical and transport properties of a mixture at T (and P): liquid "
                "and vapour density, viscosity, thermal conductivity, surface tension, "
                "heat of vaporization, and liquid heat capacity (the inputs for "
                "equipment sizing and heat-transfer calculations)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "x": {"type": "array", "items": {"type": "number"}},
                    "temperature": {"type": "number", "description": "Temperature (K)"},
                    "pressure": {
                        "type": "number",
                        "description": "Pressure (Pa) for the vapour density; default 1 atm",
                    },
                },
                "required": ["components", "x", "temperature"],
            },
            run=_physical_properties,
        ),
        ToolSpec(
            name="binary_diffusivity",
            description=(
                "Binary diffusion coefficients of a solute in a solvent: gas-phase "
                "D_AB by Fuller at (T, P) and infinite-dilution liquid D_AB by "
                "Wilke-Chang at T (mass-transfer and column-efficiency inputs)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "solute": {"type": "string"},
                    "solvent": {"type": "string"},
                    "temperature": {"type": "number", "description": "Temperature (K)"},
                    "pressure": {
                        "type": "number",
                        "description": "Pressure (Pa) for the gas-phase value; default 1 atm",
                    },
                },
                "required": ["solute", "solvent", "temperature"],
            },
            run=_binary_diffusivity,
        ),
        ToolSpec(
            name="saturation_pressure",
            description="Compute the pure-component saturation pressure (Pa) at a temperature.",
            parameters={
                "type": "object",
                "properties": {
                    "component": {"type": "string"},
                    "temperature": {"type": "number", "description": "Temperature in K"},
                },
                "required": ["component", "temperature"],
            },
            run=_saturation_pressure,
        ),
        ToolSpec(
            name="bubble_pressure",
            description="Bubble-point pressure and vapor composition of a liquid mixture.",
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "x": {"type": "array", "items": {"type": "number"}},
                    "temperature": {"type": "number"},
                },
                "required": ["components", "x", "temperature"],
            },
            run=_bubble_pressure,
        ),
        ToolSpec(
            name="flash_drum",
            description="Isothermal-isobaric flash of a feed; returns vapor/liquid products.",
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number", "description": "Total feed flow (mol/s)"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                },
                "required": ["components", "z", "flow", "temperature", "pressure"],
            },
            run=_flash,
        ),
        ToolSpec(
            name="heat_exchanger",
            description=(
                "Heat or cool a stream. Provide exactly one of t_out (target outlet "
                "temperature, K) or duty (signed heat added, W); returns the other."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "t_out": {"type": "number", "description": "Target outlet temperature (K)"},
                    "duty": {"type": "number", "description": "Signed heat added (W)"},
                },
                "required": ["components", "z", "flow", "temperature", "pressure"],
            },
            run=_heat_exchanger,
        ),
        ToolSpec(
            name="compressor",
            description=(
                "Isentropic-efficiency compressor or turbine. Set machine='turbine' to "
                "expand; returns outlet temperature and shaft work."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "pressure_out": {"type": "number"},
                    "efficiency": {"type": "number"},
                    "machine": {"type": "string", "enum": ["compressor", "turbine"]},
                },
                "required": ["components", "z", "flow", "temperature", "pressure", "pressure_out"],
            },
            run=_compressor,
        ),
        ToolSpec(
            name="pump",
            description=(
                "Pump an incompressible liquid to a higher pressure; returns work and outlet T."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "pressure_out": {"type": "number"},
                    "efficiency": {"type": "number"},
                },
                "required": ["components", "z", "flow", "temperature", "pressure", "pressure_out"],
            },
            run=_pump,
        ),
        ToolSpec(
            name="valve",
            description=(
                "Isenthalpic (Joule-Thomson) pressure letdown; returns the outlet temperature."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "pressure_out": {"type": "number"},
                },
                "required": ["components", "z", "flow", "temperature", "pressure", "pressure_out"],
            },
            run=_valve,
        ),
        ToolSpec(
            name="shortcut_distillation",
            description=(
                "Fenske-Underwood-Gilliland shortcut design from key recoveries: returns "
                "minimum stages, minimum reflux, actual stages, and feed location."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "feed_flow": {"type": "number"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "light_key": {"type": "integer", "description": "Index of the light key"},
                    "heavy_key": {"type": "integer", "description": "Index of the heavy key"},
                    "lk_recovery": {"type": "number", "description": "Light-key recovery to top"},
                    "hk_recovery": {"type": "number", "description": "Heavy-key recovery to top"},
                    "q": {"type": "number", "description": "Feed quality (1 = sat. liquid)"},
                },
                "required": [
                    "components",
                    "z",
                    "feed_flow",
                    "temperature",
                    "pressure",
                    "light_key",
                    "heavy_key",
                ],
            },
            run=_shortcut_distillation,
        ),
        ToolSpec(
            name="rigorous_distillation",
            description=(
                "Rigorous multistage column (Wang-Henke, constant molar overflow) with EOS "
                "K-values; returns product compositions, temperatures, and column duties."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "feed_flow": {"type": "number"},
                    "feed_temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "n_stages": {"type": "integer"},
                    "feed_stage": {"type": "integer"},
                    "reflux": {"type": "number"},
                    "distillate_rate": {"type": "number"},
                    "q": {"type": "number"},
                },
                "required": [
                    "components",
                    "z",
                    "feed_flow",
                    "feed_temperature",
                    "pressure",
                    "n_stages",
                    "feed_stage",
                    "reflux",
                    "distillate_rate",
                ],
            },
            run=_rigorous_distillation,
        ),
        ToolSpec(
            name="optimize_flash_temperature",
            description=(
                "Gradient-based solve for the flash temperature (K) that hits a target vapor "
                "fraction, differentiating the equilibrium flash."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "pressure": {"type": "number"},
                    "target_vapor_fraction": {"type": "number"},
                },
                "required": ["components", "z", "pressure", "target_vapor_fraction"],
            },
            run=_optimize_flash_temperature,
        ),
        ToolSpec(
            name="optimize_column_reflux",
            description=(
                "Gradient-based solve for the reflux ratio that achieves a target distillate "
                "purity of the light key, differentiating through the rigorous column."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "feed_flow": {"type": "number"},
                    "feed_temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "n_stages": {"type": "integer"},
                    "feed_stage": {"type": "integer"},
                    "distillate_rate": {"type": "number"},
                    "light_key": {"type": "integer"},
                    "target_purity": {"type": "number"},
                    "q": {"type": "number"},
                },
                "required": [
                    "components",
                    "z",
                    "feed_flow",
                    "feed_temperature",
                    "pressure",
                    "n_stages",
                    "feed_stage",
                    "distillate_rate",
                    "light_key",
                    "target_purity",
                ],
            },
            run=_optimize_column_reflux,
        ),
        ToolSpec(
            name="activity_coefficients",
            description=(
                "Liquid-phase activity coefficients (non-ideality) at a composition and "
                "temperature using NRTL/UNIQUAC (curated) or UNIFAC/Dortmund (predictive)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "x": {"type": "array", "items": {"type": "number"}},
                    "temperature": {"type": "number"},
                    "method": {"type": "string", "enum": list(_ACTIVITY_METHODS)},
                },
                "required": ["components", "x", "temperature"],
            },
            run=_activity_coefficients,
        ),
        ToolSpec(
            name="vle_diagram",
            description=(
                "Binary P-x-y (fixed T) or T-x-y (fixed P) phase-envelope data from a "
                "gamma-phi model; returns the liquid x1, vapour y1, and P or T arrays."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "method": {"type": "string", "enum": list(_ACTIVITY_METHODS)},
                    "kind": {"type": "string", "enum": ["Pxy", "Txy"]},
                    "temperature": {"type": "number", "description": "Required for Pxy (K)"},
                    "pressure": {"type": "number", "description": "Required for Txy (Pa)"},
                    "points": {"type": "integer", "description": "Grid resolution"},
                },
                "required": ["components"],
            },
            run=_vle_diagram,
        ),
        ToolSpec(
            name="find_azeotrope",
            description=(
                "Locate a binary azeotrope (where vapour = liquid composition) at a fixed "
                "temperature or pressure; reports whether one exists and its composition."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "method": {"type": "string", "enum": list(_ACTIVITY_METHODS)},
                    "temperature": {"type": "number", "description": "Fix T (K); finds P"},
                    "pressure": {"type": "number", "description": "Fix P (Pa); finds T"},
                },
                "required": ["components"],
            },
            run=_find_azeotrope,
        ),
        ToolSpec(
            name="liquid_liquid_split",
            description=(
                "Test a liquid mixture for a miscibility gap and, if it splits, return the "
                "two conjugate liquid compositions and phase fractions (decanter design)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "temperature": {"type": "number"},
                    "method": {"type": "string", "enum": list(_ACTIVITY_METHODS)},
                },
                "required": ["components", "z", "temperature"],
            },
            run=_liquid_liquid_split,
        ),
        ToolSpec(
            name="three_phase_flash",
            description=(
                "Vapour-liquid-liquid (V-L-L) flash for heterogeneous systems; returns the "
                "vapour and two liquid phases with their fractions and compositions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "method": {"type": "string", "enum": list(_ACTIVITY_METHODS)},
                },
                "required": ["components", "z", "temperature", "pressure"],
            },
            run=_three_phase_flash,
        ),
        ToolSpec(
            name="residue_curve_map",
            description=(
                "Ternary residue-curve map at fixed pressure: simple-distillation liquid "
                "paths (and their boiling temperatures) used to sequence distillation and "
                "spot distillation boundaries. Defaults to predictive UNIFAC."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "pressure": {"type": "number", "description": "Fixed pressure (Pa)"},
                    "method": {"type": "string", "enum": list(_ACTIVITY_METHODS)},
                    "starts": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Optional list of starting compositions (each sums to 1)",
                    },
                    "steps": {"type": "integer", "description": "Steps per direction"},
                    "t_min": {"type": "number"},
                    "t_max": {"type": "number"},
                },
                "required": ["components", "pressure"],
            },
            run=_residue_curve_map,
        ),
        ToolSpec(
            name="solvent_screening",
            description=(
                "Rank candidate solvents for a solute by the solute's infinite-dilution "
                "activity coefficient (lower = more soluble); predictive UNIFAC by default."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "solute": {"type": "string"},
                    "solvents": {"type": "array", "items": {"type": "string"}},
                    "temperature": {"type": "number"},
                    "method": {"type": "string", "enum": list(_ACTIVITY_METHODS)},
                },
                "required": ["solute", "solvents"],
            },
            run=_solvent_screening,
        ),
        ToolSpec(
            name="fit_activity_parameters",
            description=(
                "Regress binary NRTL interaction parameters (b12, b21) to measured "
                "bubble-pressure data by differentiable least squares; returns the fit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "x1": {"type": "array", "items": {"type": "number"}},
                    "temperature": {"type": "array", "items": {"type": "number"}},
                    "bubble_pressure": {"type": "array", "items": {"type": "number"}},
                    "alpha": {"type": "number", "description": "NRTL non-randomness (fixed)"},
                },
                "required": ["components", "x1", "temperature", "bubble_pressure"],
            },
            run=_fit_activity_parameters,
        ),
        ToolSpec(
            name="heat_of_reaction",
            description=(
                "Standard enthalpy, entropy and Gibbs energy of reaction and the "
                "equilibrium constant K(T) for a reaction equation, from ideal-gas "
                "formation data (e.g. 'nitrogen + 3 hydrogen = 2 ammonia')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "Reaction, sides split by '='"},
                    "components": {"type": "array", "items": {"type": "string"}},
                    "temperature": {"type": "number", "description": "Temperature (K)"},
                },
                "required": ["equation", "components"],
            },
            run=_heat_of_reaction,
        ),
        ToolSpec(
            name="reaction_equilibrium",
            description=(
                "Gas-phase chemical-equilibrium composition for one or more reactions "
                "at (T, P) from a feed (moles). Use basis='phi' for EOS fugacity "
                "corrections at high pressure."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "equations": {"type": "array", "items": {"type": "string"}},
                    "n": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Feed amounts per component (mol)",
                    },
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "basis": {"type": "string", "enum": ["ideal-gas", "phi"]},
                },
                "required": ["components", "equations", "n", "temperature", "pressure"],
            },
            run=_reaction_equilibrium,
        ),
        ToolSpec(
            name="reactor",
            description=(
                "Single-reaction reactor with energy balance. mode='equilibrium' "
                "(equilibrium outlet), 'stoichiometric' (give conversion or extent), "
                "or 'cstr'/'pfr' (give power-law a, ea, orders and volume). Isothermal "
                "(t_out) returns duty; adiabatic=true returns the outlet temperature."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "equation": {"type": "string"},
                    "n": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Feed amounts/flows per component",
                    },
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "mode": {
                        "type": "string",
                        "enum": ["equilibrium", "stoichiometric", "cstr", "pfr"],
                    },
                    "adiabatic": {"type": "boolean"},
                    "t_out": {"type": "number"},
                    "conversion": {
                        "type": "number",
                        "description": "Limiting-reactant conversion (stoichiometric)",
                    },
                    "extent": {"type": "array", "items": {"type": "number"}},
                    "a": {"type": "number", "description": "Power-law pre-exponential (cstr/pfr)"},
                    "ea": {"type": "number", "description": "Activation energy J/mol (cstr/pfr)"},
                    "orders": {"type": "array", "items": {"type": "number"}},
                    "volume": {"type": "number", "description": "Reactor volume m^3 (cstr/pfr)"},
                },
                "required": ["components", "equation", "n", "temperature", "pressure"],
            },
            run=_reactor,
        ),
        ToolSpec(
            name="reactive_flash",
            description=(
                "Flash with simultaneous chemical and vapour-liquid equilibrium for a "
                "non-ideal (gamma-phi) liquid, e.g. esterification. Returns the V/L "
                "split, product compositions and the reaction extent."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "equation": {"type": "string"},
                    "n": {"type": "array", "items": {"type": "number"}},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "method": {"type": "string", "enum": list(_ACTIVITY_METHODS)},
                },
                "required": ["components", "equation", "n", "temperature", "pressure"],
            },
            run=_reactive_flash,
        ),
        ToolSpec(
            name="fit_kinetics",
            description=(
                "Fit Arrhenius parameters (pre-exponential A, activation energy Ea) to "
                "measured rate constants k(T) by exact log-linear least squares."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "temperature": {"type": "array", "items": {"type": "number"}},
                    "rate_constant": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["temperature", "rate_constant"],
            },
            run=_fit_kinetics,
        ),
        ToolSpec(
            name="flash_sensitivity",
            description=(
                "Exact sensitivities of an isothermal-isobaric flash to temperature "
                "and pressure, differentiated straight through the equation-of-state "
                "phase equilibrium (autodiff, not finite differences)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number", "description": "Total feed flow (mol/s)"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                },
                "required": ["components", "z", "flow", "temperature", "pressure"],
            },
            run=_flash_sensitivity,
        ),
        ToolSpec(
            name="equipment_cost",
            description=(
                "Turton purchased and installed (bare-module) cost of one equipment "
                f"item. kind is one of {list(_EQUIPMENT_KINDS)}; size is the capacity "
                "attribute (area m^2 for heat_exchanger/tray, power kW for pump/"
                "compressor, volume m^3 for vessel, duty kW for fired_heater)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(_EQUIPMENT_KINDS)},
                    "size": {"type": "number"},
                    "pressure_barg": {"type": "number"},
                    "material": {"type": "string", "enum": ["CS", "SS", "Ni", "Ti"]},
                },
                "required": ["kind", "size"],
            },
            run=_equipment_cost,
        ),
        ToolSpec(
            name="heat_exchanger_cost",
            description=(
                "Size a heat exchanger from its duty, overall U and terminal "
                "temperature approaches (via the log-mean DT), then cost it (Turton)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "duty": {"type": "number", "description": "Heat duty (W)"},
                    "u": {"type": "number", "description": "Overall U (W/m^2/K)"},
                    "dt_hot": {"type": "number", "description": "Hot-end approach (K)"},
                    "dt_cold": {"type": "number", "description": "Cold-end approach (K)"},
                    "pressure_barg": {"type": "number"},
                    "material": {"type": "string", "enum": ["CS", "SS", "Ni", "Ti"]},
                },
                "required": ["duty", "u", "dt_hot", "dt_cold"],
            },
            run=_heat_exchanger_cost,
        ),
        ToolSpec(
            name="utility_cost",
            description=(
                "Annual cost of a heating or cooling duty for a named utility "
                "(cooling_water, chilled_water, refrigeration, lp/mp/hp_steam, "
                "fired_heat, electricity)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "duty": {"type": "number", "description": "Duty (W)"},
                    "utility": {"type": "string"},
                    "hours_per_year": {"type": "number"},
                },
                "required": ["duty", "utility"],
            },
            run=_utility_cost,
        ),
        ToolSpec(
            name="annual_cost",
            description=(
                "Annualized capital and total annual cost (TAC = CRF*CAPEX + OPEX) "
                "from a capital/operating split, interest rate and project life."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "capex": {"type": "number", "description": "Installed capital ($)"},
                    "opex": {"type": "number", "description": "Operating cost ($/yr)"},
                    "interest_rate": {"type": "number"},
                    "years": {"type": "number"},
                },
                "required": ["capex", "opex"],
            },
            run=_annual_cost,
        ),
        ToolSpec(
            name="net_present_value",
            description="Net present value of a yearly cash-flow stream (index 0 is now).",
            parameters={
                "type": "object",
                "properties": {
                    "cash_flows": {"type": "array", "items": {"type": "number"}},
                    "discount_rate": {"type": "number"},
                },
                "required": ["cash_flows"],
            },
            run=_net_present_value,
        ),
        ToolSpec(
            name="size_column",
            description=(
                "Size a distillation column: diameter from the Souders-Brown "
                "flooding velocity, height from the stage count, then cost the shell "
                "(vessel) and trays."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "vapor_molar_flow": {"type": "number", "description": "Vapor load (mol/s)"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "n_stages": {"type": "integer"},
                    "molar_mass": {"type": "number", "description": "Vapor molar mass (kg/mol)"},
                    "liquid_density": {"type": "number", "description": "Liquid density (kg/m^3)"},
                    "pressure_barg": {"type": "number"},
                    "material": {"type": "string", "enum": ["CS", "SS", "Ni", "Ti"]},
                },
                "required": ["vapor_molar_flow", "temperature", "pressure", "n_stages"],
            },
            run=_size_column,
        ),
        ToolSpec(
            name="steam_state",
            description=(
                "IAPWS-95 steam tables: resolve a water/steam state from pressure "
                "plus exactly one of temperature (K), quality (0-1), enthalpy "
                "(kJ/kg) or entropy (kJ/kg/K). Returns density, h, s, cp, speed of "
                "sound, quality and (single-phase) IAPWS viscosity/conductivity. "
                "Reference-grade accuracy, identical to REFPROP/CoolProp."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pressure": {"type": "number", "description": "Pressure (Pa)"},
                    "temperature": {"type": "number", "description": "Temperature (K)"},
                    "quality": {"type": "number", "description": "Vapor quality (0-1)"},
                    "enthalpy_kj_kg": {"type": "number"},
                    "entropy_kj_kg_k": {"type": "number"},
                },
                "required": ["pressure"],
            },
            run=_steam_state,
        ),
        ToolSpec(
            name="reference_fluid_state",
            description=(
                "Reference multiparameter-EOS (REFPROP-class) single-phase state for "
                "a pure fluid at (T, P): density, Z, h, s, cp, cv, speed of sound. "
                "Available fluids include water/steam, CO2, ammonia, light "
                "hydrocarbons (methane..octane), H2, N2, O2, Ar, refrigerants "
                "(R134a, R32, R1234yf). Use steam_state for two-phase water."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fluid": {"type": "string", "description": "Fluid name, e.g. 'co2'"},
                    "temperature": {"type": "number", "description": "Temperature (K)"},
                    "pressure": {"type": "number", "description": "Pressure (Pa)"},
                },
                "required": ["fluid", "temperature", "pressure"],
            },
            run=_reference_fluid_state,
        ),
        ToolSpec(
            name="reference_saturation",
            description=(
                "Reference-EOS vapor-liquid saturation state of a pure fluid at a "
                "temperature or a pressure (exactly one): psat/Tsat, phase "
                "densities, latent heat, surface tension. Solved from the EOS by a "
                "Maxwell construction, not an Antoine fit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fluid": {"type": "string"},
                    "temperature": {"type": "number", "description": "Temperature (K)"},
                    "pressure": {"type": "number", "description": "Pressure (Pa)"},
                },
                "required": ["fluid"],
            },
            run=_reference_saturation,
        ),
        ToolSpec(
            name="steam_utility_requirements",
            description=(
                "Size a heating/cooling utility for a duty on real IAPWS-95 water: "
                "steam mass flow at the lp/mp/hp header (real latent heat at "
                "pressure) or cooling-water circulation, plus the priced annual "
                "cost. Accepts duties of either sign."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "duty": {"type": "number", "description": "Duty (W), any sign"},
                    "utility": {
                        "type": "string",
                        "enum": ["lp_steam", "mp_steam", "hp_steam", "cooling_water"],
                    },
                    "hours_per_year": {"type": "number"},
                },
                "required": ["duty", "utility"],
            },
            run=_steam_utility_requirements,
        ),
        ToolSpec(
            name="steam_turbine",
            description=(
                "Expand steam through a turbine (IAPWS-95): shaft power from an "
                "isentropic-efficiency expansion, with wet-outlet quality from the "
                "two-phase dome logic. The workhorse of Rankine-cycle and "
                "letdown-train design."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mass_flow_kg_s": {"type": "number"},
                    "inlet_pressure_pa": {"type": "number"},
                    "inlet_temperature_k": {"type": "number"},
                    "outlet_pressure_pa": {"type": "number"},
                    "isentropic_efficiency": {"type": "number"},
                },
                "required": [
                    "mass_flow_kg_s",
                    "inlet_pressure_pa",
                    "inlet_temperature_k",
                    "outlet_pressure_pa",
                ],
            },
            run=_steam_turbine_tool,
        ),
    ]
    specs.extend(dynamics_tool_specs())
    specs.extend(integration_tool_specs())
    specs.extend(mpc_tool_specs())
    specs.extend(saft_tool_specs())
    return {spec.name: spec for spec in specs}


def tool_schemas(registry: dict[str, ToolSpec] | None = None) -> list[JsonDict]:
    """Return the function-calling schemas for every tool in the registry."""
    registry = default_registry() if registry is None else registry
    return [
        {"name": s.name, "description": s.description, "parameters": s.parameters}
        for s in registry.values()
    ]


def call_tool(
    name: str,
    arguments: JsonDict,
    registry: dict[str, ToolSpec] | None = None,
) -> JsonDict:
    """Dispatch a tool call by name with a JSON argument dict.

    Raises:
        KeyError: if ``name`` is not a registered tool.
    """
    registry = default_registry() if registry is None else registry
    if name not in registry:
        raise KeyError(f"unknown tool {name!r}; available: {sorted(registry)}")
    return registry[name].run(**arguments)
