"""Human-readable (Markdown) summaries of simulation and design results.

The copilot returns numbers; these helpers turn them into the tables and
summaries an engineer expects (a stream table, an optimization summary, an
equipment cost breakdown, and a rendered agent transcript) so a language model
(or a notebook) can present results cleanly. Everything here is pure Python
string formatting over already-computed results; no JAX, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from fugacio.sim import Stream, bubble_pressure

if TYPE_CHECKING:  # avoid any import-order coupling at runtime
    from fugacio.copilot.agent import AgentResult
    from fugacio.sim import EquipmentCost

Antoine = tuple[float, float, float]


def summarize_bubble_point(
    x1: float,
    temperature: float,
    antoine1: Antoine,
    antoine2: Antoine,
    a12: float = 0.0,
    a21: float = 0.0,
) -> str:
    """Return a one-line natural-language summary of a binary bubble point."""
    pressure, y1 = bubble_pressure(x1, temperature, antoine1, antoine2, a12, a21)
    return (
        f"At T={temperature:g} and x1={x1:g}, the bubble-point pressure is "
        f"{float(pressure):.4g} with vapor composition y1={float(y1):.4f}."
    )


def stream_table(streams: Mapping[str, Stream], *, digits: int = 4) -> str:
    """Render a Markdown stream table: per-stream T, P, total flow, and mole fractions.

    Args:
        streams: Named streams (e.g. the output of a flowsheet solve).
        digits: Significant digits for the mole fractions.

    Returns:
        A Markdown table with one column per stream.
    """
    if not streams:
        return "_(no streams)_"
    names = list(streams)
    first = streams[names[0]]
    components = first.components

    header = "| Property | " + " | ".join(names) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(names)) + " |"
    rows = [header, sep]

    rows.append("| T (K) | " + " | ".join(f"{float(streams[n].t):.2f}" for n in names) + " |")
    rows.append("| P (Pa) | " + " | ".join(f"{float(streams[n].p):.4g}" for n in names) + " |")
    rows.append(
        "| Flow (mol/s) | " + " | ".join(f"{float(streams[n].total):.4g}" for n in names) + " |"
    )
    for i, comp in enumerate(components):
        cells = " | ".join(f"{float(streams[n].z[i]):.{digits}f}" for n in names)
        rows.append(f"| x[{comp}] | {cells} |")
    return "\n".join(rows)


def summarize_optimization(result: Any, *, title: str = "Optimization") -> str:
    """Summarize an `OptimizeResult` or `FlowsheetOptResult` as Markdown.

    Reads the common fields by duck typing, so it accepts either the raw optimizer
    result (``fun``) or a flowsheet optimization result (``objective``).
    """
    objective = getattr(result, "objective", None)
    if objective is None:
        objective = getattr(result, "fun", None)
    converged = bool(getattr(result, "converged", False))
    n_iter = getattr(result, "n_iter", None)
    lines = [f"### {title}", ""]
    lines.append(f"- Converged: **{converged}**")
    if objective is not None:
        lines.append(f"- Objective: **{float(objective):.6g}**")
    if n_iter is not None:
        lines.append(f"- Iterations: {int(n_iter)}")
    theta = getattr(result, "theta", None)
    if isinstance(theta, Mapping):
        lines.append("- Variables:")
        for key, value in theta.items():
            lines.append(f"  - `{key}` = {float(value):.6g}")
    return "\n".join(lines)


def summarize_economics(
    items: Sequence[EquipmentCost],
    *,
    operating_cost: float = 0.0,
    interest_rate: float = 0.1,
    years: float = 10.0,
) -> str:
    """Render an equipment cost breakdown and the resulting total annual cost.

    Args:
        items: Costed equipment (from `fugacio.sim.bare_module_cost`).
        operating_cost: Annual operating (utility) cost ($/yr).
        interest_rate: Annual interest rate for the capital-recovery factor.
        years: Project life (years).

    Returns:
        A Markdown report: a per-item cost table plus annualized capital, OPEX,
        and total annual cost.
    """
    from fugacio.sim import annualized_capital, total_annual_cost

    lines = [
        "### Economics",
        "",
        "| Equipment | Size | Installed cost ($) |",
        "| --- | --- | --- |",
    ]
    capex = 0.0
    for it in items:
        capex += float(it.bare_module)
        lines.append(f"| {it.kind} | {float(it.size):.4g} | {float(it.bare_module):,.0f} |")
    lines.append(f"| **Total CAPEX** |  | **{capex:,.0f}** |")
    ann_cap = float(annualized_capital(capex, rate=interest_rate, years=years))
    tac = float(total_annual_cost(capex, operating_cost, rate=interest_rate, years=years))
    lines += [
        "",
        f"- Annualized capital: **{ann_cap:,.0f}** $/yr "
        f"(i = {interest_rate:.0%}, n = {years:g} yr)",
        f"- Operating cost: **{operating_cost:,.0f}** $/yr",
        f"- **Total annual cost: {tac:,.0f} $/yr**",
    ]
    return "\n".join(lines)


def summarize_pid_tuning(recommendation: Mapping[str, Any]) -> str:
    """Render a `recommend_pid_tuning` result as a Markdown comparison table.

    Accepts the dict returned by the ``recommend_pid_tuning`` copilot tool (a list
    of candidate rules with their gains and closed-loop metrics) and renders the
    comparison plus the recommended rule.
    """
    candidates = list(recommendation.get("candidates", []))
    controller = recommendation.get("controller", "PI")
    recommended = recommendation.get("recommended_rule")
    lines = [
        f"### PID tuning comparison ({controller})",
        "",
        "| Rule | kc | tau_i (s) | tau_d (s) | IAE | Overshoot | Settling (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in candidates:
        star = " *" if c.get("rule") == recommended else ""
        lines.append(
            f"| {c['rule']}{star} | {float(c['kc']):.4g} | {float(c['tau_i_s']):.4g} | "
            f"{float(c['tau_d_s']):.4g} | {float(c['iae']):.4g} | "
            f"{float(c['overshoot_fraction']):.1%} | {float(c['settling_time_s']):.4g} |"
        )
    if recommended is not None:
        lines += ["", f"**Recommended:** `{recommended}` (lowest closed-loop IAE)."]
    return "\n".join(lines)


def summarize_heat_integration(targets: Mapping[str, Any]) -> str:
    """Render heat-integration targets as a Markdown summary.

    Accepts the dict returned by the ``heat_integration_targets`` copilot tool
    (minimum utilities, pinch, recovery, area/unit/cost targets) and lays it out
    as a compact engineer-facing report.
    """
    pinch = targets.get("pinch", {})
    lines = [
        f"### Heat integration (dt_min = {float(targets.get('dt_min', 0.0)):g} K)",
        "",
        "| Target | Value |",
        "| --- | --- |",
        f"| Hot utility | {float(targets.get('hot_utility_w', 0.0)):,.0f} W |",
        f"| Cold utility | {float(targets.get('cold_utility_w', 0.0)):,.0f} W |",
        f"| Heat recovery | {float(targets.get('heat_recovery_w', 0.0)):,.0f} W |",
    ]
    if "area_target_m2" in targets:
        lines.append(f"| Area target | {float(targets['area_target_m2']):,.1f} m^2 |")
    if "minimum_units" in targets:
        lines.append(f"| Minimum units | {int(targets['minimum_units'])} |")
    if "total_annual_cost_usd_yr" in targets:
        lines.append(
            f"| Total annual cost | {float(targets['total_annual_cost_usd_yr']):,.0f} $/yr |"
        )
    if pinch.get("exists"):
        lines += [
            "",
            f"**Pinch:** {float(pinch['hot_temperature_k']):g} K (hot) / "
            f"{float(pinch['cold_temperature_k']):g} K (cold).",
        ]
    else:
        lines += ["", "**Threshold problem** (no pinch): a single utility suffices."]
    return "\n".join(lines)


def summarize_lqr_design(design: Mapping[str, Any]) -> str:
    """Render an `lqr_design` result (gain, poles, stability) as Markdown.

    Accepts the dict returned by the ``lqr_design`` copilot tool.
    """
    continuous = bool(design.get("continuous", False))
    gain = design.get("gain", [])
    poles = design.get("pole_real_parts" if continuous else "pole_magnitudes", [])
    kind = "continuous" if continuous else "discrete"
    lines = [
        f"### LQR design ({kind})",
        "",
        "- Feedback law: **u = -K x**",
        "- Gain K:",
    ]
    for row in gain:
        lines.append("  - [" + ", ".join(f"{float(v):.4g}" for v in row) + "]")
    pole_word = "Re(eig)" if continuous else "|eig|"
    lines.append(
        f"- Closed-loop poles ({pole_word}): " + ", ".join(f"{float(v):.4g}" for v in poles)
    )
    lines.append(f"- Stable: **{bool(design.get('stable', False))}**")
    return "\n".join(lines)


def summarize_mpc_simulation(result: Mapping[str, Any]) -> str:
    """Render a `simulate_mpc` result as a Markdown step-response report.

    Accepts the dict returned by the ``simulate_mpc`` copilot tool (per-output step
    metrics, setpoint, and final value) and lays it out as an engineer-facing table.
    """
    setpoint = list(result.get("setpoint", []))
    final = list(result.get("final_output", []))
    metrics = list(result.get("metrics", []))
    lines = [
        "### MPC closed-loop response",
        "",
        "| Output | Setpoint | Final | Offset | Overshoot | Settling (s) | IAE |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, m in enumerate(metrics):
        sp = float(setpoint[i]) if i < len(setpoint) else float("nan")
        fv = float(final[i]) if i < len(final) else float("nan")
        offset = fv - sp
        lines.append(
            f"| y[{m.get('output', i)}] | {sp:.4g} | {fv:.4g} | {offset:+.2e} | "
            f"{float(m.get('overshoot_fraction', 0.0)):.1%} | "
            f"{float(m.get('settling_time_s', 0.0)):.4g} | "
            f"{float(m.get('iae', 0.0)):.4g} |"
        )
    lines += ["", "_Offset-free tracking: the steady-state offset is driven to zero._"]
    return "\n".join(lines)


def summarize_transcript(result: AgentResult, *, max_chars: int = 200) -> str:
    """Render an agent run (its tool calls and final answer) as a Markdown report."""
    lines = ["### Copilot run", ""]
    for i, step in enumerate(result.transcript, start=1):
        args = ", ".join(f"{k}={v!r}" for k, v in step["arguments"].items())
        result_str = str(step["result"])
        if len(result_str) > max_chars:
            result_str = result_str[:max_chars] + "..."
        lines.append(f"{i}. **{step['tool']}**({args}) -> `{result_str}`")
    lines += ["", f"**Answer:** {result.answer}"]
    return "\n".join(lines)
