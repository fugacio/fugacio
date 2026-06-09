"""Human-readable (Markdown) summaries of simulation and design results.

The copilot returns numbers; these helpers turn them into the tables and
summaries an engineer expects -- a stream table, an optimization summary, an
equipment cost breakdown, and a rendered agent transcript -- so a language model
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
    """Summarize an :class:`OptimizeResult` or :class:`FlowsheetOptResult` as Markdown.

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
        items: Costed equipment (from :func:`fugacio.sim.bare_module_cost`).
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
