"""Human-readable summaries of simulation results."""

from __future__ import annotations

from fugacio.sim import bubble_pressure

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
