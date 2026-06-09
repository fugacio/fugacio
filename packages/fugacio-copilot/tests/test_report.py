"""Markdown report rendering.

Checks the stream table, optimization summary, economics breakdown, and agent
transcript renderers produce well-formed Markdown containing the key numbers.
"""

import jax.numpy as jnp

from fugacio.copilot import (
    AgentResult,
    stream_table,
    summarize_economics,
    summarize_optimization,
    summarize_transcript,
)
from fugacio.sim import Stream, bare_module_cost
from fugacio.sim.design import FlowsheetOptResult


def test_stream_table_has_components_and_columns() -> None:
    s1 = Stream.from_fractions(("a", "b"), jnp.array([0.6, 0.4]), 100.0, 320.0, 1e5)
    s2 = Stream.from_fractions(("a", "b"), jnp.array([0.9, 0.1]), 40.0, 300.0, 1e5)
    table = stream_table({"feed": s1, "product": s2})
    assert "| feed | product |" in table
    assert "x[a]" in table and "x[b]" in table
    assert "Flow (mol/s)" in table


def test_stream_table_empty() -> None:
    assert "no streams" in stream_table({})


def test_summarize_optimization_flowsheet_result() -> None:
    res = FlowsheetOptResult(
        theta={"reflux": jnp.asarray(1.8)},
        streams={},
        objective=jnp.asarray(123456.0),
        converged=jnp.asarray(True),
        n_iter=jnp.asarray(12),
    )
    md = summarize_optimization(res, title="Column design")
    assert "Column design" in md
    assert "123456" in md.replace(",", "")
    assert "reflux" in md


def test_summarize_economics_table_and_tac() -> None:
    items = [
        bare_module_cost("heat_exchanger", 100.0),
        bare_module_cost("pump", 20.0),
    ]
    md = summarize_economics(items, operating_cost=2.0e5)
    assert "Total CAPEX" in md
    assert "Total annual cost" in md


def test_summarize_transcript_lists_calls_and_answer() -> None:
    result = AgentResult(
        answer="The bubble pressure is 9.5 bar.",
        transcript=[
            {"tool": "bubble_pressure", "arguments": {"temperature": 300.0}, "result": {"p": 9.5e5}}
        ],
    )
    md = summarize_transcript(result)
    assert "bubble_pressure" in md
    assert "The bubble pressure is 9.5 bar." in md
