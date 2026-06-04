"""Chemical-engineering design copilot for Fugacio (depends on ``fugacio.sim``).

The baseline ships a single deterministic helper so the dependency layering
(``copilot`` -> ``sim`` -> ``thermo``) is exercised end-to-end. LLM-backed
planning will live behind the optional ``llm`` extra.
"""

from fugacio.copilot.report import summarize_bubble_point

__all__ = ["summarize_bubble_point"]

__version__ = "0.0.1"
