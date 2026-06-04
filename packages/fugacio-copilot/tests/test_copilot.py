from fugacio.copilot import summarize_bubble_point

COMP1 = (8.07131, 1730.63, 233.426)
COMP2 = (7.43155, 1554.68, 240.337)


def test_summary_mentions_pressure_and_composition() -> None:
    summary = summarize_bubble_point(0.4, 80.0, COMP1, COMP2, a12=0.5, a21=0.8)
    assert "bubble-point pressure" in summary
    assert "y1=" in summary
