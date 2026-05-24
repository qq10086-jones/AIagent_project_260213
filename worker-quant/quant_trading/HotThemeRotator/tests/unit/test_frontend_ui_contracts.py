"""Static frontend UI contract checks for zero-build React variants."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_frontend_file(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_v1_price_panel_uses_dedicated_ladder_rail():
    """P8-17: V1 should not spend chart padding on seven ladder labels."""
    v1 = read_frontend_file("frontend/v1.jsx")

    assert "function V1KLineLadderPanel" in v1
    assert "function V1LadderRail" in v1
    assert 'gridTemplateColumns: "minmax(0, 1fr) 184px"' in v1
    assert "chartLadder={top.ladder}" in v1
    assert "ladder={null}" in v1
    assert "padding={{ top: 18, right: 28, bottom: 22, left: 12 }}" in v1


def test_v2_decision_log_section_has_surface_and_empty_state():
    """P8-17: V2 lower section should not look broken when log is empty."""
    v2 = read_frontend_file("frontend/v2.jsx")

    assert "function V2SurfacePane" in v2
    assert "function V2DecisionLogPane" in v2
    assert "暂无 §8.6 决策日志" in v2
    assert 'background: "var(--htr-surface)"' in v2
    assert "minHeight: 220" in v2


def test_v2_page_paper_background_extends_with_scrolled_content():
    """P8-17 follow-up: V2 page wrapper must not stretch to viewport only."""
    v2 = read_frontend_file("frontend/v2.jsx")

    assert 'alignItems: "flex-start"' in v2
