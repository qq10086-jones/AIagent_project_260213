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


# ─── P8-18 Interactive Exploration Layer (Rule 11) ───────────────────────────


def test_shared_jsx_exports_p8_18_hooks():
    """Three hooks must be defined and exported on `window` for cross-file use."""
    shared = read_frontend_file("frontend/shared.jsx")

    for sym in ("useSelectedSymbol", "useSymbolKline", "useSymbolProfile"):
        assert f"function {sym}(" in shared, f"missing hook definition: {sym}"
        assert sym in shared.split("Object.assign(window,")[1], f"hook not exported: {sym}"


def test_shared_jsx_hooks_are_get_only():
    """Rule 11.2 — hooks may only GET from /api/symbol/{ticker}/* paths."""
    shared = read_frontend_file("frontend/shared.jsx")

    # No POST / PUT / DELETE / PATCH inside hooks.
    for method in ('"POST"', '"PUT"', '"DELETE"', '"PATCH"',
                   "method: 'POST'", "method: 'PUT'", "method: 'DELETE'", "method: 'PATCH'"):
        assert method not in shared, f"forbidden write method appeared in shared.jsx: {method}"


def test_v1_uses_selected_symbol_hook_and_passes_klinebars():
    """V1 must drive K-line and hero from selectedSymbol, not data.candidates[0] only."""
    v1 = read_frontend_file("frontend/v1.jsx")

    assert "useSelectedSymbol" in v1
    assert "useSymbolKline" in v1
    assert "klineBars" in v1
    assert "const fallbackKline = top.symbol === defaultSymbol ? data.kline : [];" in v1
    assert "fallback: fallbackKline" in v1
    assert "key={top.symbol}" in v1
    assert "klineError={kline.error}" in v1
    # CandidateRowMini must receive onClick + active
    assert "active={c.symbol === top.symbol}" in v1
    assert "onClick={() => setSelectedSymbol(c.symbol)}" in v1


def test_v1_renders_daily_cockpit_panel_from_dashboard_payload():
    """P10-20: V1 should surface Stage 0 dailyCockpit without write actions."""
    v1 = read_frontend_file("frontend/v1.jsx")

    assert "function V1DailyCockpitPanel" in v1
    assert "cockpit={data.dailyCockpit}" in v1
    assert "cockpit.activationStage" in v1
    assert "cockpit.notificationsInvoked" in v1
    assert "cockpit.execution?.orders" in v1
    assert "dailyCockpit" in v1


def test_v1_daily_cockpit_adds_no_write_methods_or_push_copy():
    v1 = read_frontend_file("frontend/v1.jsx")

    for forbidden in ('"POST"', '"PUT"', '"DELETE"', '"PATCH"', "push_allowed"):
        assert forbidden not in v1, f"forbidden frontend cockpit text appeared: {forbidden}"


def test_v3_candidate_rows_clickable_and_drive_leader_card():
    v3 = read_frontend_file("frontend/v3.jsx")

    assert "useSelectedSymbol" in v3
    assert "active={c.symbol === top.symbol}" in v3
    assert "onClick={() => setSelectedSymbol(c.symbol)}" in v3


def test_v1_v3_localstorage_key_is_htr_symbol():
    """Rule 11.3 — selection persists to localStorage user-state only."""
    shared = read_frontend_file("frontend/shared.jsx")
    assert 'localStorage.getItem("htr_symbol")' in shared
    assert 'localStorage.setItem("htr_symbol"' in shared
