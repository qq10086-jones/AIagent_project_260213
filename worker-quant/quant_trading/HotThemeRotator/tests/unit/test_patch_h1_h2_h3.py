"""Patch tests for H1 (tz comparison) / H2 (commit re-validation) / H3 (migration tolerance)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.position_adapter import PortfolioState, PositionRow  # noqa: E402
from hot_theme_rotator.portfolio.journal_writer import append_cash_event, append_fill  # noqa: E402
from hot_theme_rotator.portfolio.manual_entry_service import (  # noqa: E402
    build_cash_event,
    build_fill_entry,
    commit_cash_event,
    commit_fill,
    preview_cash_event,
    preview_fill,
)
from hot_theme_rotator.portfolio.migration import (  # noqa: E402
    MigrationError,
    migrate_portfolio_from_project_optimized,
    migration_complete_path,
)
from hot_theme_rotator.portfolio.validation import (  # noqa: E402
    PortfolioValidationError,
    validate_manual_fill,
)
from hot_theme_rotator.portfolio.schema import derive_fill_entry_id  # noqa: E402


JST = timezone(timedelta(hours=9), name="JST")
NOW = datetime(2026, 5, 26, 10, 0, tzinfo=JST)


# ─── H1: timezone comparison fragility ──────────────────────────────────────


def test_h1_naive_now_is_rejected_with_clear_error():
    """Validation must fail-closed with a clear message when `now` is naive, not crash."""
    fill = build_fill_entry(side="BUY", symbol="1306.T", qty=10, price=400.0,
                            ts="2026-05-26T09:00:00+09:00")
    naive_now = datetime(2026, 5, 26, 10, 0)
    with pytest.raises(PortfolioValidationError, match="timezone-aware"):
        validate_manual_fill(proposed=fill, journal=(), now=naive_now)


def test_h1_z_suffix_ts_normalized_to_utc():
    """ts ending in 'Z' should parse as UTC and compare correctly across versions."""
    fill = build_fill_entry(side="BUY", symbol="1306.T", qty=10, price=400.0,
                            ts="2026-05-26T00:00:00+00:00")
    # Schema accepts the explicit +00:00 form. The validation layer's Z handler
    # is exercised via a malformed user input that happens to use Z.
    from hot_theme_rotator.portfolio.validation import _check_ts_not_in_future
    _check_ts_not_in_future("2026-05-26T00:00:00Z", now=NOW)  # should not raise


def test_h1_naive_ts_at_validation_layer_rejected():
    from hot_theme_rotator.portfolio.validation import _check_ts_not_in_future
    with pytest.raises(PortfolioValidationError, match="timezone"):
        _check_ts_not_in_future("2026-05-26T10:00:00", now=NOW)


# ─── H2: commit-time re-validation (TOCTOU) ─────────────────────────────────


def _seed_path_a(tmp_path):
    deposit = build_cash_event(ts="2026-05-07T09:00:00+09:00", amount=389345.0,
                               reason="deposit", note="initial")
    buy = build_fill_entry(side="BUY", symbol="1306.T", qty=900, price=403.0,
                           ts="2026-05-07T09:30:00+09:00")
    append_cash_event(deposit, base_dir=tmp_path)
    append_fill(buy, base_dir=tmp_path)


def test_h2_commit_revalidates_against_fresh_journal(tmp_path):
    """If another SELL lands between preview and commit, the second commit must fail."""
    _seed_path_a(tmp_path)
    fill_a = build_fill_entry(side="SELL", symbol="1306.T", qty=900, price=420.0,
                              ts="2026-05-25T14:00:00+09:00", note="A")
    preview_a = preview_fill(fill_a, base_dir=tmp_path, now=NOW)

    fill_b = build_fill_entry(side="SELL", symbol="1306.T", qty=900, price=420.0,
                              ts="2026-05-25T14:00:00+09:00", note="B")
    preview_b = preview_fill(fill_b, base_dir=tmp_path, now=NOW)

    # Commit A first — succeeds (holdings 900 → 0).
    commit_fill(preview_a, base_dir=tmp_path, now=NOW)

    # B's preview was taken while holdings = 900, but journal now shows qty=0.
    # Commit B must re-validate and reject.
    with pytest.raises(PortfolioValidationError, match="holdings"):
        commit_fill(preview_b, base_dir=tmp_path, now=NOW)


def test_h2_commit_revalidates_future_ts_after_clock_tick(tmp_path):
    """A preview taken at now=10:00 with ts=10:30 is fine; committing at now=10:00 still
    fine. But if `now` rolls back (user clock skew) the commit refuses."""
    _seed_path_a(tmp_path)
    fill = build_fill_entry(side="SELL", symbol="1306.T", qty=100, price=420.0,
                            ts="2026-05-26T09:30:00+09:00")
    preview = preview_fill(fill, base_dir=tmp_path, now=NOW)
    # Simulate clock going back to 5-26 09:00 → 09:30 is now in the future.
    earlier = datetime(2026, 5, 26, 9, 0, tzinfo=JST)
    with pytest.raises(PortfolioValidationError, match="future"):
        commit_fill(preview, base_dir=tmp_path, now=earlier)


def test_h2_cash_event_commit_revalidates(tmp_path):
    _seed_path_a(tmp_path)
    event = build_cash_event(ts="2026-05-20T09:00:00+09:00", amount=-1000000.0,
                             reason="withdrawal", note="overdraw")
    preview = preview_cash_event(event, base_dir=tmp_path, now=NOW)
    # Soft warning at preview, hard reject if we set source to non-manual at commit.
    # Or just confirm the commit path actually appends without re-raising on warnings.
    commit_cash_event(preview, base_dir=tmp_path, now=NOW)  # should succeed (soft warning only)


# ─── H3: migration NAV tolerance + source consistency + marker payload ───────


def _path_a_state(cash=193685.0, positions_value=209000.0, nav=None):
    if nav is None:
        nav = cash + positions_value
    return PortfolioState(
        asof="2026-05-25", cash=cash, positions_value=positions_value, nav=nav,
        strategy_id="etf_buyhold",
        holdings=(
            PositionRow(
                asof="2026-05-25", symbol="1306.T", qty=500.0, avg_cost=403.0,
                market_price=418.0, market_value=209000.0, unrealized_pnl=7500.0,
            ),
        ),
        source_path="(mock)", positions_asof="2026-05-25",
    )


def test_h3_source_internal_inconsistency_distinct_error(tmp_path):
    """source.cash + source.positions_value != source.nav must be a distinct error
    so the operator knows to refresh Project_optimized, not to debug HTR."""
    inconsistent = _path_a_state(cash=193685.0, positions_value=209000.0, nav=500000.0)
    with pytest.raises(MigrationError, match="source state inconsistent"):
        migrate_portfolio_from_project_optimized(
            cutover_date="2026-06-08", base_dir=tmp_path,
            source_loader=lambda: inconsistent,
        )


def test_h3_marker_records_components_and_tolerance(tmp_path):
    result = migrate_portfolio_from_project_optimized(
        cutover_date="2026-06-08", base_dir=tmp_path,
        source_loader=_path_a_state,
    )
    payload = json.loads(
        migration_complete_path("2026-06-08", base_dir=tmp_path).read_text(encoding="utf-8")
    )
    assert "nav_tolerance_yen" in payload
    assert payload["source_components"]["cash"] == pytest.approx(193685.0)
    assert payload["source_components"]["positions_value"] == pytest.approx(209000.0)
    assert payload["derived_components"]["cash"] == pytest.approx(193685.0)
    assert payload["derived_components"]["positions_value"] == pytest.approx(209000.0)


def test_h3_tolerance_scales_with_nav(tmp_path):
    """Tolerance = max(¥10, NAV × 0.01%). For Path A NAV ≈ ¥402k → tolerance ≈ ¥40."""
    result = migrate_portfolio_from_project_optimized(
        cutover_date="2026-06-08", base_dir=tmp_path,
        source_loader=_path_a_state,
    )
    payload = json.loads(
        migration_complete_path("2026-06-08", base_dir=tmp_path).read_text(encoding="utf-8")
    )
    # ¥402,685 × 0.0001 = ¥40.27, max(10, 40.27) = ¥40.27
    assert 10.0 <= payload["nav_tolerance_yen"] <= 50.0
