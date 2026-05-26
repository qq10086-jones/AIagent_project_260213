"""P10-22 — Project_optimized → HTR migration script tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.position_adapter import PortfolioState, PositionRow  # noqa: E402
from hot_theme_rotator.portfolio.derive import (  # noqa: E402
    derive_cash_balance,
    derive_positions,
)
from hot_theme_rotator.portfolio.journal_writer import read_journal  # noqa: E402
from hot_theme_rotator.portfolio.migration import (  # noqa: E402
    MigrationError,
    migrate_portfolio_from_project_optimized,
    migration_complete_path,
)


def _path_a_state(cash=193685.0):
    """Mirror the Path A state right after the 5-25 SELL救急."""
    return PortfolioState(
        asof="2026-05-25",
        cash=cash,
        positions_value=209000.0,
        nav=cash + 209000.0,
        strategy_id="etf_buyhold",
        holdings=(
            PositionRow(
                asof="2026-05-25", symbol="1306.T", qty=500.0, avg_cost=403.0,
                market_price=418.0, market_value=209000.0, unrealized_pnl=7500.0,
            ),
        ),
        source_path="(mock)",
        positions_asof="2026-05-25",
    )


def test_dry_run_returns_preview_without_writing(tmp_path):
    result = migrate_portfolio_from_project_optimized(
        cutover_date="2026-06-08",
        base_dir=tmp_path,
        source_loader=_path_a_state,
        dry_run=True,
    )
    assert result.dry_run is True
    assert result.cash_event_count == 1
    assert result.fill_count == 1
    assert len(result.entries_preview) == 2
    assert result.entries_preview[0]["_type"] == "cash_event"
    assert result.entries_preview[1]["_type"] == "fill"
    # No journal or marker written
    assert not (tmp_path / "reports" / "portfolio" / "journal" / "2026-06-08.jsonl").exists()
    assert not migration_complete_path("2026-06-08", base_dir=tmp_path).exists()


def test_real_run_writes_journal_and_marker(tmp_path):
    result = migrate_portfolio_from_project_optimized(
        cutover_date="2026-06-08",
        base_dir=tmp_path,
        source_loader=_path_a_state,
    )
    assert result.dry_run is False
    assert result.nav_consistent
    assert result.fill_count == 1
    journal_file = tmp_path / "reports" / "portfolio" / "journal" / "2026-06-08.jsonl"
    assert journal_file.exists()
    assert migration_complete_path("2026-06-08", base_dir=tmp_path).exists()


def test_migrated_journal_derives_to_same_positions_and_cash(tmp_path):
    migrate_portfolio_from_project_optimized(
        cutover_date="2026-06-08",
        base_dir=tmp_path,
        source_loader=_path_a_state,
    )
    entries = read_journal("2026-06-08", base_dir=tmp_path)
    positions = derive_positions(entries)
    cash = derive_cash_balance(entries)
    assert positions["1306.T"].qty == 500
    assert positions["1306.T"].avg_cost == pytest.approx(403.0)
    assert cash == pytest.approx(193685.0)


def test_idempotency_blocks_rerun(tmp_path):
    migrate_portfolio_from_project_optimized(
        cutover_date="2026-06-08",
        base_dir=tmp_path,
        source_loader=_path_a_state,
    )
    with pytest.raises(MigrationError, match="already completed"):
        migrate_portfolio_from_project_optimized(
            cutover_date="2026-06-08",
            base_dir=tmp_path,
            source_loader=_path_a_state,
        )


def test_nav_mismatch_fail_closed(tmp_path):
    """If source.nav is wrong vs cash + positions_value, the migration must fail-closed.

    Construct a state where source claims NAV is much higher than what derived
    journal would compute, e.g. by lying about positions_value relative to qty
    × market_price.
    """
    bad = PortfolioState(
        asof="2026-05-25",
        cash=100000.0,
        positions_value=100000.0,
        nav=999999.0,  # lie
        strategy_id="etf_buyhold",
        holdings=(
            PositionRow(
                asof="2026-05-25", symbol="1306.T", qty=500.0, avg_cost=403.0,
                market_price=418.0, market_value=209000.0, unrealized_pnl=7500.0,
            ),
        ),
        source_path="(mock)",
        positions_asof="2026-05-25",
    )
    # After H3 patch, this lying state is now caught by the *earlier* source-internal
    # consistency check (cash + positions_value != nav). Either error counts as
    # fail-closed on NAV inconsistency.
    with pytest.raises(MigrationError, match="(NAV verification|source state inconsistent)"):
        migrate_portfolio_from_project_optimized(
            cutover_date="2026-06-08",
            base_dir=tmp_path,
            source_loader=lambda: bad,
        )
    # marker must not exist after failed run
    assert not migration_complete_path("2026-06-08", base_dir=tmp_path).exists()


def test_migration_complete_marker_payload(tmp_path):
    result = migrate_portfolio_from_project_optimized(
        cutover_date="2026-06-08",
        base_dir=tmp_path,
        source_loader=_path_a_state,
    )
    marker = migration_complete_path("2026-06-08", base_dir=tmp_path)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["cutover_date"] == "2026-06-08"
    assert payload["adr"] == "ADR-0008"
    assert payload["fill_count"] == 1
    assert payload["nav_diff_yen"] == pytest.approx(result.nav_diff_yen)


def test_non_iso_cutover_date_rejected(tmp_path):
    with pytest.raises(MigrationError, match="cutover_date"):
        migrate_portfolio_from_project_optimized(
            cutover_date="2026/06/08",
            base_dir=tmp_path,
            source_loader=_path_a_state,
        )


def test_non_integer_qty_rejected(tmp_path):
    fractional = PortfolioState(
        asof="2026-05-25", cash=100000.0, positions_value=100000.0, nav=200000.0,
        strategy_id="etf_buyhold",
        holdings=(
            PositionRow(
                asof="2026-05-25", symbol="1306.T", qty=500.5, avg_cost=403.0,
                market_price=418.0, market_value=209100.5, unrealized_pnl=7500.0,
            ),
        ),
        source_path="(mock)", positions_asof="2026-05-25",
    )
    with pytest.raises(MigrationError, match="non-integer"):
        migrate_portfolio_from_project_optimized(
            cutover_date="2026-06-08",
            base_dir=tmp_path,
            source_loader=lambda: fractional,
        )


def test_zero_qty_holding_rejected(tmp_path):
    bad = PortfolioState(
        asof="2026-05-25", cash=100000.0, positions_value=0.0, nav=100000.0,
        strategy_id="etf_buyhold",
        holdings=(
            PositionRow(
                asof="2026-05-25", symbol="1306.T", qty=0.0, avg_cost=403.0,
                market_price=418.0, market_value=0.0, unrealized_pnl=0.0,
            ),
        ),
        source_path="(mock)", positions_asof="2026-05-25",
    )
    with pytest.raises(MigrationError, match="non-positive"):
        migrate_portfolio_from_project_optimized(
            cutover_date="2026-06-08",
            base_dir=tmp_path,
            source_loader=lambda: bad,
        )
