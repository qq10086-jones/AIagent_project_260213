"""P10-23 — manual_entry_service tests."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.portfolio.journal_writer import (  # noqa: E402
    append_cash_event,
    append_fill,
)
from hot_theme_rotator.portfolio.manual_entry_service import (  # noqa: E402
    build_cash_event,
    build_fill_entry,
    commit_cash_event,
    commit_fill,
    preview_cash_event,
    preview_fill,
)
from hot_theme_rotator.portfolio.schema import PortfolioSchemaError  # noqa: E402
from hot_theme_rotator.portfolio.validation import PortfolioValidationError  # noqa: E402


JST = timezone(timedelta(hours=9), name="JST")
NOW = datetime(2026, 5, 26, 10, 0, tzinfo=JST)


def _seed_path_a(tmp_path):
    deposit = build_cash_event(ts="2026-05-07T09:00:00+09:00", amount=389345.0,
                               reason="deposit", note="initial")
    buy = build_fill_entry(side="BUY", symbol="1306.T", qty=900, price=403.0,
                           ts="2026-05-07T09:30:00+09:00")
    append_cash_event(deposit, base_dir=tmp_path)
    append_fill(buy, base_dir=tmp_path)


def test_preview_fill_returns_before_after_state(tmp_path):
    _seed_path_a(tmp_path)
    fill = build_fill_entry(side="SELL", symbol="1306.T", qty=400, price=417.6,
                            ts="2026-05-25T14:00:00+09:00")
    preview = preview_fill(fill, base_dir=tmp_path, now=NOW)
    assert preview.before["qty"] == 900
    assert preview.before["avg_cost"] == pytest.approx(403.0)
    assert preview.before["cash"] == pytest.approx(389345.0 - 900*403.0)
    assert preview.after["qty"] == 500
    assert preview.after["cash"] == pytest.approx(preview.before["cash"] + 400*417.6)
    assert preview.after["realized_pnl_delta"] == pytest.approx(400 * (417.6 - 403.0))
    assert preview.all_warnings == ()


def test_preview_fill_surfaces_validation_warning_for_buy_exceeding_cash(tmp_path):
    _seed_path_a(tmp_path)
    fill = build_fill_entry(side="BUY", symbol="1306.T", qty=1000, price=1000.0,
                            ts="2026-05-26T09:30:00+09:00", note="overspend")
    preview = preview_fill(fill, base_dir=tmp_path, now=NOW)
    assert any("cash" in w.lower() for w in preview.all_warnings)


def test_preview_fill_magnitude_warning_fires_above_10pct_nav(tmp_path):
    _seed_path_a(tmp_path)
    fill = build_fill_entry(side="SELL", symbol="1306.T", qty=900, price=417.6,
                            ts="2026-05-26T09:30:00+09:00", note="sell all")
    preview = preview_fill(
        fill, base_dir=tmp_path, now=NOW,
        current_prices={"1306.T": 418.0},
    )
    # NAV ≈ 26645 + 900*418 = 402,845; notional 900*417.6 = 375,840 = 93% NAV → warning
    assert preview.magnitude_warning is not None
    assert "Rule 14.5" in preview.magnitude_warning


def test_preview_fill_no_magnitude_warning_when_prices_omitted(tmp_path):
    _seed_path_a(tmp_path)
    fill = build_fill_entry(side="SELL", symbol="1306.T", qty=900, price=417.6,
                            ts="2026-05-26T09:30:00+09:00")
    preview = preview_fill(fill, base_dir=tmp_path, now=NOW)
    assert preview.magnitude_warning is None  # skipped without prices


def test_preview_fill_hard_rejects_oversell(tmp_path):
    _seed_path_a(tmp_path)
    fill = build_fill_entry(side="SELL", symbol="1306.T", qty=10000, price=417.6,
                            ts="2026-05-25T14:00:00+09:00")
    with pytest.raises(PortfolioValidationError, match="holdings"):
        preview_fill(fill, base_dir=tmp_path, now=NOW)


def test_preview_fill_hard_rejects_future_ts(tmp_path):
    _seed_path_a(tmp_path)
    fill = build_fill_entry(side="SELL", symbol="1306.T", qty=10, price=417.6,
                            ts="2026-05-27T14:00:00+09:00")
    with pytest.raises(PortfolioValidationError, match="future"):
        preview_fill(fill, base_dir=tmp_path, now=NOW)


def test_commit_fill_writes_to_journal(tmp_path):
    _seed_path_a(tmp_path)
    fill = build_fill_entry(side="SELL", symbol="1306.T", qty=400, price=417.6,
                            ts="2026-05-25T14:00:00+09:00")
    preview = preview_fill(fill, base_dir=tmp_path, now=NOW)
    path = commit_fill(preview, base_dir=tmp_path)
    assert path.exists()
    # Re-preview to confirm the state actually moved
    after_preview = preview_fill(
        build_fill_entry(side="SELL", symbol="1306.T", qty=10, price=420.0,
                         ts="2026-05-26T09:30:00+09:00"),
        base_dir=tmp_path, now=NOW,
    )
    assert after_preview.before["qty"] == 500


def test_preview_cash_event_dividend(tmp_path):
    _seed_path_a(tmp_path)
    event = build_cash_event(ts="2026-05-20T09:00:00+09:00", amount=1500.0,
                             reason="dividend", symbol="1306.T")
    preview = preview_cash_event(event, base_dir=tmp_path, now=NOW)
    assert preview.before["cash"] == pytest.approx(389345.0 - 900*403.0)
    assert preview.after["cash"] == pytest.approx(preview.before["cash"] + 1500.0)
    assert preview.all_warnings == ()


def test_commit_cash_event_writes_to_journal(tmp_path):
    event = build_cash_event(ts="2026-05-08T09:00:00+09:00", amount=100000.0,
                             reason="deposit")
    preview = preview_cash_event(event, base_dir=tmp_path, now=NOW)
    path = commit_cash_event(preview, base_dir=tmp_path)
    assert path.exists()


def test_build_fill_entry_invalid_side_raises_schema_error():
    with pytest.raises(PortfolioSchemaError, match="side"):
        build_fill_entry(side="SHORT", symbol="1306.T", qty=10, price=400.0,
                         ts="2026-05-26T10:00:00+09:00")
