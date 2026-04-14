"""v3 C-5 / C-5b — SBI manual order lifecycle tests.

Covers the Codex review findings:
- B: record_sbi_fill must be idempotent (no double-booking on retry)
- D: briefing should surface 'partial' orders (not just open/proposed)
- partial fill accounting: remaining qty updates correctly
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


record_sbi_order = _load("record_sbi_order", "tools/record_sbi_order.py")
record_sbi_fill = _load("record_sbi_fill", "tools/record_sbi_fill.py")


def _seed_db(tmp_path: Path) -> Path:
    """Minimal schema used by lifecycle tests."""
    db = tmp_path / "m.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE decision_runs(
            run_id TEXT PRIMARY KEY, ts TEXT, asof TEXT,
            strategy_id TEXT, status TEXT, snapshot_path TEXT
        );
        CREATE TABLE orders(
            order_id TEXT PRIMARY KEY, run_id TEXT, asof TEXT,
            symbol TEXT, side TEXT, qty REAL, order_type TEXT,
            limit_price REAL, tif TEXT, reason TEXT,
            expected_fee REAL, expected_slippage REAL, expected_value REAL,
            status TEXT, created_ts TEXT, strategy_id TEXT
        );
        CREATE TABLE fills(
            fill_id TEXT PRIMARY KEY, order_id TEXT, run_id TEXT,
            asof TEXT, strategy_id TEXT, ts TEXT, symbol TEXT,
            side TEXT, qty REAL, price REAL, fee REAL, tax REAL,
            venue TEXT, external_ref TEXT, source TEXT,
            price_source TEXT, price_ts TEXT, price_mode TEXT,
            quote_open REAL, quote_high REAL, quote_low REAL,
            quote_close REAL, price_validated INTEGER,
            validation_note TEXT
        );
        CREATE TABLE positions(
            asof TEXT, strategy_id TEXT, symbol TEXT, qty REAL,
            avg_cost REAL, market_price REAL, market_value REAL,
            unrealized_pnl REAL, high_since_entry REAL, entry_date TEXT
        );
        CREATE TABLE account_snapshots(
            asof TEXT, strategy_id TEXT, ts TEXT, run_id TEXT,
            cash REAL, positions_value REAL, nav REAL,
            net_trade_cashflow REAL, fees REAL, tax REAL, notes TEXT
        );
        CREATE TABLE daily_prices(
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL
        );
        """
    )
    conn.commit()
    conn.close()
    return db


def _place(db: Path, side: str, symbol: str, qty: float, price: float) -> str:
    return record_sbi_order.record_order(
        db=str(db), side=side, symbol=symbol, qty=qty, limit_price=price,
    )


def test_record_order_and_list(tmp_path: Path) -> None:
    db = _seed_db(tmp_path)
    oid = _place(db, "BUY", "3041.T", 400, 585.0)
    assert oid.startswith("2") and "sbi" in oid

    orders = record_sbi_order.list_open_orders(str(db))
    assert len(orders) == 1
    assert orders[0]["symbol"] == "3041.T"
    assert orders[0]["status"] == "open"


def test_fill_full_transitions_to_filled(tmp_path: Path) -> None:
    db = _seed_db(tmp_path)
    oid = _place(db, "BUY", "3041.T", 400, 585.0)
    out = record_sbi_fill.confirm_fill(
        db=str(db), order_id=oid, price=585.0, rebuild=False,
    )
    assert out["new_status"] == "filled"
    assert out["fill_qty"] == 400
    assert out["fills_inserted"] == 1


def test_fill_is_idempotent_refuses_double_booking(tmp_path: Path) -> None:
    """Codex B: running confirm_fill twice on the same order must not
    double-insert fills or double-count qty."""
    db = _seed_db(tmp_path)
    oid = _place(db, "BUY", "3041.T", 400, 585.0)
    record_sbi_fill.confirm_fill(db=str(db), order_id=oid, price=585.0, rebuild=False)

    with pytest.raises(RuntimeError) as exc:
        record_sbi_fill.confirm_fill(db=str(db), order_id=oid, price=585.0, rebuild=False)
    assert "cannot fill" in str(exc.value).lower() or "status" in str(exc.value).lower()

    # Verify only one fill row exists for this order
    conn = sqlite3.connect(db)
    cnt = conn.execute(
        "SELECT COUNT(*) FROM fills WHERE external_ref=?", (oid,)
    ).fetchone()[0]
    conn.close()
    assert cnt == 1


def test_partial_fill_decrements_remaining_and_stays_visible(tmp_path: Path) -> None:
    """Codex D/E: after partial fill, order.status='partial' and order.qty
    tracks remaining. The open-order listing must still surface it."""
    db = _seed_db(tmp_path)
    oid = _place(db, "BUY", "3041.T", 400, 585.0)

    # First partial
    out1 = record_sbi_fill.confirm_fill(
        db=str(db), order_id=oid, price=585.0, qty=200, rebuild=False,
    )
    assert out1["new_status"] == "partial"

    orders = record_sbi_order.list_open_orders(str(db))
    assert len(orders) == 1
    assert orders[0]["status"] == "partial"
    assert orders[0]["qty"] == 200  # remaining

    # Finish it
    out2 = record_sbi_fill.confirm_fill(
        db=str(db), order_id=oid, price=586.0, qty=200, rebuild=False,
    )
    assert out2["new_status"] == "filled"

    orders_after = record_sbi_order.list_open_orders(str(db))
    assert orders_after == []

    # Total 2 fills, combined qty=400
    conn = sqlite3.connect(db)
    total = conn.execute(
        "SELECT COUNT(*), SUM(qty) FROM fills WHERE external_ref=?", (oid,)
    ).fetchone()
    conn.close()
    assert total == (2, 400.0)


def test_fill_qty_over_remaining_is_rejected(tmp_path: Path) -> None:
    db = _seed_db(tmp_path)
    oid = _place(db, "BUY", "3041.T", 400, 585.0)
    with pytest.raises(ValueError):
        record_sbi_fill.confirm_fill(
            db=str(db), order_id=oid, price=585.0, qty=500, rebuild=False,
        )


def test_cancel_order(tmp_path: Path) -> None:
    db = _seed_db(tmp_path)
    oid = _place(db, "SELL", "3041.T", 100, 600.0)
    ok = record_sbi_order.cancel_order(str(db), oid)
    assert ok is True
    assert record_sbi_order.list_open_orders(str(db)) == []

    # Already cancelled → no-op
    ok2 = record_sbi_order.cancel_order(str(db), oid)
    assert ok2 is False
