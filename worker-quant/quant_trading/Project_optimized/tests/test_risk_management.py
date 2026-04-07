"""Tests for risk management hardening (2026-04-07 patch).

Covers:
  - ATR computation (_compute_atr_pct)
  - sprint_exit_check_v2 (price stop-loss + trailing protect)
  - Stop-loss integration in make_decision (_check_stop_losses)
  - build_positions high_since_entry tracking
"""
import sqlite3
import math
import pytest
import pandas as pd

from sprint_signal import sprint_exit_check_v2, sprint_exit_check
from make_decision import _compute_atr_pct, _check_stop_losses
from trade_schema import ensure_trade_tables
from execution_model import sbi_fee


def _create_test_db():
    """Create an in-memory DB with required tables and sample data."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE daily_prices (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("""
        CREATE TABLE tickers (
            symbol TEXT PRIMARY KEY, sector TEXT
        )
    """)
    ensure_trade_tables(conn)
    return conn


def _insert_prices(conn, symbol, prices):
    """Insert [(date, open, high, low, close, volume), ...]."""
    conn.executemany(
        "INSERT INTO daily_prices(symbol, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
        [(symbol,) + p for p in prices],
    )
    conn.commit()


# ============================================================
# T1: ATR computation
# ============================================================

class TestATRComputation:
    def test_normal_20_day(self):
        conn = _create_test_db()
        # 21 days of data (need window+1)
        prices = []
        for i in range(21):
            d = f"2026-03-{10+i:02d}"
            c = 1000.0 + i * 5  # close goes from 1000 to 1100
            h = c + 10
            l = c - 10
            prices.append((d, c, h, l, c, 1000000))
        _insert_prices(conn, "TEST.T", prices)
        atr_pct = _compute_atr_pct(conn, "TEST.T", "2026-03-30", window=20)
        assert atr_pct is not None
        assert 0.0 < atr_pct < 0.1  # reasonable range

    def test_insufficient_data(self):
        conn = _create_test_db()
        _insert_prices(conn, "TEST.T", [("2026-01-01", 100, 105, 95, 100, 1000)])
        atr_pct = _compute_atr_pct(conn, "TEST.T", "2026-01-01", window=20)
        assert atr_pct is None

    def test_high_volatility(self):
        conn = _create_test_db()
        prices = []
        for i in range(25):
            d = f"2026-03-{1+i:02d}" if i < 28 else f"2026-04-{i-27:02d}"
            c = 500.0 + (50 if i % 2 == 0 else -50)  # oscillates 450-550
            h = c + 30
            l = c - 30
            prices.append((d, c, h, l, c, 1000000))
        _insert_prices(conn, "VOL.T", prices)
        atr_pct = _compute_atr_pct(conn, "VOL.T", "2026-03-25", window=20)
        assert atr_pct is not None
        assert atr_pct > 0.05  # high vol stock


# ============================================================
# T2: sprint_exit_check_v2
# ============================================================

class TestSprintExitCheckV2:
    def _base_row(self):
        return pd.Series({
            "vol_z": 0.5,
            "holding_period_target": 5,
        })

    def test_price_stop_loss_triggered(self):
        """Price drops below ATR×3 stop line → exit."""
        row = self._base_row()
        should_exit, reason = sprint_exit_check_v2(
            row, holding_days=2, benchmark_state="on",
            avg_cost=1000.0, current_price=940.0,  # -6%, ATR×3=4.5% → triggered
            atr_pct=0.015,
            stop_loss_config={"stop_loss_vol_mult": 3.0, "stop_loss_min_pct": 0.04, "stop_loss_max_pct": 0.12},
        )
        assert should_exit is True
        assert reason == "price_stop_loss"

    def test_price_above_stop_line(self):
        """Price above stop line → no exit from stop-loss."""
        row = self._base_row()
        should_exit, reason = sprint_exit_check_v2(
            row, holding_days=2, benchmark_state="on",
            avg_cost=1000.0, current_price=980.0,  # -2%, stop is ~4.5%
            atr_pct=0.015,
            stop_loss_config={"stop_loss_vol_mult": 3.0, "stop_loss_min_pct": 0.04, "stop_loss_max_pct": 0.12},
        )
        assert should_exit is False

    def test_trailing_protect_triggered(self):
        """Profit > 3% then drops 2% from high → trailing exit."""
        row = self._base_row()
        should_exit, reason = sprint_exit_check_v2(
            row, holding_days=3, benchmark_state="on",
            avg_cost=1000.0,
            current_price=1045.0,
            high_since_entry=1070.0,  # was up 7%, now 4.5%
            atr_pct=0.015,
            stop_loss_config={
                "stop_loss_vol_mult": 3.0, "stop_loss_min_pct": 0.04, "stop_loss_max_pct": 0.12,
                "trailing_activate_pct": 0.03, "trailing_stop_pct": 0.02,
            },
        )
        assert should_exit is True
        assert reason == "trailing_protect"

    def test_trailing_not_activated(self):
        """Profit < 3% → trailing not activated even if price drops."""
        row = self._base_row()
        should_exit, reason = sprint_exit_check_v2(
            row, holding_days=2, benchmark_state="on",
            avg_cost=1000.0,
            current_price=1010.0,
            high_since_entry=1020.0,  # only 2% profit, not enough to activate
            atr_pct=0.015,
            stop_loss_config={
                "stop_loss_vol_mult": 3.0, "stop_loss_min_pct": 0.04, "stop_loss_max_pct": 0.12,
                "trailing_activate_pct": 0.03, "trailing_stop_pct": 0.02,
            },
        )
        assert should_exit is False

    def test_no_cost_data_degrades(self):
        """Without avg_cost → degrades to original 3-condition check."""
        row = self._base_row()
        should_exit, reason = sprint_exit_check_v2(
            row, holding_days=2, benchmark_state="on",
        )
        assert should_exit is False

    def test_benchmark_off_priority(self):
        """benchmark_off still takes priority over price stop-loss."""
        row = self._base_row()
        should_exit, reason = sprint_exit_check_v2(
            row, holding_days=0, benchmark_state="off",
            avg_cost=1000.0, current_price=999.0,
        )
        assert should_exit is True
        assert reason == "benchmark_off"

    def test_stop_loss_min_pct_floor(self):
        """When ATR is very small, min_pct acts as floor."""
        row = self._base_row()
        # ATR×3 = 0.3%, but min_pct = 4%
        should_exit, reason = sprint_exit_check_v2(
            row, holding_days=2, benchmark_state="on",
            avg_cost=1000.0, current_price=965.0,  # -3.5%, below 4% stop
            atr_pct=0.001,
            stop_loss_config={"stop_loss_vol_mult": 3.0, "stop_loss_min_pct": 0.04, "stop_loss_max_pct": 0.12},
        )
        assert should_exit is False  # 3.5% < 4% min floor, not triggered

    def test_stop_loss_max_pct_cap(self):
        """When ATR×mult > max_pct, max_pct acts as cap."""
        row = self._base_row()
        # ATR×3 = 30%, but max_pct = 12%
        should_exit, reason = sprint_exit_check_v2(
            row, holding_days=2, benchmark_state="on",
            avg_cost=1000.0, current_price=870.0,  # -13%, beyond max 12%
            atr_pct=0.10,
            stop_loss_config={"stop_loss_vol_mult": 3.0, "stop_loss_min_pct": 0.04, "stop_loss_max_pct": 0.12},
        )
        assert should_exit is True
        assert reason == "price_stop_loss"

    def test_no_atr_uses_min_pct(self):
        """When atr_pct is None, fallback to min_pct."""
        row = self._base_row()
        should_exit, reason = sprint_exit_check_v2(
            row, holding_days=2, benchmark_state="on",
            avg_cost=1000.0, current_price=950.0,  # -5% > 4% min
            atr_pct=None,
            stop_loss_config={"stop_loss_vol_mult": 3.0, "stop_loss_min_pct": 0.04, "stop_loss_max_pct": 0.12},
        )
        assert should_exit is True
        assert reason == "price_stop_loss"


# ============================================================
# T3: _check_stop_losses integration
# ============================================================

class TestCheckStopLosses:
    def test_stop_loss_triggers_sell(self):
        conn = _create_test_db()
        # Insert 25 days of low-volatility prices (ATR ~1%), then a crash
        prices = []
        for i in range(24):
            d = f"2026-03-{5+i:02d}" if 5+i <= 31 else f"2026-04-{5+i-31:02d}"
            prices.append((d, 1000, 1005, 995, 1000, 500000))
        # Day 25: crash to 940 (-6%)
        prices.append(("2026-03-29", 960, 965, 935, 940, 1000000))
        _insert_prices(conn, "FALL.T", prices)

        # Position bought at 1000, now crashed to 940
        conn.execute("DELETE FROM positions")
        conn.execute("""
            INSERT INTO positions(asof, strategy_id, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl, high_since_entry, entry_date)
            VALUES ('2026-03-29', 'sprint', 'FALL.T', 100, 1000.0, 940.0, 94000, -6000, 1005, '2026-03-05')
        """)
        conn.commit()

        stop_cfg = {
            "stop_loss_vol_mult": 3.0,
            "stop_loss_min_pct": 0.04,
            "stop_loss_max_pct": 0.12,
            "trailing_activate_pct": 0.03,
            "trailing_stop_pct": 0.02,
            "atr_window": 20,
        }
        forced, diag = _check_stop_losses(conn, "2026-03-29", "sprint", stop_cfg)
        assert len(diag) == 1
        assert diag[0]["symbol"] == "FALL.T"
        # ATR ~1%, stop_pct = max(1%*3, 4%) = 4%, stop_price = 960
        # current_price 940 < 960 → triggered
        assert diag[0]["triggered"] is True
        assert "FALL.T" in forced

    def test_no_trigger_when_price_healthy(self):
        conn = _create_test_db()
        prices = []
        for i in range(25):
            d = f"2026-03-{5+i:02d}" if 5+i <= 31 else f"2026-04-{5+i-31:02d}"
            prices.append((d, 1000, 1015, 995, 1000 + i, 500000))
        _insert_prices(conn, "RISE.T", prices)

        conn.execute("DELETE FROM positions")
        conn.execute("""
            INSERT INTO positions(asof, strategy_id, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl, high_since_entry, entry_date)
            VALUES ('2026-03-29', 'sprint', 'RISE.T', 100, 1000.0, 1024.0, 102400, 2400, 1024, '2026-03-05')
        """)
        conn.commit()

        stop_cfg = {
            "stop_loss_vol_mult": 3.0,
            "stop_loss_min_pct": 0.04,
            "stop_loss_max_pct": 0.12,
            "trailing_activate_pct": 0.03,
            "trailing_stop_pct": 0.02,
            "atr_window": 20,
        }
        forced, diag = _check_stop_losses(conn, "2026-03-29", "sprint", stop_cfg)
        assert len(forced) == 0


# ============================================================
# T4: build_positions high_since_entry tracking
# ============================================================

class TestBuildPositionsTracking:
    def test_new_entry_sets_high(self):
        from build_positions import build_positions
        conn = _create_test_db()
        _insert_prices(conn, "NEW.T", [("2026-04-01", 500, 520, 490, 510, 100000)])

        # Insert a fill for a new buy
        conn.execute("""
            INSERT INTO fills(fill_id, run_id, asof, strategy_id, ts, symbol, side, qty, price)
            VALUES ('f1', 'run1', '2026-04-01', 'sprint', '2026-04-01 14:30', 'NEW.T', 'BUY', 100, 505)
        """)
        conn.commit()

        build_positions(conn, "run1", "2026-04-01", strategy_id="sprint")
        row = conn.execute(
            "SELECT high_since_entry, entry_date FROM positions WHERE symbol='NEW.T' AND strategy_id='sprint'"
        ).fetchone()
        assert row is not None
        high, entry = row
        assert high >= 505  # at least entry price
        assert entry == "2026-04-01"

    def test_high_updates_on_new_day(self):
        from build_positions import build_positions
        conn = _create_test_db()
        _insert_prices(conn, "UP.T", [
            ("2026-04-01", 500, 520, 490, 510, 100000),
            ("2026-04-02", 510, 550, 505, 540, 120000),  # new high 550
        ])

        # Day 1: buy
        conn.execute("""
            INSERT INTO fills(fill_id, run_id, asof, strategy_id, ts, symbol, side, qty, price)
            VALUES ('f1', 'run1', '2026-04-01', 'sprint', '2026-04-01 14:30', 'UP.T', 'BUY', 100, 505)
        """)
        conn.commit()
        build_positions(conn, "run1", "2026-04-01", strategy_id="sprint")

        # Day 2: no fills, just rebuild with new market data
        conn.execute("""
            INSERT INTO fills(fill_id, run_id, asof, strategy_id, ts, symbol, side, qty, price)
            VALUES ('f2', 'run2', '2026-04-02', 'sprint', '2026-04-02 14:30', 'UP.T', 'BUY', 0, 540)
        """)
        conn.commit()
        # Actually we need a fill to trigger rebuild, but qty=0 won't work.
        # Instead, let's just call build with an empty run
        build_positions(conn, "run_empty", "2026-04-02", strategy_id="sprint")

        row = conn.execute(
            "SELECT high_since_entry FROM positions WHERE symbol='UP.T' AND asof='2026-04-02' AND strategy_id='sprint'"
        ).fetchone()
        assert row is not None
        assert row[0] >= 550  # should pick up today's high

    def test_sell_clears_tracking(self):
        from build_positions import build_positions
        conn = _create_test_db()
        _insert_prices(conn, "SELL.T", [("2026-04-01", 500, 520, 490, 510, 100000),
                                         ("2026-04-02", 510, 515, 505, 512, 100000)])

        # Day 1: buy
        conn.execute("""
            INSERT INTO fills(fill_id, run_id, asof, strategy_id, ts, symbol, side, qty, price)
            VALUES ('f1', 'run1', '2026-04-01', 'sprint', '2026-04-01 14:30', 'SELL.T', 'BUY', 100, 505)
        """)
        conn.commit()
        build_positions(conn, "run1", "2026-04-01", strategy_id="sprint")

        # Day 2: sell all (uses day-1 position as prev)
        conn.execute("""
            INSERT INTO fills(fill_id, run_id, asof, strategy_id, ts, symbol, side, qty, price)
            VALUES ('f2', 'run2', '2026-04-02', 'sprint', '2026-04-02 09:30', 'SELL.T', 'SELL', 100, 510)
        """)
        conn.commit()
        build_positions(conn, "run2", "2026-04-02", strategy_id="sprint")

        rows = conn.execute(
            "SELECT * FROM positions WHERE symbol='SELL.T' AND asof='2026-04-02' AND strategy_id='sprint'"
        ).fetchall()
        assert len(rows) == 0  # position fully closed


# ============================================================
# Backward compatibility
# ============================================================

# ============================================================
# T5: SBI fee schedule
# ============================================================

class TestSBIFee:
    def test_zero_revolution(self):
        assert sbi_fee(50000, "sbi_zero") == 0.0
        assert sbi_fee(1000000, "sbi_zero") == 0.0

    def test_standard_plan_tiers(self):
        """SBI スタンダードプラン 手数料テーブル検証"""
        assert sbi_fee(30000, "sbi_standard") == 55.0      # ~5万
        assert sbi_fee(50000, "sbi_standard") == 55.0      # ちょうど5万
        assert sbi_fee(63200, "sbi_standard") == 99.0      # 5万超~10万 (9432.T売却相当)
        assert sbi_fee(124750, "sbi_standard") == 115.0    # 10万超~20万 (7267.T売却相当)
        assert sbi_fee(300000, "sbi_standard") == 275.0    # 20万超~50万
        assert sbi_fee(800000, "sbi_standard") == 535.0    # 50万超~100万
        assert sbi_fee(1200000, "sbi_standard") == 640.0   # 100万超~150万
        assert sbi_fee(5000000, "sbi_standard") == 1013.0  # 150万超~3000万
        assert sbi_fee(50000000, "sbi_standard") == 1070.0 # 3000万超

    def test_unknown_mode_returns_zero(self):
        assert sbi_fee(100000, "unknown_mode") == 0.0


class TestBackwardCompatibility:
    def test_original_exit_check_unchanged(self):
        """Original sprint_exit_check still works independently."""
        row = pd.Series({"vol_z": 0.5, "holding_period_target": 5})
        ok, reason = sprint_exit_check(row, 2, "on")
        assert ok is False
        ok2, reason2 = sprint_exit_check(row, 2, "off")
        assert ok2 is True
        assert reason2 == "benchmark_off"

    def test_v2_without_new_params_matches_v1(self):
        """v2 without cost params behaves like v1."""
        row = pd.Series({"vol_z": -0.8, "holding_period_target": 5})
        ok1, r1 = sprint_exit_check(row, 2, "on")
        ok2, r2 = sprint_exit_check_v2(row, 2, "on")
        assert ok1 == ok2
        assert r1 == r2
