"""Tests for cross-asset leading indicator system."""
import sqlite3
import math
import pytest

from cross_asset_signals import (
    compute_cross_asset_regime_signal,
    ensure_cross_asset_table,
    save_cross_asset_snapshot,
    load_latest_cross_asset,
    _sigmoid,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    ensure_cross_asset_table(c)
    return c


class TestCrossAssetScore:
    def test_normal_data_score_in_range(self, conn):
        snapshot = {
            "asof": "2026-04-08",
            "sp500_overnight_pct": 1.2,
            "usdjpy_change_pct": 0.3,
            "vix_change_pct": -2.5,
            "nk_futures_gap_pct": 0.8,
        }
        result = compute_cross_asset_regime_signal(snapshot)
        assert 0.0 <= result["cross_asset_score"] <= 1.0

    def test_strong_risk_on(self, conn):
        """美股涨 + VIX跌 + 期货涨 → score > 0.65"""
        snapshot = {
            "asof": "2026-04-08",
            "sp500_overnight_pct": 2.5,
            "usdjpy_change_pct": 0.8,
            "vix_change_pct": -8.0,
            "nk_futures_gap_pct": 2.0,
        }
        result = compute_cross_asset_regime_signal(snapshot)
        assert result["cross_asset_score"] > 0.65
        assert result["regime_adjustment"] == "upgrade"

    def test_strong_risk_off(self, conn):
        """美股跌 + VIX暴涨 + 期货跌 → score < 0.35"""
        snapshot = {
            "asof": "2026-04-08",
            "sp500_overnight_pct": -3.0,
            "usdjpy_change_pct": -1.0,
            "vix_change_pct": 15.0,
            "nk_futures_gap_pct": -2.5,
        }
        result = compute_cross_asset_regime_signal(snapshot)
        assert result["cross_asset_score"] < 0.35
        assert result["regime_adjustment"] == "downgrade"

    def test_partial_data_no_error(self, conn):
        """部分データ欠損でもエラーにならない"""
        snapshot = {
            "asof": "2026-04-08",
            "sp500_overnight_pct": 0.5,
            # usdjpy, vix, futures all missing
        }
        result = compute_cross_asset_regime_signal(snapshot)
        assert 0.0 <= result["cross_asset_score"] <= 1.0

    def test_all_missing_returns_neutral(self, conn):
        """全データ欠損 → 中性(≈0.5)"""
        snapshot = {"asof": "2026-04-08"}
        result = compute_cross_asset_regime_signal(snapshot)
        assert abs(result["cross_asset_score"] - 0.5) < 0.05
        assert result["regime_adjustment"] == "neutral"

    def test_db_roundtrip(self, conn):
        snapshot = {
            "asof": "2026-04-08",
            "fetch_ts": "2026-04-08T07:30:00+09:00",
            "sp500_close": 5123.4,
            "sp500_overnight_pct": 1.2,
            "usdjpy": 148.3,
            "usdjpy_change_pct": 0.45,
            "vix_close": 22.1,
            "vix_change_pct": -3.2,
            "nk_futures": 37250.0,
            "nk_futures_gap_pct": 1.2,
        }
        signal = compute_cross_asset_regime_signal(snapshot)
        save_cross_asset_snapshot(conn, snapshot, signal)

        loaded = load_latest_cross_asset(conn, "2026-04-08")
        assert loaded is not None
        assert loaded["asof"] == "2026-04-08"
        assert abs(loaded["cross_asset_score"] - signal["cross_asset_score"]) < 0.001
        assert loaded["sp500_close"] == 5123.4

    def test_with_db_history_zscore(self, conn):
        """DB に履歴があるとき z-score が使われる"""
        # 20日分のダミーデータ
        for i in range(20):
            conn.execute(
                """INSERT INTO cross_asset_snapshots
                   (asof, ts, sp500_overnight_pct, usdjpy_change_pct,
                    vix_change_pct, nk_futures_gap_pct)
                   VALUES (?, '', ?, ?, ?, ?)""",
                (f"2026-03-{10+i:02d}", 0.1 * (i % 3 - 1), 0.05, -0.5, 0.2),
            )
        conn.commit()

        snapshot = {
            "asof": "2026-04-08",
            "sp500_overnight_pct": 3.0,  # 大幅プラス
            "usdjpy_change_pct": 0.5,
            "vix_change_pct": -5.0,
            "nk_futures_gap_pct": 1.5,
        }
        result = compute_cross_asset_regime_signal(snapshot, conn=conn)
        assert result["cross_asset_score"] > 0.6


class TestSigmoid:
    def test_zero(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-6

    def test_large_positive(self):
        assert _sigmoid(100.0) > 0.999

    def test_large_negative(self):
        assert _sigmoid(-100.0) < 0.001
