"""Tests for regime v2 continuous scoring."""
import sqlite3
import pytest
import numpy as np

from benchmark_regime import (
    compute_regime_score_v2,
    regime_score_to_scale,
    compute_ma_slope,
    compute_volume_ratio,
    benchmark_regime_scale_v2,  # v1 backward compat
)


def _create_db_with_prices(n_days=30, base_close=56000.0, trend=0.0, base_vol=1000000):
    """Helper: create in-memory DB with daily_prices."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE daily_prices (
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, date))
    """)
    for i in range(n_days):
        d = f"2026-03-{1+i:02d}" if 1+i <= 31 else f"2026-04-{1+i-31:02d}"
        c = base_close + trend * i
        conn.execute(
            "INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?)",
            ("1321.T", d, c - 50, c + 100, c - 100, c, base_vol),
        )
    conn.commit()
    return conn


class TestRegimeScoreV2:
    def test_strong_risk_on(self):
        """MA金叉 + 正斜率 + 价格在MA上方 + 强量能 + 跨资产看多 → score > 0.7"""
        score, details = compute_regime_score_v2(
            px_b=58000.0,
            fast_ma=57000.0,
            slow_ma=56000.0,   # 金叉: fast > slow
            ma_slope_5d=0.003,  # +0.3%/day
            volume_ratio=3.0,
            cross_asset_score=0.8,
        )
        assert score > 0.70
        assert details["components"]["ma_signal"] > 0

    def test_strong_risk_off(self):
        """MA死叉 + 负斜率 + 价格在MA下方 + 低量能 + 跨资产看空 → score < 0.3"""
        score, details = compute_regime_score_v2(
            px_b=54000.0,
            fast_ma=55000.0,
            slow_ma=56500.0,   # 死叉
            ma_slope_5d=-0.002,
            volume_ratio=0.5,
            cross_asset_score=0.15,
        )
        assert score < 0.30

    def test_04_08_scenario(self):
        """模拟 04-07→04-08: MA死叉但跨资产和量能大幅正面 → score 在中间区域"""
        score, _ = compute_regime_score_v2(
            px_b=55950.0,        # 04-07 收盘
            fast_ma=55559.0,
            slow_ma=56638.0,     # 死叉
            ma_slope_5d=-0.0004, # 轻微下降
            volume_ratio=1.0,    # 正常量能
            cross_asset_score=0.75,  # 美股隔夜大涨
        )
        # v1 给 OFF (scale=0.25), v2 应该允许部分仓位
        assert 0.30 < score < 0.65

    def test_missing_optional_inputs(self):
        """slope/volume/cross_asset 全 None → 不报错，回到纯 MA 判断"""
        score, details = compute_regime_score_v2(
            px_b=57000.0,
            fast_ma=57500.0,
            slow_ma=56000.0,
        )
        assert 0.0 <= score <= 1.0

    def test_missing_benchmark_returns_zero(self):
        score, details = compute_regime_score_v2(
            px_b=float("nan"), fast_ma=0.0, slow_ma=0.0,
        )
        assert score == 0.0
        assert "missing" in details["diagnosis"]


class TestRegimeScoreToScale:
    def test_above_full_threshold(self):
        assert regime_score_to_scale(0.85) == 1.0

    def test_below_zero_threshold(self):
        assert regime_score_to_scale(0.10, off_scale=0.0) == 0.0
        assert regime_score_to_scale(0.10, off_scale=0.25) == 0.25

    def test_linear_interpolation(self):
        # midpoint between 0.15 and 0.70
        mid = (0.15 + 0.70) / 2  # 0.425
        scale = regime_score_to_scale(mid, full_threshold=0.70, zero_threshold=0.15, off_scale=0.0)
        assert abs(scale - 0.5) < 0.01

    def test_exact_thresholds(self):
        assert regime_score_to_scale(0.70) == 1.0
        assert regime_score_to_scale(0.15, off_scale=0.0) == 0.0


class TestMASlope:
    def test_uptrend_positive_slope(self):
        conn = _create_db_with_prices(n_days=30, base_close=55000, trend=100)
        slope = compute_ma_slope(conn, "1321.T", "2026-03-30")
        assert slope is not None
        assert slope > 0

    def test_downtrend_negative_slope(self):
        conn = _create_db_with_prices(n_days=30, base_close=58000, trend=-100)
        slope = compute_ma_slope(conn, "1321.T", "2026-03-30")
        assert slope is not None
        assert slope < 0

    def test_insufficient_data(self):
        conn = _create_db_with_prices(n_days=5)
        slope = compute_ma_slope(conn, "1321.T", "2026-03-05")
        assert slope is None


class TestVolumeRatio:
    def test_normal_volume(self):
        conn = _create_db_with_prices(n_days=25, base_vol=1000000)
        ratio = compute_volume_ratio(conn, "1321.T", "2026-03-25")
        assert ratio is not None
        assert abs(ratio - 1.0) < 0.1  # 定量のため≈1

    def test_insufficient_data(self):
        conn = _create_db_with_prices(n_days=1)
        ratio = compute_volume_ratio(conn, "1321.T", "2026-03-01")
        assert ratio is None


class TestV1BackwardCompat:
    def test_v1_unchanged(self):
        """v1 逻辑不受 v2 新增代码影响"""
        state, scale, info = benchmark_regime_scale_v2(
            px_b=55950.0, fast_ma_b=55559.0, slow_ma_b=56638.0,
            prev_state="off", enter_pct=0.01, exit_pct=0.01,
            off_scale=0.25, caution_scale=0.6,
        )
        assert state == "off"
        assert scale == 0.25
