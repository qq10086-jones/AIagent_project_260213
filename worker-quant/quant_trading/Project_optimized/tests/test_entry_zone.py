"""Tests for entry zone (limit price recommendation) system."""
import pytest

from make_decision import compute_entry_zone


class TestEntryZone:
    def test_buy_order_relationships(self):
        """BUY: target < last_close < aggressive < walk_away"""
        ez = compute_entry_zone("TEST.T", "BUY", 1000.0, atr_pct=0.03)
        assert ez["target_limit"] < 1000.0
        assert ez["aggressive_limit"] > 1000.0
        assert ez["walk_away_price"] > ez["aggressive_limit"]

    def test_sell_order_relationships(self):
        """SELL: walk_away < aggressive < last_close < target"""
        ez = compute_entry_zone("TEST.T", "SELL", 1000.0, atr_pct=0.03)
        assert ez["target_limit"] > 1000.0
        assert ez["aggressive_limit"] < 1000.0
        assert ez["walk_away_price"] < ez["aggressive_limit"]

    def test_high_volatility_wider_zone(self):
        """高 ATR → 区間が広い"""
        ez_low = compute_entry_zone("A.T", "BUY", 1000.0, atr_pct=0.01)
        ez_high = compute_entry_zone("A.T", "BUY", 1000.0, atr_pct=0.05)
        spread_low = ez_high["walk_away_price"] - ez_low["walk_away_price"]
        assert spread_low > 0  # high vol has wider walk_away

    def test_low_volatility_narrower_zone(self):
        ez_low = compute_entry_zone("A.T", "BUY", 1000.0, atr_pct=0.005)
        assert ez_low["walk_away_price"] - 1000.0 < 10  # very narrow

    def test_none_atr_fallback(self):
        """ATR=None → fallback 2%, 不报错"""
        ez = compute_entry_zone("A.T", "BUY", 1000.0, atr_pct=None)
        assert ez["target_limit"] < 1000.0
        assert ez["walk_away_price"] > 1000.0

    def test_extreme_atr_capped(self):
        """ATR > 10% → cap at 10%"""
        ez = compute_entry_zone("A.T", "BUY", 1000.0, atr_pct=0.30)
        # walk_away = 1000 * (1 + 0.10 * 1.0) = 1100 (capped)
        assert ez["walk_away_price"] <= 1100.0

    def test_output_rounded(self):
        """出力は小数点1位に丸め"""
        ez = compute_entry_zone("A.T", "BUY", 1234.5678, atr_pct=0.025)
        for key in ["target_limit", "aggressive_limit", "walk_away_price"]:
            assert ez[key] == round(ez[key], 1)
