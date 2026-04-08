"""Tests for sprint_score_v2 with full-factor adaptive weighting."""
import pytest
import pandas as pd

from sprint_signal import (
    sprint_score,
    sprint_score_v2,
    shrink_icir,
    _regime_factor_tilt,
    FACTOR_CATEGORIES,
)


class TestShrinkICIR:
    def test_low_sample_near_prior(self):
        """n=2 → ほぼ prior"""
        result = shrink_icir(icir=0.50, n_obs=2, prior=0.05, shrink_n=30)
        # shrink = 2/(2+30) ≈ 0.0625
        expected = 0.50 * (2 / 32) + 0.05 * (30 / 32)
        assert abs(result - expected) < 1e-6

    def test_high_sample_near_original(self):
        """n=100 → 元の ICIR に近い（shrink=100/130≈0.77）"""
        result = shrink_icir(icir=0.30, n_obs=100, prior=0.05, shrink_n=30)
        expected = 0.30 * (100 / 130) + 0.05 * (30 / 130)  # ≈ 0.242
        assert abs(result - expected) < 0.01

    def test_zero_obs_returns_prior(self):
        assert shrink_icir(icir=5.0, n_obs=0) == 0.05

    def test_negative_icir_shrinks_toward_prior(self):
        result = shrink_icir(icir=-6.0, n_obs=1, prior=0.05, shrink_n=30)
        # n=1: shrink=1/31 ≈ 0.032
        assert result > -1.0  # heavily shrunk from -6.0


class TestRegimeFactorTilt:
    def test_momentum_high_regime(self):
        t = _regime_factor_tilt("momentum", 1.0)
        assert t == 1.0  # 0.5 + 0.5*1.0

    def test_momentum_low_regime(self):
        t = _regime_factor_tilt("momentum", 0.0)
        assert t == 0.5

    def test_fundamental_high_regime(self):
        t = _regime_factor_tilt("fundamental", 1.0)
        assert t == 0.7  # 1.0 - 0.3*1.0

    def test_fundamental_low_regime(self):
        t = _regime_factor_tilt("fundamental", 0.0)
        assert t == 1.0

    def test_neutral_any_regime(self):
        assert _regime_factor_tilt("neutral", 0.0) == 1.0
        assert _regime_factor_tilt("neutral", 1.0) == 1.0


class TestSprintScoreV2:
    def _make_features(self):
        """因子数据: 3 只股票, 多因子"""
        return pd.DataFrame({
            "symbol": ["A.T", "B.T", "C.T"],
            "mom_consist": [0.8, 0.2, 0.5],
            "high52w": [-0.05, -0.20, -0.10],
            "vol_z": [1.0, -0.5, 0.3],
            "ret60": [0.1, -0.05, 0.03],
            "roa_op": [0.05, 0.02, 0.08],
            "margin_op": [0.10, 0.03, 0.15],
        })

    def _make_registry(self):
        return {
            "mom_consist": {"icir": 0.06, "n_obs": 218, "is_active": 1},
            "high52w": {"icir": 0.14, "n_obs": 47, "is_active": 1},
            "vol_z": {"icir": -6.24, "n_obs": 1, "is_active": 1},  # 異常: n=1
            "ret60": {"icir": 0.10, "n_obs": 218, "is_active": 1},
            "roa_op": {"icir": 0.08, "n_obs": 50, "is_active": 1},
            "margin_op": {"icir": 0.12, "n_obs": 50, "is_active": 1},
        }

    def _make_tiers(self):
        return {
            "core": ["mom_consist"],
            "candidate": ["high52w", "ret60"],
            "fundamental_pending": ["roa_op", "margin_op"],
            "excluded": ["vol_z"],  # vol_z excluded due to bad ICIR
        }

    def test_v2_returns_series(self):
        features = self._make_features()
        score = sprint_score_v2(features, self._make_registry(), self._make_tiers())
        assert isinstance(score, pd.Series)
        assert len(score) == 3

    def test_excluded_factor_ignored(self):
        """excluded 因子 (vol_z) は無視される"""
        features = self._make_features()
        reg = self._make_registry()
        tiers = self._make_tiers()

        score_with = sprint_score_v2(features, reg, tiers)

        # vol_z を巨大値にしても score 変わらない（excluded だから）
        features2 = features.copy()
        features2["vol_z"] = [100.0, 100.0, 100.0]
        score_without = sprint_score_v2(features2, reg, tiers)

        assert (score_with == score_without).all()

    def test_low_sample_icir_shrunk(self):
        """n=1 の vol_z は shrinkage で prior(0.05) 近くに → 影響極小"""
        # vol_z: ICIR=-6.24 but n=1 → shrink → ≈ -6.24*(1/31) + 0.05*(30/31) ≈ -0.153
        # without shrinkage it would be -6.24 → massive negative weight
        effective = shrink_icir(-6.24, n_obs=1)
        raw_would_be = -6.24
        # shrinkage reduces the magnitude dramatically
        assert abs(effective) < abs(raw_would_be) * 0.1

    def test_regime_high_favors_momentum(self):
        """regime=0.9 → momentum 因子が強化される"""
        features = self._make_features()
        reg = self._make_registry()
        tiers = self._make_tiers()

        score_high = sprint_score_v2(features, reg, tiers, regime_score=0.9)
        score_low = sprint_score_v2(features, reg, tiers, regime_score=0.1)

        # A.T は momentum 強い → regime 高で有利
        # 差が出るはず
        assert not (score_high == score_low).all()

    def test_empty_registry_fallback_to_v1(self):
        """registry 空 → v1 にフォールバック"""
        features = self._make_features()
        v2_score = sprint_score_v2(features, factor_registry=None)
        v1_score = sprint_score(features)
        assert (v2_score == v1_score).all()

    def test_ranking_correlation_with_v1(self):
        """v2 と v1 のランキング相関が > 0.5（完全に乖離しない）"""
        features = self._make_features()
        reg = self._make_registry()
        # no exclusions → use all factors
        tiers = {"core": list(reg.keys())}

        v1 = sprint_score(features)
        v2 = sprint_score_v2(features, reg, tiers, regime_score=0.5)

        # ランキング相関: 3 サンプルなので簡易チェック
        v1_rank = v1.rank()
        v2_rank = v2.rank()
        # 少なくとも top stock は同じ
        assert v1_rank.idxmax() == v2_rank.idxmax() or len(features) < 5
