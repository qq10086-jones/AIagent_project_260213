"""Model primitives extracted from ss7_sqlite_news_overlay.py.

Contains: RSI, slope_log_price, make_features, make_target, PanelRidge.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ── Technical helpers ────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)
    ru = up.ewm(alpha=1 / period, adjust=False).mean()
    rd = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def slope_log_price(close: pd.Series, window: int = 60) -> pd.Series:
    arr = close.to_numpy(dtype=float)
    y = np.log(np.clip(arr, 1e-12, None))
    w = int(window)
    if w < 2:
        return pd.Series(np.full_like(y, np.nan, dtype=float), index=close.index)
    x = np.arange(w, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum()) + 1e-12
    numer = np.convolve(y, x[::-1], mode="valid")
    out = np.full_like(y, np.nan, dtype=float)
    out[w - 1 :] = numer / denom
    out[~np.isfinite(out)] = np.nan
    return pd.Series(out, index=close.index)


# ── Feature engineering ──────────────────────────────────────

def make_features(prices: pd.DataFrame, volumes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    feats = {}
    for tkr in prices.columns:
        c = prices[tkr]
        df = pd.DataFrame(index=prices.index)

        # ── original 10 factors ──────────────────────────────────────────
        df["ret1"] = c.pct_change()
        df["ret5"] = c.pct_change(5)
        df["ret20"] = c.pct_change(20)
        df["ret60"] = c.pct_change(60)
        df["vol20"] = df["ret1"].rolling(20).std()
        df["vol60"] = df["ret1"].rolling(60).std()
        ma50 = c.rolling(50).mean()
        ma200 = c.rolling(200).mean()
        df["ma_gap"] = (ma50 / (ma200 + 1e-12)) - 1.0
        df["z_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-12)
        df["rsi14"] = rsi(c, 14) / 100.0
        df["slope60"] = slope_log_price(c, 60)

        # ── academic alpha factors ───────────────────────────────────
        df["mom_12_1"] = c.pct_change(252) - c.pct_change(21)
        df["high52w"] = (c / c.rolling(252).max().clip(lower=1e-6)) - 1.0
        df["vol_adj_mom20"] = df["ret20"] / (df["vol20"] + 1e-12)
        df["mom_consist"] = df["ret1"].rolling(63).apply(
            lambda x: float((x > 0).mean()), raw=True
        )

        if volumes is not None and tkr in volumes.columns:
            v = volumes[tkr].replace(0, np.nan)
            log_v = np.log(v.clip(lower=1.0))
            df["vol_z"] = (log_v - log_v.rolling(60).mean()) / (log_v.rolling(60).std() + 1e-12)
        else:
            df["vol_z"] = 0.0

        feats[tkr] = df
    out = pd.concat(feats, axis=1).sort_index()
    return out


def make_target(prices: pd.DataFrame, H: int = 20) -> pd.DataFrame:
    """Target: forward return / vol20 (risk-adjusted forward)."""
    ret1 = prices.pct_change()
    fwd = prices.shift(-int(H)) / prices - 1.0
    vol20 = ret1.rolling(20).std()
    return fwd / (vol20 + 1e-12)


# ── Ridge model ──────────────────────────────────────────────

class PanelRidge:
    """Ridge with z-score standardization + intercept."""

    def __init__(self, alpha: float = 50.0):
        self.alpha = float(alpha)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0) + 1e-12
        Xs = (X - self.mean_) / self.std_
        Xd = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
        n_feat = Xd.shape[1]
        I = np.eye(n_feat, dtype=float)
        I[0, 0] = 0.0
        A = Xd.T @ Xd + self.alpha * I
        b = Xd.T @ y
        self.beta_ = np.linalg.solve(A, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        Xd = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
        return Xd @ self.beta_

    def fit_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        candidate_alphas: Optional[List[float]] = None,
        n_splits: int = 5,
    ) -> dict:
        """Time-series cross-validation to select optimal alpha.

        Evaluation metric: mean cross-section Spearman IC on held-out folds.
        After selection, refits on full data with the best alpha.

        Returns dict with cv_results, best_alpha, best_ic.
        """
        if candidate_alphas is None:
            candidate_alphas = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]

        if n < n_splits + 1:
            self.fit(X, y)
            return {"best_alpha": self.alpha, "best_ic": float("nan"), "cv_results": {}, "note": "too_few_samples"}

        fold_size = n // (n_splits + 1)
        splits = []
        for i in range(n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end
            val_end = min(train_end + fold_size, n)
            if val_end <= val_start:
                continue
            splits.append((slice(0, train_end), slice(val_start, val_end)))

        if not splits:
            self.fit(X, y)
            return {"best_alpha": self.alpha, "best_ic": float("nan"), "cv_results": {}, "note": "no_valid_splits"}

        cv_results = {}
        for alpha in candidate_alphas:
            fold_ics = []
            for train_sl, val_sl in splits:
                model = PanelRidge(alpha=alpha)
                model.fit(X[train_sl], y[train_sl])
                pred = model.predict(X[val_sl])
                actual = y[val_sl]
                valid = np.isfinite(pred) & np.isfinite(actual)
                if valid.sum() >= 5:
                    rho, _ = stats.spearmanr(pred[valid], actual[valid])
                    if np.isfinite(rho):
                        fold_ics.append(float(rho))
            cv_results[alpha] = {
                "mean_ic": float(np.mean(fold_ics)) if fold_ics else float("nan"),
                "std_ic": float(np.std(fold_ics)) if len(fold_ics) > 1 else 0.0,
                "n_folds": len(fold_ics),
            }

        valid_results = {a: r for a, r in cv_results.items() if np.isfinite(r["mean_ic"])}
        if valid_results:
            best_alpha = max(valid_results, key=lambda a: valid_results[a]["mean_ic"])
            best_ic = valid_results[best_alpha]["mean_ic"]
        else:
            best_alpha = 50.0
            best_ic = float("nan")

        self.alpha = float(best_alpha)
        self.fit(X, y)
        return {"best_alpha": best_alpha, "best_ic": best_ic, "cv_results": cv_results}


__all__ = [
    "rsi",
    "slope_log_price",
    "make_features",
    "make_target",
    "PanelRidge",
]
