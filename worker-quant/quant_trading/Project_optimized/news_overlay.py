from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class NewsConfig:
    enabled: bool = False
    csv_path: Optional[str] = None
    db_path: Optional[str] = None
    half_life_days: float = 3.0
    lookback_days: int = 10
    A_max: float = 4.0
    U_high: float = 0.6
    absF_min: float = 0.5
    g_min: float = 0.15
    k_absF: float = 1.0
    k_U: float = 3.0
    k_A: float = 0.6
    shadow_only: bool = False
    sprint_gating: bool = False


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_news_items(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    def _col(name: str, default):
        if name in cols:
            return df[cols[name]]
        return default

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(_col("date", None), errors="coerce")
    out["ticker"] = _col("ticker", None)
    out["sent"] = pd.to_numeric(_col("sent", 0.0), errors="coerce")
    out["weight"] = pd.to_numeric(_col("weight", 1.0), errors="coerce").fillna(1.0).clip(lower=0.0)
    out["conf"] = pd.to_numeric(_col("conf", 1.0), errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    out = out.dropna(subset=["date", "ticker", "sent"])
    out["ticker"] = out["ticker"].astype(str)
    out["sent"] = out["sent"].clip(-1.0, 1.0)
    return out.sort_values("date")


def load_news_items_from_db(db_path: str, cutoff_ts: Optional[str] = None) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["date", "ticker", "sent", "weight", "conf"])
    try:
        conn = sqlite3.connect(db_path)
        pit_clause = "AND nf.published_ts < :cutoff AND nf.ingested_ts < :cutoff" if cutoff_ts else ""
        rows = conn.execute(
            f"""
            SELECT
                nf.published_ts AS date,
                nf.symbol AS ticker,
                ns.sentiment_score AS sent,
                COALESCE(ns.urgency, 1.0) AS weight,
                1.0 AS conf
            FROM news_feed nf
            INNER JOIN (
                SELECT news_id, sentiment_score, urgency
                FROM news_sentiment
                WHERE scored_ts != 'TIMEOUT'
                GROUP BY news_id
                HAVING scored_ts = MAX(scored_ts)
            ) ns ON nf.news_id = ns.news_id
            WHERE (
                nf.event_cluster_id IS NULL
                OR nf.ingested_ts = (
                    SELECT MIN(nf2.ingested_ts)
                    FROM news_feed nf2
                    WHERE nf2.event_cluster_id = nf.event_cluster_id
                )
            )
            {pit_clause}
            ORDER BY nf.published_ts
            """,
            {"cutoff": cutoff_ts} if cutoff_ts else {},
        ).fetchall()
        if not rows:
            legacy_rows = conn.execute(
                f"""
                SELECT asof, symbol, value
                FROM feature_daily
                WHERE feature_name='news_risk_raw'
                {'AND asof <= :cutoff' if cutoff_ts else ''}
                """,
                {"cutoff": cutoff_ts} if cutoff_ts else {},
            ).fetchall()
            conn.close()
            if not legacy_rows:
                return empty
            out = pd.DataFrame(legacy_rows, columns=["date", "ticker", "sent"])
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out["sent"] = pd.to_numeric(out["sent"], errors="coerce").fillna(0.0).apply(lambda v: float(-v)).clip(-1.0, 1.0)
            out["weight"] = 1.0
            out["conf"] = 1.0
            out = out.dropna(subset=["date", "ticker"])
            out["ticker"] = out["ticker"].astype(str)
            return out.sort_values("date")
        conn.close()
    except Exception:
        return empty

    out = pd.DataFrame(rows, columns=["date", "ticker", "sent", "weight", "conf"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["sent"] = pd.to_numeric(out["sent"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0).clip(lower=0.0)
    out["conf"] = 1.0
    out = out.dropna(subset=["date", "ticker"])
    out["ticker"] = out["ticker"].astype(str)
    return out.sort_values("date")


def build_news_factors(
    dates: pd.Index,
    tickers: List[str],
    items: pd.DataFrame,
    cfg: NewsConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if items is None or len(items) == 0:
        zero = pd.DataFrame(0.0, index=dates, columns=tickers)
        return zero.copy(), zero.copy(), zero.copy()

    half_life = max(float(cfg.half_life_days), 1e-6)
    lam = math.log(2.0) / half_life
    items = items.copy()
    items["d"] = items["date"].dt.normalize()

    def _agg(g: pd.DataFrame) -> pd.Series:
        w = (g["weight"] * g["conf"]).to_numpy(dtype=float)
        s = g["sent"].to_numpy(dtype=float)
        num = float((w * s).sum())
        att = float(w.sum())
        denom = float((w * abs(s)).sum()) + 1e-12
        dis = 1.0 - abs(num) / denom
        return pd.Series({"dir": num, "att": att, "dis": dis})

    daily = items.groupby(["d", "ticker"], sort=False).apply(_agg).reset_index()
    daily = daily[daily["ticker"].isin(set(tickers))].copy()
    if len(daily) == 0:
        zero = pd.DataFrame(0.0, index=dates, columns=tickers)
        return zero.copy(), zero.copy(), zero.copy()

    cal_days = pd.date_range(daily["d"].min(), daily["d"].max(), freq="D")
    dir_cal = pd.DataFrame(0.0, index=cal_days, columns=tickers)
    att_cal = pd.DataFrame(0.0, index=cal_days, columns=tickers)
    dis_cal = pd.DataFrame(0.0, index=cal_days, columns=tickers)
    for _, row in daily.iterrows():
        d = pd.Timestamp(row["d"])
        ticker = str(row["ticker"])
        dir_cal.at[d, ticker] = float(row["dir"])
        att_cal.at[d, ticker] = float(row["att"])
        dis_cal.at[d, ticker] = float(row["dis"])

    lookback = int(max(cfg.lookback_days, 1))
    weights = pd.Series([math.exp(-lam * i) for i in range(lookback)], dtype=float)
    weights = weights / max(float(weights.sum()), 1e-12)

    def _decay_apply(cal_df: pd.DataFrame) -> pd.DataFrame:
        arr = cal_df.to_numpy(dtype=float)
        out = arr * 0.0
        for i in range(len(cal_days)):
            start = max(0, i - lookback + 1)
            window = arr[start : i + 1, :]
            ww = weights.iloc[: window.shape[0]].iloc[::-1].to_numpy().reshape(-1, 1)
            out[i, :] = (window * ww).sum(axis=0)
        return pd.DataFrame(out, index=cal_days, columns=tickers)

    F_cal = _decay_apply(dir_cal)
    A_cal = _decay_apply(att_cal)
    U_cal = _decay_apply(dis_cal)
    normalized_dates = pd.to_datetime(dates).normalize()
    F = F_cal.reindex(normalized_dates).ffill().reindex(dates).fillna(0.0)
    A = A_cal.reindex(normalized_dates).ffill().reindex(dates).fillna(0.0)
    U = U_cal.reindex(normalized_dates).ffill().reindex(dates).fillna(0.0)
    return F, A, U


def apply_news_overlay_to_weights(
    w_target: pd.Series,
    dt: pd.Timestamp,
    F: pd.DataFrame,
    A: pd.DataFrame,
    U: pd.DataFrame,
    cfg: NewsConfig,
) -> Tuple[pd.Series, float]:
    if (not cfg.enabled) or (F is None) or (dt not in F.index):
        return w_target, 1.0

    f = F.loc[dt].reindex(w_target.index).fillna(0.0).astype(float)
    a = A.loc[dt].reindex(w_target.index).fillna(0.0).astype(float)
    u = U.loc[dt].reindex(w_target.index).fillna(0.0).astype(float).clip(0.0, 1.0)
    g = pd.Series(1.0, index=w_target.index, dtype=float)
    g[a >= float(cfg.A_max)] = float(cfg.g_min)
    g[u >= float(cfg.U_high)] = float(cfg.g_min)
    absf = f.abs()
    trust = (absf >= float(cfg.absF_min)).astype(float)
    score = (
        float(cfg.k_absF) * (absf - float(cfg.absF_min)).clip(lower=0.0) * trust
        - float(cfg.k_U) * u
        - float(cfg.k_A) * a
    )
    g_soft = score.apply(_sigmoid)
    g_soft = float(cfg.g_min) + (1.0 - float(cfg.g_min)) * g_soft
    g = pd.concat([g, g_soft], axis=1).min(axis=1)
    w_new = (w_target.fillna(0.0).clip(lower=0.0) * g).astype(float)
    total = float(w_new.sum())
    if total <= 1e-12:
        return w_target * 0.0, 0.0
    g_port = float((w_target.fillna(0.0).clip(lower=0.0) * g).sum() / (float(w_target.sum()) + 1e-12))
    return w_new, g_port


__all__ = [
    "NewsConfig",
    "load_news_items",
    "load_news_items_from_db",
    "build_news_factors",
    "apply_news_overlay_to_weights",
]
