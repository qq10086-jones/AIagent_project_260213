"""Execution model extracted from ss7_sqlite_news_overlay.py.

Contains: ExecConfig, lot_size, execute_rebalance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def sbi_fee(notional: float, fee_mode: str = "sbi_zero") -> float:
    """SBI証券の手数料計算。

    fee_mode:
      "sbi_zero"     — ゼロ革命（電子交付設定済み）→ 手数料 0
      "sbi_standard" — スタンダードプラン（1注文ごと、税込）
      "flat_bps"     — 従来の固定bps方式（呼び出し側で計算）
    """
    if fee_mode == "sbi_zero":
        return 0.0
    if fee_mode != "sbi_standard":
        return 0.0  # unknown mode → safe default

    # SBI スタンダードプラン 現物取引 手数料（税込）
    n = abs(float(notional))
    if n <= 50_000:
        return 55.0
    elif n <= 100_000:
        return 99.0
    elif n <= 200_000:
        return 115.0
    elif n <= 500_000:
        return 275.0
    elif n <= 1_000_000:
        return 535.0
    elif n <= 1_500_000:
        return 640.0
    elif n <= 30_000_000:
        return 1013.0
    else:
        return 1070.0


@dataclass
class ExecConfig:
    initial_capital: float = 200_000.0
    lot_size_default: int = 1
    lot_size_by_ticker: Optional[Dict[str, int]] = None
    fee_bps: float = 0.0
    fee_mode: str = "sbi_zero"  # "sbi_zero" | "sbi_standard" | "flat_bps"
    slippage_bps: float = 0.0
    impact_k: float = 0.0
    adv_lookback: int = 20
    max_adv_frac: float = 1.0
    cash_rate_daily: float = 0.0
    target_vol_annual_pct: float = 0.0
    vol_target_lookback: int = 20
    vol_target_min_scale: float = 0.35
    vol_target_max_scale: float = 1.0


def lot_size(ticker: str, cfg: ExecConfig) -> int:
    if cfg.lot_size_by_ticker and ticker in cfg.lot_size_by_ticker:
        return int(cfg.lot_size_by_ticker[ticker])
    return int(cfg.lot_size_default)


def _round_to_lot(shares: int, lot: int) -> int:
    if lot <= 1:
        return int(shares)
    return int((shares // lot) * lot)


def execute_rebalance(
    prices: pd.Series,
    volumes: Optional[pd.Series],
    target_w: pd.Series,
    holdings: pd.Series,
    cash: float,
    cfg: ExecConfig,
    forced_exit_tickers: Optional[List[str]] = None,
) -> Tuple[pd.Series, float, float, float]:
    """
    Rebalance at close price.
    Returns: new_holdings, new_cash, traded_notional, total_cost
    """
    tickers = list(target_w.index)
    px = prices.reindex(tickers).astype(float)
    px = px.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    px = px.clip(lower=1e-6)

    # Current portfolio value
    cur_val = float((holdings.reindex(tickers).fillna(0).astype(float) * px).sum() + cash)
    if not np.isfinite(cur_val) or cur_val <= 0:
        cur_val = max(float(cash), 1e-6)

    # Desired dollar allocation
    target_w = target_w.fillna(0.0).clip(lower=0.0)
    sw = float(target_w.sum())
    if sw <= 1e-12:
        target_w = target_w * 0.0
    elif sw > 1.0 + 1e-12:
        target_w = target_w / sw
    target_val = target_w * cur_val

    # Convert to desired shares with lots
    desired_shares = {}
    for t in tickers:
        lot = lot_size(t, cfg)
        raw = int(target_val[t] // px[t])
        desired_shares[t] = _round_to_lot(raw, lot)

    desired = pd.Series(desired_shares, index=tickers, dtype=int)
    if forced_exit_tickers:
        for ticker in forced_exit_tickers:
            if ticker in desired.index:
                desired.loc[ticker] = 0
    cur = holdings.reindex(tickers).fillna(0).astype(int)

    # Liquidity constraint
    trade = desired - cur
    if volumes is not None and cfg.max_adv_frac < 1.0:
        adv = volumes.reindex(tickers).fillna(0.0).astype(float)
        cap = (adv * float(cfg.max_adv_frac)).fillna(0.0)
        trade = trade.clip(lower=-cap, upper=cap).round().astype(int)
        desired = cur + trade

    # Compute trade notional and costs
    trade_notional = float((trade.abs().astype(float) * px).sum())
    if cfg.fee_mode == "flat_bps":
        fee = trade_notional * (float(cfg.fee_bps) / 10000.0)
    else:
        # SBI per-order fee: sum fee for each ticker's trade
        fee = sum(
            sbi_fee(abs(float(trade.get(t, 0))) * float(px.get(t, 0)), cfg.fee_mode)
            for t in tickers if abs(trade.get(t, 0)) > 0
        )
    slip = trade_notional * (float(cfg.slippage_bps) / 10000.0)

    impact = 0.0
    if volumes is not None and float(cfg.impact_k) > 0.0:
        adv = volumes.reindex(tickers).fillna(0.0).astype(float)
        adv_notional = float((adv * px).mean())
        denom = max(adv_notional, 1e-6)
        impact_bps = float(cfg.impact_k) * math.sqrt(trade_notional / denom)
        impact = trade_notional * (impact_bps / 10000.0)

    total_cost = fee + slip + impact

    # Update cash and holdings
    cash_after_trades = cash - float((trade.astype(float) * px).sum()) - total_cost

    # If cash becomes negative, scale down buys
    if cash_after_trades < -1e-6:
        buys = trade.clip(lower=0)
        buy_notional = float((buys.astype(float) * px).sum())
        if buy_notional > 1e-6:
            scale = max((cash - total_cost) / buy_notional, 0.0)
            adj_trade = trade.copy()
            for t in tickers:
                if adj_trade[t] > 0:
                    lot = lot_size(t, cfg)
                    adj = int(adj_trade[t] * scale)
                    adj_trade[t] = _round_to_lot(adj, lot)
            trade = adj_trade
            desired = cur + trade
            trade_notional = float((trade.abs().astype(float) * px).sum())
            if cfg.fee_mode == "flat_bps":
                fee = trade_notional * (float(cfg.fee_bps) / 10000.0)
            else:
                fee = sum(
                    sbi_fee(abs(float(trade.get(t, 0))) * float(px.get(t, 0)), cfg.fee_mode)
                    for t in tickers if abs(trade.get(t, 0)) > 0
                )
            slip = trade_notional * (float(cfg.slippage_bps) / 10000.0)
            impact = 0.0
            if volumes is not None and float(cfg.impact_k) > 0.0:
                adv = volumes.reindex(tickers).fillna(0.0).astype(float)
                adv_notional = float((adv * px).mean())
                denom = max(adv_notional, 1e-6)
                impact_bps = float(cfg.impact_k) * math.sqrt(trade_notional / denom)
                impact = trade_notional * (impact_bps / 10000.0)
            total_cost = fee + slip + impact
            cash_after_trades = cash - float((trade.astype(float) * px).sum()) - total_cost

    new_holdings = desired.astype(int)
    new_cash = float(cash_after_trades)
    return new_holdings, new_cash, trade_notional, total_cost


__all__ = ["ExecConfig", "execute_rebalance", "lot_size"]
