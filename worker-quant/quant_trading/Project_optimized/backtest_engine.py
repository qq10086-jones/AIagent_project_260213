from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    drawdown = equity_curve / peak - 1.0
    return float(drawdown.min())


@dataclass
class BacktestResult:
    close: pd.DataFrame
    trade_tickers: list[str]
    weights: pd.DataFrame
    equity: pd.Series
    stats: pd.DataFrame
    summary: dict[str, Any]


def run_backtest(bt: Any, ex: Any) -> BacktestResult:
    from ss7_sqlite_news_overlay import backtest, summary_from_equity, summary_from_stats

    close, trade_tickers, weights, equity, stats = backtest(bt, ex)
    summary = {}
    summary.update(summary_from_equity(equity, float(ex.initial_capital)))
    summary.update(summary_from_stats(stats, float(ex.initial_capital)))
    return BacktestResult(
        close=close,
        trade_tickers=list(trade_tickers),
        weights=weights,
        equity=equity,
        stats=stats,
        summary=summary,
    )


__all__ = ["BacktestResult", "run_backtest", "max_drawdown"]
