"""Small vectorbt spike for take-profit / stop-loss grid checks."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class MissingCostConfigError(ValueError):
    """Raised when a backtest is requested without explicit trading costs."""


@dataclass(frozen=True)
class BacktestCostConfig:
    fee_bps: float
    slippage_bps: float

    @property
    def total_bps(self) -> float:
        return self.fee_bps + self.slippage_bps


@dataclass(frozen=True)
class BacktestInput:
    close: pd.Series
    entries: pd.Series
    stop_loss_pct: float
    take_profit_pcts: tuple[float, ...]
    costs: BacktestCostConfig | None
    init_cash: float = 100_000.0


@dataclass(frozen=True)
class BacktestGridRow:
    take_profit_pct: float
    stop_loss_pct: float
    total_return_pct: float
    profit_loss_ratio: float
    max_drawdown_pct: float
    trade_count: int
    max_single_loss_pct: float
    cost_bps: float


@dataclass(frozen=True)
class BacktestGridResult:
    rows: tuple[BacktestGridRow, ...]
    engine: str


def run_take_profit_stop_loss_grid(payload: BacktestInput) -> BacktestGridResult:
    """Run a vectorbt TP/SL grid and return compact research metrics."""
    if payload.costs is None:
        raise MissingCostConfigError("explicit fee_bps and slippage_bps are required")

    import vectorbt as vbt

    close = payload.close.astype(float)
    entries = payload.entries.reindex(close.index, fill_value=False).astype(bool)
    fee_rate = payload.costs.fee_bps / 10_000.0
    slippage_rate = payload.costs.slippage_bps / 10_000.0

    rows: list[BacktestGridRow] = []
    for take_profit_pct in payload.take_profit_pcts:
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=None,
            init_cash=payload.init_cash,
            fees=fee_rate,
            slippage=slippage_rate,
            tp_stop=take_profit_pct,
            sl_stop=payload.stop_loss_pct,
        )
        rows.append(
            BacktestGridRow(
                take_profit_pct=take_profit_pct,
                stop_loss_pct=payload.stop_loss_pct,
                total_return_pct=round(float(portfolio.total_return()) * 100.0, 4),
                profit_loss_ratio=_profit_loss_ratio(portfolio.trades.records_readable),
                max_drawdown_pct=round(-abs(float(portfolio.max_drawdown())) * 100.0, 4),
                trade_count=int(portfolio.trades.count()),
                max_single_loss_pct=_max_single_loss_pct(portfolio.trades.records_readable),
                cost_bps=payload.costs.total_bps,
            )
        )

    return BacktestGridResult(rows=tuple(rows), engine=f"vectorbt {vbt.__version__}")


def _max_single_loss_pct(trades: pd.DataFrame) -> float:
    if trades.empty or "Return" not in trades:
        return 0.0
    losing_returns = trades["Return"][trades["Return"] < 0]
    if losing_returns.empty:
        return 0.0
    return round(float(losing_returns.min()) * 100.0, 4)


def _profit_loss_ratio(trades: pd.DataFrame) -> float:
    if trades.empty or "Return" not in trades:
        return 0.0
    winning_returns = trades["Return"][trades["Return"] > 0]
    losing_returns = trades["Return"][trades["Return"] < 0]
    if winning_returns.empty or losing_returns.empty:
        return 0.0
    return round(float(winning_returns.mean() / abs(losing_returns.mean())), 4)
