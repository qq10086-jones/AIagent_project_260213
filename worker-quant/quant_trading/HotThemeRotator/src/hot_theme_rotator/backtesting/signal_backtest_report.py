"""Research-only backtest report from generated HotThemeRotator signals."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hot_theme_rotator.backtesting.vectorbt_spike import (
    BacktestCostConfig,
    BacktestGridResult,
    BacktestInput,
    run_take_profit_stop_loss_grid,
)
from hot_theme_rotator.common.schema import TradingSignal


ENTRY_ACTIONS = {"BUY", "ROTATE"}


@dataclass(frozen=True)
class SignalBacktestInput:
    signals: list[TradingSignal]
    close_by_symbol: dict[str, pd.Series]
    costs: BacktestCostConfig
    stop_loss_pct: float
    take_profit_pcts: tuple[float, ...]
    init_cash: float = 100_000.0


@dataclass(frozen=True)
class SymbolBacktestResult:
    symbol: str
    grid: BacktestGridResult


@dataclass(frozen=True)
class SignalBacktestResult:
    symbol_results: tuple[SymbolBacktestResult, ...]
    entry_signal_count: int
    symbol_count: int
    cost_bps: float


def run_signal_backtest_grid(payload: SignalBacktestInput) -> SignalBacktestResult:
    """Run TP/SL grids for symbols with generated BUY/ROTATE signals."""
    entry_signals = [signal for signal in payload.signals if signal.action in ENTRY_ACTIONS]
    symbol_results: list[SymbolBacktestResult] = []
    for symbol in sorted({signal.symbol for signal in entry_signals}):
        close = payload.close_by_symbol.get(symbol)
        if close is None or close.empty:
            continue
        entries = _entries_for_symbol(close.index, entry_signals, symbol)
        if not entries.any():
            continue
        grid = run_take_profit_stop_loss_grid(
            BacktestInput(
                close=close,
                entries=entries,
                stop_loss_pct=payload.stop_loss_pct,
                take_profit_pcts=payload.take_profit_pcts,
                costs=payload.costs,
                init_cash=payload.init_cash,
            )
        )
        symbol_results.append(SymbolBacktestResult(symbol=symbol, grid=grid))

    return SignalBacktestResult(
        symbol_results=tuple(symbol_results),
        entry_signal_count=len(entry_signals),
        symbol_count=len(symbol_results),
        cost_bps=payload.costs.total_bps,
    )


def render_signal_backtest_markdown(result: SignalBacktestResult, asof: str) -> str:
    """Render a compact Markdown report for research-only review."""
    lines = [
        f"# HotThemeRotator Signal Backtest - {asof}",
        "",
        "Status: research-only. No automatic execution.",
        "",
        f"- Entry signals: {result.entry_signal_count}",
        f"- Symbols tested: {result.symbol_count}",
        f"- Cost assumption: {result.cost_bps:.1f} bps round-trip input",
        "",
        "| symbol | take_profit | stop_loss | return | profit_loss | max_dd | trades | max_single_loss | cost_bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for symbol_result in result.symbol_results:
        for row in symbol_result.grid.rows:
            lines.append(
                f"| {symbol_result.symbol} | "
                f"{row.take_profit_pct:.2%} | "
                f"{row.stop_loss_pct:.2%} | "
                f"{row.total_return_pct:.2f}% | "
                f"{row.profit_loss_ratio:.2f} | "
                f"{row.max_drawdown_pct:.2f}% | "
                f"{row.trade_count} | "
                f"{row.max_single_loss_pct:.2f}% | "
                f"{row.cost_bps:.1f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _entries_for_symbol(
    index: pd.Index,
    signals: list[TradingSignal],
    symbol: str,
) -> pd.Series:
    entries = pd.Series(False, index=index)
    normalized_index = pd.Index(str(item)[:10] for item in index)
    for signal in signals:
        if signal.symbol != symbol:
            continue
        signal_date = signal.asof[:10]
        matches = normalized_index == signal_date
        if matches.any():
            entries.iloc[list(matches).index(True)] = True
    return entries
