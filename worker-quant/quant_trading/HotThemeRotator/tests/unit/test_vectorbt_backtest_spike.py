import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.backtesting.vectorbt_spike import (  # noqa: E402
    BacktestCostConfig,
    BacktestInput,
    MissingCostConfigError,
    run_take_profit_stop_loss_grid,
)


def _sample_prices() -> pd.Series:
    return pd.Series(
        [100.0, 101.0, 103.0, 106.0, 104.0, 99.0, 96.0, 98.0],
        index=pd.date_range("2026-05-01", periods=8, freq="D"),
        name="8035.T",
    )


def test_vectorbt_grid_requires_explicit_costs():
    payload = BacktestInput(
        close=_sample_prices(),
        entries=pd.Series(
            [True, False, False, False, False, False, False, False],
            index=_sample_prices().index,
        ),
        stop_loss_pct=0.04,
        take_profit_pcts=(0.02, 0.03, 0.05),
        costs=None,
    )

    with pytest.raises(MissingCostConfigError):
        run_take_profit_stop_loss_grid(payload)


def test_vectorbt_grid_reports_take_profit_variants_with_costs():
    prices = _sample_prices()
    result = run_take_profit_stop_loss_grid(
        BacktestInput(
            close=prices,
            entries=pd.Series(
                [True, False, False, False, False, False, False, False],
                index=prices.index,
            ),
            stop_loss_pct=0.04,
            take_profit_pcts=(0.02, 0.03, 0.05),
            costs=BacktestCostConfig(fee_bps=5.0, slippage_bps=5.0),
            init_cash=100_000.0,
        )
    )

    assert [row.take_profit_pct for row in result.rows] == [0.02, 0.03, 0.05]
    assert all(row.trade_count >= 1 for row in result.rows)
    assert all(row.cost_bps == 10.0 for row in result.rows)
    assert all(row.profit_loss_ratio >= 0 for row in result.rows)
    assert result.rows[0].total_return_pct > 0
    assert result.rows[2].max_drawdown_pct <= 0
