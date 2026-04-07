#!/usr/bin/env python3
"""
T6-1/T6-2: 风控加固回测对比
Baseline: vol_mult=6.0, 无 trailing, dd_half=0.12, dd_full=0.18
Variant:  vol_mult=3.0, trailing 3%/2%, dd_half=0.08, dd_full=0.12, max_pos=0.35

对比指标: Sharpe, MaxDD, 胜率, 平均持有天数, 平均盈亏比
"""
import sys, os, warnings, json, math
from pathlib import Path
from dataclasses import replace
from copy import deepcopy

# 确保当前目录可导入
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from ss7_sqlite_news_overlay import (
    BTConfig, ExecConfig, backtest,
    summary_from_equity, summary_from_stats,
)


def extract_trade_stats(stats: pd.DataFrame) -> dict:
    """从 stats DataFrame 提取交易级统计"""
    if stats is None or stats.empty:
        return {"win_rate": 0, "avg_holding_days": 0, "profit_factor": 0, "n_trades": 0}

    # 止损触发次数
    stop_col = stats.get("stop_loss_count", pd.Series(dtype=float))
    total_stop_triggers = int(stop_col.fillna(0).sum()) if stop_col is not None else 0

    # 从 equity 变化推算交易统计
    equity_col = stats.get("equity", pd.Series(dtype=float))
    if equity_col is not None and len(equity_col) > 1:
        rets = equity_col.astype(float).pct_change().dropna()
        pos_rets = rets[rets > 0]
        neg_rets = rets[rets < 0]
        win_rate = len(pos_rets) / len(rets) * 100 if len(rets) > 0 else 0
        avg_gain = float(pos_rets.mean()) if len(pos_rets) > 0 else 0
        avg_loss = float(neg_rets.mean()) if len(neg_rets) > 0 else 0
        profit_factor = abs(avg_gain / avg_loss) if abs(avg_loss) > 1e-12 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0

    # 平均持仓天数（从 rebalance 频率估算）
    rebal_col = stats.get("rebalance_due", pd.Series(dtype=float))
    if rebal_col is not None:
        rebal_days = rebal_col.fillna(0).astype(int)
        n_rebal = int(rebal_days.sum())
        avg_holding = len(stats) / max(n_rebal, 1)
    else:
        avg_holding = 0

    return {
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "stop_loss_triggers": total_stop_triggers,
        "trading_days": len(stats),
    }


def run_comparison():
    # 使用 screener 输出的 tickers 或默认 universe
    tickers_env = os.getenv("SS7_TICKERS", "9432.T,9433.T,2914.T,3382.T,8604.T,7201.T")
    TICKERS = [t.strip() for t in tickers_env.split(",") if t.strip()]
    BENCHMARK = os.getenv("SS7_BENCHMARK", "1321.T")

    db_path = os.getenv("SS7_DB_PATH", None)

    # 共用的 ExecConfig
    ex = ExecConfig(initial_capital=400_000)

    # ── Baseline: 旧参数（宽松止损，无 trailing）──
    bt_baseline = BTConfig(
        tickers=TICKERS,
        benchmark_ticker=BENCHMARK,
        start="2020-01-01",
        signal_mode="ridge",
        compare_signal_modes=[],
        db_path=db_path,
        output_dir="reports/bt_baseline",
        # 旧参数
        stop_loss_mode="atr",
        stop_loss_vol_mult=6.0,
        stop_loss_min_pct=0.06,
        stop_loss_max_pct=0.20,
        trailing_activate_pct=0.0,     # 无 trailing
        trailing_stop_pct=0.02,
        max_dd_half=0.12,
        max_dd_full=0.18,
        max_single_position_pct=0.50,
        safe_plot=True,
    )

    # ── Variant: Sprint 加固参数 ──
    bt_variant = BTConfig(
        tickers=TICKERS,
        benchmark_ticker=BENCHMARK,
        start="2020-01-01",
        signal_mode="ridge",
        compare_signal_modes=[],
        db_path=db_path,
        output_dir="reports/bt_variant",
        # 新参数
        stop_loss_mode="atr",
        stop_loss_vol_mult=3.0,
        stop_loss_min_pct=0.04,
        stop_loss_max_pct=0.12,
        trailing_activate_pct=0.03,    # 浮盈 3% 激活
        trailing_stop_pct=0.02,        # 最高点回撤 2% 触发
        max_dd_half=0.08,
        max_dd_full=0.12,
        max_single_position_pct=0.35,
        safe_plot=True,
    )

    results = {}
    for label, bt in [("baseline", bt_baseline), ("variant", bt_variant)]:
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"  stop_loss_vol_mult={bt.stop_loss_vol_mult}, trailing_activate={bt.trailing_activate_pct}")
        print(f"  max_dd_half={bt.max_dd_half}, max_dd_full={bt.max_dd_full}")
        print(f"  max_single_position_pct={bt.max_single_position_pct}")
        print(f"{'='*60}")

        Path(bt.output_dir).mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            close, trade_tickers, w_df, equity, stats = backtest(bt, ex)

        summary = {}
        summary.update(summary_from_equity(equity, ex.initial_capital))
        summary.update(summary_from_stats(stats, ex.initial_capital))
        summary.update(extract_trade_stats(stats))

        results[label] = summary
        print(f"\n  Sharpe: {summary['sharpe']:.3f}")
        print(f"  MaxDD:  {summary['max_drawdown_pct']:.2f}%")
        print(f"  Return: {summary['total_return_pct']:.2f}%")
        print(f"  StopLoss triggers: {summary['stop_loss_triggers']}")

    # ── 生成对比报告 ──
    b = results["baseline"]
    v = results["variant"]

    report_lines = [
        "# 风控加固回测对比报告",
        f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        "",
        "## 参数对比",
        "",
        "| 参数 | Baseline (旧) | Variant (Sprint加固) |",
        "|------|---------------|---------------------|",
        "| stop_loss_vol_mult | 6.0 | 3.0 |",
        "| stop_loss_min_pct | 6% | 4% |",
        "| stop_loss_max_pct | 20% | 12% |",
        "| trailing_activate | OFF | 3% |",
        "| trailing_stop | - | 2% |",
        "| max_dd_half | 12% | 8% |",
        "| max_dd_full | 18% | 12% |",
        "| max_position_pct | 50% | 35% |",
        "",
        "## 回测结果",
        "",
        "| 指标 | Baseline | Variant | 变化 |",
        "|------|----------|---------|------|",
        f"| Final Equity | ¥{b['final_equity']:,.0f} | ¥{v['final_equity']:,.0f} | {v['final_equity']-b['final_equity']:+,.0f} |",
        f"| Total Return | {b['total_return_pct']:.2f}% | {v['total_return_pct']:.2f}% | {v['total_return_pct']-b['total_return_pct']:+.2f}pp |",
        f"| Sharpe | {b['sharpe']:.3f} | {v['sharpe']:.3f} | {v['sharpe']-b['sharpe']:+.3f} |",
        f"| Sortino | {b['sortino']:.3f} | {v['sortino']:.3f} | {v['sortino']-b['sortino']:+.3f} |",
        f"| Max Drawdown | {b['max_drawdown_pct']:.2f}% | {v['max_drawdown_pct']:.2f}% | {v['max_drawdown_pct']-b['max_drawdown_pct']:+.2f}pp |",
        f"| Annual Vol | {b['annual_vol_pct']:.2f}% | {v['annual_vol_pct']:.2f}% | {v['annual_vol_pct']-b['annual_vol_pct']:+.2f}pp |",
        f"| Win Rate | {b['win_rate_pct']:.1f}% | {v['win_rate_pct']:.1f}% | {v['win_rate_pct']-b['win_rate_pct']:+.1f}pp |",
        f"| Profit Factor | {b['profit_factor']:.2f} | {v['profit_factor']:.2f} | {v['profit_factor']-b['profit_factor']:+.2f} |",
        f"| StopLoss Triggers | {b['stop_loss_triggers']} | {v['stop_loss_triggers']} | {v['stop_loss_triggers']-b['stop_loss_triggers']:+d} |",
        f"| Avg Turnover | ¥{b['avg_turnover_notional']:,.0f} | ¥{v['avg_turnover_notional']:,.0f} | - |",
        f"| Avg Cost | ¥{b['avg_cost_paid']:,.0f} | ¥{v['avg_cost_paid']:,.0f} | - |",
        "",
        "## 分析",
        "",
    ]

    # 自动评估（MaxDD 可能是负值，用绝对值比较）
    sharpe_better = v['sharpe'] > b['sharpe']
    dd_better = abs(v['max_drawdown_pct']) < abs(b['max_drawdown_pct'])
    ret_diff = v['total_return_pct'] - b['total_return_pct']

    if sharpe_better and dd_better:
        report_lines.append("**结论**: Variant 在风险调整收益(Sharpe)和最大回撤两方面均优于 Baseline。风控加固有效。")
    elif dd_better and abs(ret_diff) < 5:
        report_lines.append("**结论**: Variant 显著降低了最大回撤，收益略有让步但在可接受范围。风控加固值得采用。")
    elif sharpe_better:
        report_lines.append("**结论**: Variant Sharpe 更高但回撤未改善，需进一步调参。")
    else:
        report_lines.append(f"**结论**: Variant 收益下降 {abs(ret_diff):.1f}pp，回撤{'改善' if dd_better else '未改善'}。需权衡风控收紧的代价。")

    report_lines.extend([
        "",
        f"- 止损触发次数变化: {b['stop_loss_triggers']} → {v['stop_loss_triggers']}（更严格的止损线导致更频繁触发）",
        f"- Trailing stop 新增，预期减少盈利回吐",
        "",
        "---",
        "*自动生成 by run_risk_backtest_comparison.py*",
    ])

    report_path = Path("reports/risk_hardening_backtest_comparison.md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告已保存: {report_path}")

    # JSON 版本
    json_path = Path("reports/risk_hardening_backtest_comparison.json")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON 已保存: {json_path}")


if __name__ == "__main__":
    run_comparison()
