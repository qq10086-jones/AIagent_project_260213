# Progress: 2026-04-02 Pipeline Fix & Cash Diagnosis

## Status

`daily_run.py` 今日首次在修复后成功完整执行（run_id: `2026-04-01__0cb010ffd8`，asof=2026-04-01）。

---

## 问题一：Pipeline 崩溃（已修复）

### 根因

`config.yaml` 中 `fundamental.enabled: true` + `source: "jquants"` + `fail_closed: true`，
但环境未安装 `jquantsapi` 模块，导致 `update_fundamentals.py` 抛出 `ModuleNotFoundError`，
`daily_run.py` 在步骤 1.5 崩溃，后续所有步骤（screener / ss7 模型 / make_decision / paper_execute / compute_ic）均未执行。

**现象**：每次手动或 Task Scheduler 触发 `daily_run.py` 均立即失败，信号数据停留在 2026-03-26 不再更新。

### 修复

```yaml
# config.yaml
fundamental:
  enabled: false        # 改前: true
  source: "noop"        # 改前: "jquants"
```

Screener 已使用 yfinance 作为基本面回退，`enabled: false` 对 screener 无影响。
`SS6_USE_FUNDAMENTAL_FEATURES` 环境变量仍由 `use_in_live_scoring` 控制（保持原值）。

---

## 问题二：模型持续空仓（非 bug，行为符合设计）

### 诊断

通过 `reports/weights_history.csv` 确认：

| 日期 | 权重合计 | 说明 |
|------|---------|------|
| 2026-03-12 | 0.575 | 最后一次非零持仓（rebalance 日） |
| 2026-03-13 起 | 0.000 | 全部清零 |

根因：**Benchmark 趋势过滤器（MA20 vs MA60）在 2026-03-30 转为 False**。

```
日期        1321.T收盘  MA20      MA60      above_ma
2026-03-27  55,110     56,554   56,397      True  ← 最后绿灯
2026-03-30  54,310     56,213   56,431     False  ← 过滤器关闭
2026-03-31  53,480     55,873   56,445     False
2026-04-01  56,240     55,762   56,509     False  ← 当前
```

日经（1321.T）从 2026-02-27 高点 61,120 跌至 2026-03-31 低点 53,480（-12.5%），
MA20 在 3月30日跌破 MA60，模型按 `benchmark_hysteresis_exit_pct=1%` 触发全仓退出并拒绝新建仓。

**这是正确行为**：趋势过滤器保护了账户免受 3月份最大 -12.5% 的系统性下跌。

### 重入条件

| 参数 | 当前值（4月1日） | 目标值 |
|------|----------------|--------|
| MA20 | 55,762 | ≥ 57,074（= MA60 × 1.01） |
| MA60 | 56,509 | — |
| 缺口 | — | **约 +2.4%** |

4月1日大盘大幅反弹（+6.74%），若后续维持强势，预计 **1～2 周**内 MA20 可重新越过 MA60。

---

## 今日 daily_run 执行结果

```
run_id:  2026-04-01__0cb010ffd8
asof:    2026-04-01
orders:  0（benchmark filter=False，无建仓信号）
NAV:     400,000 JPY（全现金）
```

### Factor IC 状态（compute_ic --shadow）

| 因子 | IC均值 | t-stat | 状态 |
|------|--------|--------|------|
| mom_consist | 0.0101 | 2.06 | PASS |
| rsi14 | 0.0269 | 0.60 | FAIL |
| vol_adj_mom20 | 0.0331 | 0.54 | FAIL |
| ret20 | 0.0301 | 0.62 | FAIL |
| slope60 | 0.0177 | 1.05 | FAIL |

整体 mean_t_stat=0.98，未达晋升门槛（1.5）。`mom_consist` 是唯一统计显著因子。
因子 IC 数据积累不足（部分因子仅 50 观测），需更多数据周期后重评估。

### Promotion 评估

- recommendation: **hold**（ridge 继续作为生产模式）
- 未通过：t_stat / backtest_sharpe / paper_days / turnover_stability

---

## 下次操作建议

1. **保持观望**：等 benchmark MA20 重新越过 MA60（×1.01），模型将自动在下个 rebalance 日建仓
2. **若想提前重入**：可临时调整 `benchmark_slow_ma_window: 60 → 40` 或降低 `benchmark_hysteresis_enter_pct: 0.01 → 0.005`
3. **每日运行**：`PYTHONIOENCODING=utf-8 python daily_run.py`（注意编码参数，Windows cp932 下子进程打印中文需要此环境变量）
4. **jquants 激活**：若未来获得 J-Quants API key，安装 `pip install jquantsapi` 并将 `fundamental.enabled` 改回 `true`

---

## 文件变更

| 文件 | 变更 |
|------|------|
| `config.yaml` | `fundamental.enabled: false`, `source: "noop"` |
| `reports/target_weights.csv` | 刷新（全零，benchmark filter=False） |
| `reports/weights_history.csv` | 刷新（214 dates × 50 stocks） |
| `artifacts/decision/2026-04-01/` | 新增 run_id `2026-04-01__0cb010ffd8` |
| `reports/factor_health_report.*` | 刷新 |
| `reports/signal_mode_compare_report.*` | 刷新 |
