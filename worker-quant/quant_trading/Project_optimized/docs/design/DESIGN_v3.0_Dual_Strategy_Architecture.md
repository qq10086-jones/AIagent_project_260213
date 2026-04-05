# Worker-Quant — 双策略架构设计文档 v3.0

**作者**: PM + Quant Architect (Senior Review)
**创建日期**: 2026-04-04
**状态**: DRAFT — 待审批
**对应任务清单**: `../tasks/TASKS_2026-04-04_Dual_Strategy_Implementation.md`
**上承文档**: `DESIGN_v2.0_Daily_Analysis_Pipeline.md`, `DESIGN_v1.5_Intelligence_Augmented_Quant.md`
**治理文档**: `../governance/GOVERNANCE_v2_DUAL_STRATEGY.md`

---

## 0. 设计背景与动机

### 0.1 现状诊断（2026-04-04 Senior Review）

| 维度 | 当前分 | 核心瓶颈 |
|------|--------|----------|
| 工程架构 | 8/10 | Pipeline 完整，模块化需改善（ss7 单文件 127KB） |
| 风控意识 | 9/10 | 多层风控 + 晋升门控，纪律性强 |
| 因子研究 | 5/10 | 20 因子中仅 3 个通过 t-stat guard，多数 n_obs < 50 |
| 资金管理 | 3/10 | 40万 JPY 跑多因子分散，策略容量严重不匹配 |
| 数据治理 | 7/10 | PIT 完整，但 JSON/SQLite 双数据源并存 |
| 实盘就绪度 | 4/10 | 新闻/基本面因子未真正接入，paper 幂等性存疑 |

### 0.2 核心矛盾

**资金规模（40万 JPY ≈ 2700 USD）与多因子分散策略的容量需求（最低 200万 JPY）存在结构性矛盾。**

- `top_k=50` 候选池 + `lot=100` + `max_single_position_pct=0.25`，实际仅能持有 3-4 只
- 单只暴露 25-33%，分散化形同虚设
- 交易成本（10bps fee + 5bps slippage）对小资金侵蚀比例更高

### 0.3 设计目标

1. 引入双策略架构，解决资金-容量矛盾
2. 将所有弱项维度拉升至 8.5+ 分
3. 保持现有 v2.0 pipeline 不退化，增量式改造

---

## 1. 双策略总架构

```
总资金 400,000 JPY
  │
  ├── Phase 1（资金 < 2,000,000 JPY）
  │   └── Strategy B "Sprint" — 100% 资金
  │       短期集中动量，Half-Kelly 仓位管理
  │       目标：快速滚资到 200万+
  │
  ├── Phase 2（资金 >= 2,000,000 JPY）── 自动触发分仓
  │   ├── Strategy A "Harvest" — 70% 资金
  │   │   中长期多因子分散，IC-weighted equal-risk
  │   │   目标：年化 Sharpe > 1.5，MaxDD < 10%
  │   │
  │   └── Strategy B "Sprint" — 30% 资金
  │       继续短期集中，但资金占比缩小
  │       目标：高 alpha 补充收益
  │
  └── Phase 3（资金 >= 5,000,000 JPY）── 可选扩展
      ├── Strategy A "Harvest" — 60%
      ├── Strategy B "Sprint" — 20%
      └── Strategy C "Event" — 20%（新闻/财报事件驱动）
```

### 1.1 Phase 触发与资金再平衡

```python
def resolve_phase(total_nav: float) -> str:
    if total_nav < 2_000_000:
        return "phase_1"   # Sprint only
    elif total_nav < 5_000_000:
        return "phase_2"   # Sprint + Harvest
    else:
        return "phase_3"   # Sprint + Harvest + Event (future)
```

Phase 升级为**单向门控**（只升不降），避免短期 NAV 波动导致频繁切换：
- Phase 1 -> Phase 2：总 NAV 连续 5 个交易日 >= 200万
- Phase 2 -> Phase 3：总 NAV 连续 5 个交易日 >= 500万（v3.0 暂不实现）

---

## 2. Strategy B "Sprint" 详细设计

### 2.1 设计哲学

Sprint 不是赌博策略。它是在资金约束下做出的**数学最优选择**：

- 小资金无法有效分散 -> 不如集中在最强信号上
- 集中度风险用 Kelly 公式数学约束 -> 破产概率趋近于零
- 持有期缩短 -> 降低单笔最大回撤暴露
- benchmark risk-off 更激进 -> 宁可踏空不亏损

### 2.2 参数配置

```yaml
strategy_profiles:
  sprint:
    enabled: true
    strategy_id: "sprint"
    capital_allocation_pct: 1.0       # Phase 1: 100%，Phase 2: 0.30
    activation_threshold: 0           # 始终激活

    # 选股
    max_positions: 3
    top_k: 20                          # 较小候选池，只看最强信号
    min_adv_floor: 5000000             # 流动性要求更高 (500万 JPY)
    max_cost_per_lot: 100000           # 1手上限 10万 JPY，40万能持 4 只

    # 因子
    signal_mode: "sprint_momentum"
    factor_set: ["mom_consist", "high52w", "vol_z"]
    weighting: "ic_rank"               # 不经过 Ridge，直接 IC 加权排名

    # 仓位管理
    position_sizing: "half_kelly"
    kelly_fraction: 0.5
    max_single_position_pct: 0.50      # 允许集中
    max_sector_weight: 0.60
    min_position_pct: 0.05

    # 风控
    holding_period_target: 5           # 目标持有 3-5 天
    stop_loss_mode: "atr"
    stop_loss_vol_mult: 3.0            # 收紧止损
    stop_loss_min_pct: 0.03
    stop_loss_max_pct: 0.10
    max_dd_half: 0.10                  # 10% 回撤降半仓
    max_dd_full: 0.15                  # 15% 回撤全平
    rebalance_every: 1                 # 每日评估

    # Benchmark regime
    benchmark_off_scale: 0.0           # risk-off = 完全清仓
    benchmark_caution_scale: 0.40
    use_vix_confirmation: true
    vix_off_threshold: 30.0
    vix_ticker: "1552.T"              # 日经 VI 指数
```

### 2.3 Sprint 选股器

与 Harvest 的 `screener.py` 分离，Sprint 专用选股逻辑：

```
输入：screened universe (from screener.py, top_k=20)
过滤条件（全部满足才入池）：
  1. 日均成交额 > 500万 JPY
  2. 1手成本 < 10万 JPY
  3. 近5日 mom_consist 排名前 20%
  4. vol_z > 0.5（放量确认）
  5. fundamental_score > 0.5（不买基本面硬否决的）
  6. benchmark_state != "off"
输出：排名前 5 只候选，由 Kelly sizer 决定最终持仓数和权重
```

### 2.4 Half-Kelly 仓位管理器

核心公式：

```
完整 Kelly：f* = (p * b - q) / b
  其中 p = 滚动胜率，b = 盈亏比，q = 1 - p
实际仓位：position_weight = f* × kelly_fraction (0.5)
约束：min_position_pct <= position_weight <= max_single_position_pct
```

Kelly 参数从历史信号回测滚动计算：

| 参数 | 计算方式 | 滚动窗口 |
|------|---------|---------|
| 胜率 p | 过去 N 次信号中收盘盈利的比例 | 60 个交易日 |
| 平均盈利 | 盈利交易的平均收益率 | 60 个交易日 |
| 平均亏损 | 亏损交易的平均亏损率（绝对值） | 60 个交易日 |
| 盈亏比 b | 平均盈利 / 平均亏损 | 60 个交易日 |

**安全护栏**：
- edge <= 0（无正期望）-> 仓位 = 0，不开仓
- edge 计算不足 30 个样本 -> 回退到固定仓位 10%
- 连续 3 次止损触发 -> 暂停 Sprint 5 个交易日（冷却期）

### 2.5 Sprint 进出场规则

```
进场信号（全部满足）：
  1. benchmark_state != "off"
  2. mom_consist 当日截面排名前 20%
  3. high52w > -0.10（距52周高点不超过10%）
  4. vol_z > 0.5（放量确认）
  5. Kelly edge > 0 且 suggested_weight > min_position_pct

出场信号（任一触发）：
  1. ATR 3x 止损触发（硬止损，不可覆盖）
  2. 持有超过 holding_period_target 天且浮亏
  3. vol_z 从 > 1.5 急降到 < -0.5（量能衰竭信号）
  4. benchmark_state 转 "off"（全仓立即退出）
  5. 组合回撤触发 max_dd_half 或 max_dd_full
```

### 2.6 Sprint 数学期望分析

保守估计（基于日股短期动量历史统计）：

| 参数 | 保守值 | 来源 |
|------|--------|------|
| 胜率 p | 55% | 日股 5 日动量回测中位数 |
| 盈亏比 b | 1.8:1 | ATR 3x 止损 vs ATR 5x 止盈期望 |
| Kelly edge | 0.30 | (0.55 × 1.8 - 0.45) / 1.8 |
| Half-Kelly 仓位 | 15% | 0.30 / 2 |
| 单笔期望收益 | 0.90% | 0.55 × 3% - 0.45 × 1.67% |
| 月均交易次数 | 4-5 | 每周约 1 次换仓 |
| 月收益期望 | 3.6-4.5% | 不含复利 |

40万 -> 200万预计 8-12 个月（复利计算，含回撤周期）。

**风险提示**：以上为数学期望，非保证收益。实际表现取决于市场环境、信号衰减、执行偏差。Sprint 策略必须先通过 30 天 paper trading 验证才能投入真钱。

---

## 3. Strategy A "Harvest" 详细设计

### 3.1 设计哲学

Harvest 是经典的机构级多因子策略，追求稳定复利而非高收益：

- 持仓 8-12 只，真正分散化
- 因子以 IC 验证后的核心集为主
- 换仓频率低（10-20 天），降低交易成本
- 风控宽松（ATR 6x 止损），给趋势足够空间

### 3.2 参数配置

```yaml
strategy_profiles:
  harvest:
    enabled: false                      # Phase 1 暂不启用
    strategy_id: "harvest"
    capital_allocation_pct: 0.70        # Phase 2: 70%
    activation_threshold: 2000000       # 总 NAV >= 200万 JPY 触发
    activation_consecutive_days: 5      # 连续 5 日达标才激活

    # 选股
    max_positions: 12
    top_k: 50                           # 沿用现有候选池大小
    min_adv_floor: 2000000
    max_cost_per_lot: 500000            # Phase 2 资金充足

    # 因子
    signal_mode: "shadow_hybrid_ic"     # 沿用现有晋升路径
    factor_set: ["mom_consist", "ma_gap", "sharpe_60", "roa_op", "cfo_assets"]
    weighting: "ridge_cv"               # Ridge + Time-Series CV

    # 仓位管理
    position_sizing: "equal_risk"       # 等风险贡献
    max_single_position_pct: 0.15       # 更低集中度
    max_sector_weight: 0.30

    # 风控
    holding_period_target: 15
    stop_loss_mode: "atr"
    stop_loss_vol_mult: 6.0             # 沿用宽止损
    stop_loss_min_pct: 0.06
    stop_loss_max_pct: 0.20
    max_dd_half: 0.12
    max_dd_full: 0.18
    rebalance_every: 10

    # Benchmark regime
    benchmark_off_scale: 0.45           # 保留 45% 底仓（长期策略不怕短期波动）
    benchmark_caution_scale: 0.70
    use_vix_confirmation: false
```

### 3.3 Harvest 因子体系（分层管理）

| 层级 | 因子 | 进入条件 | 退出条件 |
|------|------|---------|---------|
| **Core（生产权重）** | `mom_consist`, `ma_gap`, `sharpe_60` | t-stat >= 1.5 且 n_obs >= 100 | 连续 60 期 t-stat < 0.5 |
| **Candidate（shadow 跟踪）** | `high52w`, `mom_12_1`, `ret60`, `vol_z` | 自动跟踪 IC | 无自动退出 |
| **Fundamental（待激活）** | `roa_op`, `cfo_assets`, `accruals_inv`, `margin_op`, `leverage_safety` | 基本面数据覆盖率 > 60% 且 IC t-stat >= 1.5 | 数据源中断 > 30 天 |
| **Excluded（已排除）** | `ret20`, `rsi14`, `slope60`, `vol_adj_mom20`, `ret1`, `ret5`, `vol20`, `vol60`, `z_20` | 需 180 天重新评估 | — |

### 3.4 Ridge 模型 alpha 选择改造

当前 `PanelRidge(alpha=50.0)` 为硬编码，改为 Time-Series CV：

```
候选 alpha 值：[1, 5, 10, 25, 50, 100, 200]
验证方式：TimeSeriesSplit(n_splits=5)
评价指标：Spearman IC（排序相关，不是 MSE）
选出最优 alpha 后用全量数据 refit
```

---

## 4. 资金管理子系统设计

### 4.1 策略间资金隔离

每个策略维护独立的虚拟账户：

```sql
-- trade_schema.py 新增字段
ALTER TABLE positions ADD COLUMN strategy_id TEXT DEFAULT 'default';
ALTER TABLE orders ADD COLUMN strategy_id TEXT DEFAULT 'default';
ALTER TABLE fills ADD COLUMN strategy_id TEXT DEFAULT 'default';
ALTER TABLE account_snapshots ADD COLUMN strategy_id TEXT DEFAULT 'default';
ALTER TABLE decision_runs ADD COLUMN strategy_id TEXT DEFAULT 'default';
```

每日 `daily_run.py` 流程：

```
1. 读取总 NAV（所有策略的 positions + cash 之和）
2. resolve_phase(total_nav) -> 决定激活哪些策略
3. 计算各策略资金分配：
     sprint_capital = total_nav × sprint.capital_allocation_pct
     harvest_capital = total_nav × harvest.capital_allocation_pct (if enabled)
4. 对每个激活策略，分别运行：
     screener(strategy_config) -> signal(strategy_config) -> decision(strategy_config) -> paper(strategy_config)
5. 各策略的 orders/fills/positions 写入 SQLite，带 strategy_id 标识
```

### 4.2 资金再平衡规则

- **月度再平衡**：每月第一个交易日，按目标比例重新分配
- **触发式再平衡**：当某策略偏离目标比例 > 10 个百分点时触发
- **单向资金转移**：Sprint 盈利可转入 Harvest，但 Harvest 不向 Sprint 补仓

### 4.3 废弃 paper_trading_account.json

`paper_trading_account.json` 降级为只读诊断快照，不再作为数据源：

```python
# daily_run.py 末尾
def export_readonly_snapshot(conn, reports_dir: Path):
    """仅供人类快速查看，不参与任何决策逻辑"""
    snapshot = {
        "exported_at": _utc_now_iso(),
        "read_only": True,
        "source_of_truth": "japan_market.db (positions + account_snapshots tables)",
        "strategies": {}
    }
    for sid in ["sprint", "harvest"]:
        # ... 从 SQLite 读取并填充 ...
    (reports_dir / "paper_trading_account.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
```

---

## 5. Benchmark Risk-Off 改造

### 5.1 当前问题

`benchmark_off_scale=0.25` 在急跌后直接砍仓到 25%，导致：
- 2026-03-30 日经暴跌 -12.5% 后触发 risk-off，持续空仓至今
- V 型反弹时完全踏空

### 5.2 分策略差异化 + VIX 二次确认

| 状态 | Sprint scale | Harvest scale | 二次确认 |
|------|-------------|---------------|---------|
| on | 1.0 | 1.0 | — |
| caution | 0.40 | 0.70 | — |
| off（MA 触发） | 0.0 | 0.45 | Sprint 需 VI > 30 确认 |
| off（MA + VI 确认） | 0.0 | 0.45 | — |
| off（MA 触发但 VI < 30） | 降级为 caution 0.40 | 0.45 | Sprint 不完全清仓 |

VIX 二次确认逻辑：

```python
def benchmark_regime_scale_v2(
    px_b, fast_ma_b, slow_ma_b, prev_state,
    enter_pct, exit_pct, off_scale, caution_scale,
    vix_value=None, vix_off_threshold=30.0,
    use_vix_confirmation=False,
):
    """增强版：MA 交叉 + VIX 二次确认

    当 MA 信号判定 off 但 VIX 低于阈值时，降级为 caution，
    避免温和回调被误判为趋势反转。
    """
    state, scale = benchmark_regime_scale(
        px_b, fast_ma_b, slow_ma_b, prev_state,
        enter_pct, exit_pct, off_scale, caution_scale,
    )
    if use_vix_confirmation and state == "off":
        if vix_value is not None and vix_value < vix_off_threshold:
            state = "caution"
            scale = float(np.clip(caution_scale, 0.0, 1.0))
    return state, scale
```

数据源：`1552.T`（日经平均VI）的日线收盘价，从 `daily_prices` 表读取。需要 `db_update.py` 将 `1552.T` 加入默认更新列表。

---

## 6. 新闻模块分阶段接入

### 6.1 接入路径

| 阶段 | config 状态 | 行为 | 验证标准 | 预计时间 |
|------|------------|------|---------|---------|
| Phase 1 | `news.enabled: true, news.shadow_only: true` | 新闻写入 DB，计算 F/A/U 三维度，写入 shadow 日志，不影响信号 | 跑 30 天，观察 news_risk 与次日收益的相关性 | 1 个月 |
| Phase 2 | `news.sprint_gating: true` | 新闻进入 Sprint 策略的 gating 层（仅降低仓位，不加仓） | Sprint Sharpe 不低于无新闻版本 | 2 周 |
| Phase 3 | `news.harvest_factor: true` | 新闻因子进入 Harvest 的 candidate_factors | IC t-stat >= 1.5 | 数据积累决定 |

### 6.2 Sprint 新闻门控规则

```
如果某只候选股在过去 48h 内有 news_risk = "HIGH"：
  -> Kelly 仓位 × 0.5（减半）
如果 news_risk = "CRITICAL"：
  -> 跳过该标的
如果全市场 macro_sentiment < -0.5：
  -> 全策略 benchmark_caution_scale × 0.8
```

---

## 7. 数据治理改造

### 7.1 SQLite 单一真实来源

| 改造项 | 现状 | 目标 |
|--------|------|------|
| positions | SQLite + JSON 并存 | SQLite only，JSON 降级为只读快照 |
| orders | SQLite | 不变，加 `strategy_id` |
| fills | SQLite | 不变，加 `strategy_id` |
| account_snapshots | SQLite | 不变，加 `strategy_id` |
| decision_runs | SQLite | 不变，加 `strategy_id` |

### 7.2 Paper 幂等性保证

```python
def check_idempotent(conn, asof: str, strategy_id: str) -> bool:
    """同一天同一策略不重复执行"""
    row = conn.execute(
        """SELECT COUNT(*) FROM decision_runs
           WHERE asof=? AND strategy_id=?
           AND status IN ('paper_filled', 'paper_no_orders', 'paper_checked_no_fill')""",
        (asof, strategy_id)
    ).fetchone()
    return row[0] > 0
```

`daily_run.py` 在每个策略的 decision 步骤前调用此检查。重跑不产生重复 paper 记录。

### 7.3 执行质量监控

每次 paper fill 后自动计算：

```python
def post_trade_analytics(conn, run_id: str, strategy_id: str) -> dict:
    return {
        "fill_count": ...,
        "avg_slippage_bps": ...,
        "total_commission": ...,
        "implementation_shortfall_bps": ...,   # 相对决策价的实现差距
        "fill_validation_rate": ...,            # 成交价在当日 high-low 内的比例
    }
```

---

## 8. ss7 模块拆分

当前 `ss7_sqlite_news_overlay.py`（127KB / ~3200 行）拆分为：

| 新文件 | 职责 | 预计行数 |
|--------|------|---------|
| `model_ridge.py` | PanelRidge + fit_with_cv + make_features + make_target | ~400 |
| `backtest_engine.py` | run_backtest + equity curve + drawdown 计算 | ~600 |
| `execution_model.py` | ExecConfig + execute_rebalance + lot sizing | ~250 |
| `news_overlay.py` | NewsConfig + load_news_items + F/A/U gating | ~300 |
| `benchmark_regime.py` | benchmark_regime_state + _scale + _v2 + VIX 确认 | ~150 |
| `portfolio_optimizer.py` | solve_long_only_meanvar + simplex projection + sector cap | ~250 |
| `ss7_sqlite_news_overlay.py` | 保留为 facade，import 并委托给上述模块 | ~200 |

facade 模式保证所有现有调用方不需要修改 import 路径。

---

## 9. 改造后预期分值

| 维度 | 当前 | 改造后目标 | 达成条件 |
|------|------|-----------|---------|
| 工程架构 | 8.0 | 9.0 | ss7 拆分 + 双策略 config + strategy_id 隔离 |
| 风控意识 | 9.0 | 9.5 | Kelly 约束 + VIX 确认 + Sprint 冷却期 |
| 因子研究 | 5.0 | 8.5 | 因子分层 + CV alpha + 核心集 3 因子 |
| 资金管理 | 3.0 | 9.0 | 双策略分仓 + Phase 门控 + Kelly sizing |
| 数据治理 | 7.0 | 8.5 | SQLite 单源 + 幂等性 + 执行质量监控 |
| 实盘就绪度 | 4.0 | 8.5 | 新闻分阶段接入 + VIX 确认 + paper 验证 |

---

## 10. 设计约束与不可变量

1. **不修改 ss7 核心算法逻辑**（只拆分文件结构）
2. **不删除任何现有功能**（只新增 + 重构）
3. **SQLite 为唯一持久化存储**（不引入新数据库）
4. **Sprint 策略必须先通过 30 天 paper 验证**才能投入真钱
5. **Harvest 只在资金达标后激活**，不提前运行生产信号
6. **所有改造均向后兼容**，现有 `daily_run.py --config config.yaml` 不 break

---

*最后更新：2026-04-04*
