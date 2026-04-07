# Worker-Quant — 执行纪律与 Regime 优化补丁设计文档

**作者**: Senior Quant Architect
**创建日期**: 2026-04-07
**状态**: APPROVED — 待实施
**对应任务清单**: `../tasks/TASKS_2026-04-07_Execution_Discipline_and_Regime_Optimization.md`
**上承文档**: `PATCH_2026-04-07_Risk_Management_Hardening.md`, `DESIGN_v3.0_Dual_Strategy_Architecture.md`
**触发原因**: 04-06/04-07 实盘操作复盘——建仓无视 benchmark off 信号、清仓延迟导致 Honda -1,200 JPY 亏损。模型信号正确但操作者未遵从。

---

## 0. 问题审计

### 0.1 事件时间线

| 时间 | 事件 | 模型建议 | 实际操作 | 偏差 |
|------|------|----------|----------|------|
| 04-06 | Benchmark regime = off (MA20 < MA60) | 不建仓 | 买入 3 只 | Override |
| 04-06 | Decision: SELL 7267.T, SELL 9432.T | 立刻卖出 | 未执行 | 延迟 ~19h |
| 04-07 09:30 | 7267.T 开盘 1260, 盘中下探 1246 | - | 卖出 @ 1247.5 | 较 04-06 收盘多亏 1,450 |
| 04-07 09:30 | 9432.T 开盘 157.7 | - | 卖出 @ 158.0 | 幸运高于开盘 |

### 0.2 损益归因

| 标的 | 总盈亏 | 模型信号正确性 | 亏损归因 |
|------|--------|----------------|----------|
| 7267.T | -1,200 | 正确（建议卖出） | 100% 执行延迟 |
| 9432.T | +440 | 正确（建议卖出） | 卖出执行偶然获利 |
| 4005.T | +440 (浮盈) | 正确（唯一推荐持有） | 模型选股 alpha |
| **合计** | **-320 (已实现)** | 3/3 正确 | 执行偏差 |

### 0.3 核心矛盾

模型胜率 3/3，但操作者净亏。**当前系统最大的 alpha 泄漏不在模型层，在执行层。**

### 0.4 历史回测支撑

Sprint 策略在 risk-on 窗口的爆发力已被回测验证：

| 窗口 | 收益 | 年化 |
|------|------|------|
| 最佳 20 天 | +30.2% | ~2,678% |
| 最佳 60 天 | +67.1% | ~764% |
| 全周期 1004 天 (variant) | +50.6% | Sharpe 1.18 |

**结论**: risk-on 区间日均 1-2% 有数据支撑。关键不是优化模型的绝对收益率，而是：
1. 精准识别 risk-on 窗口
2. 窗口内最大化执行效率
3. 窗口外零亏损

---

## 0.5 交易日时间轴与系统介入点（核心调度设计）

### 0.5.1 问题

当前系统仅在两个时间点运行：08:30（晨报）和 16:30（收盘全链路）。
盘中 7 小时无任何系统介入，导致：
- SELL 信号产出后无人跟踪执行
- 操作者可以无摩擦地 override 模型
- 盘中价格变动（突破止损线/止盈线）无法触发实时提醒

### 0.5.2 六段式交易日调度

以 JST（日本标准时间）为基准，北京时间 = JST - 1h：

```
┌─────────────────────────────────────────────────────────────────────┐
│  JST       事件             系统动作              操作者动作         │
├─────────────────────────────────────────────────────────────────────┤
│  07:30   ① 盘前准备        db_update + briefing   阅读晨报         │
│          (pre_market)       + 推送 action_plan                     │
│                             到 Discord/LINE                        │
│                                                                    │
│  09:00   ② 开盘监控        intraday_update        根据 action_plan │
│          (open_watch)       (每5min×6次=30min)     执行挂单         │
│                             + 开盘价偏离预警                       │
│                             + 止损线检查                           │
│                                                                    │
│  09:30   ③ 开盘执行确认    check_pending_actions   确认/dismiss     │
│          (open_confirm)     对比 action_plan vs     未执行信号       │
│                             实际 fills                             │
│                             未执行 → 第一次提醒                    │
│                                                                    │
│  11:30   ④ 午盘检查        intraday_update        review 午盘状态  │
│          (midday_check)     + 止损/止盈线检查                      │
│                             + 未执行信号第二次提醒                 │
│                             + regime 实时状态                      │
│                                                                    │
│  14:00   ⑤ 尾盘决策窗口    intraday_update        最后执行机会     │
│          (pre_close)        + 未执行信号最终提醒                   │
│                             + "今日不执行则信号                    │
│                               失效" 警告                          │
│                             + 尾盘限价建议                         │
│                                                                    │
│  16:30   ⑥ 收盘全链路      daily_run.py           录入 fills       │
│          (post_close)       (数据+模型+决策)       post_trade.bat   │
│                             + 次日 action_plan     build_positions  │
│                             + compliance 记录      build_snapshot   │
│                             + outcome 回填                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 0.5.3 各时段详细规格

#### ① 盘前准备 (07:30 JST / 06:30 北京)

**触发**: Windows Task Scheduler / cron
**脚本**: `morning_briefing.bat`（已有，需扩展）
**新增输出**: `reports/action_plan_today.json`

```json
{
  "asof": "2026-04-07",
  "generated_at": "2026-04-07T07:30:00+09:00",
  "regime": "risk_off",
  "regime_detail": {"ma_gap_pct": -1.9, "enter_line": 56072},
  "pending_sells": [
    {"symbol": "7267.T", "reason": "model_signal", "signal_price": 1262.0,
     "suggested_limit": 1255.0, "urgency": "HIGH"}
  ],
  "pending_buys": [],
  "held_positions": [
    {"symbol": "4005.T", "qty": 100, "cost": 528.8, "stop_loss": 508.6,
     "take_profit": 687.0, "current_status": "HOLD"}
  ],
  "risk_alerts": [],
  "action_summary": "今日需执行: SELL 7267.T。无新建仓。持有 4005.T 观察。"
}
```

**推送**: 如配置 `alerts.discord_webhook_url` 或 `alerts.line_notify_token`，
推送 `action_summary` + pending 操作列表。

#### ② 开盘监控 (09:00-09:30 JST)

**触发**: Task Scheduler 09:00，运行 `intraday_monitor.py`（新建）
**行为**:
- 调用 `intraday_update.py` 拉取持仓标的 1m K 线（每 5 分钟一次，共 6 次）
- 每次拉取后检查：
  - 开盘价 vs 昨收偏离 > 2%? → 推送 `price_gap_alert`
  - 当前价 < 止损线? → 推送 `stop_loss_triggered`（urgency=CRITICAL）
  - 当前价 > 止盈线? → 推送 `take_profit_triggered`（urgency=HIGH）
- 首次运行时将开盘价写入 `reports/runtime_latest_event.json`

**分析间隔**: 5 分钟（09:00-09:30 共 6 次检查）

#### ③ 开盘执行确认 (09:30 JST)

**触发**: Task Scheduler 09:30，运行 `check_pending_actions.py`（新建）
**行为**:
- 读取 `action_plan_today.json` 中的 pending_sells / pending_buys
- 查询 `fills` 表：是否已有对应 fill？
- 未执行 → 写入 runtime_events + 推送**第一次提醒**
- 提醒内容: `[09:30] 7267.T SELL 信号未执行。当前价 1258，建议卖出价 1255。请确认或 dismiss。`

#### ④ 午盘检查 (11:30 JST)

**触发**: Task Scheduler 11:30
**脚本**: `intraday_monitor.py --mode midday`
**行为**:
- 更新 intraday 数据
- 止损/止盈线再检查（价格可能盘中触发）
- 未执行信号 → **第二次提醒**（措辞加重）
- 内容: `[11:30] ⚠️ 7267.T SELL 仍未执行（已过 2h）。当前价 1250，较信号价 -0.95%。预计额外损失 -1,200 JPY。`
- 同时输出 regime 实时状态（MA 是否在盘中变化）

**分析间隔**: 单次运行（不重复）

#### ⑤ 尾盘决策窗口 (14:00 JST)

**触发**: Task Scheduler 14:00
**脚本**: `intraday_monitor.py --mode pre_close`
**行为**:
- 更新 intraday 数据
- 未执行信号 → **最终提醒**（红色警告）
- 内容: `[14:00] 🔴 7267.T SELL 今日最后执行窗口。收盘前 1h 内建议 MARKET ORDER。今日不执行则信号失效，明日开盘价不可预测。`
- 对 pending_buys：输出尾盘限价建议（基于日内 VWAP 和剩余 1h 预估）
- 对持仓：输出日内 PnL 变动和止损/止盈距离

**分析间隔**: 单次运行

#### ⑥ 收盘全链路 (16:30 JST)

**触发**: Task Scheduler 16:30（已有）
**脚本**: `daily_run.bat`（已有，需扩展）
**新增行为**:
- daily_run 完成后自动生成次日 `action_plan_today.json`（预生成）
- 运行 `compliance_tracker.py`：对比今日 action_plan vs 实际 fills，写入 decision_journal
- 运行 outcome 回填：7 天前的 override 操作计算实际 PnL

### 0.5.4 提醒升级策略

| 时段 | 距信号产出 | 提醒级别 | 措辞 | 渠道 |
|------|-----------|---------|------|------|
| 09:30 | +0.5h | INFO | "信号未执行，请确认" | runtime_events + Discord |
| 11:30 | +2.5h | WARNING | "仍未执行，预计额外损失 ¥X" | runtime_events + Discord |
| 14:00 | +5h | CRITICAL | "最后执行窗口，今日不执行则失效" | runtime_events + Discord + LINE |
| 16:30 | 收盘 | AUDIT | 记录为 miss/override/follow | decision_journal |

### 0.5.5 风险评估时间点

| 评估类型 | 时间 | 频率 | 数据源 |
|----------|------|------|--------|
| 止损线检查 | 09:00-09:30 | 每 5 min | intraday_quotes |
| 止盈线检查 | 09:00-09:30 | 每 5 min | intraday_quotes |
| 组合回撤检查 | 11:30, 14:00 | 各 1 次 | positions + intraday |
| Regime 状态 | 07:30, 11:30, 14:00 | 各 1 次 | daily_prices (ETF) |
| 新闻风险扫描 | 07:30 | 1 次 | news_feed |
| 执行偏差检查 | 09:30, 11:30, 14:00, 16:30 | 4 次 | action_plan vs fills |

### 0.5.6 Windows Task Scheduler 注册

新增 4 个定时任务（含已有 2 个的更新）:

| 任务名 | 时间 (JST) | 脚本 | 状态 |
|--------|-----------|------|------|
| QuantPreMarket | 07:30 | `pre_market.bat` | **新增** |
| QuantOpenWatch | 09:00 | `open_watch.bat` | **新增** |
| QuantOpenConfirm | 09:30 | `open_confirm.bat` | **新增** |
| QuantMiddayCheck | 11:30 | `midday_check.bat` | **新增** |
| QuantPreClose | 14:00 | `pre_close.bat` | **新增** |
| QuantDailyRun | 16:30 | `daily_run.bat` | 已有，扩展 |

注意：JST 与本机时间的差异需在 .bat 中处理（当前机器为北京时间 = JST - 1h）。

### 0.5.7 最小可行实现（MVP）

如果 6 个时段全部实现工程量过大，**MVP 只需 3 个**：

| 优先级 | 时段 | 理由 |
|--------|------|------|
| P0 | ⑥ 16:30 收盘（扩展） | 已有，加 action_plan 生成 + compliance 记录 |
| P0 | ① 07:30 盘前（扩展） | 已有，加 action_plan 推送 |
| P0 | ③ 09:30 执行确认 | **新建**，堵住"信号产出但不执行"的漏洞 |

其余 ②④⑤ 为 P1，在 MVP 验证有效后再加。

---

## 1. 设计方向一：执行纪律辅助系统

### 1.1 问题

操作者手动执行时存在 action bias（空仓焦虑导致 override）和 disposition effect（延迟止损）。纯靠自律不可靠，需要系统级约束。

### 1.2 Decision Journal（决策日志）

在 `fills` 表之外新增 `decision_journal` 表：

```sql
CREATE TABLE IF NOT EXISTS decision_journal (
  journal_id TEXT PRIMARY KEY,
  asof TEXT NOT NULL,
  ts TEXT NOT NULL,
  strategy_id TEXT DEFAULT 'sprint',
  action_type TEXT NOT NULL,        -- 'model_follow' | 'model_override' | 'manual_entry'
  model_signal TEXT,                -- 模型当时的建议 (JSON)
  actual_action TEXT NOT NULL,      -- 实际执行的操作 (JSON)
  override_reason TEXT,             -- override 时必须填写理由
  outcome_pnl REAL,                -- 事后回填：这笔操作的盈亏
  outcome_filled_at TEXT,           -- outcome 回填时间
  compliance_score REAL             -- 0.0~1.0，自动计算
);
```

**规则**：
- 每次通过 Streamlit 或 CLI 录入 fill 时，如果当天存在 `orders_proposal.csv` 且实际操作与建议不一致，强制弹出 override_reason 输入框
- 不填理由则 fill 不入库（hard block）
- 每周日自动回填 outcome_pnl（对比 override 操作 vs 假设遵从模型的收益差）

### 1.3 Compliance Dashboard

在 `app.py` (Streamlit) 中新增 tab：

- **遵从率**: 最近 30 天 `model_follow / total_actions`
- **Override 盈亏**: override 操作的累计 PnL vs 遵从模型的模拟 PnL
- **行为模式**: 是否在特定时间/市场状态下更容易 override（统计检验）

### 1.4 Alert 推送

当模型产出 SELL 信号后，如果 2 小时内未在 fills 表中看到对应 SELL fill：

- 第一次提醒：写入 `reports/runtime_events.jsonl`
- 第二次提醒（+2h）：如果配置了 Discord webhook，推送消息
- 每次提醒内容包含：标的、当前价、模型建议卖出价、已亏损金额

实现位置：`daily_run.py` 末尾新增 `_check_pending_exits()` 函数。

---

## 2. 设计方向二：入场时机优化

### 2.1 问题

当前系统只输出"买不买"，不输出"什么价位买"。操作者倾向于尾盘一次性 market order 建仓，错过日内更优价格。

### 2.2 Limit Order 建议引擎

在 `make_decision.py` 的 orders_proposal 输出中，新增限价建议列：

```csv
symbol,side,qty,suggested_type,suggested_limit,est_notional,comment
4005.T,BUY,100,LIMIT,525.0,52500,limit=prev_close*(1-slippage_target_pct)
```

**计算逻辑**：

```python
def suggest_limit_price(symbol, side, current_price, atr_pct, conn, asof):
    """基于 ATR 和日内波动特征，建议限价"""
    # 取最近 20 天的 (open - low) / close 中位数作为日内下探空间
    intraday_dips = fetch_intraday_dip_pct(conn, symbol, asof, lookback=20)
    median_dip = np.median(intraday_dips)  # 典型值 0.5%~1.5%
    
    if side == 'BUY':
        # 限价 = 前收盘 × (1 - median_dip × aggression)
        # aggression: 0.5 = 保守（更容易成交）, 1.0 = 激进（价格更优但可能不成交）
        limit = current_price * (1 - median_dip * 0.7)
    elif side == 'SELL':
        limit = current_price * (1 + median_dip * 0.5)
    
    return round_to_tick(limit, symbol)
```

### 2.3 分批建仓策略

对于大于 NAV 15% 的订单，建议拆分为 2-3 批：

| 批次 | 时间 | 数量 | 价格 |
|------|------|------|------|
| 1/3 | 开盘 9:00-9:15 | 33% | Market |
| 2/3 | 盘中 10:00-11:00 | 33% | Limit (开盘价 - 0.5%) |
| 3/3 | 尾盘 14:30-15:00 | 34% | Limit (VWAP - 0.3%) |

输出到 `orders_proposal.csv` 的 `execution_plan` 列（JSON 格式）。

### 2.4 成交价 Benchmark 对比

每笔 fill 入库时，自动记录当日 VWAP 和 TWAP：

```python
# build_positions.py 或 fill 录入时
fill.price_benchmark_vwap = calc_vwap(conn, symbol, asof)
fill.price_benchmark_twap = calc_twap(conn, symbol, asof)
fill.execution_quality = (vwap - fill.price) / vwap  # 正数 = 比 VWAP 好
```

新增 fills 表列：`benchmark_vwap REAL`, `benchmark_twap REAL`, `execution_quality REAL`

---

## 3. 设计方向三：Benchmark Regime 分级优化

### 3.1 问题

当前 regime 是二元切换：MA20 >= MA60 → on, MA20 < MA60 → off。
实际上 MA20 接近 MA60 时（差距 < 1%），市场可能即将反转，完全 risk-off 会错过反转初期的 alpha 窗口。

### 3.2 三级 Regime 模型

替换 `benchmark_regime.py` 中的二元状态为三级：

```python
def compute_regime_v2(px_b, fast_ma, slow_ma, hysteresis_pct=0.01):
    gap_pct = (fast_ma - slow_ma) / slow_ma
    
    if gap_pct >= hysteresis_pct:
        return "risk_on", 1.0          # MA20 明确在 MA60 上方
    elif gap_pct >= -hysteresis_pct:
        return "transition", 0.50      # MA20 在 MA60 附近 (±1%)
    else:
        return "risk_off", 0.25        # MA20 明确在 MA60 下方
```

| Regime | MA20 vs MA60 | 允许操作 | Scale |
|--------|-------------|----------|-------|
| risk_on | gap >= +1% | 全仓位 | 1.0 |
| transition | -1% < gap < +1% | 半仓试探，仅 Top 1 候选 | 0.50 |
| risk_off | gap <= -1% | 仅持有，不开新仓 | 0.25 |

### 3.3 趋势动量确认

transition 区间内额外要求动量确认才允许建仓：

```python
def transition_entry_allowed(conn, asof, lookback=5):
    """transition 区间内，需要 MA20 连续 N 天上升才允许入场"""
    ma20_series = fetch_ma20_series(conn, asof, lookback)
    rising_days = sum(1 for i in range(1, len(ma20_series)) 
                      if ma20_series[i] > ma20_series[i-1])
    return rising_days >= 3  # 最近 5 天中至少 3 天 MA20 在上升
```

### 3.4 行业强度 Exception

即使大盘 risk-off，如果个股所在行业近 10 天相对强度 > 0（跑赢大盘），允许以 25% 仓位试探：

```python
def sector_strength_exception(conn, symbol, asof, lookback=10):
    """检查个股行业是否逆势走强"""
    sector = get_sector(conn, symbol)
    sector_return = calc_sector_return(conn, sector, asof, lookback)
    benchmark_return = calc_benchmark_return(conn, asof, lookback)
    relative_strength = sector_return - benchmark_return
    return relative_strength > 0
```

**限制条件**：exception 最多允许 1 只标的，仓位上限 NAV × 15%。

### 3.5 当前状态分析

以当前数据为例：
- MA20: 55,559 | MA60: 56,638 | gap: **-1.9%**
- 状态: **risk_off**（差距超过 -1%，不在 transition 区间）
- 重入 transition: MA20 需升至 56,074 (+0.9%)
- 重入 risk_on: MA20 需升至 57,204 (+3.0%)

---

## 4. 设计方向四：Risk-On 窗口收益最大化

### 4.1 问题

回测显示最佳 20 天窗口 +30%、60 天窗口 +67%，但实盘 Kelly sample_count=2（远低于 min_samples=30）。当前无法区分"好的 risk-on"和"平庸的 risk-on"。

### 4.2 窗口质量评分

新增 `regime_quality_scorer.py`，在 risk-on 激活时评估窗口质量：

```python
def score_risk_on_window(conn, asof):
    """
    评估当前 risk-on 窗口的质量，输出 0.0~1.0
    
    因子:
    1. MA gap magnitude — gap 越大，趋势越强
    2. Breadth — screener 候选池中上涨股占比
    3. Volume confirmation — 大盘成交量 vs 20 日均量
    4. Momentum persistence — 连续上涨天数
    5. Sector rotation — 领涨行业是否从防御转向进攻
    """
    scores = {}
    
    # Factor 1: MA gap (0~1, capped at 3%)
    gap_pct = compute_ma_gap(conn, asof)
    scores['ma_gap'] = min(gap_pct / 0.03, 1.0)
    
    # Factor 2: Market breadth
    advancing = count_advancing_stocks(conn, asof)
    total = count_screened_stocks(conn, asof)
    scores['breadth'] = advancing / max(total, 1)
    
    # Factor 3: Volume confirmation
    vol_ratio = today_volume(conn, asof) / ma_volume(conn, asof, 20)
    scores['volume'] = min(vol_ratio / 1.5, 1.0)
    
    # Factor 4: Momentum persistence
    up_streak = count_consecutive_up_days(conn, asof)
    scores['momentum'] = min(up_streak / 5, 1.0)
    
    # Weighted average
    weights = {'ma_gap': 0.3, 'breadth': 0.3, 'volume': 0.2, 'momentum': 0.2}
    quality = sum(scores[k] * weights[k] for k in scores)
    
    return quality, scores
```

### 4.3 Quality-Adjusted Position Sizing

将窗口质量分数接入 Kelly 仓位计算：

```python
# 现有: suggested_weight = kelly_fraction * fallback_position_pct
# 改为:
effective_weight = base_weight * window_quality_score
# 高质量窗口 (score > 0.7): 仓位接近满配
# 低质量窗口 (score < 0.4): 仓位打折，保守入场
```

### 4.4 快速 Kelly 积累

当前 Kelly min_samples=30 导致前 2-3 个月只能用 fallback。优化方案：

1. **纳入模拟交易样本**: 将 risk-off 期间的"如果遵从模型会怎样"的模拟结果计入 Kelly 样本（标记 `source='simulated'`，权重 0.5）
2. **Bootstrap 置信区间**: sample_count < 30 时，用 bootstrap 估计 Kelly fraction 的 95% CI，取下界作为保守估计（而非直接 fallback 到固定 25%）
3. **目标**: 2 周内积累 15+ 真实样本 + 15 模拟样本，脱离 fallback 模式

### 4.5 Exit 优化: Trailing Protect 精细化

当前 trailing protect: activate 3%, stop 2%（固定）。优化为动态：

```python
def dynamic_trailing(unrealized_pnl_pct, holding_days, atr_pct):
    """浮盈越大、持有越久，trailing 越紧"""
    if unrealized_pnl_pct < 0.03:
        return None  # 未激活
    
    # Base trailing = 2%
    base = 0.02
    
    # 浮盈越大，trailing 越紧 (保护利润)
    profit_tightening = max(0, (unrealized_pnl_pct - 0.05)) * 0.5
    
    # 持有越久，trailing 越紧 (sprint 策略不恋战)
    time_tightening = min(holding_days * 0.002, 0.01)
    
    trailing_stop_pct = max(base - profit_tightening - time_tightening, 0.01)
    return trailing_stop_pct
```

---

## 5. 架构影响与约束

### 5.1 新增文件

| 文件 | 用途 |
|------|------|
| `regime_quality_scorer.py` | 窗口质量评分 |
| `execution_advisor.py` | 限价建议 + 分批建仓 + 执行质量评估 |
| `compliance_tracker.py` | 决策日志 + 遵从率统计 + Override 分析 |

### 5.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `benchmark_regime.py` | 二元 → 三级 regime |
| `make_decision.py` | 接入限价建议、regime 分级、窗口质量 |
| `kelly_sizer.py` | Bootstrap CI + 模拟样本纳入 |
| `sprint_signal.py` | 动态 trailing |
| `trade_schema.py` | 新增 decision_journal 表、fills 扩展列 |
| `app.py` | 新增 Compliance Dashboard tab |
| `daily_run.py` | 新增 pending exit check |
| `build_account_snapshot.py` | outcome 回填逻辑 |

### 5.3 不修改的文件

- `ss7_sqlite_news_overlay.py` — 核心回测引擎不动
- `screener.py` — 选股逻辑独立
- `config.yaml` 中已有参数不变，仅新增参数

### 5.4 风险

| 风险 | 缓解 |
|------|------|
| 三级 regime 增加过拟合风险 | hysteresis 参数可配置，默认 1% 保守 |
| 限价单不成交导致错过机会 | 第 3 批用 market order 兜底 |
| 模拟样本污染 Kelly | 模拟样本权重 0.5 且标记来源，可随时排除 |
| Compliance hard block 太烦 | 第一个月先 soft warning，看遵从率再决定是否 hard block |

---

## 6. 成功标准

| 指标 | 当前值 | 30天目标 | 90天目标 |
|------|--------|----------|----------|
| 模型遵从率 | ~33% (1/3) | >= 80% | >= 95% |
| Override 盈亏比 | 未追踪 | 有数据可查 | Override 胜率 < 遵从胜率则收紧 |
| Kelly sample_count | 2 | >= 15 (含模拟) | >= 30 (真实) |
| Risk-on 窗口捕获率 | 未追踪 | >= 70% | >= 85% |
| 执行质量 (vs VWAP) | 未追踪 | > 0 (买入优于 VWAP) | 持续 > 0 |
| Risk-off 期间亏损 | -760 JPY | 0 | 0 |

---

*文档结束*
