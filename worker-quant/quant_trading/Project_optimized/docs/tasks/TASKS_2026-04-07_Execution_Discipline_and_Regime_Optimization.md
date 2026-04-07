# TASKS: 2026-04-07 执行纪律与 Regime 优化

参考设计文档：
`docs/design/PATCH_2026-04-07_Execution_Discipline_and_Regime_Optimization.md`

优先级说明：[P0] 阻塞性 / [P1] 高优先 / [P2] 中等 / [P3] 低优先/长期
估时单位：天（1天 = 有效工作日，含测试）

---

## 阶段零：交易日调度系统（P0，预计 2 天）

**最高优先级**：当前盘中 7 小时无系统介入，是执行纪律缺失的根本原因。

### [P0] T0. Action Plan 生成 + 盘前推送

- [ ] **T0-1** `action_plan_builder.py` 新建模块
  - `build_action_plan(conn, asof, strategy_id)` → 生成 `action_plan_today.json`
  - 读取最新 `orders_proposal.csv`（pending_sells/pending_buys）
  - 读取当前 positions（含 stop_loss/take_profit 线）
  - 读取 regime 状态
  - 生成 `action_summary` 一句话概括今日操作
  - 输出路径：`reports/action_plan_today.json`
  - 文件路径：`action_plan_builder.py`（新建）

- [ ] **T0-2** `morning_briefing.bat` 扩展
  - 在 briefing 生成后调用 `action_plan_builder.py`
  - 如配置 `alerts.discord_webhook_url`，推送 action_summary + pending 操作
  - 文件路径：`morning_briefing.bat`

- [ ] **T0-3** `daily_run.py` 末尾扩展
  - decision 产出后自动生成次日 action_plan（预生成）
  - 次日 07:30 推送时直接使用，不需重新计算
  - 文件路径：`daily_run.py`

- [ ] **T0-4** 单元测试：`tests/test_action_plan_builder.py`
  - Case 1: 有 SELL 信号 → action_plan 包含 pending_sells
  - Case 2: 无信号 → action_plan 只含持仓 + regime
  - Case 3: action_summary 文本生成正确

---

### [P0] T0B. 盘中执行确认检查（09:30 JST）

- [ ] **T0B-1** `check_pending_actions.py` 新建模块
  - 读取 `action_plan_today.json` 的 pending 操作
  - 查询 `fills` 表：是否已有对应 fill
  - 未执行的信号：写入 `runtime_events.jsonl` (event_type=`pending_action_alert`)
  - 包含：symbol, signal_price, current_price (从 yfinance 实时取), hours_since_signal, urgency
  - 文件路径：`check_pending_actions.py`（新建）

- [ ] **T0B-2** `open_confirm.bat` 新建
  - 09:30 JST 触发
  - 调用 `intraday_update.py --symbols [持仓标的]`（刷新价格）
  - 调用 `check_pending_actions.py`
  - 如有未执行信号 + 配置了 webhook → 推送提醒
  - 文件路径：`open_confirm.bat`（新建）

- [ ] **T0B-3** `scripts/register_open_confirm_task.bat` 新建
  - 注册 Windows Task Scheduler：`QuantOpenConfirm` 工作日 09:30 JST
  - 注意 JST 与本机时区差异（本机北京时间 = JST - 1h → 实际注册 08:30 本地）
  - 文件路径：`scripts/register_open_confirm_task.bat`（新建）

- [ ] **T0B-4** 单元测试
  - Case 1: action_plan 有 SELL 7267.T + fills 无对应 → alert 写入 runtime_events
  - Case 2: action_plan 有 SELL 7267.T + fills 已有 SELL 7267.T → 无 alert
  - Case 3: action_plan 为空 → 无 alert

---

### [P0] T0C. 盘中提醒升级（11:30 + 14:00 JST）

- [ ] **T0C-1** `intraday_monitor.py` 新建模块
  - `--mode midday` (11:30)：更新 intraday + 止损检查 + 未执行信号第二次提醒
  - `--mode pre_close` (14:00)：最终提醒 + 限价建议 + "信号今日失效" 警告
  - 止损线检查：读取 positions 的 stop_loss_price，对比 intraday 最新价
  - 止盈线检查：同理
  - 文件路径：`intraday_monitor.py`（新建）

- [ ] **T0C-2** 提醒级别升级逻辑
  - 09:30 INFO → 11:30 WARNING（加额外亏损估算）→ 14:00 CRITICAL（"最后窗口"）
  - 级别存储在 runtime_events.jsonl，前端/Discord 可按级别过滤
  - 文件路径：`intraday_monitor.py`

- [ ] **T0C-3** 注册 3 个 Task Scheduler 任务
  - `QuantMiddayCheck` 工作日 11:30 JST (10:30 北京)
  - `QuantPreClose` 工作日 14:00 JST (13:00 北京)
  - `scripts/register_intraday_tasks.bat`（新建）

- [ ] **T0C-4** 单元测试
  - Case 1: midday 模式，持仓价格跌破止损 → stop_loss_triggered alert
  - Case 2: pre_close 模式，未执行信号 → CRITICAL 级别提醒
  - Case 3: 无持仓、无未执行信号 → 静默通过

---

### [P0] T0D. 开盘价格监控（09:00-09:30 JST）

- [ ] **T0D-1** `open_watch.bat` 新建
  - 09:00 JST 触发
  - 循环 6 次（每 5 分钟）：`intraday_update.py` + 止损检查
  - 开盘价 vs 昨收偏离 > 2%: 推送 `price_gap_alert`
  - 止损触发: 推送 `stop_loss_triggered` (CRITICAL)
  - 文件路径：`open_watch.bat`（新建）

- [ ] **T0D-2** `scripts/register_open_watch_task.bat`
  - 注册 `QuantOpenWatch` 工作日 09:00 JST (08:00 北京)

---

## 阶段一：执行纪律基础设施（P0，预计 2 天）

**次高优先级**：有了调度系统后，需要数据层支撑（journal + compliance tracking）。

### [P0] T1. Decision Journal 表 + 录入强制校验

- [ ] **T1-1** `trade_schema.py` 新增 `decision_journal` 表
  - 字段：`journal_id, asof, ts, strategy_id, action_type, model_signal, actual_action, override_reason, outcome_pnl, outcome_filled_at, compliance_score`
  - `action_type` 枚举：`model_follow | model_override | manual_entry`
  - 幂等 CREATE TABLE IF NOT EXISTS
  - 文件路径：`trade_schema.py`

- [ ] **T1-2** `compliance_tracker.py` 新增模块
  - `record_action(conn, asof, strategy_id, model_signal, actual_action, override_reason=None)` — 写入 journal
  - `check_model_deviation(conn, run_id, asof, fills)` — 对比 `orders_proposal.csv` 与实际 fills，返回偏差列表
  - `compute_compliance_score(conn, asof_from, asof_to)` — 计算区间遵从率
  - 文件路径：`compliance_tracker.py`（新建）

- [ ] **T1-3** `app.py` Streamlit fill 录入改造
  - 录入 fill 时自动调用 `check_model_deviation`
  - 如果 deviation detected：弹出 `override_reason` 文本框（st.text_area）
  - Phase 1（前 30 天）：soft warning，允许跳过
  - Phase 2（30 天后看遵从率决定）：hard block，不填理由不入库
  - 文件路径：`app.py`

- [ ] **T1-4** 单元测试：`tests/test_compliance_tracker.py`
  - Case 1: 模型建议 SELL 7267.T，实际 fill 是 BUY 7267.T → deviation detected, action_type=model_override
  - Case 2: 模型建议 SELL 7267.T，实际 fill 是 SELL 7267.T → no deviation, action_type=model_follow
  - Case 3: 模型无建议，用户手动 fill → action_type=manual_entry
  - Case 4: compliance_score 计算：3/5 follow → score=0.6

---

### [P0] T2. 未执行信号提醒

- [ ] **T2-1** `daily_run.py` 末尾新增 `_check_pending_exits(conn, asof, strategy_id)`
  - 读取当天 `orders_proposal.csv` 中的 SELL 信号
  - 检查 `fills` 表中是否有对应的 SELL fill
  - 未执行的信号写入 `reports/runtime_events.jsonl`，event_type=`pending_exit_alert`
  - 包含字段：symbol, model_sell_price, current_price, hours_since_signal, estimated_loss
  - 文件路径：`daily_run.py`

- [ ] **T2-2** Discord webhook 推送（可选）
  - 如果 `config.yaml` 中配置了 `alerts.discord_webhook_url`
  - 发送格式：`[ALERT] 模型建议卖出 7267.T，已过 4h 未执行。当前价 1252 vs 建议卖出时 1262，预计额外亏损 -1,000 JPY`
  - 不配置 webhook 时静默跳过（不报错）
  - 文件路径：`daily_run.py` 或独立 `alert_sender.py`

- [ ] **T2-3** 单元测试
  - Case 1: 有 SELL 信号 + 无对应 fill → alert 写入 runtime_events
  - Case 2: 有 SELL 信号 + 有对应 fill → 无 alert
  - Case 3: 无 SELL 信号 → 无 alert

---

## 阶段二：入场时机优化（P1，预计 1.5 天）

### [P1] T3. 限价建议引擎

- [ ] **T3-1** `execution_advisor.py` 新建模块
  - `suggest_limit_price(conn, symbol, side, current_price, asof, aggression=0.7)`
    - 取最近 20 天 `(open - low) / close` 中位数作为日内下探空间
    - BUY: `limit = current_price × (1 - median_dip × aggression)`
    - SELL: `limit = current_price × (1 + median_dip × (1 - aggression))`
    - 输出: `round_to_tick(limit)` (日股 tick size 规则)
  - `round_to_tick(price, symbol)` — 根据价格区间匹配 TSE tick size
  - 文件路径：`execution_advisor.py`（新建）

- [ ] **T3-2** `make_decision.py` 集成限价建议
  - `orders_proposal.csv` 的 `suggested_limit` 列填入计算值（当前为空）
  - `suggested_type` 从 `MKT` 改为 `LIMIT`（保留 MKT 作为 fallback）
  - 文件路径：`make_decision.py`

- [ ] **T3-3** 单元测试：`tests/test_execution_advisor.py`
  - Case 1: 正常 20 天数据，median_dip=1.0%，current=1000 → limit=993
  - Case 2: 低波动股 median_dip=0.3% → limit 接近 current
  - Case 3: tick size 舍入正确（如 1000~3000 区间 tick=1）
  - Case 4: SELL 方向限价高于 current

---

### [P1] T4. 执行质量追踪

- [ ] **T4-1** `trade_schema.py` fills 表新增列
  - `benchmark_vwap REAL` — 当日 VWAP
  - `benchmark_twap REAL` — 当日 TWAP
  - `execution_quality REAL` — (vwap - fill_price) / vwap，正数=买入优于 VWAP
  - ALTER TABLE + try/except 幂等模式
  - 文件路径：`trade_schema.py`

- [ ] **T4-2** `execution_advisor.py` 新增 VWAP/TWAP 计算
  - `calc_vwap(conn, symbol, asof)` — 从 `daily_prices` 取 OHLCV 近似: `(open+high+low+close)/4` 加权 volume（精确 VWAP 需 intraday 数据，先用近似）
  - `calc_execution_quality(fill_price, vwap, side)` — BUY: (vwap-price)/vwap, SELL: (price-vwap)/vwap
  - 文件路径：`execution_advisor.py`

- [ ] **T4-3** `build_positions.py` 或 fill 录入时自动回填执行质量
  - 每笔 fill 入库后计算并更新 benchmark_vwap, execution_quality
  - 文件路径：`build_positions.py` 或 `app.py`

---

## 阶段三：Regime 分级（P1，预计 1.5 天）

### [P1] T5. 三级 Regime 实现

- [ ] **T5-1** `benchmark_regime.py` 新增 `compute_regime_v2()`
  - 输入：px_b, fast_ma, slow_ma, hysteresis_pct (default=0.01)
  - 输出：(state, scale, gap_pct)
  - 三级: risk_on (gap>=+1%, scale=1.0), transition (-1%<gap<+1%, scale=0.50), risk_off (gap<=-1%, scale=0.25)
  - 保留原 `compute_regime()` 不动，新函数独立
  - 文件路径：`benchmark_regime.py`

- [ ] **T5-2** Transition 区间动量确认
  - `transition_entry_allowed(conn, asof, lookback=5)` — 最近 5 天中 MA20 上升天数 >= 3
  - transition + 动量确认 → 允许建仓（仅 Top 1 候选，仓位上限 15%）
  - transition + 无动量确认 → 等同 risk_off
  - 文件路径：`benchmark_regime.py`

- [ ] **T5-3** `make_decision.py` 接入 regime_v2
  - 替换现有 regime 调用为 v2
  - transition 区间限制：max_positions=1, max_single_position_pct=0.15
  - 文件路径：`make_decision.py`

- [ ] **T5-4** `config.yaml` 新增参数
  ```yaml
  benchmark:
    regime_version: "v2"        # "v1" = 二元, "v2" = 三级
    hysteresis_pct: 0.01        # transition 区间阈值
    transition_momentum_days: 5
    transition_momentum_min_rising: 3
    transition_max_positions: 1
    transition_max_position_pct: 0.15
  ```

- [ ] **T5-5** 单元测试：`tests/test_regime_v2.py`
  - Case 1: gap=+2% → risk_on, scale=1.0
  - Case 2: gap=-0.5% → transition, scale=0.50
  - Case 3: gap=-2% → risk_off, scale=0.25
  - Case 4: transition + MA20 连升 4 天 → entry allowed
  - Case 5: transition + MA20 仅升 1 天 → entry blocked
  - Case 6: 从 risk_off 进入 transition → scale 变化验证

---

### [P1] T6. 行业强度 Exception

- [ ] **T6-1** `benchmark_regime.py` 新增 `sector_strength_exception()`
  - 计算个股所在行业近 10 天 return vs benchmark return
  - 相对强度 > 0 且 regime=risk_off → 允许 1 只、仓位上限 NAV×15%
  - 行业映射：使用 `tickers` 表的 `sector` 列
  - 文件路径：`benchmark_regime.py`

- [ ] **T6-2** 集成到 `make_decision.py`
  - risk_off 时额外检查 sector exception
  - exception 触发时在 decision_snapshot 中记录 reason

- [ ] **T6-3** 单元测试
  - Case 1: 大盘 risk_off, 化学板块 +3% vs 大盘 -1% → exception allowed
  - Case 2: 大盘 risk_off, 化学板块 -2% vs 大盘 -1% → no exception
  - Case 3: 已有 1 只 exception 持仓 → 不允许第 2 只

---

## 阶段四：Risk-On 窗口收益最大化（P2，预计 2 天）

### [P2] T7. 窗口质量评分

- [ ] **T7-1** `regime_quality_scorer.py` 新建模块
  - `score_risk_on_window(conn, asof)` → (quality: float, breakdown: dict)
  - 4 个因子: ma_gap (30%), breadth (30%), volume (20%), momentum (20%)
  - 输出: 0.0~1.0 的综合质量分
  - 写入 decision_snapshot 的 `window_quality` 字段
  - 文件路径：`regime_quality_scorer.py`（新建）

- [ ] **T7-2** 接入 Kelly 仓位
  - `kelly_sizer.py` 的 `suggested_weight` 乘以 `window_quality_score`
  - 高质量窗口 (>0.7): 接近满配
  - 低质量窗口 (<0.4): 仓位打折
  - 文件路径：`kelly_sizer.py`

- [ ] **T7-3** 单元测试
  - Case 1: 强趋势 + 宽breadth + 放量 → score > 0.7
  - Case 2: 弱趋势 + 窄breadth + 缩量 → score < 0.4
  - Case 3: Kelly weight 被 quality score 调整验证

---

### [P2] T8. Kelly 快速积累

- [ ] **T8-1** 模拟样本纳入 Kelly
  - risk-off 期间，读取 `orders_proposal.csv` 中的 BUY 信号
  - 模拟执行：假设次日开盘买入，持有 N 天后模拟止损/止盈/到期卖出
  - 结果写入 `kelly_simulated_samples` 表（新建）
  - 标记 `source='simulated'`，Kelly 计算时权重 0.5
  - 文件路径：`kelly_sizer.py`

- [ ] **T8-2** Bootstrap 置信区间
  - `kelly_sizer.py` 新增 `kelly_bootstrap_ci(samples, n_bootstrap=1000, ci=0.95)`
  - sample_count < 30 时，取 Kelly CI 下界（而非固定 fallback 25%）
  - sample_count >= 30 时，正常 Kelly 计算
  - 文件路径：`kelly_sizer.py`

- [ ] **T8-3** 单元测试
  - Case 1: 10 真实 + 10 模拟样本 → 计算出 bootstrap CI
  - Case 2: 模拟样本权重 0.5 → 影响力低于真实样本
  - Case 3: sample_count >= 30 → 不使用 bootstrap，直接 Kelly

---

### [P2] T9. 动态 Trailing Protect

- [ ] **T9-1** `sprint_signal.py` 新增 `dynamic_trailing_stop_pct()`
  - 输入: unrealized_pnl_pct, holding_days, atr_pct
  - 逻辑: 浮盈越大 → trailing 越紧；持有越久 → trailing 越紧
  - 最低 trailing: 1%（防止噪声触发）
  - 文件路径：`sprint_signal.py`

- [ ] **T9-2** 集成到 `sprint_exit_check_v2()`
  - 替换现有固定 `trailing_stop_pct=0.02` 为动态计算
  - config.yaml 新增 `trailing_dynamic: true` 开关（默认 false，手动激活）
  - 文件路径：`sprint_signal.py`, `config.yaml`

- [ ] **T9-3** 回测对比
  - Baseline: 固定 trailing 2%
  - Variant: 动态 trailing (1%~3%)
  - 对比 Sharpe、MaxDD、盈利回吐率
  - 文件路径：`run_risk_backtest_comparison.py`（扩展）

- [ ] **T9-4** 单元测试
  - Case 1: 浮盈 3%（刚激活）→ trailing=2.0%
  - Case 2: 浮盈 8% + 持有 5 天 → trailing 收紧至 ~1.5%
  - Case 3: 浮盈 2%（未激活）→ trailing=None

---

## 阶段五：Compliance Dashboard（P2，预计 1 天）

### [P2] T10. Streamlit Dashboard

- [ ] **T10-1** `app.py` 新增 "Compliance" tab
  - 遵从率折线图（最近 30 天滚动）
  - Override 明细表（日期、标的、模型建议、实际操作、理由、结果盈亏）
  - Override vs Follow 的累计 PnL 对比曲线
  - 行为模式分析：哪些时段/市场状态下容易 override
  - 文件路径：`app.py`

- [ ] **T10-2** 自动 outcome 回填
  - 每次 `daily_run.py` 运行时，回填 7 天前的 override 操作的 outcome_pnl
  - 计算方式：override fill 的盈亏 vs 假设遵从模型的盈亏
  - 文件路径：`compliance_tracker.py`

---

## 验收标准

| 阶段 | 验收条件 | 阻塞关系 | 估时 |
|------|----------|----------|------|
| 阶段零 | T0-T0D 完成，6 段调度可运行，action_plan 可生成推送 | 无前置 | 2 天 |
| 阶段一 | T1-T2 完成，journal 可写入，alert 可触发 | 依赖阶段零（action_plan） | 2 天 |
| 阶段二 | T3-T4 完成，限价建议输出，执行质量可追踪 | 无前置 | 1.5 天 |
| 阶段三 | T5-T6 完成，三级 regime 可切换，sector exception 可用 | 无前置 | 1.5 天 |
| 阶段四 | T7-T9 完成，窗口评分输出，Kelly 脱离 fallback | 依赖阶段三 | 2 天 |
| 阶段五 | T10 完成，Dashboard 可用 | 依赖阶段一 | 1 天 |

**总估时**: 阶段零~五合计约 **10 天**

**建议执行顺序**: 阶段零 → 阶段一 → 阶段三 → 阶段二 → 阶段四 → 阶段五

**MVP（3 天可交付）**: 阶段零 T0+T0B（action_plan + 09:30 确认）+ 阶段一 T1（journal）
仅此三项就能堵住"信号产出但不执行"的核心漏洞。

---

## 风险与注意事项

1. **Compliance hard block 的时机**: 第一个月先 soft warning 收集数据，看遵从率和 override 胜率。如果 override 胜率确实 > 遵从胜率（目前仅 1 个样本，不显著），则保持 soft。
2. **Regime v2 向后兼容**: 保留 v1 函数，config.yaml 中 `regime_version` 控制切换，默认仍为 v1。
3. **模拟样本质量**: 模拟假设次日开盘买入，实际可能不会成交在开盘价。需要加 slippage_bps 模拟。
4. **不修改 ss7 核心**: 所有改动在 decision/execution 层，回测引擎独立。
5. **测试不退化**: 每个阶段完成后跑 `pytest tests/ -v`，确保全绿。

---

*文档结束*
