# TASKS: 2026-04-04 双策略架构实施

参考设计文档：
`docs/design/DESIGN_v3.0_Dual_Strategy_Architecture.md`

参考治理文档：
`docs/governance/GOVERNANCE_v2_DUAL_STRATEGY.md`

优先级说明：[P0] 阻塞性 / [P1] 高优先 / [P2] 中等 / [P3] 低优先/长期
估时单位：天（1天 = 有效工作日，含测试）

---

## 阶段一：基础设施（P0，预计 3 天）

所有后续工作依赖此阶段完成。

### [P0] T1. strategy_profiles 配置层

- [x] **T1-1** `config.yaml` 新增 `strategy_profiles` 节点，包含 `sprint` 和 `harvest` 两个完整策略配置块
  - Sprint: `strategy_id=sprint`, `position_sizing=half_kelly`, `max_positions=3`, `stop_loss_vol_mult=3.0`
  - Harvest: `strategy_id=harvest`, `enabled=false`, `activation_threshold=2000000`, `max_positions=12`
  - 文件路径：`config.yaml`
  - 验证：`python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); assert 'sprint' in cfg['strategy_profiles']"`

- [x] **T1-2** `daily_run.py` 新增 `load_strategy_profiles(cfg)` 函数，解析 strategy_profiles 并返回激活策略列表
  - 逻辑：读取总 NAV -> `resolve_phase()` -> 根据 phase 决定哪些策略 enabled
  - 向后兼容：无 `strategy_profiles` 节点时，行为等同当前单策略模式
  - QA 修复（2026-04-04）：返回类型改为 `list[dict]`，Phase 2 返回全部激活策略
  - 文件路径：`daily_run.py`

- [x] **T1-3** `daily_run.py` 主循环改造为 per-strategy 执行
  - 对每个激活策略分别运行：screener -> model -> decision -> paper
  - 各步骤接受 `strategy_config` 参数
  - 文件路径：`daily_run.py` `main()` 函数

---

### [P0] T2. SQLite schema 扩展（strategy_id 隔离）

- [x] **T2-1** `trade_schema.py` 新增 migration：为 `positions`, `orders`, `fills`, `account_snapshots`, `decision_runs` 表添加 `strategy_id TEXT DEFAULT 'default'` 列
  - 幂等性：使用 `ALTER TABLE ... ADD COLUMN` + try/except（SQLite 不支持 IF NOT EXISTS 对 column）
  - 向后兼容：现有数据的 strategy_id 默认为 `'default'`
  - 文件路径：`trade_schema.py` `ensure_trade_tables()`

- [x] **T2-2** 所有写入上述表的代码路径添加 `strategy_id` 参数
  - 涉及文件：`make_decision.py`, `paper_execute.py`, `build_positions.py`, `build_account_snapshot.py`, `import_fills.py`
  - 查找方法：`grep -rn "INSERT INTO positions\|INSERT INTO orders\|INSERT INTO fills\|INSERT INTO account_snapshots\|INSERT INTO decision_runs" *.py`

- [x] **T2-3** 所有读取上述表的代码路径添加 `WHERE strategy_id=?` 过滤
  - 涉及文件：`make_decision.py`, `paper_execute.py`, `live_trade_advisor.py`, `evaluate_promotion.py`, `quant_briefing.py`

- [x] **T2-4** Paper 幂等性检查
  - 新增 `check_idempotent(conn, asof, strategy_id) -> bool`
  - `daily_run.py` 在 decision 步骤前调用，已完成则跳过
  - 文件路径：`daily_run.py`

---

### [P0] T3. Half-Kelly 仓位管理器

- [x] **T3-1** 新增 `kelly_sizer.py`，实现 `KellyPositionSizer` dataclass
  - 输入：滚动胜率、平均盈利、平均亏损、kelly_fraction、max/min_position_pct
  - 输出：`suggested_weight` 属性
  - 安全护栏：edge <= 0 返回 0，样本不足 30 回退固定 10%
  - 连续 3 次止损冷却期（`cooldown_remaining_days` 属性）

- [x] **T3-2** 新增 `compute_kelly_params(conn, strategy_id, lookback_days=60) -> dict`
  - 从 `fills` + `daily_prices` 计算滚动胜率、盈亏比
  - 返回 `{win_rate, avg_win, avg_loss, sample_count, edge, suggested_weight}`
  - 文件路径：`kelly_sizer.py`

- [x] **T3-3** `make_decision.py` 集成 Kelly sizer
  - 当 `strategy_config.position_sizing == "half_kelly"` 时，用 Kelly 替换等权/Ridge 权重
  - 文件路径：`make_decision.py`

- [x] **T3-4** 测试（5 tests PASS）
  - 单元测试：edge > 0 / edge = 0 / edge < 0 / 样本不足 / 冷却期
  - 文件路径：`tests/test_kelly_sizer.py`

---

## 阶段二：因子研究改造（P1，预计 2 天）

### [P1] T4. 因子分层体系

- [x] **T4-1** `config.yaml` 新增因子分层配置
  ```yaml
  factor_tiers:
    core: ["mom_consist", "ma_gap", "sharpe_60"]
    candidate: ["high52w", "mom_12_1", "ret60", "vol_z"]
    fundamental_pending: ["roa_op", "cfo_assets", "accruals_inv", "margin_op", "leverage_safety"]
    excluded: ["ret20", "rsi14", "slope60", "vol_adj_mom20", "ret1", "ret5", "vol20", "vol60", "z_20"]
  factor_promotion_rules:
    min_observations: 100
    min_t_stat: 1.5
    min_ic_mean: 0.01
    demotion_t_stat: 0.5
    demotion_lookback_periods: 60
    review_frequency_days: 30
  ```

- [x] **T4-2** `compute_ic.py` 改造：区分 core/candidate/excluded 因子
  - Core 因子：计算 IC 并���新生产权重
  - Candidate 因子：计算 IC 但只写 shadow 记录，不更新生产权重
  - Excluded 因子：跳过计算
  - 自动晋升/降级：candidate 达标 -> core，core 不达标 -> candidate
  - 晋升/降级事件写入 `learning_audit` 表

- [x] **T4-3** `factor_health_report.py` 更新：输出因子层级状态
  - 报告新增列：`tier`（core/candidate/excluded）、`promotion_eligible`、`demotion_risk`

---

### [P1] T5. Ridge alpha Time-Series CV

- [x] **T5-1** `model_ridge.py` 的 `PanelRidge` 新增 `fit_with_cv()` 方法（已从 ss7 提取到独立模块）
  - 候选 alpha：`[1, 5, 10, 25, 50, 100, 200]`
  - 验证方式：`TimeSeriesSplit(n_splits=5)`
  - 评价指标：截面 Spearman IC（非 MSE）
  - 返回最优 alpha 并用全量数据 refit

- [x] **T5-2** `config.yaml` 新增 `model.ridge_alpha_cv: false`（默认 false，启用后走 CV）
  - 向后兼容：`false` 时沿用 `alpha=50.0` 硬编码
  - `daily_run.py` 通过 `SS7_RIDGE_ALPHA_CV` 环境变量传递给 ss7

- [x] **T5-3** 回测对比：CV alpha vs 固定 alpha=50 �� Sharpe / IC 差异
  - 输出到 `reports/ridge_cv_comparison.json`
  - ss7 backtest 结束时自动输出 CV 结果统计（mean/median best_alpha, mean CV IC）

---

### [P1] T6. Sprint 专用信号生成器

- [x] **T6-1** 新增 `sprint_signal.py`
  - `sprint_screener(conn, asof, config) -> pd.DataFrame`：Sprint 专用选股
  - `sprint_score(features, ic_weights) -> pd.Series`：IC 加权 z-score 排名（不经过 Ridge）
  - `sprint_entry_check(row, benchmark_state) -> bool`：进场条件检查
  - `sprint_exit_check(row, holding_days, benchmark_state) -> tuple[bool, str]`：出场条件检查

- [x] **T6-2** `daily_run.py` 集成：当 `strategy_config.signal_mode == "sprint_momentum"` 时调用 `sprint_signal.py`

- [x] **T6-3** 测试（4 tests PASS）
  - 测试：进场条件全满足 / 部分不满足 / benchmark off 阻止进场
  - 测试：出场条件：止损 / 持有期 / 量能衰竭 / benchmark off
  - QA 修复（2026-04-04）：`mom_consist` 改为截面百分位 `mom_consist_pctile >= 0.80`
  - QA 修复（2026-04-04）：`prev_state` 从 `regime_diagnosis.json` 读取，不再硬编码 "off"
  - 文件路径：`tests/test_sprint_signal.py`

---

## 阶段三：风控与 Benchmark 改造（P1，预计 1.5 天）

### [P1] T7. Benchmark Risk-Off 增强

- [x] **T7-1** `benchmark_regime.py` 实现 `benchmark_regime_scale_v2()`（独立模块，非 ss7 内）
  - 在现有 `benchmark_regime_scale()` 基础上增加 VIX 二次确认参数
  - MA 判定 off + VIX < threshold -> 降级为 caution
  - 文件路径：`ss7_sqlite_news_overlay.py`

- [x] **T7-2** `db_update.py` 将 `1552.T`（日经 VI 指数）加入默认更新列表
  - 确保 `daily_prices` 表中有 VI 指数数据
  - 如果 `1552.T` 数据不可用，VIX 确认静默降级为 disabled

- [x] **T7-3** `daily_run.py` / `make_decision.py` 读取策略配置的 benchmark 参数
  - Sprint 用 `sprint.benchmark_off_scale`
  - Harvest 用 `harvest.benchmark_off_scale`
  - 向后兼容：无 strategy_profiles 时沿用全局 `model.benchmark_off_scale`

- [x] **T7-4** 测试（3 tests PASS）
  - 测试：MA off + VIX 高 -> 确认 off
  - 测试：MA off + VIX 低 -> 降级 caution
  - 测试：VIX 数据缺失 -> 静默 fallback 到旧逻辑
  - 文件路径：`tests/test_benchmark_regime_v2.py`

---

### [P1] T8. Sprint 冷却期机制

- [x] **T8-1** `kelly_sizer.py` 新增冷却期逻辑
  - 连续止损计数：从 `fills` 表查询最近 N 笔 Sprint 成交的盈亏
  - 连续 3 笔止损 -> `cooldown_remaining_days = 5`
  - 冷却期内 `suggested_weight = 0`

- [x] **T8-2** `daily_run.py` 在 Sprint decision 前检查冷却期
  - 冷却期内跳过 Sprint，只记录 `runtime_event: sprint_cooldown`

---

## 阶段四：数据治理收口（P2，预计 1.5 天）

### [P2] T9. 废弃 paper_trading_account.json 作为数据源

- [x] **T9-1** 审计所有读取 `paper_trading_account.json` 的代码路径
  - `grep -rn "paper_trading_account" *.py`
  - 预计涉及：`quant_briefing.py`, `live_trade_advisor.py`

- [x] **T9-2** 将这些代码路径改为从 SQLite 读取（带 `strategy_id` 过滤）

- [x] **T9-3** `daily_run.py` 末尾保留 JSON 导出，但标记 `"read_only": true, "source_of_truth": "sqlite"`

进展备注（2026-04-04）：
- `quant_briefing.py` 与 `live_trade_advisor.py` 已改为直接读取 SQLite，并支持 `strategy_id` 过滤。
- 当前仓库内不再有任何决策路径读取 `paper_trading_account.json`；该文件仅作为只读诊断快照导出。

---

### [P2] T10. 执行质量监控

- [x] **T10-1** `paper_execute.py` 新增 `post_trade_analytics(conn, run_id, strategy_id) -> dict`
  - 计算：fill_count, avg_slippage_bps, total_commission, fill_validation_rate
  - 输出到 `reports/execution_quality.json`

- [x] **T10-2** `daily_run.py` 在 paper 步骤后调用 `post_trade_analytics()`
  - 结果写入 `runtime_events.jsonl`（level=info，含 strategy_id）

进展备注（2026-04-04）：
- `paper_execute.py` 已输出 `reports/execution_quality.json`。
- `daily_run.py` 已写入 `execution_quality` 运行时事件，并补充了回归测试。

---

## 阶段五：新闻分阶段接入（P2，预计 1 天）

### [P2] T11. 新闻 Phase 1（Shadow 模式）

- [x] **T11-1** `config.yaml` 修改：`news.enabled: true`, 新增 `news.shadow_only: true`

- [x] **T11-2** `ss7_sqlite_news_overlay.py` 的新闻 gating 逻辑加入 shadow 开关
  - `shadow_only=true` 时：计算 F/A/U gate 值，写入日志，但 `g_min=1.0`（gate 不生效）
  - 日志写入 `learning_audit` 表：`{factor: "news_gate", gate_value, would_have_applied}`

- [ ] **T11-3** 30 天后评估：分析 shadow 日志中 news_gate 与次日收益的相关性
  - 输出 `reports/news_shadow_evaluation.json`

进展备注（2026-04-04）：
- `T11-1/T11-2` 的代码与配置已落地，`news_shadow_evaluation.json` 已开始产出。
- 但 30 天 shadow 观察窗口尚未完成，所以 `T11-3` 继续保持未完成状态。

---

### [P2] T12. 新闻 Phase 2（Sprint Gating）

**依赖 T11 完成 + 30 天 shadow 数据积累**

- [x] **T12-1** `config.yaml` 新增 `news.sprint_gating: true`
- [x] **T12-2** `sprint_signal.py` 集成新闻门控
  - `news_risk = "HIGH"` -> Kelly 仓位 × 0.5
  - `news_risk = "CRITICAL"` -> 跳过该标的
  - macro_sentiment < -0.5 -> 全策略 caution_scale × 0.8
- [x] **T12-3** A/B 对比：Sprint with news gating vs Sprint without
  - 输出 `reports/news_gating_ab_test.json`

进展备注（2026-04-04 QA 修复后更新）：
- Sprint 新闻门控与 A/B 报告生成已实现。
- QA 修复：`config.yaml` 中 `sprint_gating` 改为 `false`，治理要求 shadow 运行满 30 天后才可启用（GOVERNANCE_v2 6.1）。
- 正式治理启用仍应等待 T11 的 shadow 前置条件满足。

---

## 阶段六：工程优化（P3，预计 2 天）

### [P3] T13. ss7 模块拆分

- [x] **T13-1** 新建以下文件并从 `ss7_sqlite_news_overlay.py` 迁移对应代码：
  - `model_ridge.py` — PanelRidge, make_features, make_target, fit_with_cv
  - `backtest_engine.py` — run_backtest, equity curve, drawdown
  - `execution_model.py` — ExecConfig, execute_rebalance, lot_size
  - `news_overlay.py` — NewsConfig, load_news_items, F/A/U gating
  - `benchmark_regime.py` — benchmark_regime_state, _scale, _v2
  - `portfolio_optimizer.py` — solve_long_only_meanvar, project_to_simplex, sector_cap

- [x] **T13-2** `ss7_sqlite_news_overlay.py` 改为 facade 模式
  - 保留所有现有的公开函数签名
  - 函数体改为 `from xxx import yyy; return yyy(...)`
  - 所有现有 import 路径不 break

- [x] **T13-3** 全量测试回归
  - `python daily_run.py --config config.yaml` 运行通过
  - 回测结果与拆分前一致（Sharpe/MaxDD 差异 < 0.001）

进展备注（2026-04-04 QA 修复后更新）：
- **真正拆分完成**：三个模块均包含从 ss7 提取的完整实现代码，不再是 re-export。
  - `model_ridge.py`（120 行）：rsi, slope_log_price, make_features, make_target, PanelRidge
  - `execution_model.py`（140 行）：ExecConfig, lot_size, _round_to_lot, execute_rebalance
  - `portfolio_optimizer.py`（148 行）：project_to_simplex, shrink_cov, solve_long_only_meanvar, apply_sector_cap, apply_single_name_cap
- `ss7_sqlite_news_overlay.py` 现在从这三个模块 import，本地定义已删除，减少约 280 行重复代码。
- 25/25 测试全绿。

---

### [P3] T14. 文档与 CLAUDE.md 更新

- [x] **T14-1** `CLAUDE.md` 新增场景八（双策略日常运行）、场景九（Sprint 独立运行）
- [x] **T14-2** `CLAUDE.md` 更新关键参数表，新增 Sprint 策略参数
- [x] **T14-3** `CLAUDE.md` 更新关键文件路径，新增 `kelly_sizer.py`, `sprint_signal.py`, `benchmark_regime.py`

进展备注（2026-04-04）：
- `CLAUDE.md` 已加入双策略运行说明、推荐命令以及 Sprint/Kelly/benchmark 关键文件索引。

---

## 长期/条件性任务

### [P3] L1. Harvest 策略激活（条件：总 NAV >= 200万 JPY 连续 5 日）

- [ ] **L1-1** `daily_run.py` 的 `resolve_phase()` 自动检测并激活 Harvest
- [ ] **L1-2** 首次激活时，按 70/30 比例分配资金到两个策略的虚拟账户
- [ ] **L1-3** 发送 `runtime_event: harvest_activated`，level=warning（重要事件）

### [P3] L2. 基本面因子激活（条件：数据覆盖率 > 60% 且 IC t-stat >= 1.5）

- [ ] **L2-1** `compute_ic.py` 中 `fundamental_pending` 层因子定期评估 IC
- [ ] **L2-2** 达标后自动晋升到 core，写入 `learning_audit`

### [P3] L3. Phase 3 Event 策略（条件：总 NAV >= 500万 JPY）

- [ ] **L3-1** 设计 Event 策略：财报发布日前后 3 天的短期交易
- [ ] **L3-2** 集成 `earnings_events` 表数据
- [ ] **L3-3** 独立 paper 验证 30 天

---

## 完成标准

### 阶段一完成标志
1. `config.yaml` 包含完整 `strategy_profiles` 节点
2. `python daily_run.py --config config.yaml` 不报错，且输出包含 `[sprint]` 前缀日志
3. `positions` 表中有 `strategy_id` 列
4. 同一天重跑 `daily_run.py` 不产生重复 paper 记录

### 阶段二完成标志
1. `factor_health_report` 输出包含 `tier` 列
2. `reports/ridge_cv_comparison.json` 存在且 CV alpha != 50（证明 CV 生效）
3. `sprint_signal.py` 单元测试全通过

### 阶段三完成标志
1. `1552.T` 出现在 `daily_prices` 表中
2. `benchmark_regime_scale_v2()` 单元测试全通过
3. Sprint 冷却期逻辑有测试覆盖

### 全部完成标志
1. Sprint paper trading 运行 30 天，无异常中断
2. 治理 scorecard 所有维度 >= 8.5
3. `daily_run.py` 向后兼容（删除 `strategy_profiles` 节点后仍能正常运行）

---

## 工期总结

| 阶段 | 优先级 | 预计工期 | 依赖 |
|------|--------|---------|------|
| 阶段一：基础设施 | P0 | 3 天 | 无 |
| 阶段二：因子研究 | P1 | 2 天 | T1, T2 |
| 阶段三：风控/Benchmark | P1 | 1.5 天 | T1, T3 |
| 阶段四：数据治理 | P2 | 1.5 天 | T2 |
| 阶段五：新闻接入 | P2 | 1 天 | T6 |
| 阶段六：工程优化 | P3 | 2 天 | 全部 |
| **总计** | | **~11 天** | |

P0+P1 完成（约 6.5 天）后 Sprint 策略即可开始 paper trading。

---

*最后更新：2026-04-04*
