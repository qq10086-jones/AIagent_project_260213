# Governance v2: Dual Strategy Operating Standard

**创建日期**: 2026-04-04
**状态**: DRAFT
**设计文档**: `../design/DESIGN_v3.0_Dual_Strategy_Architecture.md`
**任务清单**: `../tasks/TASKS_2026-04-04_Dual_Strategy_Implementation.md`
**上承文档**: `GOVERNANCE_v1_85_SCORECARD.md`（继续有效，本文档为扩展）

---

## 0. 治理范围

本文档定义双策略架构（Sprint / Harvest）的运营治理标准。v1 Scorecard 继续有效，
本文档新增双策略特有的治理维度，并更新 8.5+ 达标条件。

---

## 1. 策略隔离治理

### 1.1 资金隔离（硬性）

| 规则 | 描述 | 违规后果 |
|------|------|---------|
| ISO-1 | 每个策略的 positions / orders / fills / account_snapshots 必须通过 `strategy_id` 隔离 | 混合记录需立即回滚并修复 |
| ISO-2 | 策略间不共享持仓。同一只股票可分别被两个策略持有（不同 strategy_id） | 交叉查询需带 strategy_id 过滤 |
| ISO-3 | 总 NAV 计算必须汇总所有 strategy_id 的 positions + cash | 单策略 NAV 仅用于该策略内部决策 |

### 1.2 Phase 门控（硬性）

| 规则 | 描述 |
|------|------|
| PH-1 | Phase 升级为单向：Phase 1 -> Phase 2 -> Phase 3，不可回退 |
| PH-2 | Phase 2 激活条件：总 NAV >= 2,000,000 JPY，**连续 5 个交易日** |
| PH-3 | Phase 升级时自动触发 `runtime_event: phase_upgrade`（level=warning） |
| PH-4 | Harvest 策略在 Phase 1 期间 `enabled=false`，任何 Harvest 相关代码路径不得执行 |

### 1.3 资金再平衡（软性，可配置）

| 规则 | 默认值 | 可配置 |
|------|--------|--------|
| RB-1 | 月度再平衡：每月第一个交易日 | 是 |
| RB-2 | 偏离触发：某策略偏离目标比例 > 10pp | 是 |
| RB-3 | 单向转移：Sprint 盈利可转入 Harvest，反向不允许 | 否（硬性） |

---

## 2. Sprint 策略专属治理

### 2.1 Kelly 仓位约束（硬性）

| 规则 | 描述 | 阈值 |
|------|------|------|
| KL-1 | edge <= 0 时，仓位 = 0，不开仓 | edge = (p*b - q) / b |
| KL-2 | 样本不足时回退固定仓位 | min_samples = 30 笔 |
| KL-3 | 单只仓位上限 | max_single_position_pct = 0.50 |
| KL-4 | 单只仓位下限（低于此值不值得开仓） | min_position_pct = 0.05 |
| KL-5 | Kelly fraction 始终 <= 0.5（Half-Kelly） | kelly_fraction = 0.5 |

违规场景与应对：
- Kelly 计算得出的仓位 > 50%：硬 cap 到 50%，记录 `runtime_event: kelly_capped`
- Kelly edge 计算出错（NaN/Inf）：回退固定 10%，记录 `runtime_event: kelly_fallback`

### 2.2 冷却期机制（硬性）

| 规则 | 描述 |
|------|------|
| CD-1 | 连续 3 笔止损触发后，Sprint 暂停 5 个交易日 |
| CD-2 | 冷却期内 `suggested_weight = 0`，所有 Sprint 信号被屏蔽 |
| CD-3 | 冷却期事件写入 `runtime_events.jsonl`：`sprint_cooldown_activated` / `sprint_cooldown_expired` |
| CD-4 | 冷却期不影响 Harvest 策略（如已激活） |

### 2.3 Sprint 进出场纪律（硬性）

进场五要素（全部满足才开仓）：

| # | 条件 | 可配置 |
|---|------|--------|
| E1 | `benchmark_state != "off"` | 否 |
| E2 | `mom_consist` 截面排名前 20% | 是（百分位阈值） |
| E3 | `high52w > -0.10` | 是（阈值） |
| E4 | `vol_z > 0.5` | 是（阈值） |
| E5 | `kelly_edge > 0 且 suggested_weight > min_position_pct` | 否 |

出场五触发（任一即退出）：

| # | 条件 | 可覆盖 |
|---|------|--------|
| X1 | ATR 3x 止损 | **不可覆盖** |
| X2 | 持有 > holding_period_target 天且浮亏 | 是（天数） |
| X3 | vol_z 从 > 1.5 急降到 < -0.5 | 是（阈值） |
| X4 | `benchmark_state` 转 "off" | **不可覆盖** |
| X5 | 组合回撤 >= max_dd_half | **不可覆盖** |

**X1、X4、X5 为不可覆盖的硬止损，任何配置修改不得绕过。**

### 2.4 Sprint Paper 验证门控（硬性）

| 规则 | 描述 |
|------|------|
| SP-1 | Sprint 策略必须先通过 30 天 paper trading 才能投入真钱 |
| SP-2 | Paper 期间要求：win_rate >= 0.45，盈亏比 >= 1.3，MaxDD < 20% |
| SP-3 | Paper 验证由 `evaluate_promotion.py` 扩展，新增 `strategy_id=sprint` 路径 |
| SP-4 | Paper 验证不通过时，禁止切换到 live 模式 |

---

## 3. Harvest 策略专属治理

### 3.1 激活条件

| 条件 | 阈值 | 硬性/软性 |
|------|------|----------|
| 总 NAV | >= 2,000,000 JPY | 硬性 |
| 连续达标天数 | >= 5 个交易日 | 硬性 |
| Sprint paper 验证 | 已通过（SP-1~SP-4） | 软性（建议但不阻塞） |

### 3.2 因子治理

| 规则 | 描述 |
|------|------|
| HF-1 | 仅 Core 层因子参与 Harvest 生产权重 |
| HF-2 | Core 层准入条件：t-stat >= 1.5 且 n_obs >= 100 |
| HF-3 | Core 层降级条件：连续 60 个 rebalance 期 t-stat < 0.5 |
| HF-4 | Candidate 层不参与 Harvest 生产权重，仅 shadow 跟踪 |
| HF-5 | 因子晋升/降级事件记录到 `learning_audit` 表 |
| HF-6 | 每 30 天审查一次因子表现（`factor_promotion_rules.review_frequency_days`） |

### 3.3 Harvest Paper 验证门控

沿用 v1 晋升条件（`evaluate_promotion.py` 现有逻辑）：
- Sharpe >= 1.5（tolerance 0.01）
- paper_days >= 30
- IC t-stat >= 1.5
- 相对 baseline 提升 >= 0.1

---

## 4. Benchmark Regime 增强治理

### 4.1 分策略 Regime 参数

| 参数 | Sprint | Harvest | 理由 |
|------|--------|---------|------|
| off_scale | 0.0 | 0.45 | Sprint 宁可踏空；Harvest 保留底仓 |
| caution_scale | 0.40 | 0.70 | Sprint 更保守 |
| use_vix_confirmation | true | false | Sprint 用 VIX 防误判 |
| vix_off_threshold | 30.0 | — | VI > 30 才确认 risk-off |

### 4.2 VIX 二次确认规则

| 场景 | MA 信号 | VIX 状态 | Sprint 最终判定 | Harvest 最终判定 |
|------|---------|----------|----------------|-----------------|
| A | off | VI >= 30 | off（确认） | off |
| B | off | VI < 30 | **caution**（降级） | off |
| C | off | VI 数据缺失 | off（保守 fallback） | off |
| D | caution | 任意 | caution | caution |
| E | on | 任意 | on | on |

### 4.3 Regime 诊断输出

每次 daily_run 输出 `reports/regime_diagnosis.json`：

```json
{
  "asof": "2026-04-04",
  "benchmark_ticker": "1321.T",
  "px_b": 37500.0,
  "fast_ma": 37200.0,
  "slow_ma": 38100.0,
  "ma_signal": "off",
  "vix_ticker": "1552.T",
  "vix_value": 25.3,
  "sprint_final_state": "caution",
  "sprint_scale": 0.40,
  "harvest_final_state": "off",
  "harvest_scale": 0.45,
  "diagnosis": "MA says off but VIX low -> Sprint downgraded to caution"
}
```

---

## 5. 数据治理增强

### 5.1 单一真实来源

| 规则 | 描述 |
|------|------|
| DG-1 | SQLite (`japan_market.db`) 为唯一持久化存储 |
| DG-2 | `paper_trading_account.json` 降级为只读诊断快照，不参与决策 |
| DG-3 | 所有交易表查询必须带 `strategy_id` 过滤（除汇总 NAV） |

### 5.2 Paper 幂等性

| 规则 | 描述 |
|------|------|
| DG-4 | 同一天同一 strategy_id 不重复执行 paper |
| DG-5 | `check_idempotent(conn, asof, strategy_id)` 在 decision 步骤前强制调用 |
| DG-6 | 重跑 daily_run 不产生重复记录（幂等保证） |

### 5.3 执行质量监控

| 指标 | 阈值 | 响应 |
|------|------|------|
| fill_validation_rate | < 90% | warning（成交价超出当日 high-low 范围） |
| avg_slippage_bps | > 20 | warning（滑点异常） |
| implementation_shortfall_bps | > 50 | error（执行偏差过大，检查价格源） |

---

## 6. 新闻接入治理

### 6.1 分阶段门控

| 阶段 | 启用条件 | 回退条件 |
|------|---------|---------|
| Phase 1 Shadow | 手动 `news.shadow_only: true` | 任何时候可关闭 |
| Phase 2 Sprint Gating | Shadow 运行 30 天 + 新闻 IC 与次日收益正相关 | Sprint Sharpe 下降 > 0.3 时自动回退到 Shadow |
| Phase 3 Harvest Factor | Sprint Gating 运行 30 天 + IC t-stat >= 1.5 | t-stat 跌破 1.0 时降级 |

### 6.2 新闻门控安全限制

| 规则 | 描述 |
|------|------|
| NW-1 | 新闻只能降低仓位或阻止开仓，永远不能作为加仓理由 |
| NW-2 | 新闻数据缺失时，gate = 1.0（等同无新闻，不影响信号） |
| NW-3 | 新闻极端评分（|score| >= 0.99）需记录触发 prompt（沿用 Governance Rule 4.1） |

---

## 7. 更新后的 8.5+ Scorecard

### 7.1 各维度达标条件（v2 更新）

#### 1. 架构设计 (目标 9.0)

v1 条件继续有效，新增：
- [ ] 双策略 config 层可独立配置且互不干扰
- [ ] strategy_id 隔离在所有交易表中一致
- [ ] ss7 已拆分为独立模块（facade 兼容）
- [ ] `daily_run.py` 向后兼容（删除 strategy_profiles 不 break）

#### 2. 因子工程 (目标 8.5)

v1 条件继续有效，新增：
- [ ] 因子分层体系（core/candidate/excluded）已在 config 和代码中生效
- [ ] Ridge alpha 通过 Time-Series CV 选择（非硬编码）
- [ ] Sprint 使用独立因子集（不依赖 Ridge）
- [ ] 因子晋升/降级有审计记录

#### 3. 风控管理 (目标 9.5)

v1 条件继续有效，新增：
- [ ] Kelly 仓位约束（KL-1~KL-5）已实现且有测试
- [ ] Sprint 冷却期机制（CD-1~CD-4）已实现
- [ ] VIX 二次确认已实现（T7-1~T7-4）
- [ ] Sprint 硬止损（X1/X4/X5）不可被配置覆盖

#### 4. 执行质量 (目标 8.5)

v1 条件继续有效，新增：
- [ ] Paper 幂等性保证（DG-4~DG-6）
- [ ] 执行质量监控自动运行并输出
- [ ] Sprint 和 Harvest 的 paper 记录独立且不混淆

#### 5. 信号质量 (目标 8.5)

v1 条件继续有效，新增：
- [ ] Sprint 信号生成器独立于 Ridge 模型
- [ ] Sprint 进出场规则全部有测试覆盖
- [ ] Sprint paper 验证门控（SP-1~SP-4）已实现

#### 6. 数据质量 (目标 8.5)

v1 条件继续有效，新增：
- [ ] SQLite 为唯一真实来源（DG-1~DG-3）
- [ ] `paper_trading_account.json` 不再被任何决策代码读取
- [ ] 1552.T (VIX) 数据可用且纳入日更新

#### 7. 运营成熟度 (目标 8.5)

v1 条件继续有效，新增：
- [ ] Phase 升级事件有 runtime_event 记录
- [ ] Sprint 冷却期有 runtime_event 记录
- [ ] `regime_diagnosis.json` 每日输出
- [ ] 操作员可从报告中区分两个策略的状态

#### 8. 整体生产就绪度 (目标 8.5)

- [ ] Sprint paper trading 运行 >= 30 天
- [ ] Sprint paper 验证通过（win_rate >= 0.45, PnL ratio >= 1.3, MaxDD < 20%）
- [ ] 所有 P0 治理规则已实现且有测试
- [ ] 无未解决的硬性治理违规

### 7.2 当前 vs 目标对照

| 维度 | v1 当前分 | v2 目标 | 关键差距 |
|------|----------|---------|---------|
| 架构设计 | 8.0 -> 8.5 (v1 addendum) | 9.0 | ss7 拆分 + strategy_id 隔离 |
| 因子工程 | 5.0 -> 8.5 (v1 addendum) | 8.5 | 因子分层 + CV alpha |
| 风控管理 | 9.0 | 9.5 | Kelly + VIX + 冷却期 |
| 执行质量 | 4.0 -> 8.5 (v1 addendum) | 8.5 | 幂等性 + 质量监控 |
| 信号质量 | 4.0 -> 8.5 (partial) | 8.5 | Sprint 独立信号链 |
| 数据质量 | 7.0 | 8.5 | 单一 SQLite 来源 |
| 运营成熟度 | 7.0 | 8.5 | regime 诊断 + 事件日志 |
| 整体就绪度 | 4.0 | 8.5 | Sprint 30天 paper 验证 |

---

## 8. 治理审查节奏

| 审查类型 | 频率 | 内容 |
|---------|------|------|
| 日审 | 每个交易日 | regime_diagnosis.json + runtime_events 检查 |
| 周审 | 每周一 | 因子 IC 更新 + Kelly 参数检查 + paper 盈亏汇总 |
| 月审 | 每月第一个交易日 | 资金再平衡 + 因子层级审查 + 新闻 Phase 评估 |
| 季审 | 每季度 | 策略整体 Sharpe/MaxDD 评估 + Phase 升级检查 |

---

## 9. 紧急回退流程

### 9.1 Sprint 紧急停机

触发条件（任一）：
- Sprint 连续 5 个交易日亏损且累计亏损 > 10%
- Kelly edge 持续为负 > 10 个交易日
- 代码 bug 导致非预期交易

操作：
1. `config.yaml` 中 `strategy_profiles.sprint.enabled: false`
2. 当天不再执行 Sprint decision/paper
3. 现有 Sprint 持仓通过 `live_trade_advisor.py` 手动平仓
4. 记录 `runtime_event: sprint_emergency_halt`

### 9.2 Phase 降级（极端情况）

Phase 门控为单向，但极端情况下操作员可手动降级：
1. 修改 `config.yaml` 中 `harvest.enabled: false`
2. Harvest 持仓需手动平仓后转移资金到 Sprint
3. 记录 `runtime_event: phase_manual_downgrade`（level=error）
4. **此操作需要在 governance 审查中记录原因**

---

*最后更新：2026-04-04*
