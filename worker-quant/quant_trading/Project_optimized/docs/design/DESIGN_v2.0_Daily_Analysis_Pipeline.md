# Worker-Quant — 每日分析链路设计文档 v2.0

**作者**: PM + Quant Architect
**创建日期**: 2026-04-02
**状态**: APPROVED — 待实施
**对应任务清单**: `../tasks/TASKS_2026-04-02_Analysis_Pipeline_Rebuild.md`
**上承文档**: `DESIGN_v1.5_Intelligence_Augmented_Quant.md`

---

## 0. 设计背景与目标

v1.x 版本已建立了完整的模型骨架（29因子 + PanelRidge + MVO优化器），但日常运行的"分析报告"质量不符合实战需求：输出内容分散、新闻情报过于宽泛、因子更新逻辑不清晰、报告缺乏明确的操作结论。

**v2.0 目标**：将 worker-quant 的每日运行固化为两大类有序链路，使每个交易日产出一份"有数据支撑、有逻辑推理、有明确结论"的完整分析报告。

---

## 1. 系统总架构：两大链路

```
┌─────────────────────────────────────────────────────────────────┐
│  Category A: 动态因子更新链路 (Factor Refresh Pipeline)          │
│  触发时机: 收盘后 15:30 JST（Task Scheduler, daily_run.py）      │
│                                                                   │
│  A1. 精准新闻采集 & 情感分析                                      │
│      ↓                                                            │
│  A2. 行情数据库更新 & 基本面更新                                   │
│      ↓                                                            │
│  A3. Screener 选股器运行（含基本面叠加层）                         │
│      ↓                                                            │
│  A4. 因子 IC 更新（每周一次，非每日）                              │
└─────────────────────────────────────────────────────────────────┘
         ↓ 产出：更新后的 DB + 因子权重 + 新闻情感打分
┌─────────────────────────────────────────────────────────────────┐
│  Category B: 股票与策略分析链路 (Strategy Analysis Pipeline)     │
│  触发时机: 次日开盘前 08:00 JST，或用户随时手动触发               │
│                                                                   │
│  B1. 市场状态判断（Regime Detection）                             │
│      ↓                                                            │
│  B2. 持仓健康度扫描（PnL + 止损检查）                             │
│      ↓                                                            │
│  B3. 信号生成 & 新闻交叉验证                                       │
│      ↓                                                            │
│  B4. 策略推理 & 操作建议生成                                       │
│      ↓                                                            │
│  B5. 结构化报告输出（Markdown + JSON）                             │
└─────────────────────────────────────────────────────────────────┘
         ↓ 产出：每日分析报告 briefing_v2_latest.md/.json
```

---

## 2. Category A 详细设计

### A1. 精准新闻采集 & 情感分析

**设计原则：精准情报 > 广泛覆盖**

v1 "世界新闻"过于宽泛，大量不相关内容引入噪音。v2 聚焦以下四类有效情报：

| 情报类别 | 优先级 | 来源 | 对应影响 |
|---------|--------|------|---------|
| 日本宏观/BOJ 货币政策 | P0 | Reuters JP / NHK | 日经整体方向 |
| 美日贸易/关税动态 | P0 | AP News / Bloomberg | 制造业/出口股 |
| 标的公司公告/业绩修正 | P0 | TDnet / Yahoo Finance JP | 个股直接冲击 |
| 半导体/汽车/钢铁行业动态 | P1 | 行业 RSS 订阅 | 选股池行业标的 |
| 一般世界新闻 | 排除 | — | 相关性弱，不采集 |

**情感处理流程**：
1. 采集 → 去重 → 按标的代码打标签
2. 每条新闻输出：`sentiment_score (-1~1)` + `impact_category` + `summary_cn (50字)`
3. 标的无关新闻打 `market_wide` 标签，作为大盘情绪参考
4. 写入 DB `news_items` 表，供 ss7 新闻门控（F/A/U gate）消费

**现有代码**：`news_to_db.py`（已有骨架，需升级情报源精准度）

---

### A2. 行情数据库更新 & 基本面更新

**执行顺序**（逻辑更新，彻底免费化）：
```
db_update.py          → K线数据更新（yfinance）
update_fundamentals.py → 基本面快照更新（剥离 J-Quants，全面切换至纯 yfinance 免费数据源）
```

**注意**：此链路必须保证无任何外部收费 API 依赖，以确保自动化流水线每日稳定跑通。

---

### A3. Screener 选股器

**执行**：`screener.py`（含 FundamentalOverlayConfig）

**输出字段**（供 B 链路消费）：
- `tech_score`：纯技术因子得分
- `fundamental_score (0.0~1.0)`：基本面调权系数
- `score_adjusted`：最终综合得分
- `fundamental_note`：降权理由说明（如"季度EPS亏损-8%"）

**关键设计原则**（不可更改）：
- 评分降权而非硬否决（binary veto 已废弃）
- 硬否决仅当：营业利润率 < -15% **且** OCF 为负（真正经营危机）
- 净利润不作为硬门槛（易被并购摊销/重组扭曲）

---

### A4. 因子 IC 更新

**重要约束：每周更新一次，不是每日。**

理由：IC 统计显著性需要 20-30 个观测周期；每日更新会导致权重过拟合，信号漂移加剧。

执行条件：
- `paper_days >= 30`（当前未达标，处于准备阶段）
- 每周一收盘后执行一次 `compute_ic.py`
- 更新 `factor_registry` 表中的 `ic_mean / icir / weight`

当前生产信号模式：`ridge`（待 IC 验证达标后考虑切换 `shadow_hybrid_ic`）

---

## 3. Category B 详细设计

### B1. 市场状态判断（Regime Detection）

**必须前置，不可跳过。** 市场状态决定整体仓位激进度。

```python
# 判断逻辑（基于日经ETF 1570.T）
regime = {
    "trend":    "UP" | "DOWN" | "SIDEWAYS",
    "vol_level": "LOW" | "NORMAL" | "HIGH",   # 采用近期真实日内振幅：过去5日 (ATR / Close) 与 过去60日均值 对比，若 > 1.5 倍即为 HIGH vol
    "bias":     "BULLISH" | "BEARISH" | "NEUTRAL"
}
```

**对策略的影响**：
| Regime | 仓位激进度 | 挂单距离 |
|--------|-----------|---------|
| UP + LOW vol | 正常 | 正常限价 |
| SIDEWAYS | 保守 | 更保守限价（patient档） |
| DOWN | 防御 | 不新开仓，检查止损 |
| HIGH vol（任意趋势）| 减半 | 扩大限价距离 |

---

### B2. 持仓健康度扫描

**数据来源**：`positions` / `fills` / `orders` 表 + 实时报价

每只持仓输出：
```
{
  "symbol": "9432.T",
  "cost_price": 156.7,
  "current_price": 158.2,
  "pnl_pct": +0.96%,
  "stop_loss_price": 147.3,   # cost × (1 - ATR动态止损%)
  "stop_triggered": false,
  "holding_days": 3,
  "signal_current": "Overweight",
  "action_hint": "HOLD"
}
```

**止损触发条件**（来自 config.yaml，不可随意修改）：
- ATR 止损：6%（动态，vol_mult=6.0）
- 组合半仓线：回撤 12%
- 组合全平线：回撤 18%

---

### B3. 信号生成 & 新闻交叉验证

**信号生成**：运行 `quant_briefing.py --mode market`，从 `briefing_latest.json` 读取信号

**新闻交叉验证（关键新增步骤）**：

```
对每个候选买入信号：
  IF 该标的有 P0/P1 级负面新闻（过去48h）:
    → 降低置信度，输出风险提示
    → 不自动取消信号，由用户最终判断
  IF 该行业有系统性负面新闻（关税/BOJ加息）:
    → 整体候选池保守化处理
```

这是"模型信号"与"新闻情报"的融合层，避免纯量化盲区（如 5401.T 并购摊销案例）。

---

### B4. 策略推理 & 操作建议生成

**推理框架（标准化，每日一致）**：

```
1. [市场背景]  今日大盘 Regime 判断 → 整体操作基调
2. [持仓状况]  各持仓 PnL / 止损距离 → HOLD/STOP/REDUCE
3. [候选信号]  Screener Top 5 + 新闻交叉验证结果
4. [操作指令]  每只标的明确的 BUY/HOLD/SELL/WATCH 结论
               附带：建议挂单价 / 挂单数量 / 理由
5. [风险提示]  本日特殊注意事项（异常量能/重大新闻/临近财报）
```

**输出格式约束**：
- 每条操作指令必须有**量化依据**（信号分值、PnL数据、新闻来源）
- 禁止输出无依据的"感觉性"建议
- 操作指令字段需符合 nexus JSON schema（供 Discord → nexus 链路解析）

---

### B5. 结构化报告输出

**输出文件**：
- `reports/briefing_v2_latest.md`（人类可读，Markdown 格式）
- `reports/briefing_v2_latest.json`（机器可读，nexus 解析用）

**报告 Markdown 固定结构**：
```markdown
# 每日量化简报 [日期] [时段]

## 一、市场状态
（大盘 Regime + 日经/TOPIX 涨跌幅 + 情绪判断）

## 二、今日有效情报
（精准新闻摘要，按影响等级排列，P0优先）

## 三、持仓健康度
（每只持仓：成本/现价/浮盈亏/止损距离/建议动作）

## 四、候选信号 Top 5
（代码 / 调整分 / 基本面注记 / 新闻交叉验证结论）

## 五、今日操作指令
（明确的 BUY/HOLD/SELL/WATCH，含挂单价和数量）

## 六、风险提示
（本日特殊注意事项）
```

**JSON schema（nexus 解析字段）**：
```json
{
  "date": "2026-04-02",
  "regime": {"trend": "UP", "bias": "BULLISH"},
  "positions": [...],
  "candidates": [...],
  "orders": [
    {"symbol": "9432.T", "action": "BUY", "price": 157.0, "qty": 1000, "reason": "..."}
  ],
  "risk_alerts": [...]
}
```

---

## 4. 关键设计决策记录

| 决策 | 结论 | 理由 |
|------|------|------|
| 因子 IC 更新频率 | 每周，非每日 | 每日更新导致过拟合，统计功效不足 |
| 新闻范围 | 精准4类，排除泛世界新闻 | 降低噪音，提升情报有效性 |
| 基本面硬否决条件 | 双重条件（利润率<-15% 且 OCF<0）| 单一EPS亏损可能是会计项目，非经营危机 |
| 报告触发方式 | A链路: 收盘后自动；B链路: 开盘前手动或自动 | A需时效性；B需要当日市场开盘状态 |
| 操作指令格式 | 结构化 JSON | 供 nexus → Discord 链路透传给用户 |

---

## 5. 与现有代码的对应关系

| v2.0 步骤 | 现有文件 | 状态 |
|----------|---------|------|
| A1 新闻采集 | `news_to_db.py` | 需升级情报源精准度 |
| A2 数据更新 | `db_update.py` + `update_fundamentals.py` | 可用，保持 |
| A3 选股器 | `screener.py` | 可用，保持 |
| A4 IC 更新 | `compute_ic.py` | 需改为每周触发 |
| B1 Regime 判断 | 缺失 | **需新增** |
| B2 持仓扫描 | `build_positions.py` + `live_trade_advisor.py` | 需整合 |
| B3 信号+新闻交叉 | 部分在 `ss7_sqlite_news_overlay.py` | 需新增交叉验证逻辑 |
| B4 策略推理 | `make_decision.py` | 需扩展推理框架 |
| B5 报告输出 | `quant_briefing.py` | 需升级输出格式至 v2 结构 |

---

*最后更新：2026-04-02*
## Runtime Addendum (2026-04-02)

This addendum records the current validated runtime state and should override older
aspirational notes when the two differ.

### Stable Default Production Mode

- Default unattended daily-run mode should be:
  - `fundamental.enabled: true`
  - `fundamental.source: "yfinance"`
  - `fail_closed: false`
  - `require_available_ts: false`
- `jquants_v2` remains an optional enhancement path, not the default production path.
- Reason: local validation showed that full-universe J-Quants refresh can hit repeated
  `429` rate limits and materially extend runtime.

### Current Operational Truth

- The daily pipeline is operational in the local environment.
- The current live outcome is still:
  - `recommendation = hold`
  - `orders = 0`
  - `paper_days = 0`
- Therefore the primary problem is not "pipeline cannot run". The primary problem is
  "pipeline runs, but governance evidence is not accumulating."

### Zero-Exposure Risk Before Benchmark Gating

- `Sharpe = 0` must not be explained only by the benchmark MA20/MA60 regime filter.
- The observed zero-exposure period started before the later benchmark risk-off trigger.
- This creates a first-class design risk in:
  `signal -> target weights -> order proposal -> min_trade / sizing filters`
- Current working hypothesis:
  `ridge` may be over-regularized, producing weak near-equal-weight output that is then
  rounded or filtered into zero effective exposure.

### Governance Risk Statement

- A governance deadlock risk exists because paper statistics are not accumulating.
- This should be treated as a diagnosis item, not yet as a proven claim that
  `paper_execute.py` itself is broken.
- Two cases must be separated explicitly:
  1. execution path runs, but no orders survive sizing and risk filters
  2. execution status, fills, or account state are not written back consistently

### Factor Quality Caution

- The factor set should currently be treated as partially validated.
- The next phase should prioritize:
  - factor quality cleanup
  - paper-loop diagnosis
  - signal-to-weight diagnosis
- The next phase should not prioritize expanding factor count.

## Risk-Control Addendum (2026-04-03)

This addendum should override any older wording that frames the project as mainly blocked
by research breadth or infrastructure survivability. The latest local evidence shows the
main gap is risk-control execution fidelity.

### Current validated truth

- the pipeline runs end to end on this machine
- governance now blocks promotion for the right reasons
- the strategy is still not production-ready because key controls are only partially
  enforced or enforced too late in the chain

### P0 design gaps

1. Stop-loss handling is not yet an independent execution guarantee.
   Current behavior identifies `stop_loss_tickers` and sets their target weights to zero,
   but this still depends on portfolio construction and rebalance flow. The intended design
   must include:
   - stop trigger detection
   - explicit flatten instruction
   - auditable exit reason in reports and artifacts
   - consistent behavior across backtest, paper, and live advice

2. Concentration control is not yet symmetric.
   Sector cap exists in the optimizer path, but `max_single_position_pct` is not yet
   treated as a hard post-optimization invariant. The design must require:
   - hard single-name cap after optimization
   - hard sector cap after optimization
   - recheck before order generation
   - QA failure when either cap is violated

3. Zero exposure still lacks a mandatory runtime response.
   Current governance can detect `latest_zero_exposure_days > threshold`, but design does
   not yet require the system to react. The intended runtime behavior should be:
   - alert when the zero-exposure window is breached
   - report the dominant cause
   - optionally auto-fallback to a safer baseline mode
   - prevent silent repeated `paper_no_orders` loops

### Production-safety priorities

Before new alpha expansion work, the project should satisfy all of the following:

- stop-loss exits are executable and traceable
- single-name and sector caps are both hard-checked
- prolonged zero exposure emits alert plus fallback behavior
- promotion remains blocked until at least one actionable mode exports non-zero weights
- reports distinguish benchmark de-risking, news suppression, lot-size suppression, and
  hard risk-control exits as separate causes

### Deferred until after P0 closure

These remain valid but are not the current lead items:

- expanding IC universe size
- continuous regime scoring
- richer news-overlay shaping
- optimizer sophistication upgrades
- additional factor families beyond the currently eligible set

## Governance Addendum (2026-04-03)

The project now uses an explicit score-based governance target:

- reference: `docs/governance/GOVERNANCE_v1_85_SCORECARD.md`
- policy: no dimension may be claimed as `8.5+` based on design intent alone
- rule: implementation readiness without validated runtime evidence is capped below `8.5`

This changes how roadmap work is prioritized:

1. design quality alone is no longer sufficient for a high score
2. every claimed maturity gain must be tied to reports, tests, or paper-trading evidence
3. the weakest dimensions now govern the release narrative:
   - risk management
   - execution quality
   - signal quality
   - operations maturity

### Immediate design consequence

The project should be managed against a two-part bar:

- `code-complete`
- `evidence-complete`

Until both are satisfied, the design may be improved, but the score should not be raised
to the user-facing `8.5+` target.

## Execution Addendum (2026-04-03, later)

The latest implementation adds a practical constraint that was previously only implicit:
the exported portfolio must be executable for the configured account size.

### New runtime rule

- `target_weights.csv` now exports the latest actionable non-zero target row rather than
  blindly exporting the latest history row when the latest row is zero because of
  benchmark de-risking on a non-rebalance day
- decision packaging applies a second-stage `lot-feasible concentration` pass after
  single-name and sector caps
- this pass is allowed to compress a fragmented low-weight basket into a smaller subset of
  affordable names so long as it remains inside:
  - `max_single_position_pct`
  - `max_sector_weight`
  - lot-size and min-trade constraints

### Why this matters

- for a `JPY 400,000` account, a mathematically valid 50-name weight vector can still be
  operationally non-tradable because most names cannot clear one JP board lot
- therefore execution quality must be measured on `lot-feasible target weights`, not only
  optimizer output

### Required artifacts

- `target_weights.csv`: executable target row used by decision packaging
- `target_weights_latest.csv`: the latest raw history row, even if zero
- `target_weights_last_nonzero.csv`: the last non-zero target row for audit
- decision snapshot `lot_feasibility` diagnostics
- `reports/runtime_events.jsonl`: machine-readable daily runtime events and fallback traces

## Governance Addendum (2026-04-03)

The promotion layer now distinguishes between:

- theoretical mode factor families
- governed production-eligible factors

For `shadow_hybrid_ic`, runtime weighting and promotion gating must use the governed
production subset once at least three factors satisfy:

- latest learning guard = `PASS`
- `n_observations >= 80`

This avoids penalizing the production signal for stale or ineligible factors that are
still tracked for research. Promotion also now supports a configured
`backtest_sharpe_tolerance` so boundary-case Sharpe estimates are handled explicitly in
code and governance output rather than by manual interpretation.
