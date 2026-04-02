# TASKS: 2026-04-02 每日分析链路重构

参考设计文档：
`docs/design/DESIGN_v2.0_Daily_Analysis_Pipeline.md`

优先级说明：[P0] 阻塞性 / [P1] 高优先 / [P2] 中等 / [P3] 低优先/长期

---

## Category A：动态因子更新链路

### [P1] A1. 新闻采集模块升级

- [x] **A1-1** 梳理 `news_to_db.py` 现有情报源（Kabutan/Google/GDELT），已评估覆盖率
- [x] **A1-2** 新增两类精准情报源：`fetch_google_news_boj()`（日银/金利/BOJ）+ `fetch_google_news_trade()`（関税/貿易摩擦）；`--sources` 默认值更新为 `kabutan,google,boj,trade,gdelt`
- [x] **A1-3** 每条新闻输出格式标准化完成（sentiment_score / impact_category / related_tickers / summary_cn）
- [x] **A1-4** 新增 `news_items` 表（含两个索引），`write_to_db` 每条 item 同步写入；impact_category 推导：BOJ/TRADE 源直接设置，有 symbol→COMPANY，gdelt→MARKET_WIDE，其余→SECTOR

---

### [P0] A2. 行情与基本面数据源彻底免费化（替换 J-Quants）

- [x] **A2-1** 修改 `update_fundamentals.py`，彻底移除 `jquantsapi` 依赖及其相关逻辑
- [x] **A2-2** 使用 `yfinance` 的季度财务报表 (quarterly_income_stmt/balance_sheet/cashflow) + info 替代基本面数据抓取；`--source` 默认值改为 `yfinance`
- [x] **A2-3** 清理 `daily_run.bat` 中的 `JQUANTS_API_KEY` 明文，改为注释说明（已弃用）

---

### [P2] A4. 因子 IC 更新频率修正

- [x] **A4-1** `daily_run.py` 第6步加入周一门控（`weekday != 0` 时跳过），支持 `learning.ic_update_force=true` 强制触发
- [x] **A4-2** 门控日志打印当日 weekday 及跳过原因
- [x] **A4-3** 添加 paper_days 防护：查询 `signals` 表实际天数，不足30天时打印警告（compute_ic.py 仍运行但权重覆盖由其自身拦截）

---

## Category B：股票与策略分析链路

### [P0] B1. 新增市场状态判断模块（Regime Detection）

**此步骤缺失，是当前报告质量差的核心原因之一**

- [x] **B1-1** 在 `quant_briefing.py` 中新增 `_detect_market_regime()` 函数（70日K线，SMA5/SMA20趋势 + ATR相对振幅波动率）
- [x] **B1-2** Regime 结果写入 `briefing_v2_latest.json` 的 `regime` 字段（所有模式均执行）
- [x] **B1-3** Regime 的 `action_bias` 字段在 MD 报告一、市场状态节中明确显示

---

### [P1] B2. 持仓健康度扫描整合

- [x] **B2-1** 新增 `_enrich_positions_with_stop_loss()` 函数：拉取20日K线计算ATR14，动态止损=cost×(1-max(ATR%×6.0, 6%))，上限20%；`build_briefing` 中自动调用
- [x] **B2-2** 持仓对象新增字段：`stop_loss_price / stop_loss_pct / stop_triggered / stop_note / pnl_pct`
- [x] **B2-3** v2报告"三、持仓健康度"表格中止损触发显示 `⚠️止损` 图标；risk_alerts 同步包含止损警告

---

### [P0] B3. 新增新闻交叉验证逻辑

**这是防止量化盲区（如 5401.T 案例）的关键步骤**

- [x] **B3-1** 新增 `_cross_validate_with_news(db_path, symbol)` 函数；对每个候选信号查询 `news_items` 表过去48h新闻；news_items 表不存在时静默降级为 NONE（不影响主流程）
- [x] **B3-2** 交叉验证结果以 `news_validation: {news_risk, news_notes}` 字段写入每个候选信号对象
- [x] **B3-3** 报告"候选信号 Top 5"中明确标注 🔴公司负面 / 🟡宏观 / — 三级图标

---

### [P1] B5. 报告输出格式升级至 v2 结构

- [x] **B5-1** 新增 `--output-version {v1,v2}` 参数（默认 v2），同时输出 v1（兼容）和 v2
- [x] **B5-2** 新增 `write_report_v2()` 函数，输出固定6节 Markdown 结构
- [x] **B5-3** JSON 输出符合 nexus schema（regime / positions / candidates[:5] / orders / risk_alerts）
- [x] **B5-4** 新文件 `briefing_v2_latest.md` / `briefing_v2_latest.json`；旧版 `briefing_latest.*` 保留

---

### [P1] CLAUDE.md 更新

- [x] **DOC-1** 场景对照表新增场景六（每日完整分析）、场景七（仅更新数据）
- [x] **DOC-2** 关键文件路径表新增 `briefing_v2_latest.md/.json`
- [x] **DOC-3** 新增完整的"六、新增场景"章节，含触发时机说明、报告六节解读顺序

---

## 长期/条件性任务

### [P3] 待 paper_days >= 30 后

- [ ] 运行 `compute_ic.py --shadow` 用真实 IC 替换 `shadow_hybrid_ic` 占位权重
- [ ] 评估 Sharpe / IC t-stat 是否达到晋升条件（Sharpe≥1.5, IC t-stat≥1.5）
- [ ] 若达标，在 `config.yaml` 中将 `signal_mode` 从 `ridge` 切换为 `shadow_hybrid_ic`

### [P3] 持仓挂单跟进（来自 2026-03-26 操作记录）

- [ ] 9432.T 预挂单 ¥156.7 — 考虑上调至 ¥157.0（差 0.1 JPY 未成交，连续多日）
- [ ] 4005.T 预挂单 ¥492.8 — 建议撤销（基本面降权 + 当前价远离限价 10 JPY）

---

### [P2] A5. 基本面数据源升级：J-Quants + yfinance 混合方案（待用户实现）

**背景**：2026-04-02 测试确认 J-Quants 免费版（API Key via `ClientV2(api_key=...)`）可获取高质量季报数据，
yfinance 日本股票季报数据稀疏（quarterly_income_stmt 多数只有 EPS），而 J-Quants `get_fin_summary` 直接提供 OP/Sales/CFO/BPS。

**推荐分工**：
- **价格/技术因子** → 继续 yfinance（实时，无截止限制）
- **基本面季报因子** → J-Quants `get_fin_summary(code)`（OP、Sales、CFO、BPS 直接可用）

**实现要点**：
- `update_fundamentals.py` 新增 `import_jquants_v2(db_path)` 函数，基于 `ClientV2(api_key=os.environ["JQUANTS_API_KEY"])`
- 速率限制 5次/分 → 每只股票间隔 12s，30只约需 6 分钟
- 字段映射：`OP→operating_income`，`Sales→revenue`，`CFO→operating_cf`，`NP→net_income`，`BPS→book_value_per_share`
- `CurPerType` 用于识别季度（1Q/2Q/3Q/FY），`CurPerEn` 作为 `fiscal_period_end`
- `--source` 参数新增选项 `jquants_v2`，`daily_run.py` 中对应切换
- `JQUANTS_API_KEY` 继续通过环境变量注入（勿硬编码）

**验证方式**：跑完后检查 `fundamental_snapshots` 中 `operating_income` 是否非 NULL，与 J-Quants 网站季报对比。

- [x] 由用户（及 AI 协助）最终实现强壮版 `import_jquants_v2()`
- [x] 加入 `429 Rate Limit` 指数退避重试保护（Exponential Backoff）
- [x] 引入基于 14天缓存 的断点试跳过机制，支持分批增量拉取大盘
- [x] 在 `daily_run.bat` / 本地环境变量中设置 `JQUANTS_API_KEY`
- [x] `config.yaml` 中 `fundamental.source` 改为 `jquants_v2`

---

## 完成标准

当以下条件全部满足时，本 milestone 视为完成：

1. `python quant_briefing.py --mode full` 输出包含6节完整结构的 `briefing_v2_latest.md`
2. `briefing_v2_latest.json` 包含 `regime` / `positions` / `orders` / `risk_alerts` 四个顶层字段
3. 每个候选买入信号都有 `news_validation` 字段（即使无相关新闻也显示 `"news_risk": "NONE"`）
4. 止损触发时，报告第六节自动置顶显示相关警告
5. 因子 IC 更新频率已改为每周，有防护不覆盖未验证权重

---

*最后更新：2026-04-02（全部原 P0/P1/P2 任务已完成；新增 A5 J-Quants 混合方案遗留待用户实现）*
## Runtime Follow-ups Added After Local QA (2026-04-02)

These tasks were added after validating the current local runtime behavior. They are
intended to close the gap between "pipeline runs" and "governance can accumulate real
evidence".

### [P0] R1. Diagnose paper-loop evidence gap

- [x] **R1-1** Verify whether `paper_execute.py` is actually writing fills/account state correctly or whether the system is simply producing zero executable orders.
- [x] **R1-2** Trace `decision_runs.status`, `paper_executed`, fills insertion, and account snapshot updates for the most recent runs and document the exact break point.
- [x] **R1-3** Produce a short QA note that distinguishes "execution path broken" from "execution path runs but no orders survive filters".

Status note:
`paper_execute.py` runs successfully on this machine. The current evidence shows "execution path runs, but current runs have zero orders / zero fills", not a hard execution-path break. `evaluate_promotion.py` and `paper_execute.py` were updated so governance stats now reflect snapshots and status transitions more accurately.

### [P0] R2. Diagnose zero-exposure path

- [x] **R2-1** Trace `signal -> target_weights -> orders_proposal` and identify where non-zero alpha turns into zero effective exposure.
- [x] **R2-2** Quantify the impact of `min_trade`, lot sizing, cash sizing, and post-processing filters on order suppression.
- [x] **R2-3** Confirm whether the benchmark regime filter is the primary blocker or merely a later secondary blocker.

Status note:
A formal `zero_exposure_report.json/md` is now produced by `ss7_sqlite_news_overlay.py`. Current finding: the latest direct blocker is `benchmark_risk_off`, while the last non-zero target remains `2026-03-12`. Order suppression is currently happening before `min_trade` filtering because the latest exported target weights are already zero.

### [P1] R3. Compare primary signal modes in paper governance

- [x] **R3-1** Run paper-governance comparison for `ridge` and `shadow_ic` in parallel rather than relying on backtest-only comparison.
- [x] **R3-2** Compare weight concentration, non-zero order count, and surviving executable orders across modes.
- [ ] **R3-3** Do not switch the production default mode until paper-layer evidence exists for both candidates.

Status note:
The comparison report now includes actionable status, last non-zero asof, and shared latest zero-exposure cause. Current result: all four compared modes are zero now, so there is no evidence-based reason to switch away from `ridge` yet.

### [P1] R4. Clean up factor registry participation

- [x] **R4-1** Identify factors with insufficient observations or failing statistical guards and mark them as non-production candidates.
- [ ] **R4-2** Prevent low-confidence factors from diluting the production weight calculation without explicit override.
- [x] **R4-3** Publish a compact factor-quality summary listing observation count, IC mean, t-stat, and production eligibility.

Status note:
`factor_health_report.py` now emits `factor_registry_cleanup_candidates.csv` and cleanup reports. Current production-eligible factor set contains only `mom_consist`; `ret20`, `rsi14`, `slope60`, and `vol_adj_mom20` are marked as exclusion candidates.
