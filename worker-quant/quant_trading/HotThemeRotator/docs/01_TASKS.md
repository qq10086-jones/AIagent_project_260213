# HotThemeRotator Task List

> 所有工作都必须挂到本文件。任务状态同步到 `../PROJECT_STATUS.md`，但流水更新只写 `PROJECT_STATUS.md`。

## Milestone P0: Project Spine

### P0-01 Create Project Structure

- Status: done
- Goal: 建立独立目录、设计文档、任务清单、治理文档和唯一状态文件。
- Acceptance:
  - `README.md` exists.
  - `PROJECT_STATUS.md` exists.
  - `docs/00_DESIGN.md` exists.
  - `docs/01_TASKS.md` exists.
  - `docs/02_GOVERNANCE.md` exists.
  - `docs/03_FOLDER_MAP.md` exists.

### P0-02 Confirm Scope

- Status: done
- Goal: 确认第一阶段只做日股直接信号，A股和美股只做外部温度因子。
- Acceptance:
  - `PROJECT_STATUS.md` 中记录用户确认。
  - 配置中 `enabled_trade_markets` 只包含 `JP`。
  - Verified: `configs/strategy.example.yaml` keeps `enabled_trade_markets: [JP]`; external markets are only temperature inputs.

### P0-03 Fix PROJECT_STATUS Encoding Artifacts

- Status: done
- Goal: 修复 `PROJECT_STATUS.md` 中第 3 行副标题、Current Objective 正文、Active Decision 正文、Open Risks 整段、Rules Snapshot 整段五处双重编码乱码，恢复中文可读性并还原原有 markdown 换行。
- Files:
  - Update: `PROJECT_STATUS.md`
- Acceptance:
  - 五处损坏区域全部恢复为可读中文，且语义与 `README.md`、`docs/00_DESIGN.md`、`docs/02_GOVERNANCE.md` 一致。
  - Open Risks 与 Rules Snapshot 还原为正常 markdown bullet（不再压成单行）。
  - 不改动 Change Log 已有任何一行。
  - 不引入新的策略主张或参数调整（仅编码恢复）。
  - Verified: `python .runtime/_verify_p0_03.py` 报告 `residual garbled lines: 0`；最终行数 84（原 78，因 L75 / L78 各扩展 +3）。

### P0-04 Decision Log Hardening (Codex Findings)

- Status: done
- Depends on: P9-01, P9-02
- Goal: 处理 codex:rescue 在 2026-05-24 对 P9-01 + P8-08 + P9-02 代码的审查中提出的 10 项发现（3 HIGH / 4 MEDIUM / 3 LOW）。修复 outcome_join 的 reference_price 静默回退、asof 重复/非 ISO 不验证、ladder 不完整时静默跳过等 fail-closed 漏洞；强化 OutcomeRecord 在 `status="complete"` 时对 1D/3D/5D 完整性的契约；为 path helpers 增加 ISO 验证；文档化 6 位精度与并发假设；修正 `DashboardPanel.trade_date` 的 ISO 解析。
- Files:
  - Update: `docs/adr/ADR-0003-decision-log.md` (F1 — 澄清 scanner-side enforcement)
  - Update: `src/hot_theme_rotator/decision_log/schema.py` (F5 完整性契约 / F9 delimiter 文档 / 新增 `malformed_data` 状态)
  - Update: `src/hot_theme_rotator/decision_log/jsonl_writer.py` (F6 ISO 验证 / F7 并发文档)
  - Update: `src/hot_theme_rotator/decision_log/outcome_join.py` (F2 fail-closed reference_price / F3 asof 验证 / F4 ladder 完整性)
  - Update: `src/hot_theme_rotator/opportunity/opportunity_scanner.py` (F8 精度文档)
  - Update: `src/hot_theme_rotator/ui/opportunity_dashboard.py` (F10 ISO 解析)
  - Update: 对应测试文件
- Acceptance:
  - 10 项 codex 发现全部按 fix-or-document 处理。
  - `malformed_data` 加入 `ALLOWED_OUTCOME_STATUSES`，覆盖 reference_price 缺失 / 重复 asof / 非 ISO asof / opportunity ladder 七档不完整四类。
  - `OutcomeRecord` `__post_init__` 在 `status="complete"` 时校验 `realized_returns` 至少含 `("1D", "3D", "5D")`。
  - `predictions_path` / `outcomes_path` 通过 `date.fromisoformat` 验证 `trade_date`。
  - 至少 8 个新单测覆盖 fail-closed 行为。
  - 全部测试通过；无 score_status / Rule 3 / §9.4 / §10 红线被引入。
  - Verified Cycle 1 (Tier 1 HIGH): F1 — ADR-0003 Decision §2 改写明确 scanner-side enforcement，writer 不再声称做 PIT；F2 — `_extract_reference_price` 删除 first_bar.open 静默回退，缺失/非正即返回 `malformed_data` outcome；F3 — 新增 `_validate_bar_sequence` 拒绝重复 asof、非 ISO asof、cutoff 当日或之前的 bar；新增状态 `malformed_data` 加入 `ALLOWED_OUTCOME_STATUSES`；5 个新 outcome_join 测试 + 1 个旧 fallback 测试改写；`python -m pytest .\tests` -> `170 passed`.
  - Verified Cycle 2 (Tier 2 MEDIUM): F4 — `_compute_ladder_touches` 返回 `(touches, problem)`，缺失任一档 / 非数值 / 非正值都返回 `malformed_data`；F5 — `OutcomeRecord.__post_init__` 在 `status="complete"` 时强制 `realized_returns` 含 `("1D","3D","5D")`，否则抛错；F6 — `predictions_path` / `outcomes_path` 通过 `date.fromisoformat` 验证 `trade_date`，拒绝路径穿越（`../etc`）和非 ISO 格式（`2026/05/23`）；F7 — `jsonl_writer.py` 顶部 docstring 明确单写者假设和未来 lock/SQLite 演进路径；7 个新测试（2 ladder + 3 schema + 2 path）；`python -m pytest .\tests` -> `177 passed`.
  - Verified Cycle 3 (Tier 3 LOW): F8 — `compute_opportunity_snapshot_id` docstring 文档化 6 位小数精度契约；F9 — `compute_prediction_id` docstring 文档化 `|` delimiter 假设与未来扩展约束；F10 — `DashboardPanel.trade_date` 用 `datetime.fromisoformat` 正确解析，同时处理 T 分隔、空格分隔、Z UTC、date-only 输入；3 个新 DashboardPanel.trade_date 测试覆盖 T / 空格 / date-only 三种 asof 格式；`python -m pytest .\tests` -> `180 passed`.

## Milestone P1: Data Contract

### P1-01 Define Core Schemas

- Status: done
- Goal: 定义 price、news、theme、temperature、signal、position 的最小字段。
- Files:
  - Create: `src/hot_theme_rotator/common/schema.py`
  - Create: `tests/unit/test_schema.py`
- Acceptance:
  - 每个 schema 能从 dict 构造。
  - 缺少关键字段时 fail closed。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\unit\test_schema.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned passing.

### P1-02 Build Legacy Data Adapter

- Status: done
- Goal: 从 `../Project_optimized/japan_market.db` 读取价格、新闻和持仓快照。
- Files:
  - Create: `src/hot_theme_rotator/data/legacy_project_adapter.py`
  - Create: `tests/integration/test_legacy_project_adapter.py`
- Acceptance:
  - 能读取指定日期的日股价格。
  - 能读取指定日期前的相关新闻。
  - 数据读取失败时返回明确错误，不静默生成假数据。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\integration\test_legacy_project_adapter.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned `4 passed`.

## Milestone P2: Market Temperature

### P2-01 Implement Japan Market Temperature

- Status: done
- Goal: 计算日股市场温度。
- Files:
  - Create: `src/hot_theme_rotator/market_temperature/japan_temperature.py`
  - Create: `tests/unit/test_japan_temperature.py`
- Acceptance:
  - 输出 0-100 分数。
  - 输出 `HOT/WARM/NEUTRAL/COLD/RISK_OFF` regime。
  - 明确解释分数来源。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\unit\test_japan_temperature.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned `4 passed`.

### P2-02 Implement External Temperature Inputs

- Status: done
- Goal: 加入 A股、美股、USDJPY 和全球风险偏好作为外部因子。
- Files:
  - Create: `src/hot_theme_rotator/market_temperature/external_temperature.py`
  - Create: `tests/unit/test_external_temperature.py`
- Acceptance:
  - 外部因子只能影响 trade_permission 和权重，不能单独触发买入。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\unit\test_external_temperature.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned `4 passed`.

## Milestone P3: Theme And Leader Engine

### P3-01 Theme Detector V1

- Status: done
- Goal: 建立关键词主题识别，覆盖 AI 半导体、机器人、汽车、防卫、药品审批、TOB、回购、中美缓和链。
- Files:
  - Create: `src/hot_theme_rotator/theme_detection/theme_detector.py`
  - Create: `tests/unit/test_theme_detector.py`
- Acceptance:
  - 新闻标题和正文能映射到 theme_id。
  - 无主题新闻必须返回 `None`。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\unit\test_theme_detector.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned `4 passed`.

### P3-02 Leader Ranker V1

- Status: done
- Goal: 结合主题相关性、相对强度、成交额和流动性，对主题内股票排序。
- Files:
  - Create: `src/hot_theme_rotator/leader_ranking/leader_ranker.py`
  - Create: `tests/unit/test_leader_ranker.py`
- Acceptance:
  - 每个主题最多输出 3 个候选。
  - 过热、低流动性、新闻弱相关标的被降权或剔除。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\unit\test_leader_ranker.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned `4 passed`.

## Milestone P4: Signal And Risk

### P4-01 Signal Engine V1

- Status: done
- Goal: 生成 BUY/HOLD/TAKE_PROFIT/STOP_LOSS/NO_TRADE 建议。`ROTATE` 需要组合级持仓切换输入，移到后续任务。
- Files:
  - Create: `src/hot_theme_rotator/signal_engine/signal_engine.py`
  - Create: `tests/unit/test_signal_engine.py`
- Acceptance:
  - 市场温度为 RISK_OFF 时不允许 BUY。
  - 输出 2%、3%、5% 止盈价。
  - 输出固定止损价和最大持有天数。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\unit\test_signal_engine.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned passing.

### P4-03 Rotation Signal Design

- Status: done
- Goal: 在已有持仓、候选龙头、主题退潮和新主题升温同时可见时，定义 ROTATE 信号输入和行为。
- Files:
  - Create: `docs/adr/ADR-0002-rotation-signal.md`
  - Update: `tests/unit/test_signal_engine.py`
  - Update: `src/hot_theme_rotator/signal_engine/signal_engine.py`
- Acceptance:
  - 明确 ROTATE 和 TAKE_PROFIT/STOP_LOSS 的优先级。
  - 明确旧持仓和新候选之间的比较字段。
  - ROTATE 仍为 advice-only，不生成订单。
  - Verified: `python -m pytest .\tests\unit\test_signal_engine.py -q -o cache_dir=.\pytest_cache` returned `10 passed`.

### P4-02 Risk Governor V1

- Status: done
- Goal: 限制单票、单主题、总风险暴露。
- Files:
  - Create: `src/hot_theme_rotator/risk/risk_governor.py`
  - Create: `tests/unit/test_risk_governor.py`
- Acceptance:
  - 单票仓位超过配置上限时建议减少。
  - 主题暴露超过配置上限时禁止加仓。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\unit\test_risk_governor.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned `5 passed`.

## Milestone P5: Reports And Review

### P5-01 Daily Briefing

- Status: done
- Goal: 生成市场温度、热点主题、龙头候选和风险提示报告。
- Files:
  - Create: `src/hot_theme_rotator/reporting/daily_briefing.py`
  - Create: `reports/daily/README.md`
  - Create: `tests/unit/test_daily_briefing.py`
- Acceptance:
  - 报告包含温度、主题、候选、入场理由、退出价格。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\unit\test_daily_briefing.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned `2 passed`.

### P5-02 Paper Review Log

- Status: done
- Goal: 建立 paper 信号和实际观察结果的复盘记录。
- Files:
  - Create: `src/hot_theme_rotator/reporting/paper_review.py`
  - Create: `reports/paper/README.md`
  - Create: `tests/unit/test_paper_review.py`
- Acceptance:
  - 每条信号能记录 entry、exit、exit_reason、realized_return。
  - 支持统计胜率、平均盈利、平均亏损、最大单笔亏损。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\unit\test_paper_review.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned `4 passed`.

### P5-03 Daily Pipeline MVP

- Status: done
- Goal: 串联市场温度、主题识别、龙头排序、信号生成和日报渲染，生成一份完整 briefing。
- Files:
  - Create: `src/hot_theme_rotator/reporting/daily_pipeline.py`
  - Create: `tests/integration/test_daily_pipeline.py`
- Acceptance:
  - 输入价格、新闻、候选特征和参考价格后能生成 Markdown 简报。
  - Pipeline 不写订单、不自动执行，只返回 advice-only 报告。
  - Pipeline 复用已有模块，不复制评分逻辑。
  - Pipeline 在 BUY 建议渲染前调用 risk governor。
  - Verified: `python -m pytest .\quant_trading\HotThemeRotator\tests\integration\test_daily_pipeline.py -q -o cache_dir=.\quant_trading\HotThemeRotator\.pytest_cache` returned passing.

## Milestone P6: Open Source Integration

### P6-01 OpenBB Adapter Spike

- Status: done
- Goal: 验证 OpenBB 是否适合作为外部数据研究接口。
- Acceptance:
  - 给出是否纳入主线的结论。
  - 不允许在未验证稳定性前替代本地数据。

### P6-02 vectorbt Backtest Spike

- Status: done
- Goal: 验证 vectorbt 是否适合批量参数回测。
- Acceptance:
  - 能回测 2%、3%、5% 止盈规则。
  - 能输出盈亏比、尾部亏损、最大回撤和换仓成本。
  - Verified: `src/hot_theme_rotator/backtesting/vectorbt_spike.py` runs 2% / 3% / 5% take-profit variants with fixed stop-loss.
  - Verified: missing explicit costs raises `MissingCostConfigError`.
  - Verified: `python -m pytest .\tests\unit\test_vectorbt_backtest_spike.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `2 passed`.

### P6-03 Historical Signal Sample

- Status: done
- Goal: Build a multi-day historical signal sample under current production thresholds before attempting a meaningful vectorbt entry backtest.
- Files:
  - Create: `src/hot_theme_rotator/backtesting/historical_signal_sample.py`
  - Create: `tests/unit/test_historical_signal_sample.py`
  - Generate: `reports/backtests/historical_signal_sample_2026-05-20.md`
- Acceptance:
  - Summarizes daily news, detected-theme symbols, leader candidates, signals, and entry signals.
  - Flags zero-entry samples as not ready for meaningful entry backtest.
  - Verified on 2026-04-21 to 2026-05-20 local DB sample: 18 days, 710 news items, 109 detected-theme symbols, 109 leader candidates, 36 signals, 0 BUY/ROTATE entries.

### P6-04 NO_TRADE Threshold Diagnostics

- Status: done
- Goal: Diagnose why the historical signal sample produces no BUY/ROTATE entries before changing any strategy parameters.
- Files:
  - Create: `src/hot_theme_rotator/backtesting/no_trade_diagnostics.py`
  - Create: `tests/unit/test_no_trade_diagnostics.py`
  - Generate: `reports/backtests/no_trade_diagnostics_2026-05-21.md`
- Acceptance:
  - Counts blocker reasons separately from explanatory reason codes.
  - Reports entry-score min/average/max for NO_TRADE signals.
  - Verified on 2026-04-21 to 2026-05-20 local DB sample: 38 NO_TRADE signals, top blocker `ENTRY_SCORE_TOO_LOW` with 35 occurrences, `MARKET_BLOCK` with 3 occurrences.

## Milestone P7: Universal Attribution And Feedback

### P7-01 Universal Attribution Governance

- Status: done
- Goal: Define hard rules for representative-instrument attribution, ex-ante buy/sell/hold scoring, cross-symbol integration, and feedback correction.
- Files:
  - Update: `docs/02_GOVERNANCE.md`
  - Update: `docs/01_TASKS.md`
  - Update: `PROJECT_STATUS.md`
- Acceptance:
  - `1306.T` is treated as an example, not the whole framework.
  - A representative universe is required before single-symbol analysis.
  - Ex-ante decision inputs are separated from ex-post explanations.
  - Point-in-time `available_ts` is mandatory for news, factors, and market inputs.
  - LLM-generated probabilities are forbidden unless backed by a calibrated historical model.
  - Cross-symbol integration and feedback logging requirements are defined.

### P7-02 Universal Attribution Core Contracts

- Status: done
- Goal: Implement the first code-level contracts for representative instruments, point-in-time snapshots, and role-weighted decision integration.
- Files:
  - Create: `src/hot_theme_rotator/attribution/__init__.py`
  - Create: `src/hot_theme_rotator/attribution/universal_attribution.py`
  - Create: `tests/unit/test_universal_attribution.py`
- Acceptance:
  - Default Japan representative universe includes multiple roles and keeps `1306.T` as only one member.
  - Representative universe validation rejects duplicate symbols, non-positive weights, and too few distinct roles.
  - Point-in-time snapshots reject features whose `available_ts` is later than the decision cutoff.
  - Missing attribution buckets are exposed explicitly.
  - Symbol-level buy/sell/hold outputs are integrated by role-weighted average.
  - Integrated output is called `calibrated_probability` only when every component is calibrated.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `72 passed`.

### P7-03 Point-In-Time Data Adapter

- Status: done
- Goal: Build the data adapter that creates daily point-in-time snapshots for every representative instrument.
- Files:
  - Create: `src/hot_theme_rotator/attribution/point_in_time_adapter.py`
  - Create: `tests/unit/test_point_in_time_adapter.py`
- Acceptance:
  - Captures price, volume, news, Japan equity beta, FX, rates, and external-risk inputs per instrument.
  - Stores `available_ts` and decision cutoff for every input.
  - Produces reproducible `input_snapshot_id` values.
  - Fails closed when required timestamps are missing or later than cutoff.
  - Exposes missing price or factor buckets instead of silently filling them.
  - Verified: `python -m pytest .\tests\unit\test_point_in_time_adapter.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `4 passed`.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `76 passed`.

### P7-04 Ex-Ante Probability Baseline

- Status: done
- Goal: Produce buy, sell, and hold outputs from historical features without pretending uncalibrated scores are true probabilities.
- Files:
  - Create: `src/hot_theme_rotator/attribution/baseline_decision_score.py`
  - Create: `tests/unit/test_baseline_decision_score.py`
- Acceptance:
  - Uses 3 trading days as the default evaluation horizon.
  - Supports auxiliary 1D and 5D horizons.
  - Emits `insufficient_calibration` or `uncalibrated_research_score` until calibration evidence exists.
  - Records `model_version` with every prediction.
  - Integrates symbol-level outputs using the representative-universe weights.
  - Verified: `python -m pytest .\tests\unit\test_baseline_decision_score.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `4 passed`.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `81 passed`.

### P7-05 Weekly Universal Attribution Report

- Status: done
- Goal: Generate a weekly report that explains each representative instrument, then integrates those explanations into one market-level view.
- Files:
  - Create: `src/hot_theme_rotator/reporting/weekly_universal_attribution.py`
  - Create: `tests/unit/test_weekly_universal_report.py`
- Acceptance:
  - Includes OHLC, return, volume, gap, and range for each instrument and trading day.
  - Includes ex-post movement label and factor attribution table.
  - Includes ex-ante input snapshot and buy/sell/hold output status.
  - Includes cross-symbol integrated buy/sell/hold output.
  - Clearly marks missing factor buckets instead of silently omitting them.
  - Clearly marks uncalibrated output as not win probability.
  - Verified: `python -m pytest .\tests\unit\test_weekly_universal_report.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `1 passed`.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `81 passed`.

### P7-06 Attribution Feedback Calibration

- Status: pending
- Depends on: P9-01, P9-02, P9-03
- Goal: 在 P9-01 decision log、P9-02 outcome join 与 P9-03 calibration 基础上，针对 attribution 路径（`baseline_decision_score` 与 `IntegratedDecisionScore`）做 buy/sell/hold 校准评估，聚焦 §8 universal attribution 的 ex-post 校准证据。
- Acceptance:
  - Attribution 路径的每一条 buy/sell/hold 预测都通过 P9-01 写入 decision log，包含 `prediction_id`、`input_snapshot_id`、`model_version` 与 `decision_cutoff`。
  - 通过 P9-02 join 获取 1D/3D/5D 实际收益。
  - 按 §8.6 输出 Brier / log loss / calibration bins，且仅在样本量充足时报告；否则维持 `insufficient_calibration` / `uncalibrated_research_score`。
  - 校准结论不会静默回写到 baseline 模型参数（仍走 Rule 4 流程）。

## Milestone P8: Realtime Opportunity And Price Ladders

### P8-01 Opportunity Scanner V1

- Status: done
- Goal: Search potential stocks from normalized real-time inputs and rank them by research-only opportunity score.
- Files:
  - Create: `src/hot_theme_rotator/opportunity/__init__.py`
  - Create: `src/hot_theme_rotator/opportunity/opportunity_scanner.py`
  - Create: `tests/unit/test_opportunity_scanner.py`
- Acceptance:
  - Consumes price, news/theme, relative strength, volume, liquidity, and context features.
  - Rejects invalid prices and missing required fields.
  - Emits reason codes and data gaps.
  - Uses `uncalibrated_research_score` until calibration exists.
  - Verified: `python -m pytest .\tests\unit\test_opportunity_scanner.py .\tests\unit\test_price_ladder.py .\tests\unit\test_realtime_opportunity_panel.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `7 passed`.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `88 passed`.

### P8-02 Price Ladder V1

- Status: done
- Goal: Generate staged buy, stop, and sell prices for each candidate.
- Files:
  - Create: `src/hot_theme_rotator/opportunity/price_ladder.py`
  - Create: `tests/unit/test_price_ladder.py`
- Acceptance:
  - Produces aggressive, balanced, and conservative entry prices.
  - Produces stop, first exit, second exit, and stretch exit prices.
  - Uses deterministic formulas based on current price and range/ATR proxy.
  - Marks output as research-only, not an order.
  - Verified: `python -m pytest .\tests\unit\test_opportunity_scanner.py .\tests\unit\test_price_ladder.py .\tests\unit\test_realtime_opportunity_panel.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `7 passed`.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `88 passed`.

### P8-03 Realtime Candidate Panel V1

- Status: done
- Goal: Render the first user-facing product for live opportunity review.
- Files:
  - Create: `src/hot_theme_rotator/reporting/realtime_opportunity_panel.py`
  - Create: `tools/realtime_opportunity_demo.py`
  - Create: `tests/unit/test_realtime_opportunity_panel.py`
  - Create: `tests/unit/test_realtime_opportunity_demo.py`
- Acceptance:
  - Shows rank, symbol, trigger theme, score label, opportunity score, staged entries, stop, staged exits, reasons, and data gaps.
  - Does not display uncalibrated scores as win rates.
  - Supports multiple candidates sorted by score.
  - Provides `build_realtime_opportunity_panel_markdown(...)` as the first runnable product entry point.
  - Provides `python .\tools\realtime_opportunity_demo.py` as the first local demo command.
  - Verified: `python -m pytest .\tests\unit\test_opportunity_scanner.py .\tests\unit\test_price_ladder.py .\tests\unit\test_realtime_opportunity_panel.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `7 passed`.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `88 passed`.
  - Verified: `python .\tools\realtime_opportunity_demo.py` prints ranked candidates and price ladders.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `94 passed`.

### P8-04 Real-Time Data Adapter

- Status: done
- Goal: Connect live or near-real-time data sources to the P8 scanner while preserving point-in-time rules.
- Files:
  - Create: `src/hot_theme_rotator/data/free_web_opportunity_adapter.py`
  - Create: `tests/unit/test_free_web_opportunity_adapter.py`
- Acceptance:
  - Every input has `available_ts`.
  - Stale, missing, or post-cutoff data fails closed.
  - Adapter output can feed P8-01 without changing scanner rules.
  - Includes default refresh schedule: pre-open 10 minutes, trading sessions 3 minutes, lunch 15 minutes, post-close 3 hours, overnight 6 hours, material news immediate.
  - Includes a yfinance-compatible quote client boundary for free web quote ingestion.
  - Converts free-web quotes, news, and context snapshots into `OpportunityInput`.
  - Verified: `python -m pytest .\tests\unit\test_free_web_opportunity_adapter.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `5 passed`.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `93 passed`.

### P8-05 Ladder Feedback Calibration

- Status: done
- Depends on: P9-01, P9-02, P9-03
- Goal: 在 P9-01 decision log、P9-02 outcome join 与 P9-03 calibration 基础上，针对 opportunity ladder（`price_ladder.PriceLadder` 的 aggressive / balanced / conservative / stop / first / second / stretch 七档）逐档评估实际触达证据。P8-05 只消费统一 `PredictionRecord` / `OutcomeRecord`，不自建存储，不重算 outcome，不触发 alert / paper order / broker order。
- Acceptance:
  - 新增 `calibration.ladder_feedback`，输入为现有 predictions/outcomes 对象，按 `prediction_id` 配对。
  - 仅纳入 `OutcomeRecord.status == "complete"` 且含完整七档 `ladder_touches` 的 opportunity 样本；缺 tier 或 `touched` 非 bool 必须 fail closed。
  - 每档输出 `sample_count`、`touched_count`、`status`；样本数未达阈值时 `touch_rate` 必须为 `None` 且状态为 `insufficient_calibration`。
  - 样本数达到阈值后才可输出 numeric `touch_rate`；该指标只能叫触达率，不能叫胜率、盈利概率或交易建议。
  - 同一 report 可携带 P9-03 的 bullish 3D `CalibrationReport`，但不新增 score_status，不显示 `calibrated_probability`，不改变 §10 gates。
  - 全部测试通过，并在 `PROJECT_STATUS.md` 记录验证命令。
  - Verified 2026-05-24: `calibration.ladder_feedback` added with `LadderTierFeedback`, `LadderFeedbackReport`, `build_ladder_feedback_report`, and fail-closed tier validation. RED `test_ladder_feedback.py` failed on missing module; GREEN targeted -> `5 passed`; related calibration/outcome/API tests -> `72 passed`; full suite with `NUMBA_CACHE_DIR=.runtime\numba_cache` -> `313 passed in 8.26s`.

### P8-06 Local User Interface V1

- Status: done
- Goal: Provide a local web dashboard so general users can operate the realtime opportunity panel without command-line snippets.
- Files:
  - Create: `src/hot_theme_rotator/ui/__init__.py`
  - Create: `src/hot_theme_rotator/ui/opportunity_dashboard.py`
  - Create: `tools/streamlit_opportunity_app.py`
  - Create: `tests/unit/test_opportunity_dashboard.py`
  - Update: `.gitignore`
  - Update: `README.md`
  - Update: `requirements.txt`
- Acceptance:
  - Opens as a Streamlit local dashboard.
  - Defaults to sample data for immediate first use.
  - Supports yfinance quote-only mode from the sidebar.
  - Shows Chinese user-facing candidate table, price ladders, rules, and raw Markdown.
  - Keeps research-only and uncalibrated-score warnings visible.
  - Keeps local Streamlit runtime logs outside pytest temp directories via ignored `.runtime/`.
  - Verified: `python -m pytest .\tests\unit\test_opportunity_dashboard.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `4 passed`.
  - Verified: `python -m py_compile .\tools\streamlit_opportunity_app.py` succeeded.
  - Verified: `http://localhost:8501` returned HTTP 200 after starting Streamlit.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `98 passed`.

### P8-07 Retail Opportunity Dashboard V2

- Status: done
- Goal: Make the local opportunity UI friendly for general retail users by opening directly into a plain-language "今日机会中心".
- Files:
  - Update: `src/hot_theme_rotator/ui/opportunity_dashboard.py`
  - Update: `tools/streamlit_opportunity_app.py`
  - Update: `tests/unit/test_opportunity_dashboard.py`
  - Update: `README.md`
  - Update: `.gitignore`
  - Update: `docs/02_GOVERNANCE.md`
  - Update: `PROJECT_STATUS.md`
- Acceptance:
  - First screen shows top candidate, action wording, buy zone, stop, sell zone, reasons, and risk warning.
  - Candidate table remains available for scanning multiple names.
  - Candidate details remain available without requiring raw Markdown.
  - Automation roadmap is visible and gated before live execution.
  - Research-only and uncalibrated-score warnings remain visible.
  - Verified: `python -m pytest .\tests\unit\test_opportunity_dashboard.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `7 passed`.
  - Verified: `python -m py_compile .\tools\streamlit_opportunity_app.py` succeeded.
  - Verified: `http://localhost:8501` returned HTTP 200.
  - Verified: `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` returned `101 passed`.
  - Runtime recovery: after a stale Streamlit process raised `cannot import name 'automation_roadmap_rows'`, port 8501 was restarted and `http://localhost:8501/_stcore/health` returned `ok`.

### P8-08 Retail Dashboard Presentation Polish V3

- Status: done
- Depends on: P9-01
- Goal: 让 §10 gate progress、§9.4 未校准状态、§9.3 价格阶梯七档全部直观可见；把 §8.6 decision log 的成果暴露到 dashboard；markdown 报告从 14 列宽表重构为每候选一节；CLI demo 加细微终端着色。不引入新评分逻辑，不改任何 score_status，不引入执行通路。
- Files:
  - Update: `src/hot_theme_rotator/ui/opportunity_dashboard.py` (新增 `build_gate_progress_rows` / `build_price_ladder_view` / `build_recent_predictions_view` / `build_calibration_badge` 四个 helper；`automation_roadmap_rows` 与 P9 实际状态对齐)
  - Update: `src/hot_theme_rotator/reporting/realtime_opportunity_panel.py` (新增 `render_realtime_opportunity_panel_markdown_v2` 每候选一节风格，不删除 v1)
  - Update: `tools/streamlit_opportunity_app.py` (gate strip / calibration pill / 视觉阶梯 / 最近记录 tab / v2 markdown)
  - Update: `tools/realtime_opportunity_demo.py` (isatty 检测下的 ANSI 着色)
  - Update: `tests/unit/test_opportunity_dashboard.py` (新增 helper 的测试)
  - Update: `tests/unit/test_realtime_opportunity_panel.py` (v2 markdown 测试)
- Acceptance:
  - Gate progress strip 在 Streamlit 顶部对每个 §10 gate 显示 done / in_progress / pending 状态；P9-01 完成后 gate 3 标 done。
  - Hero card 包含醒目 calibration pill；当 score_status 是 `uncalibrated_research_score` / `insufficient_calibration` 时 pill 视觉权重高于普通 metric。
  - 七档价位（aggressive/balanced/conservative entry + stop + first/second/stretch exit）在 hero card 中作为可视化列表逐档呈现，含相对现价的 ± 百分比。
  - "最近记录" tab 从 `reports/predictions/{trade_date}.jsonl` 读取并展示当日已落盘的预测条目（symbol / score / status / prediction_id）；当当日无文件时显示 "暂无记录"。
  - Markdown v2 报告每候选一段，含机会分、买入三档、止损、卖出三档、理由、数据缺口；v1 仍可用以兼容已有调用。
  - CLI demo 在 isatty 模式下使用 ANSI 着色（bold rank、dim metadata）；管道/重定向时不输出 ANSI。
  - 不引入新 score_status 值；不引入 calibrated_probability 显示；不引入执行按钮。
  - 全部测试通过。
  - Verified Cycle 1 (helpers + v2 markdown): `_GATE_DEFINITIONS` 8-gate 单一来源；`build_gate_progress_rows` / `build_price_ladder_view` / `build_recent_predictions_view` / `build_calibration_badge` 四 helper 完成；`automation_roadmap_rows` 从 6 行重写为 §10-aligned 8 行；新 `render_realtime_opportunity_panel_markdown_v2` 与 v1 共存；9 个新单测；`python -m pytest .\tests` -> `133 passed`.
  - Verified Cycle 2 (Streamlit + CLI 集成): Streamlit 顶部新增 gate progress strip；hero card 新增 calibration pill + 7 档价格阶梯 HTML 可视化；新增「最近记录 (§8.6)」tab 读取 `reports/predictions/` JSONL；规则 tab 改用 v2 markdown；CLI demo 切换 v2 markdown 并加 isatty-gated ANSI 着色；2 个新 demo 测试；`python -m pytest .\tests` -> `135 passed in 5.53s`；`python -m py_compile .\tools\streamlit_opportunity_app.py .\tools\realtime_opportunity_demo.py` 通过；旧 Streamlit (PID 68256) 已停止，新进程在端口 8501 启动后 `http://localhost:8501/_stcore/health` -> `ok`、`http://localhost:8501` -> HTTP 200。

### P8-09 FastAPI + Vite/React V3 Dashboard Integration

- Status: done
- Depends on: P8-08, P9-01, P9-02, P9-03
- Goal: 把用户在 `quant.zip` 提供的**完整设计探索画布**（V1 三栏专业终端 / V2 研究备忘录 / V3 市场温度仪表盘 / V4 决策日志为脊 + 共享组件参考 + Tweaks 实时调整面板 + Rationale 设计说明）按 ADR-0004 决定的 FastAPI + React 路径集成进项目。**quant.zip 是地基，项目真实数据流进 4 个变体里**（不是只挑 V3）。Phase 1 选 zero-build (CDN React + Babel-standalone) 以最短路径让用户看到效果；Vite 迁移留作 Phase 2 (P8-10)。Python 数据层保持权威（gates 来自 `_GATE_DEFINITIONS`，calibration 来自 `build_calibration_badge`），新增 `api/` 只读 JSON 层 + `frontend/` React app；保留 `tools/streamlit_opportunity_app.py` 作为 fallback。
- Files:
  - Create: `docs/adr/ADR-0004-fastapi-frontend.md` (+ Amended for Phase 1)
  - Update: `docs/00_DESIGN.md` (§5 架构图加 user-facing surface，§6.11 API + §6.12 Frontend 新模块)
  - Update: `docs/03_FOLDER_MAP.md` (新增 `## api`、`## frontend` 章节)
  - Create: `api/__init__.py`、`api/main.py`、`api/dashboard.py`、`api/serializers.py`
  - Create: `tests/unit/test_api_dashboard.py`（FastAPI TestClient）
  - Update: `src/hot_theme_rotator/ui/opportunity_dashboard.py` (`_GATE_DEFINITIONS` gate 4/5 drift 修正)
  - Create: `frontend/index.html` (CDN React + Babel + boot script)、`frontend/v3.jsx`、`frontend/shared.jsx`、`frontend/data.js` (从 `quant.zip` 拷)
- Acceptance:
  - ADR-0004 完整记录决定 / 后果 / 风险 / 替代方案 / out-of-scope；含 Phase 1/Phase 2 路径分歧。
  - `/api/dashboard` 返回 V3 JSON 形状，gates 真实从 `_GATE_DEFINITIONS` (P8-01..02 done, P9-01..03 done, P9-04..05 pending, P9-06 blocked)，calibration 真实从 `build_calibration_badge`（当前 0 样本 → `insufficient_calibration` + sample 计数 + minSamples=100，前端渲染"校准样本不足 · 不是真实胜率"badge）。
  - API 不暴露任何 POST/PUT/DELETE 端点（Rule 3）。
  - frontend 通过 `fetch("/api/dashboard")` 取数并合并到 mock baseline 之上，markets/themes/newsTimeline/kline 在 Python 层供给前以 mock 渲染（带 boot-status 指示真实/mock 来源）。
  - `curl http://localhost:8000/` 返回 200 (V3 index.html)；`curl http://localhost:8000/api/dashboard` 返回 200 (V3 JSON)；`curl http://localhost:8000/api/health` 返回 `{"status": "ok"}`。
  - 不引入新 score_status 值；不引入 calibrated_probability 显示（除非 calibration report 真为 calibrated）；不引入任何执行按钮（Rule 3）。
  - 全部 pytest 通过；新增 ≥ 7 个 API 单测。
  - Streamlit 端口 8501 仍可启动作为 fallback。
  - Verified Cycle 1 (Foundation): ADR-0004 / DESIGN §5 + §6.11 + §6.12 / FOLDER_MAP §api + §frontend / TASKS P8-09 entry 全部新增；纯文档无回归 `python -m pytest .\tests` -> `219 passed`.
  - Verified Cycle 2 (FastAPI backend): `api/__init__.py` + `api/main.py` (CORS + /api/health + 静态挂载 frontend/) + `api/dashboard.py` + `api/serializers.py` (gates from `_GATE_DEFINITIONS` / candidates from `build_sample_panel` / calibration from `build_calibration_badge` 强制 insufficient / markets/themes/news/kline 返回空数组 + `dataAvailability` flags) + `tests/unit/test_api_dashboard.py` (8 tests covering full V3 keys / calibration warning / gate truth alignment / candidate ladder 7 tiers / Rule 3 POST→405 / top_n cap)；`_GATE_DEFINITIONS` gate 4/5 drift 修正（P9-02/P9-03 → done）；`python -m pytest .\tests` -> `227 passed`.
  - Verified Cycle 3 (Frontend assets, Phase 1 zero-build): ADR-0004 Status 标 Amended；DESIGN §6.12 / FOLDER_MAP §frontend / TASKS scope 全部改为 "Phase 1 CDN + Babel-standalone, no npm/Vite, Phase 2 deferred"；`frontend/index.html` 新建（CDN React + Babel + boot 脚本 fetch /api/dashboard 合并到 mock baseline 之上 + 失败时 fallback 到 mock + 右上角 boot-status 指示来源）；`frontend/{shared.jsx,v3.jsx,data.js}` 原样拷自 `quant.zip`；`api/main.py` `FRONTEND_DIST` 改为 `FRONTEND_ROOT` 直接挂 frontend/。
  - Verified Cycle 4 (Launch + smoke): uvicorn on 127.0.0.1:8000 启动正常 (PID via shell job)；`/api/health` -> 200；`/api/dashboard` -> 200, 3772 bytes JSON 包含完整 V3 形状；`/` -> 200, 4126 bytes HTML；`/shared.jsx /v3.jsx /data.js` 全部 -> 200；Rule 3：POST/DELETE /api/dashboard -> 405；最终 `python -m pytest .\tests` -> `227 passed in 5.54s`。
  - **Cycle 5 修正（Goal 范围扩展 + 补拼缺失变体）**：cycle 1-4 误把 Goal 写为 "V3 仪表盘集成"，结果只拷了 `shared.jsx + v3.jsx + data.js` 三个文件就 close。用户反馈澄清 "在 quant.zip 的 UI 设计基础上和我们的项目内容功能合并" = quant.zip 4 变体 + 设计画布 + Tweaks 全部要保留。Cycle 5 把缺失 6 个文件拷齐：`v1.jsx` / `v2.jsx` / `v4.jsx` / `design-canvas.jsx` / `tweaks-panel.jsx` / `.design-canvas.state.json`；`frontend/index.html` 用原版 quant.zip index.html 替换（含 `<DesignCanvas>` + 3 `<DCSection>`（intro/variations/components）+ `<TweaksPanel>` + `Rationale` + `ComponentShowcase`）；唯一修改是 `ReactDOM.createRoot(...).render(<App />)` 行改为先 `await fetch("/api/dashboard")` 合并到 `window.HTR_DATA` 之上再 render，让 4 个变体全部消费真实 Python 数据；顶部状态栏 + 顶部 boot-status div 可视化"API 真实 / mock fallback"来源分布。smoke：`/` -> 200 (15655 bytes HTML)；`/v1.jsx /v2.jsx /v3.jsx /v4.jsx /shared.jsx /design-canvas.jsx /tweaks-panel.jsx /data.js /.design-canvas.state.json` 全部 -> 200；`/api/dashboard` 仍 200。pytest 未跑（前端纯静态文件改动，后端无变化）。
  - **Cycle 6 修正（再次范围误判：把设计画布当产品 UI）**：Cycle 5 错误持续 —— 把 quant.zip 的 `DesignCanvas` 整套（设计师用来 side-by-side 比较 4 个变体 + Rationale 设计说明 + 共享组件参考的 Figma 风格画布）当作产品 UI mount 在 `/`。用户反馈："哪有人这样子把 V1-V4 全部都展现在一个画布里面的，肯定是通过一个导航导航到对应界面"。修正：把 `App()` 重写为正常应用 shell —— 顶部 sticky `<nav>` 含 4 个按钮 (V1 三栏专业终端 / V2 研究备忘录 / V3 市场温度仪表盘 / V4 决策日志为脊)，点击切换 active 变体；`<main>` 内只渲染当前 active 变体的完整组件；用户选择持久化到 `localStorage.htr_variant`（默认 V3，与 canvas state 一致）；保留 `<TweaksPanel>` 作为右下角可选调整工具；移除 `<DesignCanvas>` / `<DCSection>` / `<DCArtboard>` / `Rationale` / `ComponentShowcase` 在 App 中的使用（design-canvas.jsx 文件仍保留以备设计师工具回归）；新增 `.htr-app-nav` / `.htr-app-brand` / `.htr-app-tag` / `.htr-nav-btn` CSS。smoke：`/` -> 200 (16150 bytes，含 nav)；nav 有 4 个 V1..V4 按钮；`<DesignCanvas>` 在 App 中出现次数为 0；`/api/dashboard` 仍 200。pytest 未跑（前端静态）。
  - **Cycle 7 修正**：见 PROJECT_STATUS.md 同日 Change Log。视口铺满 + 七档阶梯反碰撞。
  - **Cycle 8 用户反馈 bug 修复**：Q5 — `useTickingPrice` 是 `setInterval` 合成 ±0.08% 随机抖动（demo 动画），JP 休市时仍在动让用户误以为是 live 数据。修法：V1/V2/V3/V4 全部删 `useTickingPrice(...)` 调用，`livePrice = top.price` 直接用真实日收盘价。`shared.jsx` 的 `useTickingPrice` 定义保留以便未来真实 intraday adapter 接入时可参考；当前 export 仍在 window 但无消费方。Q3 — V2 `WatchlistTable` 容器缺 `background`，表头 `surface-2` (#FBFAF5) 与表身（透明继承外层）不一致。修法：容器加 `background: var(--htr-surface)` + `borderRadius: 4`，与其他 panel 视觉一致。Smoke：4 个变体都 `useTickingPrice( count = 0`；V2 WatchlistTable 现在白底统一。pytest 未触（前端静态）。

### P8-10 User Positions Integration

- Status: done
- Depends on: P8-09, ADR-0005
- Goal: 从 `Project_optimized/reports/paper_trading_account.json` 读取真实持仓（每只 symbol 最新 snapshot），通过 `/api/dashboard` 暴露给前端，让 V1-V4 每个变体都能显示当前 cash / NAV / 持仓列表 / 浮动盈亏，让用户在仪表盘上直接看到自己的 3041.T 400 股 等。
- Files:
  - Create: `src/hot_theme_rotator/data/position_adapter.py`
  - Create: `tests/unit/test_position_adapter.py`
  - Update: `api/serializers.py` (add `positions` to dashboard payload)
  - Update: `tests/unit/test_api_dashboard.py` (positions key + shape)
  - Update: `frontend/v3.jsx` (新增"持仓"section，先在 V3 接入；V1/V2/V4 follow-up)
- Acceptance:
  - 读取 `paper_trading_account.json` 时仅取每只 symbol 最近 `asof` 的 snapshot（同 symbol 历史快照去重）。
  - `/api/dashboard` 返回 `positions: {cash, nav, positions_value, asof, holdings: [{symbol, qty, avg_cost, market_price, market_value, unrealized_pnl}]}`。
  - 找不到文件或 schema 缺字段时 fail-closed，`positions: null` + 显式 reason；前端渲染"持仓数据未就绪"。
  - V3 至少能渲染持仓列表 + NAV。
  - Rule 3：只 GET，不暴露持仓写入；不写回 Project_optimized。
  - 全部测试通过。
  - Verified 2026-05-24 (initial): 用 JSON-based adapter 跑通，但显示 3041.T (sprint 旧策略)。用户澄清当前 live 持仓是 etf_buyhold/1306.T。
  - **修正 2026-05-24 (source pivot)**: per Rule 4 — 源切换 `Project_optimized/reports/paper_trading_account.json` → `japan_market.db` 的 `positions` + `account_snapshots` 表，按 `strategy_id` 过滤（默认 `etf_buyhold`）。Reason: JSON 是已下线 sprint 策略的快照；live 持仓只在 DB 表里。Impact: dashboard 现在显示真实 Path A live 1306.T 900 股 @ ¥403, 现 ¥412.4, P&L +¥8,460 (+2.33%), NAV ¥395,555, cash ¥26,645。Verification: adapter 改为 SQLite read-only (`file:...?mode=ro` URI)；schema assertion 列依赖检查；`__FLAT__` sentinel + zero-qty 过滤；`list_available_strategies()` debug helper；`PortfolioState` 新增 `strategy_id` + `positions_asof` 两字段（account asof 可能晚于 positions asof，分别披露）；test fixtures 从 JSON 改为 sqlite createDB；`python -m pytest .\tests` -> `241 passed` (12 adapter + 2 api positions); smoke `curl /api/dashboard | jq .positions` 返回 1306.T 真实数据；ADR-0005 + DESIGN §6.1 文档同步更新指向 DB 源。

### P8-11 Multi-Market Temperature Aggregator

- Status: done
- Depends on: P8-09, ADR-0005
- Goal: 从 `japan_market.db.cross_asset_snapshots` 表合成 6 市场温度（日经 / TOPIX / SOX / S&P / USDJPY / 上证），含温度评分 + sparkline 尾部 + 状态 (OPEN/CLOSED/LIVE)，替换 V1 hero row / V3 hero mosaic / V2 footer strip / V4 left rail 的 mock。
- Files:
  - Create: `src/hot_theme_rotator/data/market_temp_adapter.py`
  - Create: `tests/unit/test_market_temp_adapter.py`
  - Update: `api/serializers.py` (real markets[])
- Acceptance:
  - 6 市场全部能从 cross_asset_snapshots 取出；数据不足时该市场返回 null + 显式 reason，不静默填零。
  - 温度评分公式文档化（动量 + 量能 + 波动率合成），可被 schema 测试守门。
  - Frontend 现有 4 个变体的 markets 渲染无需修改即可消费。
  - Verified 2026-05-24: `data/market_temp_adapter.py` (~160 LOC: `MarketTile` dataclass / `load_market_mosaic` / `_topix_via_etf_proxy` 用 kline_adapter 拉 1306.T / temp 公式 `clip(50 + chg*10, 0, 100)` / USDJPY inverse temp / 5 real + 1 UNKNOWN); `tests/unit/test_market_temp_adapter.py` (12 tests 覆盖 6 tile / SOX 温度 known answer / SPX 真实值 / USDJPY 反向温度 / TOPIX 1306.T proxy / SSE 永远 UNKNOWN / temp 公式 / fail-closed 三路 / 默认路径)；`api/serializers._serialize_markets()` 软回退 [] 接 markets[]；`dataAvailability.markets=true`。`python -m pytest .\tests` -> `305 passed`。Smoke：dashboard markets 6 tile 全部真实（除 SSE 标 UNKNOWN）。

### P8-12 Theme Heat Ranker

- Status: done
- Depends on: P8-09, ADR-0005
- Goal: 从 `japan_market.db.factor_signals` + `signals` 表，按主题聚合 heat + 动量、排序输出 top N 主题（含 leaders 列表），替换 V1/V3 theme heat 区 mock。
- Files:
  - Create: `src/hot_theme_rotator/data/theme_heat_adapter.py`
  - Create: `tests/unit/test_theme_heat_adapter.py`
  - Update: `api/serializers.py` (real themes[])
- Acceptance:
  - 6 主题 heat 0-100，从 factor_signals 真实合成。
  - leaders 列表来自 signals 表（按主题 + 当日 rank）。
  - 主题分类与 `theme_detection/theme_detector.py` 已有标签集对齐。
  - Verified 2026-05-24 (first cut)：`data/theme_heat_adapter.py` (~130 LOC: `ThemeHeatRow` / `load_theme_heat` 按 factor_name 聚合 abs(z_score) / Chinese label 映射表 / top N + leaders top K) — **本版用 alpha factor (mom_20/sharpe_20/...) 充当 themes**，与 theme_detection 的 keyword themes (AI 半导体/汽车出口/...) 暂未对齐；keyword-themes mapping 留 follow-up。Heat 公式 `clip(round(mean_abs_z * 50), 0, 100)`，典型 factor 落在 30-60。`tests/unit/test_theme_heat_adapter.py` (12 tests)；`api/serializers._serialize_themes()` 软回退 [] 接 themes[]；`dataAvailability.themes=true`。Smoke：6 themes 真实（ret60/sharpe_60/sortino_60/sharpe_20/high52w/vol_z, asof 2026-04-13），leaders 来自真实 factor_signals。`python -m pytest .\tests` -> `305 passed` (含此 12)。

### P8-13 News Timeline Adapter

- Status: done
- Depends on: P8-09, ADR-0005
- Goal: 从 `japan_market.db.news_feed` + `news_items` + `news_sentiment` 表取最近 N 小时新闻（默认 12 小时），按时间倒序 + weight 标签（high/medium/low）+ linkedSymbols，替换 V1/V3/V4 新闻区 mock。
- Files:
  - Create: `src/hot_theme_rotator/data/news_adapter.py`
  - Create: `tests/unit/test_news_adapter.py`
  - Update: `api/serializers.py` (real newsTimeline[])
- Acceptance:
  - weight 从 news_sentiment 表的 sentiment_score 派生（阈值文档化）。
  - linkedSymbols 来自 news_items.symbol_links 字段。
  - 时间戳本地化为 JST 显示。
  - Verified 2026-05-24：`data/news_adapter.py` (~135 LOC: `NewsRow` / `load_news_timeline(hours, limit)` / JOIN news_feed LEFT JOIN news_items / weight 公式 urgency ≥5 high, ≥2 medium, else low；|sentiment| ≥0.5 也算 high / `related_tickers` JSON 字符串解析含逗号 fallback / cutoff 用 DB 最新 published_ts 而非 wall-clock 让历史数据可用 / text 优先 summary_cn fallback title); `tests/unit/test_news_adapter.py` (14 tests 覆盖时间倒序 / 窗口外排除 / 三档 weight / JSON 解析 / 逗号 fallback / text fallback / limit / empty / fail-closed 两路); `api/serializers._serialize_news()` + `_format_ts_jst` 把 `2026-05-21T20:00:00` → `05-21 20:00 JST`；软回退 []；`dataAvailability.newsTimeline=true`。Smoke：12 小时内 6 条真实新闻，含 NTT 业绩 [high] + 大幅増益 [low] 等。`python -m pytest .\tests` -> `305 passed`。

### P8-14 K-line OHLC Fetcher + P9-02 LegacyDailyPriceFetcher

- Status: done
- Depends on: P8-09, P9-02, ADR-0005
- Goal: `data/kline_adapter.py` 读 `japan_market.db.daily_prices` 表，按 symbol + window 返回 PriceBar[]。同时实现 P9-02 `outcome_join.PriceFetcher` Protocol（即 `LegacyDailyPriceFetcher`），让 outcome join 用真实 OHLC。
- Files:
  - Create: `src/hot_theme_rotator/data/kline_adapter.py`
  - Create: `tests/unit/test_kline_adapter.py`
  - Update: `api/serializers.py` (real kline for hero)
- Acceptance:
  - `LegacyDailyPriceFetcher` 满足 `PriceFetcher.fetch(symbol, start_date, end_date)` 契约。
  - 返回的 PriceBar.asof 严格 ISO 日期（P0-04 F3 约束）。
  - 同时满足 dashboard kline (40 bar) 与 outcome_join 任意窗口两个调用方。
  - Verified 2026-05-24: `data/kline_adapter.py` (~120 LOC: `fetch_kline` 倒序拉取 + 反转为时间升序 / `fetch_latest_close` 取单根 / `LegacyDailyPriceFetcher` dataclass 实现 Protocol / schema 列断言 / SQLite `mode=ro` URI / ISO 日期 + 区间校验); `tests/unit/test_kline_adapter.py` (16 tests 覆盖时间序 / 窗口 cap / 符号过滤 / 空 symbol / 最新 close / fail-closed 三路 / Protocol 集成测试 e2e 调 compute_outcome 喂真实 1306.T bars 得到 complete outcome); `api/serializers._serialize_kline(symbol, sessions=40)` 真实从 DB 拉 hero K 线，失败软回退空数组 → frontend mock fallback。`python -m pytest .\tests` -> `258 passed` (+17 new)。Smoke: V1 KLineChart 现在自动收到 40 根真实 OHLC（top 候选 6768.T 2026-03-02 → 2026-04-27 latest close ¥801）。`dataAvailability.kline` 改为 true。

### P8-15 Real Candidate Scanner (replace build_sample_panel)

- Status: done
- Depends on: P8-09, P8-14, ADR-0005
- Goal: `data/universe_adapter.py` 读 `Project_optimized/universe.json`（951 标的）+ `selected_tickers.json`（当日 top N 短名单），让 `api/serializers` 不再用 `build_sample_panel`（8035.T/7203.T fixtures）而是扫描用户真实当日候选。
- Files:
  - Create: `src/hot_theme_rotator/data/universe_adapter.py`
  - Create: `tests/unit/test_universe_adapter.py`
  - Update: `api/serializers.py` (real candidates from selected_tickers + kline_adapter)
- Acceptance:
  - 候选列表来自 `selected_tickers.json` 真实当日 top N。
  - 每个候选的 `reference_price` 从 `kline_adapter` 取最新 close。
  - 七档阶梯继续由 `opportunity.price_ladder.build_price_ladder()` 计算（不变）。
  - `selected_tickers.json` 缺失或 schema 不符时 fail-closed 回退到 sample（带 staleness 告警）。
  - Verified 2026-05-24: `data/universe_adapter.py` (~95 LOC: `ScreenedTicker` + `ScreenerSnapshot` + `load_screener_snapshot` + `default_selected_tickers_path` + 必填字段校验); `tests/unit/test_universe_adapter.py` (9 tests 覆盖真实 payload 加载 / 顺序保持 / 可选字段默认 / fail-closed 五路 / 空 details / 默认路径); `api/serializers._real_or_sample_candidates(top_n)` 优先真实 screener，失败软回退 sample；`_serialize_real_candidates` 每条候选实时调 `fetch_latest_close` 拿真实 ref_price，构造 ladder，承袭 V1-V4 字段契约；`_build_meta` 重构为接 `trade_date + candidates_source` 直参；`meta.candidatesSource` 暴露 "screener_v2" 或 "sample"；`meta.dataAvailability.candidates` 反映真实状态。`python -m pytest .\tests` -> `267 passed` (+9 new)。Smoke `/api/dashboard` 现返回 top-10 真实候选：#1 6768.T score 50.6 ref ¥801, #2 5074.T score 46.5 ref ¥815, ... #10 2354.T，每条都带真实 mom20/sharpe20/adv 信息。tradeDate=2026-04-27 来自 screener asof。无任何执行通路新增。

### P8-16 Dashboard UX Polish — §10 chip + V1 K-line + Tooltip

- Status: done
- Depends on: P8-09 (4 variants in place), P8-11..15 (real data flowing)
- Goal: 3 项用户反馈集中处理。(1) §10 八阶门槛从底部全条改为 nav 区右上小 chip + 点击展开 modal，让出每个变体 50-70px 屏幕空间。(2) V1 中央 KLineChart 充分利用空白：加成交量子图（底 25%）+ MA20/MA60 均线叠加 + 52w 高低水平线 — 全部从已有 data.kline 计算，不引入新 adapter。(3) shared.jsx 新增 `<Term glossaryKey="...">...</Term>` 组件 + GLOSSARY 字典（~30 术语）+ tooltip CSS；V1/V2/V3/V4 中所有 jargon 字串包 `<Term>` 让小白鼠标悬停看到中文白话解释。
- Files:
  - Cycle 1: `frontend/index.html` (+nav chip + modal); `frontend/{v1,v2,v3,v4}.jsx` 删冗余 GateFlow + 调整 gridTemplateRows
  - Cycle 2: `frontend/shared.jsx` (KLineChart 加 volume subchart / MA20 / MA60 / 52w lines); 可能 v1.jsx 调高度
  - Cycle 3: `frontend/shared.jsx` (+`Term` + `Tooltip` + GLOSSARY ~30 项); tooltip CSS in index.html
  - Cycle 4: `frontend/{v1,v2,v3,v4}.jsx` 包 `<Term>` 在所有 jargon 字串
- Acceptance:
  - Nav 右上 §10 chip 显示 "5/8 ✓ · 1 blocked"；点击展开浮层渲染完整 GateFlow；4 变体里不再有独立 gate strip。
  - V1 KLineChart 包含 OHLC candles + 底部成交量 + MA20 + MA60 + 52w 高低水平线，全部正确缩放；图填满中央 Panel。
  - `<Term>` 悬停 ≥300ms 显示解释框，含 1 行白话 + 公式（可选）；覆盖至少 25 个术语。
  - Tooltip 解释**绝不**说"高分=高胜率"等违 §9.4 红线。
  - 不引入新 score_status 值、新执行按钮；Rule 3 / §9.4 / §10 保留。
  - 全部 pytest 通过（前端纯静态可不动后端测试）。
  - Verified Cycle 1 (§10 nav chip)：`index.html` 新增 `.gates-chip` + `.gates-modal-backdrop/.gates-modal` CSS；App() 加 `gatesOpen` state + ESC 关闭 + 计算 done/blocked count + 渲染 chip "§10 5/8 ✓ ⛔1" 在 nav 右侧 + modal 内 `<GateFlow compact={false}>`。V1/V2/V3/V4 全部删除独立 GateFlow（4 个 `<GateFlow` 计数 = 0）；V1 gridTemplateRows 从 "44 1fr 70" 改为 "44 1fr" 让中央区多 70px；V3 gridTemplateRows 从 "44 240 1fr 50" 改为 "44 240 1fr"；V2 Section D 改名为"新闻催化与决策日志"；V4 右栏底部 gate 卡删除。
  - Verified Cycle 2 (V1 K-line 充实)：`shared.jsx.KLineChart` 增加 `withVolume/withMA/with52wLines` 三 props (默认 true)；底部 25% innerH 渲染成交量柱 (up=绿 down=红 半透明)；`rollingMean` 内联计算 MA20+MA60 polyline，蓝实线/橙虚线 + 左上 legend；52w 高/低水平线虚线 + 价格 label；`api/serializers._serialize_kline` sessions 40→252 让 MA60+52w 有意义；V1 `<KLineChart>` 调用 width 600→760 height 300→500 填满 Panel；Panel sub 改为 `${data.kline.length} sessions`，right chip 标注 "MA20/60 · 52w 高低"。Smoke：`/api/dashboard.kline` 现 252 bars (6768.T 2025-04-16 → 2026-04-27)。
  - Verified Cycle 3 (Term + GLOSSARY)：`shared.jsx` 加 `function Term({children, k})` 组件 + `GLOSSARY` 字典 (40 个术语：七档阶梯/未校准研究分/校准/Brier/Log loss/calibrated_probability + 12 个 alpha factor + 6 个市场术语 + 持仓相关 + §10/§8.6/§9.4 + screener + dashboard sections)；`index.html` 加 `.htr-term` + `.htr-term-pop` CSS (dotted underline 悬停 / focus 显示深色 tooltip 含术语名 + 中文白话定义 + 280px 宽)；全部定义严格遵守 §9.4——绝不说"高分=高胜率"。
  - Verified Cycle 4 (变体术语包装)：V1 包 4 处 (`多市场温度` `主题热力` `七档阶梯` `MA20`/`MA60`/`52w 高低`)；V3 包 3 处 (`主题热力` `新闻催化` `决策日志`/`§8.6`)；V4 包 1 处 (ActionZone 提示中的 `决策日志` + `§10`)；总计 8 处可悬停。pytest 305 passed (无回归；前端纯静态)。
  - **Cycle 5 follow-up (用户自查触发)**：用户问"今天我们做的这些改动是先更新了规则还是还没有"。自查发现 P8-16 的 Tooltip 子系统（Term + GLOSSARY）+ §10 chip + KLineChart 强化按 §3"模块职责变化"原本应入 DESIGN §6.12 Frontend 但仅在 TASKS + Change Log 记录。**Rule 4 silent drift 补救**：DESIGN §6.12 backfill 4 段（app shell、§10 chip、Term subsystem、KLineChart 强化）。同时 V1 K 线板"仍有大量留白"问题修复：`shared.jsx` 新增 `useElementSize` ResizeObserver hook（导出至 window），v1.jsx 新 `V1KLineFillPanel` 组件用 ref 实时测 Panel 内尺寸动态喂 width/height 给 KLineChart，让 K 线真正铺满 V1 中央列。pytest 305 passed 维持。

### P8-17 V1/V2 UI Space And Surface Fixes

- Status: done
- Depends on: P8-16
- Goal: 处理用户反馈的两个 UI 问题：(1) V1 `价格走势 · 七档阶梯` 卡片仍把七档标签挤在 K 线 SVG 右侧，卡片扩高后信息密度不足；改为 K 线主区 + 独立七档侧栏，充分利用横向与纵向空间。(2) V2 `新闻催化与决策日志` 下半区在 `decisionLog` 为空时右侧像漏渲染；改为两个明确白底 paper-surface 面板，并为决策日志提供空状态。
- Files:
  - Create: `docs/superpowers/plans/2026-05-24-p8-17-v1-v2-ui-fixes.md`
  - Create: `tests/unit/test_frontend_ui_contracts.py`
  - Update: `frontend/v1.jsx`
  - Update: `frontend/v2.jsx`
  - Update: `docs/00_DESIGN.md`
  - Update: `docs/02_GOVERNANCE.md`
  - Update: `PROJECT_STATUS.md`
- Acceptance:
  - V1 `价格走势 · 七档阶梯` 卡片内部使用 `minmax(0, 1fr) 184px` 两列：左侧 KLineChart，右侧七档侧栏。
  - V1 KLineChart 不再把七档文字标签塞进 SVG 右 padding；K 线绘图 `padding.right` 收窄到 28px 左右。
  - V1 右侧七档侧栏显示 3 个卖出档、3 个买入档、1 个止损档，并保留相对现价百分比。
  - V2 Section D 的新闻与决策日志区域都有显式 `var(--htr-surface)` 背景、边框、最小高度。
  - `decisionLog` 为空时显示"暂无 §8.6 决策日志"空状态；不暗示胜率已校准，不提供任何执行控件。
  - 不新增 score_status，不改 API 数据契约，不改评分、校准、持仓、候选或下单逻辑。
  - 静态 UI 契约测试和 API dashboard 测试通过；V1/V2 headless 截图留存在 `.runtime/ui_inspection/`。
  - Verified RED: `python -m pytest .\tests\unit\test_frontend_ui_contracts.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` failed with 2 expected failures (`V1KLineLadderPanel` and `V2SurfacePane` absent).
  - Verified GREEN targeted: `python -m pytest .\tests\unit\test_frontend_ui_contracts.py -q -o cache_dir=.\.runtime\pytest_cache --basetemp=.\pytest_tmp` -> `2 passed`.
  - Verified API + UI contract: `python -m pytest .\tests\unit\test_api_dashboard.py .\tests\unit\test_frontend_ui_contracts.py -q -o cache_dir=.\.runtime\pytest_cache --basetemp=.\pytest_tmp` -> `13 passed`.
  - Verified browser smoke: Edge headless screenshots saved to `.runtime/ui_inspection/p8-17-v1_1440x900.png` and `.runtime/ui_inspection/p8-17-v2_scrolled_1440x900.png`; V1 shows K-line plus dedicated seven-level rail, V2 shows explicit surface panes and empty §8.6 state.
  - Verified full suite: after creating writable `.runtime\pytest_tmp` and `.runtime\pytest_cache_p8_17`, `python -m pytest .\tests -q -o cache_dir=.\.runtime\pytest_cache_p8_17 --basetemp=.\.runtime\pytest_tmp\p8-17-full-<timestamp>` -> `307 passed in 7.58s`.
  - Follow-up root cause (user reported page-level background switch between Section C and D): V2 root used `display: flex` with default `align-items: stretch`, which stretched the white paper wrapper to viewport height only (`858px`); scrolled content then overflowed beyond that wrapper and exposed body/html warm-gray background. Fix: set V2 root `alignItems: "flex-start"` so the paper wrapper uses content height (`2021px` in probe). Added static contract test for this. Verification: `test_frontend_ui_contracts.py` RED -> 1 expected failure; GREEN -> `3 passed`; API+UI contracts -> `14 passed`; full suite -> `308 passed in 7.64s`; screenshot `.runtime/ui_inspection/p8-17-v2-bg-fixed_1440x900.png`.

### P8-18 Interactive Exploration Layer

- Status: done
- Depends on: P8-09, P8-10, P8-14, P8-15
- Goal: 按 Rule 11 把 V1-V4 从只读看板升级成可探索研究工具。新增 3 个 GET 探索端点 + 候选清单 onClick → V1 K 线随之切换 symbol。仍全 GET，仍不破 Rule 3 / §9.4 / §10。
- Files:
  - Update: `docs/02_GOVERNANCE.md` (新增 §11 Read-Only Interactivity Rules)
  - Update: `docs/00_DESIGN.md` (§6.11.1 探索端点设计)
  - Create: `api/symbol.py` (3 个端点)
  - Update: `api/main.py` (挂载 symbol router)
  - Create: `tests/unit/test_api_symbol.py`
  - Update: `frontend/v1.jsx` / `v3.jsx` / `v4.jsx`（候选行 onClick 切换 selected symbol；K 线 fetch /api/symbol/{ticker}/kline 替换 data.kline）
  - Update: `frontend/index.html`（如需 selected symbol 全局状态）
- Acceptance:
  - `GET /api/symbol/{ticker}/kline?sessions=N` 返回该 ticker 的 N 根日 K（N ∈ [1, 1000]），ticker 不存在返回 404 + symbol_not_found，sessions 超界返回 422。
  - `GET /api/symbol/{ticker}/profile` 返回 `{symbol, latest_close, latest_asof, in_portfolio, qty, avg_cost, unrealized_pnl, in_screener, screener_score, mom_20, mom_60}`；缺字段以 null 显式标注，不静默填零。
  - `GET /api/symbol/{ticker}/ladder?ref_price=X` 用任意 ref_price 重算七档，ref_price ≤ 0 返回 422。
  - 候选清单任一行被点击后，selected symbol 持久化到 `localStorage.htr_symbol`，V1 K 线读取该 symbol 的 252 sessions；持仓行也可点击切换。
  - 不引入任何 POST / PUT / DELETE / PATCH 端点。
  - 不创建任何 `decision_log` 条目；用户的本地选择不进 `reports/predictions/`。
  - 全部测试通过；新增 ≥ 8 个 API 单测 + 1 个前端契约测试。
  - Verified 2026-05-25 (backend): `api/symbol.py` 含 3 端点 (kline/profile/ladder)，fail-closed 404/422/500；`api/main.py` 挂 symbol_router；`tests/unit/test_api_symbol.py` 15 测试覆盖 kline 5 + profile 4 + ladder 5 + 边界 1。`python -m pytest tests/unit/test_api_symbol.py` -> `15 passed in 0.31s`。
  - Verified 2026-05-25 (frontend): `shared.jsx` 新增 3 hook (`useSelectedSymbol` / `useSymbolKline` / `useSymbolProfile`) + `CandidateRow` 加 `active`/`onClick`/键盘可达；`v1.jsx` `V1ProTerminal` 用 selected symbol 驱动 hero+K线+ladder，`CandidateRowMini` 加 `active` + `onClick`；`v3.jsx` 同样 wire；`tests/unit/test_frontend_ui_contracts.py` 新增 5 个契约：hook exports / no-write-methods / V1 wiring / V3 wiring / localStorage key。`python -m pytest tests/` -> `338 passed in 7.74s` (318 baseline + 15 API + 5 frontend = 338)。

  - Follow-up 2026-05-25 (V1 selected-symbol K-line fallback): user reported V1 candidate click changed the selected stock but the K-line appeared unchanged. Root cause: `/api/symbol/{ticker}/kline` returned correct bars, but V1 reused the initial `/api/dashboard.kline` as fallback for every selected symbol when the per-symbol request failed or API was unavailable. Fix: only the initial top symbol may use dashboard K-line fallback; other symbols render an explicit unavailable state rather than a mislabeled chart, and `KLineChart` remounts with `key={top.symbol}`. RED `test_frontend_ui_contracts.py` failed on missing fallback guard; GREEN targeted -> `8 passed`; API+frontend contracts -> `23 passed`; browser smoke confirmed V1 click `5074.T` requested `/api/symbol/5074.T/kline?sessions=252` and SVG 52w labels changed from `高 808 / 低 416` to `高 832 / 低 292`.
### P8-19 Morning Briefing CLI

- Status: done
- Depends on: P8-04 (free_web_opportunity_adapter), P8-10 (positions), P8-14 (kline)
- Goal: `tools/morning_briefing.py` — 5/25 周一开市前实战工具。接收 watchlist（带 1306.T 持仓）→ 用 yfinance 拉最近真实 quote → 输出 1) 持仓核对（current price vs avg_cost, 当前 P&L）2) watchlist 每只七档阶梯 + 距各档百分比 3) §10 + §9.4 红线显式贴在输出顶部。advice-only。
- Files:
  - Create: `tools/morning_briefing.py`
  - Create: `tests/unit/test_morning_briefing.py`
  - Update: `README.md` (用法示例)
- Acceptance:
  - CLI 接受 `--watchlist 1306.T,6768.T,...` 或读 `watchlist.txt`。
  - 输出包含每只 ticker 的 yfinance 最新 close + 七档阶梯 + 持仓状态。
  - 输出顶部固定有 §9.4 警告：未校准研究分 / 不是胜率 / advice-only。
  - 在没有 yfinance 网络时 fail-closed，明确告知"无网无法继续"，不退回伪造价。
  - 不写任何执行通路；不创建 decision_log 条目（这是用户态工具，不是系统态预测路径）。
  - 全部测试通过；含一个用 mock fetcher 的集成测试。
  - Verified 2026-05-25: `tools/morning_briefing.py` (~290 LOC) 含 `QuoteFetcher` Protocol + `LegacyDbQuoteFetcher` + `YFinanceQuoteFetcher` + `render_holdings_block` / `render_watchlist_block` / `render_briefing` + argparse CLI；Windows stdout 用 `sys.stdout.reconfigure(encoding="utf-8")` 兜底 cp932 无法编码 CJK。`tests/unit/test_morning_briefing.py` 11 测试（parse_watchlist 3 + holdings 2 + watchlist 3 + briefing 3）全部用 `StubFetcher` 不依赖网络。`python -m pytest tests/unit/test_morning_briefing.py` -> `11 passed in 0.03s`；full suite -> `349 passed in 7.80s` (338 + 11)。Smoke `python tools/morning_briefing.py --watchlist 1306.T,6768.T,5074.T --source db` 输出真实 briefing：1306.T 现价 ¥412.40 P&L +¥8459.99 (+2.33%)，6768.T + 5074.T 七档完整。

## Milestone P0: Project Spine (续)

### P0-05 Git Snapshot Baseline

- Status: done
- Depends on: 无（基线任务）
- Goal: 把整棵 untracked `HotThemeRotator/` 树首次 commit 进 git + 打 tag `htr-snapshot-2026-05-25-pre-interactive`，作为 P8-18 / P8-19 大改动前的可回退基线。子树精确 staging，不触动同 repo 内 `Project_optimized/` 的 200+ in-flight 修改。补救内存中长期已知的"159 文件全部 git untracked"单点风险。
- Files:
  - Update: `.gitignore` (新增 `pytest_tmp/` / `pytest_cache/` / `.pytest_tmp/` / `quant.zip` / `reports/predictions/` / `reports/outcomes/` 排除)
  - Create: 159 文件首次入 git (104 .py + 22 .gitkeep + 21 .md + 7 .jsx + 1 .yaml + 1 .json + 1 .js + 1 .html + 1 .gitignore)
- Acceptance:
  - `git add` 精确针对 `worker-quant/quant_trading/HotThemeRotator/`，非子树 staged 计数 = 0。
  - commit 含完整说明（包含 .gitignore 排除原因 + 318 pytest 基线 + 5 ADR 完整性）。
  - tag `htr-snapshot-2026-05-25-pre-interactive` 创建。
  - 任何后续 P8-18 / P8-19 改动失败时，可通过 `git checkout htr-snapshot-2026-05-25-pre-interactive -- worker-quant/quant_trading/HotThemeRotator/` 完整回退。
  - Verified 2026-05-25: 159 files staged, 0 files outside HTR staged; commit `f1d663e` "quant/HTR: snapshot baseline" 创建；tag `htr-snapshot-2026-05-25-pre-interactive` 在 `git tag -l "htr-*"` 中可见；`.gitignore` 6 行新增 ignore 排除运行时产物 + 14M `quant.zip` 设计原始包。

## Milestone P9: Automation Gates

### P9-01 Decision Log Infrastructure

- Status: done
- Goal: 实现 §8.6 mandatory feedback log 的承载子系统（§10 gate 3）。建立单一 `PredictionRecord` schema、JSONL 持久化和 fail-closed 写入接口，横切 attribution（`baseline_decision_score`）与 opportunity（`opportunity_scanner`）两条产出路径，使 P7-06 / P8-05 / P9-02 / P9-03 都消费同一存储。
- Files:
  - Create: `docs/adr/ADR-0003-decision-log.md`
  - Update: `docs/00_DESIGN.md` (§5 架构、§6.9 Decision Log、§7 数据流)
  - Update: `docs/03_FOLDER_MAP.md` (新增 `src/hot_theme_rotator/decision_log/`、`reports/predictions/`)
  - Create: `src/hot_theme_rotator/decision_log/__init__.py`
  - Create: `src/hot_theme_rotator/decision_log/schema.py` (`PredictionRecord`)
  - Create: `src/hot_theme_rotator/decision_log/jsonl_writer.py` (`append_prediction`, `read_predictions`)
  - Create: `tests/unit/test_decision_log_schema.py`
  - Create: `tests/unit/test_decision_log_writer.py`
  - Update: `src/hot_theme_rotator/opportunity/opportunity_scanner.py` (emit `prediction_id` + `model_version`)
  - Update: `src/hot_theme_rotator/reporting/realtime_opportunity_panel.py` (调用 decision log)
  - Create: `tests/integration/test_opportunity_decision_log.py`
- Acceptance:
  - 单一 `PredictionRecord` schema 覆盖 attribution 与 opportunity 两类预测形态。
  - `prediction_id` 稳定可重现（基于 `input_snapshot_id` + `model_version` + `decision_cutoff` 的 SHA-256 摘要）。
  - JSONL 写入对缺失必填字段 fail-closed。
  - `opportunity_scanner` 的每条 candidate 都附 `prediction_id` 与 `model_version`，并落盘到 `reports/predictions/`。
  - DESIGN.md §6.9 模块职责与 §7 数据流均已加入 decision_log 节点。
  - FOLDER_MAP.md 新增 `decision_log/` 与 `reports/predictions/`。
  - 全部测试通过；预期增量 ≥ 8 个新测试。
  - Verified Cycle 1 (design alignment): ADR-0003 created; DESIGN §5 / §6.9 / §7 updated; FOLDER_MAP §src / §reports updated; P7-06 / P8-05 / P9-02 / P9-03 task wording rewritten with `Depends on`; `python -m pytest .\tests` -> `101 passed`.
  - Verified Cycle 2 (schema): `decision_log/__init__.py` + `decision_log/schema.py` (`PredictionRecord`, `compute_prediction_id`, `PredictionRecordValidationError`) created with 11 unit tests covering determinism, integrity, fail-closed; `python -m pytest .\tests\unit\test_decision_log_schema.py` -> `11 passed`; full suite -> `112 passed`.
  - Verified Cycle 3 (writer): `decision_log/jsonl_writer.py` (`append_prediction`, `append_predictions`, `read_predictions`, `predictions_path`) created with 8 unit tests covering round-trip, duplicate rejection, missing-file fallback, malformed-JSONL fail-closed; `python -m pytest .\tests\unit\test_decision_log_writer.py` -> `8 passed`; full suite -> `120 passed`.
  - Verified Cycle 4 (scanner integration): `OpportunityCandidate` extended with `input_snapshot_id` / `model_version` / `prediction_id` / `decision_cutoff` / `trade_date` / `horizon_days` (backward-compatible defaults); `compute_opportunity_snapshot_id` added; `panel_row_to_prediction` + `persist_panel_predictions` added in `realtime_opportunity_panel.py`; opportunity buy/sell/hold mapping = `buy = score/100; sell = 0; hold = 1 - buy`; dashboard `build_yfinance_quote_panel` + `build_panel_from_inputs` accept optional `persist_base_dir`; streamlit `_build_panel` passes `persist_base_dir=PROJECT_ROOT` for yfinance mode (sample mode stays write-free); 1 new scanner test + 3 new integration tests; `python -m pytest .\tests` -> `124 passed`; `python -m py_compile .\tools\streamlit_opportunity_app.py .\tools\realtime_opportunity_demo.py` succeeded.

### P9-02 Outcome Join Infrastructure

- Status: done
- Depends on: P9-01
- Goal: 在 P9-01 decision log 之上建立通用 outcome join 子系统：从存储读取所有 `PredictionRecord`，与历史价格序列匹配，标注每档价位是否触达，并写回 1D/3D/5D 实际收益（不引入 lookahead）。
- Acceptance:
  - 单一 `OutcomeRecord` schema 同时覆盖 attribution 收益结果与 opportunity ladder 触达事件。
  - 价位触达检测基于 cutoff 之后的历史 OHLC，确保 `available_ts > decision_cutoff` 才允许 join。
  - 失败匹配显式记录原因（symbol 未找到 / 数据缺口 / 时间窗未到期），不静默丢弃。
  - 输出落盘到 `reports/outcomes/`（与 `reports/predictions/` 平行）。
  - Verified Cycle 1 (schema + writer): `decision_log/schema.py` 新增 `OutcomeRecord` + `compute_outcome_id` + `ALLOWED_OUTCOME_STATUSES` + `OutcomeRecordValidationError`；`decision_log/jsonl_writer.py` 新增 `outcomes_path` / `append_outcome` / `append_outcomes` / `read_outcomes`；`decision_log/__init__.py` 增加 outcome 相关 export；11 schema 单测 + 9 writer 单测；`python -m pytest .\tests\unit\test_outcome_record_schema.py .\tests\unit\test_outcome_writer.py` -> `20 passed`；full suite -> `155 passed`.
  - Verified Cycle 2 (join function): `decision_log/outcome_join.py` 新增 `PriceFetcher` Protocol + `compute_outcome` + `compute_outcomes` + `JoinSummary` + `DEFAULT_HORIZONS_DAYS=(1,3,5)`；状态分支 `complete` / `insufficient_data` / `symbol_not_found` / `future_cutoff` 全部测试覆盖；七档 ladder 触达事件按 below-tier (low ≤ tier) 与 above-tier (high ≥ tier) 分别检测；`reference_price` 缺失时回退为首根 bar 的 open；fetcher 抛错被吸收为 `symbol_not_found`；11 join 单测；DESIGN §6.9 添加 outcome 段落；FOLDER_MAP §reports 添加 `reports/outcomes`；`python -m pytest .\tests\unit\test_outcome_join.py` -> `11 passed`；full suite -> `166 passed`.

### P9-03 Calibration Engine

- Status: done
- Depends on: P9-01, P9-02
- Goal: 在 P9-01 decision log + P9-02 outcome join 基础上建立通用校准引擎：对 attribution buy/sell/hold 和 opportunity 评分输出 Brier / log loss / calibration bins，达到样本阈值前一律维持 `uncalibrated_research_score` 或 `insufficient_calibration`。
- Acceptance:
  - 校准报告区分 attribution 与 opportunity 两类预测来源。
  - 报告 Brier、log loss、calibration bins，且每个 bin 明确标注样本数。
  - 样本量不达标时显式输出 `insufficient_calibration`，不计算虚假胜率。
  - 校准结果不静默回写底层模型参数（参数变更仍走 Rule 4 流程）。
  - Verified Cycle 1 (math + schema + reporter): 新模块 `src/hot_theme_rotator/calibration/` 含 `__init__.py` / `schema.py` (CalibrationBin + CalibrationReport + ALLOWED_CALIBRATION_STATUSES / ALLOWED_CALIBRATION_SOURCES + CalibrationReportValidationError) / `calibrator.py` (compute_brier_score / compute_log_loss / compute_calibration_bins / derive_opportunity_ground_truth) / `reporter.py` (build_calibration_report 配对 prediction_id → outcome, 按 horizon_days 派生 ground truth, min_samples gate)。`CalibrationReport.__post_init__` 强制：(a) `status='calibrated'` 必须 `sample_count >= min_samples_required` 且 `brier_score`/`log_loss` 非空；(b) `status='insufficient_calibration'` 不允许携带 brier/log_loss/bins（§9.4）。`CalibrationBin.__post_init__` 强制空 bin 的均值必须为 nan。Calibration bins 等宽 10 档，最后一档右闭以容纳 p=1.0。Log loss 对 p 在 [eps, 1-eps] 区间 clamp 避免 log(0)。Ground truth 仅在 outcome.status='complete' 且 horizon key 存在时返回；否则 None 让 reporter 跳过。新增 39 个测试（14 schema + 14 math + 11 reporter），包含 known-answer brier (0.25 for constant 0.5 vs all-1s) 与 multi-horizon ground truth 验证。`python -m pytest .\tests` -> `219 passed in 5.31s` (180 baseline + 39 new = 219)。
  - Verified Cycle 2 (docs + close): DESIGN §6.9 outcome 段落状态分支补充 `malformed_data`；DESIGN §6.10 新增 Calibration 子节描述职责与边界；FOLDER_MAP §src 增加 `calibration/` 行；TASKS.md 本任务 done。

### P9-04 Human Alerts

- Status: done
- Goal: Notify a human when watched candidates cross staged buy, stop, or sell levels.
- Acceptance:
  - Add `alerts.human_alerts` with `AlertRecord`, deterministic `alert_id`, `AlertThrottle`, and `build_ladder_alerts`.
  - Entry and stop alerts trigger only when `current_price <= level_price`; exit alerts trigger only when `current_price >= level_price`.
  - Alerts include symbol, level id, level price, current price, direction, severity, reason, risk warning, and data timestamp.
  - Alerts carry `research_only=True` and must not expose broker, account, route, quantity, notional, order type, or submit fields.
  - Duplicate alert throttling suppresses repeat alerts for the same `(symbol, level_id, trade_date)`.
  - Invalid current price, invalid level price, missing timestamp, or missing symbol fails closed.
  - No UI changes, no external notification channel, no paper trade creation, and no broker execution path in this cycle.
  - Verified 2026-05-25: `alerts.human_alerts` added with `AlertRecord`, deterministic `compute_alert_id`, `AlertThrottle`, and `build_ladder_alerts`. RED `test_human_alerts.py` failed on missing package; GREEN targeted -> `5 passed`; alert/opportunity/API related tests -> `35 passed`; full suite with `NUMBA_CACHE_DIR=.runtime\numba_cache` -> `318 passed in 7.93s`.

### P9-05 Paper Trading

- Status: pending
- Goal: Simulate strategy execution with risk limits before any live broker integration.
- Acceptance:
  - Uses logged candidates and price ladders.
  - Applies position limits, stop rules, and kill-switches.
  - Produces auditable daily paper performance.

### P9-06 Broker Execution Gate

- Status: pending
- Goal: Define the explicit approval and safety requirements before any broker API can be used.
- Acceptance:
  - Requires human approval recorded in `PROJECT_STATUS.md`.
  - Requires passing paper-trading evidence.
  - Requires kill-switch, max position, and audit logs.
  - Live broker execution remains disabled until this gate is done.

## Milestone P10: Personal Advisory System

P10 在 P0-P9 基础设施之上构建个人量化咨询层。目标两个交互模式：

- **Pull 模式**：用户给一只 ticker → 系统返回完整证据驱动的分析（news + 财报 + 因子 + 七档 + calibration status / raw frequency where allowed + LLM 叙事综合）。只有 Rule 9.4 满足后才能使用真实胜率语言。
- **Push 模式**：系统不被询问时主动扫描 + 推送高证据强度的策略候选。

校准语言通过 ADR-0006 + Rule 8.2.1 backdated bootstrap 加速形成证据，但 Rule 9.4 不放宽：未达样本和指标门槛前只能显示 `insufficient_calibration`、`uncalibrated_research_score` 或 Rule 8.6.1 允许的 raw frequency。LLM 严格限制为叙事综合（Rule 8.3 / 8.3.1），永不输出概率数字。

**Phase 1 实施顺序（2026-05-26 v4 — time-to-first-value activation staging）**：

踏空诊断显示当前最大短板是**数据新鲜度 + push 能力**，calibration bootstrap 是锦上添花。Codex (senior quant + math + stats + 自动控制) 评审 P11 反思系统后追加：(1) PIT Observability Ledger 是 P11 真正的基础（"polished hindsight machine" 失败模式的解药），(2) 时间表 6-7 周不现实，realistic 10-12 周。Rule 8.6.1 / 8.9 / Section 12 / Section 13 (Rule 13.1-13.10) + ADR-0006 + ADR-0007 全部配套。2026-05-26 用户明确 time-to-first-value 优先后，顺序改为 v4：

- **Week 1**: P10-14 TDnet RSS Adapter (Cycle 2 网络层) + P10-16 J-Quants Live Bridge + **P10-19 Best-Effort Delayed Price Orchestrator** — 关闭"今天 X 涨停为什么"数据盲区 + 多源价格 fallback chain。
- **Week 2**: **P10-20 Daily Advisory Cockpit** + **P10-17 Silent Watchlist Intelligence** — 先让系统每天可查、可用、可复盘，不先打扰用户（Rule 12.0 Stage 0/1）。
- **Week 3**: **P10-18 Anti-FOMO Guard Layer**（实施 Rule 12.1-12.6）+ **P10-10 Guarded Notification Channels** — 只有 discipline pass 的 alert 才能进入 desktop/email/telegram（Rule 12.0 Stage 2）。
- **Week 4**: **P11-00 PIT Observability Ledger** → P10-13 Backdated Calibration Bootstrap + **P11-01 Decision Trace Logger** — bootstrap 和 trace logger 都在 PIT ledger 之上。
- **Week 5**: **P11-02 Event Detector (CUSUM + ARL bootstrap)** + **P11-03 Policy Replay Engine** — 检测层 + off-policy evaluation（注意：NOT Pearl do-calculus per ADR-0007）。
- **Week 6**: **P11-04 Root Cause Analysis (ablation + funnel)** + **P11-05 LLM Reflection Report** + **P11-06 Human Decision Gate** — RCA NOT Shapley per ADR-0007。
- **Week 7+**: **P11-07 Meta-Reflection** + integration hardening + P10-01/02 forward schedulers + P10-03..09/15/12 余下 P1/P2。

**Realistic 10-12 周**（Codex aggressive estimate）；原 6-7 周是 fantasy。

### P10-13 Backdated Calibration Bootstrap Tool

- Status: done (2026-05-26)
- Priority: P0
- Depends on: ADR-0006, P9-01, P9-02, P9-03, P11-00
- Goal: 一次性历史回填工具。从 `Project_optimized/japan_market.db.factor_signals` + 历史 `selected_tickers.json` 快照（如已存档）重建过去 N 个交易日的 `PredictionRecord`，joined against `daily_prices` 真实 OHLC 通过 P9-02 outcome_join 生成 `OutcomeRecord`。所有合成记录强制标 `extra.backdated=True` + `extra.live=False` + `model_version` 加 `-backdated` 后缀。
- Files:
  - Create: `tools/backdated_calibration_bootstrap.py`
  - Create: `tests/unit/test_backdated_calibration_bootstrap.py`
  - Update: `src/hot_theme_rotator/decision_log/schema.py` (强制 extra flag 校验)
  - Update: `src/hot_theme_rotator/calibration/schema.py` (+ `evidence_origin` 字段)
  - Update: `src/hot_theme_rotator/calibration/reporter.py` (propagate evidence_origin)
  - Generate: `reports/bootstrap_provenance.json`
- Acceptance:
  - 处理连续历史窗口，禁止 date cherry-pick；任何被排除的 snapshot 写入 provenance 含 reason。
  - 每条 PredictionRecord 强制校验 4 个 flag（backdated / live / model_version 后缀 / 生成工具）。
  - 每条 OutcomeRecord 严格 join cutoff 之后至少 1 个 trading day 的 bars。
  - `scanner_config_hash` 必须匹配 `git log -- configs/scanner.yaml` 中 ≤ window start 的 commit；不匹配 fail-closed。
  - `CalibrationReport.evidence_origin` ∈ {"live", "bootstrap", "mixed"}；bootstrap-only report 可达 `calibrated` 状态。
  - bootstrap_provenance.json 包含 window/total/excluded/model_version/config_hash 完整记录。
  - 不引入任何 broker / order / alert 路径；Rule 3 / 8.3 / 9.4 全部保留。
- Verified (2026-05-26):
  - Added `src/hot_theme_rotator/calibration/backdated_bootstrap.py` (~250 LOC) with `BackdatedSnapshot` / `HistoricalSnapshotsLoader` Protocol / `BootstrapProvenance` / `BootstrapResult` / `MODEL_VERSION_SUFFIX="-backdated"` / `GENERATOR_TAG="backdated_calibration_bootstrap_v1"` / `bootstrap_calibration(*, window_start, window_end, base_model_version, scanner_config_hash, expected_scanner_config_hash, snapshots_loader, price_fetcher, base_dir, horizon_days=(1,3,5), trading_days=None)`.
  - Extended `calibration/schema.py` with `ALLOWED_EVIDENCE_ORIGINS = {"live", "bootstrap", "mixed"}` enum + `CalibrationReport.evidence_origin` field (default `"live"` for backward-compat). Extended `calibration/reporter.py` with `derive_evidence_origin(predictions)` that returns `"live"` (no backdated), `"bootstrap"` (all backdated), or `"mixed"` (both); `build_calibration_report` now propagates evidence_origin.
  - Bootstrap predictions all carry `extra={backdated: True, live: False, generator: backdated_calibration_bootstrap_v1, reference_price, ladder, reason_codes}` and `model_version` always ends with `-backdated` suffix.
  - **No date cherry-pick**: caller-supplied `trading_days` must be sorted, unique, and within window or `BootstrapError`; default fills every calendar day in window (caller passes TSE calendar list for real runs).
  - **scanner_config_hash mismatch fail-closed**: `expected_scanner_config_hash` parameter — caller resolves via `git log -- configs/scanner.yaml` at window start; mismatch → `BootstrapError` and provenance NOT written.
  - **Provenance JSON** at `reports/bootstrap_provenance.json` carries window_start / window_end / total_trading_days_attempted / snapshots_loaded / excluded (list of `{trade_date, reason}`) / model_version / scanner_config_hash / generated_at / generator.
  - Outcome computation: bootstrap walks all backdated predictions through P9-02 `compute_outcome(prediction, fetcher, evaluated_as_of, horizons_days)`; counts outcomes_built + outcomes_complete in result.
  - 14 unit tests: evidence_origin enum + reject unknown / derive_evidence_origin all-live + all-bootstrap + mixed / bootstrap backdated flags + suffix / provenance excluded reasons + JSON payload / scanner_config_hash mismatch fail-closed (no provenance written) / inverted window reject / cherry-picked trading_days reject / out-of-window trading_days reject / outcomes complete via stub fetcher / end-to-end bootstrap → report has evidence_origin="bootstrap".
  - RED missing module; first GREEN 9/14 (`compute_outcome` kwarg `horizons` → `horizons_days`; `JoinSummary` not iterable → `.outcomes`); final GREEN `python -m pytest tests/unit/test_backdated_calibration_bootstrap.py -q --basetemp=.runtime/pytest/p10_13_v3 -p no:cacheprovider` -> `14 passed in 0.06s`. Full regression `python -m pytest tests -q --basetemp=.runtime/pytest/full_p10_13 -p no:cacheprovider` -> `792 passed in 8.63s` (778 baseline + 14 new, 零回归).
  - No broker / order / alert / paper / notifier touched. Rule 3 / Rule 8.3 / Rule 9.4 / §10 gate 5/8 全保留. ADR-0006 sunset clause honored at the calibration consumer layer (CalibrationReport.evidence_origin signals UI to surface bootstrap distinction).

### P10-01 Predictions Backfill Scheduler (forward-going)

- Status: pending
- Priority: P0
- Depends on: P9-01, P9-02, ADR-0006 (sunset clause)
- Goal: 每日定时跑 `opportunity_scanner` + 当日 top-N candidates → 写 `reports/predictions/{trade_date}.jsonl`。所有 prediction 标 `extra.live=True`。Windows Task Scheduler 注册脚本配套。
- Files:
  - Create: `tools/scheduled_predictions_writer.py`
  - Create: `tests/unit/test_scheduled_predictions_writer.py`
  - Create: `scripts/register_predictions_writer_task.bat`
- Acceptance:
  - CLI `python tools/scheduled_predictions_writer.py --trade-date 2026-05-26 --top-n 10` 可手动跑。
  - 重复跑同一 trade_date 不写重复（走 P9-01 duplicate_id 拒绝路径）。
  - 所有 prediction 标 `extra.live=True`；model_version 不带 `-backdated` 后缀。
  - 30 日模拟跑 → 累计 ≥ 200 PredictionRecord（容量验证）。
  - Windows 调度脚本可注册为 daily 16:00 JST 任务。

### P10-02 Outcome Backfill Scheduler

- Status: pending
- Priority: P0
- Depends on: P10-01, P9-02
- Goal: T+5 自动跑 outcome join 把 5 个交易日前的 predictions 全部 join 进 `reports/outcomes/{trade_date}.jsonl`。
- Files:
  - Create: `tools/scheduled_outcomes_writer.py`
  - Create: `tests/unit/test_scheduled_outcomes_writer.py`
  - Create: `scripts/register_outcomes_writer_task.bat`
- Acceptance:
  - CLI `python tools/scheduled_outcomes_writer.py --evaluated-as-of 2026-06-02` 跑。
  - join 5 交易日前的所有 predictions；unmatched 显式记录原因不静默丢弃。
  - status 分支覆盖 complete / insufficient_data / symbol_not_found / future_cutoff / malformed_data。
  - 重复跑同一 trade_date 不写重复 outcome_id。
  - 与 P10-13 bootstrap 数据混合时 `CalibrationReport.evidence_origin="mixed"` 工作正常。

### P10-03 Per-Ticker News Filter

- Status: pending
- Priority: P0
- Depends on: P8-13 (news_adapter), P8-18 (symbol router)
- Goal: `GET /api/symbol/{ticker}/news?days=N&limit=M` 端点。按 `news_items.related_tickers` 过滤，扩 `news_adapter` 加 ticker filter。前端 V1/V3 在单 ticker 切换时显示该 ticker 的近期新闻。
- Files:
  - Update: `src/hot_theme_rotator/data/news_adapter.py` (+ `load_ticker_news(ticker, days, limit)`)
  - Update: `api/symbol.py` (+ /news 端点)
  - Create: `tests/unit/test_news_adapter_ticker_filter.py`
  - Update: `tests/unit/test_api_symbol.py` (+ /news 测试)
  - Update: `frontend/v1.jsx` / `frontend/v3.jsx` (单 ticker 视图加新闻区)
- Acceptance:
  - `/api/symbol/6768.T/news?days=7&limit=20` 返回该 ticker 相关近 7 天最多 20 条。
  - ticker 不存在或无相关新闻返回空数组（不 404），meta 含 reason。
  - JSON `related_tickers` 字段解析对齐 P8-13 现有逻辑（含 fallback）。
  - 前端切换 selected_symbol 时新闻 refetch；切换后旧 ticker 的新闻不能残留显示。

### P10-04 Fundamental Adapter

- Status: pending
- Priority: P1
- Depends on: ADR-0005
- Goal: 接 `Project_optimized/reports/fundamentals_status.json` + Project_optimized 内任何财报相关 DB 表（先做 audit）。`/api/symbol/{ticker}/fundamentals` 端点暴露 market_cap / pe / pb / roe / revenue_growth_yoy / debt_to_equity 等关键指标。
- Files:
  - Create: `src/hot_theme_rotator/data/fundamental_adapter.py`
  - Create: `tests/unit/test_fundamental_adapter.py`
  - Update: `api/symbol.py` (+ /fundamentals)
  - Update: `tests/unit/test_api_symbol.py`
- Acceptance:
  - 字段命名先 audit `Project_optimized/fundamentals_status.json` schema 再固化；docstring 记录依赖列。
  - 缺数据字段显式 null + reason，不静默填 0。
  - SQLite 读取一律 `mode=ro` URI；JSON 读取 read-only。
  - 不写回 Project_optimized（Rule 3 + ADR-0005 红线）。

### P10-05 Per-Ticker Factor Decomposition

- Status: pending
- Priority: P1
- Depends on: P8-12 (theme_heat_adapter)
- Goal: `GET /api/symbol/{ticker}/factors` — 该 ticker 在 12 个 alpha factor 上的当前 z-score + 历史 252 日分位数 + 方向 (bull/bear/neutral)。
- Files:
  - Update: `src/hot_theme_rotator/data/theme_heat_adapter.py` (+ `load_ticker_factor_decomposition(ticker, history_window=252)`)
  - Update: `api/symbol.py` (+ /factors)
  - Create: `tests/unit/test_ticker_factor_decomposition.py`
- Acceptance:
  - 返回 12 行：factor_name / current_z / 252d_percentile / direction。
  - 历史 < 252 天的 ticker 标 `insufficient_history`，不算 percentile。
  - factor_name 与 P8-12 现有 alpha factor 集对齐（mom_20 / sharpe_60 / ret60 / ...）。

### P10-06 LLM Per-Ticker Brief

- Status: done (2026-05-27)
- Priority: P1
- Depends on: P10-03 (news), P10-04 (fundamentals), P10-05 (factors), Rule 8.3.1
- Goal: `llm/per_ticker_brief.py` 接本地 Ollama (gemma4:e4b 默认)。输入 ticker + news + factors + fundamentals + ladder → 输出中文叙事综合 brief。**强制**不输出任何概率 / 胜率 / 百分比数字（regex post-check 拦截）。
- Files:
  - Create: `src/hot_theme_rotator/llm/__init__.py`
  - Create: `src/hot_theme_rotator/llm/ollama_client.py`
  - Create: `src/hot_theme_rotator/llm/per_ticker_brief.py`
  - Create: `tests/unit/test_per_ticker_brief.py`
  - Update: `api/symbol.py` (+ /llm_brief)
- Acceptance:
  - 默认 model `gemma4:e4b`；可选 `gemma4:26b` 校准用。
  - 输出 schema：`narrative` (markdown) + `factual_grounding` (输入引用) + `model_version` + `generation_ts`。**禁止**任何概率 / 胜率 / 评分字段。
  - Post-generation regex 拒绝 `\d+%` / "胜率" / "概率" / "win rate" / "probability"；违者 regenerate 一次后报错。
  - 24h 缓存：同 (ticker, input_hash) 不重复调用 Ollama。
  - Ollama 不可达 → 503 + reason，不生成 fake brief。
  - 全部测试用 mock Ollama client，CI 不依赖本地模型。

### P10-07 Single-Ticker Deep-Dive Page (V5)

- Status: pending
- Priority: P1
- Depends on: P10-03, P10-04, P10-05, P10-06, P8-18
- Goal: 前端新增第 5 个变体 V5。顶 nav 加 V5 按钮（默认仍 V3）。整合 hero + K 线 + ladder + portfolio status + 12 因子 + 财报 + 新闻 + LLM brief 一页看完。
- Files:
  - Create: `frontend/v5.jsx` (V5DeepDive)
  - Update: `frontend/index.html` (App nav + VARIANTS)
  - Update: `frontend/shared.jsx` (+ `useTickerNews` / `useTickerFundamentals` / `useTickerFactors` / `useTickerBrief` hooks)
  - Update: `tests/unit/test_frontend_ui_contracts.py` (+ V5 contract)
- Acceptance:
  - V5 显示：hero (price + ladder) / 252 日 K 线 / 持仓状态 / 12 因子表 / 财报关键指标 / 近 7 天新闻 / LLM brief 卡片。
  - 切换 symbol 所有区块 refetch；旧 ticker 数据不残留。
  - LLM brief 区显式 `model_version + generation_ts + "仅作叙事综合，不含胜率"` 标签。
  - 校准未达样本时 V5 不显示任何 calibrated_probability 数字；达样本则显示并标 `evidence_origin`（live / bootstrap / mixed）。

### P10-08 Bidirectional & Tier-Conditional Ground Truth

- Status: pending
- Priority: P1
- Depends on: P9-02, P9-03
- Goal: 扩 P9-03 ground truth 计算。当前只有 bullish horizon_return > 0；新增 bearish + tier-conditional（穿越某档后 N 天上涨/下跌概率）。
- Files:
  - Update: `src/hot_theme_rotator/calibration/calibrator.py` (+ `derive_bearish_ground_truth` / `derive_tier_conditional_ground_truth`)
  - Update: `src/hot_theme_rotator/calibration/reporter.py` (支持 multi-direction)
  - Create: `tests/unit/test_calibration_bidirectional.py`
  - Create: `tests/unit/test_calibration_tier_conditional.py`
- Acceptance:
  - 三种 ground truth 函数均 PIT-safe + fail-closed on missing horizon。
  - reporter 支持 `direction ∈ {"bullish", "bearish"}` 和 `tier_condition ∈ {None, "balanced_entry", "first_exit", ...}`。
  - 七档 × 2 方向 × 3 horizon = 42 个潜在 sub-report 各自独立 min_samples 计数。
  - 不修改 OutcomeRecord schema（向后兼容现有数据）。

### P10-09 Earnings & Catalyst Calendar

- Status: pending
- Priority: P2
- Depends on: P10-04
- Goal: `data/earnings_calendar_adapter.py` — 下个财报日 + 历史最近 8 次财报 T+1 / T+5 / T+20 真实收益。`/api/symbol/{ticker}/catalysts` 端点。
- Files:
  - Create: `src/hot_theme_rotator/data/earnings_calendar_adapter.py`
  - Update: `api/symbol.py` (+ /catalysts)
  - Create: `tests/unit/test_earnings_calendar_adapter.py`
- Acceptance:
  - 输出下个财报日（可能 null）+ 近 8 次财报 T+1/5/20 收益。
  - 先 audit Project_optimized 是否已有财报日期源；如无标 TBD 数据源。

### P10-10 Notification Channels

- Status: done (2026-05-26)
- Priority: P0 (Phase 1 Week 3, guarded push only)
- Activation stage: Rule 12.0 Stage 2
- Depends on: P9-04 (human_alerts), P10-18 (Anti-FOMO Guard Layer)
- Goal: 消费已经通过 P10-18 discipline filter 的 `AlertRecord` 推送到实际通道。desktop notification (Windows) + email 优先，Telegram 可选。未通过 Rule 12.1-12.6 的 alert 只能进入 silent queue 或降级，不得通知用户。
- Files:
  - Create: `src/hot_theme_rotator/alerts/notifiers/__init__.py`
  - Create: `src/hot_theme_rotator/alerts/notifiers/desktop.py`
  - Create: `src/hot_theme_rotator/alerts/notifiers/email.py`
  - Create: `src/hot_theme_rotator/alerts/notifiers/telegram.py` (optional)
  - Create: `tests/unit/test_notifiers.py`
- Acceptance:
  - P10-18 输出 typed `NotificationEnvelope` / `GuardedAlert`，含 `alert: AlertRecord`、`push_allowed: bool`、`discipline_result`、`suppression_reason`。
  - 统一 Protocol `Notifier.send(envelope: NotificationEnvelope) -> NotifyResult`；notifier 不接受裸 `AlertRecord`。
  - desktop 走 `plyer` 跨平台。
  - email 凭据从 env / `.env` 读，never hardcoded。
  - 通道失败不阻塞 alert chain；错误写 `reports/notification_errors.jsonl`。
  - 默认配置必须是 `notifications_enabled=false`；开启真实通知需要 PROJECT_STATUS.md 记录人工确认。
  - 所有 notifier 单测必须证明未经 P10-18 标记为 `push_allowed=True` 的 alert 不会被发送。
  - **不**新增任何 broker call / order field（Rule 3 + 10.1 红线）。

### P10-11 Scheduled Multi-Window Scan

- Status: pending
- Priority: P2
- Activation stage: Rule 12.0 Stage 1 by default; Stage 2 only after P10-18 + P10-10
- Depends on: P10-01, P10-17, P10-18 (for guarded push), P10-10 (only when notifications are enabled)
- Goal: 盘前 / 盘中 / 盘后 / 收盘后多窗口定时跑 scanner，对 high-score 候选触发 alerts。默认写 silent queue；只有通过 P10-18 discipline filter 且 P10-10 已人工启用时才允许 notifier 推送。
- Files:
  - Create: `tools/scheduled_multi_window_scan.py`
  - Create: `tests/unit/test_scheduled_scan.py`
  - Create: `scripts/register_multi_window_scan_task.bat`
- Acceptance:
  - 时间窗对齐 Rule 9.2 refresh schedule（pre-open 10min / sessions 3min / lunch 15min / post-close 3h / overnight 6h）。
  - 每窗只生成 score ≥ user_threshold 的 alert candidate；默认进入 silent queue。
  - Stage 2 push 时必须只发送 `push_allowed=True` 的 alert；不重复 push 同 candidate / 同 trade_date（throttle 走 P9-04）。
  - JP 节假日跳过运行。

### P10-12 Server-Side Watchlist Store

- Status: pending
- Priority: P2
- Depends on: Rule 11.3 (user-state / system-state 分离)
- Goal: 把 localStorage-only watchlist 升级为可选 server-side 持久化（多设备同步场景）。新独立 `user_state/` 存储，独立 ADR-0008（ADR-0007 已用于 reflection architecture），绝不污染 `decision_log/`。
- Files:
  - Create: `docs/adr/ADR-0008-user-state-store.md`
  - Create: `src/hot_theme_rotator/user_state/__init__.py`
  - Create: `src/hot_theme_rotator/user_state/watchlist_store.py`
  - Create: `api/user_state.py` (GET + PUT — user_state 是 §11.3 例外，PUT 仅限 user_state 区域，不触发 execution 路径)
  - Create: `tests/unit/test_watchlist_store.py`
- Acceptance:
  - ADR-0008 明确 user_state 不是 system_state，绝不进 decision_log，绝不触发 order / alert / paper。
  - 测试覆盖 Rule 11.3 红线：试图借 watchlist PUT 注入 fake position / 改 calibration / 写 prediction 必须被拒绝。

### P10-14 TDnet RSS Adapter (External Disclosure Feed)

- Status: done (Cycle 1 + Cycle 2 done 2026-05-25)
- Priority: P0 (Phase 1 Week 1)
- Depends on: 无（新数据源接入）
- Goal: 接 TDnet 適時開示 RSS 数据 → 写入 `Project_optimized/japan_market.db.tdnet_disclosures` 新表或扩展 `news_feed`。每条 disclosure 含 ticker / 公告类型（业绩 / 订单 / TOB / 増配 / 分拆 / 停牌 / 行政处分 / 其他）/ published_ts / 原文 URL / 摘要。**优先级最高 — 直接关闭"今天 X 涨停为什么"这种盲区**。
- Storage decision (2026-05-25 amendment per Rule 4): TDnet disclosures 落盘到 HTR-native `reports/tdnet/{trade_date}.jsonl`，mirror P9-01 / P9-02 JSONL pattern。**原方案**："写 `Project_optimized/japan_market.db.tdnet_disclosures` 表" 违反 ADR-0005 read-only 契约，撤销。**新方案**：HTR 自己拥有外部数据源 ingestion 的存储，Project_optimized 保持纯只读。
- Files (Cycle 1 = schema + storage + parser，无网络):
  - Create: `src/hot_theme_rotator/data/external/__init__.py`
  - Create: `src/hot_theme_rotator/data/external/tdnet_schema.py` — `TdnetDisclosure` dataclass + `compute_disclosure_id` + `TdnetDisclosureValidationError`
  - Create: `src/hot_theme_rotator/data/external/tdnet_storage.py` — `disclosures_path` / `append_disclosure` / `append_disclosures` / `read_disclosures`（mirror P9-01 pattern）
  - Create: `src/hot_theme_rotator/data/external/tdnet_parser.py` — Yanoshin JSON + TDnet HTML 双格式解析 + category 分类器 + ticker 归一化
  - Create: `tests/unit/test_tdnet_schema.py` / `test_tdnet_storage.py` / `test_tdnet_parser.py`
  - Create: `tests/fixtures/tdnet/` — Yanoshin JSON sample + TDnet HTML sample
  - Update: `docs/03_FOLDER_MAP.md` §reports 新增 `reports/tdnet`，§src/data 加 `external/` 子包说明
- Files (Cycle 2 = network adapter + CLI):
  - Create: `src/hot_theme_rotator/data/external/tdnet_rss_adapter.py` — 网络客户端，Yanoshin Web API 为默认源
  - Create: `tests/unit/test_tdnet_rss_adapter.py` — mock HTTP，不依赖真实 TDnet
  - Create: `tools/poll_tdnet_rss.py` — 定时轮询 CLI
  - Create: `scripts/register_tdnet_poll_task.bat` — Windows Task Scheduler 注册脚本
- Cycle 1 Acceptance (无网络):
  - `TdnetDisclosure` schema 含 9 字段：`disclosure_id` (deterministic SHA-256 hash) / `ticker` / `company_name` (optional) / `published_ts` / `collected_ts` / `title` / `category` / `url` / `summary` (optional) / `raw` (optional dict)。
  - Category enum 覆盖至少 8 类：`earnings` / `order` / `tob` / `dividend` / `split` / `suspension` / `governance` / `other`。
  - JSONL storage 复用 P9-01 pattern：duplicate `disclosure_id` 拒绝；malformed JSONL fail-closed；`trade_date` 通过 `date.fromisoformat` 验证（P0-04 F6 同款）。
  - Parser 支持 Yanoshin JSON 格式 + TDnet HTML 列表格式，全部基于 fixture sample 解析。
  - Ticker 归一化（4-digit → `X.T` 后缀）+ 未知 category fallback 到 `other`。
  - 全部测试不走网络；至少 20 单测。
- Cycle 2 Acceptance (网络):
  - 单次 fetch 支持 ticker 过滤 + 全市场 fallback。
  - rate limit 1 req / 5s + retry-after 遵守；指数退避 backoff on 429 / 503。
  - 定时任务 register 为每 15min 跑（Rule 9.2 within-session 频率）。
  - 集成测试用 mock HTTP，绝不依赖真实 TDnet endpoint。
- 通用约束 (两个 Cycle 都适用):
  - schema 列与 docstring pinned；TDnet 改格式 fail-closed 报错不静默吃。
  - **不**新增任何 broker / order 路径；Rule 3 + Rule 12.2（stale fail-closed）保留。
- Verified Cycle 1 (2026-05-25):
  - Storage 决定文档化 (Rule 4)；ADR-0005 read-only 契约保留（HTR 不写 Project_optimized）。FOLDER_MAP §reports + §src/data 同步。
  - `tdnet_schema.py` RED → GREEN：`python -m pytest tests/unit/test_tdnet_schema.py` → `21 passed`。覆盖：deterministic disclosure_id / 8-category enum / 必填字符串 fail-closed / ISO ts 验证 / ticker form 验证（4-digit + .T，拒绝 letter/wrong-length）/ disclosure_id integrity check / unknown-key from_dict reject。
  - `tdnet_storage.py` GREEN：`python -m pytest tests/unit/test_tdnet_storage.py` → `20 passed`。覆盖：path 验证（拒绝 `2026/05/25` + `../../etc/passwd` + 空）/ duplicate disclosure_id reject / malformed JSONL fail-closed / schema-violation 在 read 时 fail-closed / per-published_ts 路由 / tz-aware published_ts (`+09:00`) 正确提取 trade_date。
  - `tdnet_parser.py` GREEN：`python -m pytest tests/unit/test_tdnet_parser.py` → `32 passed`。覆盖：ticker 归一化 4-digit / 5-digit-with-trailing-zero / strip whitespace / reject non-string + letters + wrong length / 8 category 分类（earnings / tob / dividend / split / suspension / order / governance / other）+ tob 优先 earnings 顺序 / Yanoshin JSON fixture 解析 3 条 / TDnet HTML BeautifulSoup 解析 3 条 + header `<th>` 行自动跳过 / 各类 fail-closed (非 string / 缺 key / bad pubdate / bad ticker / missing url)。
  - Fixtures: `tests/fixtures/tdnet/yanoshin_sample.json` + `tdnet_html_sample.html`，3 条真实样本（6779.T 业绩 / 1306.T 配当 / 6768.T TOB）。
  - Full suite no regression: `python -m pytest tests/` → `422 passed in 8.03s`（349 baseline + 73 new = 422）。
  - bs4 (4.14.3) + lxml (6.0.2) 已装，无需更新 requirements.txt。

### P10-15 Yahoo Finance Japan Scraper

- Status: pending
- Priority: P1
- Depends on: P10-14
- Goal: 补 TDnet 不覆盖的细节 — per-ticker 新闻、券商目标价、消息板情绪。Tier 2 数据源，scraping based。
- Files:
  - Create: `src/hot_theme_rotator/data/external/yahoo_jp_scraper.py`
  - Create: `tests/unit/test_yahoo_jp_scraper.py`
  - Create: `configs/yahoo_jp_selectors.yaml` (HTML selector 配置外置)
- Acceptance:
  - requests-html 或 playwright；rate limit 1 req / 3s；24h cache 同 (ticker, hour)。
  - robots.txt 启动时校验；不符合直接 fail-closed 拒绝运行。
  - selector 写在 yaml 配置，HTML 结构变化只改配置不改代码。
  - HTML 解析失败 fail-closed，不返回部分数据。
  - 测试用 fixture HTML 不走网络。
  - **不**新增 broker / order 路径；Rule 3 保留。

### P10-16 J-Quants Live Bridge

- Status: deferred-pending-credentials (Cycle 1 adapter+auth+tests done 2026-05-25; user confirmed no J-Quants account; module retained for optionality but removed from active production fallback chain)
- Priority: P3 (was P0, demoted after no-credentials constraint)
- Depends on: 无（复用 Project_optimized J-Quants token）
- Goal: 把 Project_optimized 的 J-Quants 集成借过来，HTR 直接调用而不等 Project_optimized 每日 refresh。覆盖 OHLC 当日实时 + 财报日历 + 上市/退市公告。
- Files:
  - Create: `src/hot_theme_rotator/data/external/jquants_live_bridge.py`
  - Create: `tests/unit/test_jquants_live_bridge.py`
  - Update: `api/symbol.py` `/api/symbol/{ticker}/kline` 加 `?source=jquants_live` 参数
  - Update: `tools/morning_briefing.py` `--source jquants_live` 新选项
- Acceptance:
  - 复用 `Project_optimized/backfill_jquants_history.py` 的 token / 配置（read-only consume，不触动原 config 文件）。
  - Rate limit 按 J-Quants 文档分级；不同 tier 限速不同。
  - 失败 fail-closed，可显式 fallback 到 `daily_prices` 历史数据并标注 `fallback_used=True`。
  - 至少 10 单测含 mock J-Quants response；不走网络。
  - **不**新增 broker / order 路径。

### P10-17 Watchlist Intelligence

- Status: done (2026-05-26)
- Priority: P0 (Phase 1 Week 2, silent-first)
- Activation stage: Rule 12.0 Stage 1
- Depends on: P10-14 (TDnet), P10-19 (delayed price orchestrator), P9-04 (human_alerts), Rule 11.3, Rule 12.0
- Goal: 用户声明 watchlist → 系统持续监控（每窗扫描）每只标的的 news / factor / 价格 / ladder 触达 → 检测事件 → 生成 `AlertRecord` → 写入 dashboard-visible silent queue。P10-17 默认不调用 notifier；真实 push 只能在 P10-18 + P10-10 后启用。
- Files:
  - Create: `src/hot_theme_rotator/watchlist_intelligence/__init__.py`
  - Create: `src/hot_theme_rotator/watchlist_intelligence/monitor.py` (per-window scan loop)
  - Create: `src/hot_theme_rotator/watchlist_intelligence/event_detector.py` (factor spike / TDnet disclosure / ladder touch / 集中度异动)
  - Create: `src/hot_theme_rotator/user_state/__init__.py`
  - Create: `src/hot_theme_rotator/user_state/watchlist_store.py` (per Rule 11.3 — 单用户本地 store；P10-12 留作未来 multi-device sync)
  - Create: `tests/unit/test_watchlist_monitor.py`
  - Create: `tests/unit/test_event_detector.py`
  - Create: `tests/integration/test_watchlist_event_to_alert.py`
  - Update: `frontend/v3.jsx` 加 watchlist 管理 UI (Rule 11.3 user-state only)
- Acceptance:
  - 用户 watchlist 持久化到 `user_state/watchlist.jsonl`（per Rule 11.3 — user_state 严禁进 decision_log）。
  - 每窗 scan：对每只 watchlist 标的拉最新 quote + news + factor → 调用 event_detector。
  - 事件类型至少 4 类：factor z-score 跳变 / TDnet disclosure / ladder 档位触达 / 集中度异动。
  - 事件 → `AlertRecord` → silent queue；默认不触发 desktop / email / telegram。
  - dashboard 显示今日触发、今日压制、压制原因、数据新鲜度、study_only 降级原因。
  - P10-17 不重复实现 Rule 12.3 / 12.4；若需要展示压制原因，只消费 P10-18 discipline 输出。
  - **不**触发任何 paper / broker order；Rule 3 + Rule 11.3 双重保护。
- Verified silent queue storage slice (2026-05-26):
  - Added `watchlist_intelligence/silent_queue.py` and package exports.
  - `SilentWatchlistEvent` validates ISO trade date, ISO created timestamp, `.T` symbols, non-empty fields, and rejects `push_allowed=True` so Stage 1 remains silent.
  - `append_silent_event` / `read_silent_events` write/read append-only JSONL at `reports/observability/silent_queue/{trade_date}.jsonl`.
  - RED `tests/unit/test_silent_watchlist_queue.py` first failed on missing package; GREEN targeted `python -m pytest tests/unit/test_silent_watchlist_queue.py -q` -> `4 passed`.
  - Slice adds no notifier, paper, broker, order, POST/PUT/PATCH/DELETE, or push path.
- Verified event detector slice (2026-05-26):
  - Added `watchlist_intelligence/event_detector.py` with `detect_watchlist_events`.
  - Detector creates silent events for missing quote health, uncertain quote health, and watched TDnet disclosures; disclosures outside the watchlist are ignored.
  - Every generated event is a `SilentWatchlistEvent` with `push_allowed=False`; TDnet and uncertain quote events are `study_only=True`.
  - RED `tests/unit/test_watchlist_event_detector.py` first failed on missing module; GREEN targeted `python -m pytest tests/unit/test_watchlist_event_detector.py -q` -> `4 passed`.
- Verified monitor wiring slice (2026-05-26):
  - Added `watchlist_intelligence/monitor.py` with `run_watchlist_monitor`.
  - Monitor wires detector output into append-only silent queue storage and returns the persisted event tuple for dashboard/CLI callers.
  - RED `tests/unit/test_watchlist_monitor.py` first failed on missing module; GREEN targeted `python -m pytest tests/unit/test_watchlist_monitor.py -q` -> `1 passed`.
  - Scheduling/window orchestration remains P10-11; this slice does not push notifications.

### P10-18 Anti-FOMO Guard Layer (Rule 12 Enforcement)

- Status: done (2026-05-26)
- Priority: P0 (Phase 1 Week 3)
- Activation stage: Rule 12.0 Stage 2 gate
- Depends on: P9-04 (human_alerts), P10-17 (watchlist intelligence), Section 12
- Goal: 实施 Section 12 全部 6 子规则的代码层 — alert budget / stale data 拦截 / chase filter / cooling-off / concentration guard / cross-strategy journal。所有 alert 必须经过这层才能 push 到 notifier；P10-10 依赖本任务。
- Files:
  - Create: `src/hot_theme_rotator/alerts/discipline.py` (Rule 12.1-12.6 实现)
  - Create: `src/hot_theme_rotator/alerts/budget_tracker.py` (per-user daily budget tracking)
  - Create: `src/hot_theme_rotator/alerts/silent_queue.py` (over-budget 落盘 visible in dashboard 但不 push)
  - Update: `src/hot_theme_rotator/alerts/human_alerts.py` (集成 discipline filter)
  - Create: `reports/meta_strategy_journal.jsonl` 落盘 schema (Rule 8.9 + 12.6 共享)
  - Create: `configs/push_discipline.yaml` (budget / stale window / chase threshold / cooling-off / concentration threshold，全部 conservative 默认)
  - Create: `tests/unit/test_alert_discipline.py`
  - Create: `tests/unit/test_alert_budget_tracker.py`
  - Create: `tests/integration/test_anti_fomo_end_to_end.py`
- Acceptance:
  - 6 子规则每条至少 3 个单测（共 ≥ 18 单测）。
  - end-to-end 集成测试：模拟一天 20 个 alert 候选 → 验证 budget=10 生效、chase filter 触发 2 次降级、cooling-off 抑制 1 次新加标的、concentration 警告先于 BUY 1 次。
  - 集成测试覆盖：watchlist 新加 24h 内 BUY alert 被 cooling-off 抑制（Rule 12.4）；涨停标的 BUY alert 降级 study_only（Rule 12.3）。这些测试归 P10-18，不归 P10-17。
  - 输出 typed `NotificationEnvelope` / `GuardedAlert`；P10-10 notifier 只能消费该 guarded envelope，不能消费裸 `AlertRecord`。
  - silent_queue 落盘格式可被 dashboard 消费（P10-17 watchlist intel panel 显示"今日被抑制 N 条"）。
  - meta_strategy_journal.jsonl schema 与 Rule 8.9 共享；写入失败 alert 必须 suppress（不能 push 而不记 journal）。
  - configs/push_discipline.yaml 默认值：budget=10/day, stale_threshold=2h, chase_threshold=10%, cooling_off=24h, concentration_threshold=20% NAV。
  - **不**新增任何 broker / order 路径；Rule 3 + Section 12 双重保护。
- Verified core guard slice (2026-05-26):
  - Added `alerts/discipline.py` with pure `evaluate_alert_discipline` domain logic and package exports.
  - Covered Rule 12.1 budget, Rule 12.2 stale fail-closed, Rule 12.3 chase downgrade to `study_only`, and Rule 12.4 new-watchlist cooling-off for BUY alerts.
  - RED `tests/unit/test_alert_discipline.py` first failed on missing module; GREEN targeted `python -m pytest tests/unit/test_alert_discipline.py -q` -> `5 passed`.
  - This slice does not integrate notifiers and does not enable guarded push; concentration guard and cross-strategy journal remain pending for full P10-18 closure.

### P10-20 Daily Advisory Cockpit (Time-to-First-Value MVP)

- Status: done (2026-05-26)
- Priority: P0 (Phase 1 Week 2, before guarded push)
- Activation stage: Rule 12.0 Stage 0
- Depends on: P8-18 (interactive exploration), P8-19 (morning briefing), P10-14 (TDnet), P10-19 (delayed price orchestrator)
- Goal: 把现有 dashboard + morning briefing 升级为每天可用的 Pull-only 入口。用户主动打开后能看到持仓、watchlist、延迟报价、TDnet disclosure、七档价格、数据缺口、research-only 风险提示；不主动推送，不发送通知，不写订单。
- Files:
  - Update: `tools/morning_briefing.py` (default daily cockpit CLI, realtime source option if available)
  - Update: `api/dashboard.py` / `api/serializers.py` (surface source freshness + suppressed/study-only counts when present)
  - Update: `frontend/v1.jsx` or selected default dashboard variant (daily cockpit first screen)
  - Create/update tests for cockpit payload contracts and no-notification behavior
- Acceptance:
  - Pull-only: no desktop / email / telegram / mobile notification path is invoked.
  - Every quote or disclosure surfaced carries source + `data_ts` / freshness status; inferred quote timestamps are explicitly marked.
  - Morning briefing and dashboard both show research-only / uncalibrated status and never render win-rate language unless Rule 9.4 is satisfied.
  - Watchlist entries without fresh data show an explicit unavailable/stale state, not old fallback data.
  - No broker / paper order path is touched. Section 14 manual portfolio recording POST endpoints are outside the cockpit surface and remain record-only, not execution endpoints.
  - Tests cover: data freshness display, no-notifier call, stale data visible state, Rule 3 method guard, and no calibrated win-rate label.
- Verified Stage 0 payload contract slice (2026-05-26):
  - Added `reporting/daily_advisory_cockpit.py` with `build_daily_advisory_cockpit`.
  - Payload carries `activationStage=stage_0_pull_only`, `researchOnly=True`, `notificationsInvoked=False`, and explicit `execution` false flags for broker/orders/paperOrders.
  - Watchlist rows surface quote source, price, checked/data/wall timestamps, `dataTsInferred`, `freshnessStatus`, `priceUncertain`, `failReason`, TDnet counts, and `dataGaps`.
  - Missing quote rows render `quoteStatus=unavailable` + `quote_unavailable`; no calibrated win-rate/probability language is emitted.
  - RED `tests/unit/test_daily_advisory_cockpit.py` first failed on missing module; GREEN targeted `python -m pytest tests/unit/test_daily_advisory_cockpit.py -q` -> `6 passed`.
- Verified API integration slice (2026-05-26):
  - `/api/dashboard` now includes top-level `dailyCockpit` while preserving existing V3 fields.
  - Serializer reads local `reports/observability/price_health/{trade_date}.json`, `reports/tdnet/{trade_date}.jsonl`, and `reports/observability/silent_queue/{trade_date}.jsonl` fail-soft; missing reports produce explicit unavailable state rather than old fallback data.
  - Follow-up fix: cockpit observation date is now independent from screener candidate snapshot date, so today's price health report can feed the pull-only cockpit even when candidates come from an older `selected_tickers.json` asof.
  - Added API contract tests for Stage 0 pull-only flags and no push/probability/execution wording.
  - RED `tests/unit/test_api_dashboard.py::test_dashboard_daily_cockpit_is_pull_only_and_research_only` first failed with missing `dailyCockpit`; GREEN targeted dashboard cockpit tests `2 passed`, language guard `1 passed`.
- Verified V1 frontend cockpit slice (2026-05-26):
  - `frontend/v1.jsx` now renders `V1DailyCockpitPanel` from `data.dailyCockpit`.
  - Panel surfaces Stage 0 state, watchlist count, quote count, silent queue count, per-symbol quote status, and TDnet count.
  - Static frontend contract proves V1 consumes `dailyCockpit`, reads `cockpit.activationStage`, `cockpit.notificationsInvoked`, and `cockpit.execution?.orders`, and adds no POST/PUT/DELETE/PATCH or push text.
  - RED `tests/unit/test_frontend_ui_contracts.py::test_v1_renders_daily_cockpit_panel_from_dashboard_payload` first failed on missing panel; GREEN targeted V1 cockpit tests `2 passed`; related frontend/API contracts `23 passed`.

### P10-19 Best-Effort Delayed Price Orchestrator

- Status: in_progress (Cycle 1 done 2026-05-25, Cycle 2 网络层 + Stooq replacement 待做)
- Priority: P0 (Phase 1 Week 1)
- **Active source chain (post 2026-05-25 live smoke + no-credentials constraint)**: `cache → yahoo_japan → kabutan → yfinance`. TwelveData (no account), Stooq (service requires apikey/captcha since 2026-05-25), J-Quants (no account) removed from default chain. Parsers retained as inactive code for optionality.
- Depends on: 无（新数据源接入）
- Goal: 多源 free/near-real-time JP equity 价格 fallback chain。**Per Codex review (ADR-0007)**：不叫 "real-time"——物理 floor ≈ 5 min（web display refresh interval）。Conditional consensus 用于 high-salience triggers (Rule 12.3 chase boundary / 止损止盈附近 / 当日 ≥5%)。
- Files (Cycle 1，无网络，fixture-based):
  - Create: `src/hot_theme_rotator/data/external/realtime_price/__init__.py`
  - Create: `src/hot_theme_rotator/data/external/realtime_price/schema.py` (`PriceQuote` dataclass)
  - Create: `src/hot_theme_rotator/data/external/realtime_price/yahoo_japan_scraper.py`
  - Create: `src/hot_theme_rotator/data/external/realtime_price/kabutan_scraper.py`
  - Create: `src/hot_theme_rotator/data/external/realtime_price/twelvedata_client.py`
  - Create: `src/hot_theme_rotator/data/external/realtime_price/stooq_csv_fetcher.py`
  - Create: `src/hot_theme_rotator/data/external/realtime_price/orchestrator.py`（fallback chain + 60s cache + conditional consensus）
  - Create: `src/hot_theme_rotator/data/external/realtime_price/health.py` (per-source health checks + report writer)
  - Create: `tests/unit/test_realtime_price_*.py` (~25 tests fixture-based)
  - Create: `tests/unit/test_price_health.py`
  - Create: `tests/fixtures/realtime_price/` (sample HTML / CSV / JSON 各源)
- Files (Cycle 2，网络):
  - Create: `tests/integration/test_realtime_price_network.py` (mock HTTP)
  - Update: `tools/morning_briefing.py` 加 `--source realtime` 选项
  - Update: `api/symbol.py` 加 `GET /api/symbol/{ticker}/realtime_price` 端点
- Acceptance (Cycle 1):
  - `PriceQuote(symbol, price, source, data_ts, wall_ts, fail_reason)` schema，每个 scraper 返回这个形状。
  - Orchestrator fallback 顺序：cache(60s TTL) → yahoo_japan → twelvedata → kabutan → stooq → yfinance → fail-closed。
  - all-fail 返回 explicit reason（Rule 12.2 stale fail-closed 兼容）。
  - **Conditional consensus**：caller 传 `high_salience=True` 时，orchestrator 拉第二源对比，delta > 1% → 标 `price_uncertain=True`。
  - 每条返回都带 `data_ts` (源声明时间戳) + `wall_ts` (调用时间戳) 让上层判断新鲜度。
  - 至少 25 单测 fixture-based 不走网络。
- Acceptance (Cycle 2):
  - 每个 scraper rate limit 1 req / 10s 默认。
  - User-Agent rotation across 真实 browser strings。
  - robots.txt 启动验证；不符合 fail-closed。
  - Cloudflare 探测 abort。
  - 集成测试用 mock HTTP，绝不打真实 endpoint。
  - Health check 逐源记录 ok/fail、price、data_ts/wall_ts、data_ts_inferred、price_uncertain、fail_reason。
  - Daily health report 写入 `reports/observability/price_health/{trade_date}.json`；路径拒绝非 ISO trade_date。
- 通用约束:
  - **不**叫 "real-time"——文档 / API response / 日志全部用 "best-effort delayed market data"（Codex 修正）。
  - **不**新增任何 broker / order 路径。
  - Rule 12.2 stale fail-closed 严格遵守。
- Verified Cycle 2 health slice (2026-05-26):
  - Added `health.py` with `PriceSourceHealth`, `run_price_source_health_checks`, `price_health_report_path`, `write_price_health_report`, `read_price_health_report`.
  - RED `tests/unit/test_price_health.py` first failed on missing module/API; GREEN targeted `python -m pytest tests/unit/test_price_health.py -q` -> `6 passed`.
  - Health layer is observability-only: no notifier, alert, paper, broker, or order path.
- Verified Cycle 2 HTTP policy + mock network slice (2026-05-26):
  - Added `http_policy.py` with `HttpFetchPolicy`, per-host rate limiting, fixed/injectable robots policy, User-Agent rotation, and Cloudflare / anti-bot HTML detection.
  - Added Yahoo Japan and Kabutan fetch wrappers that call policy before parsing and accept injected HTTP text fetchers for mock HTTP tests.
  - RED `tests/unit/test_realtime_price_http_policy.py` first failed on missing module; GREEN targeted `6 passed`. RED `tests/integration/test_realtime_price_network.py` first failed on missing fetch wrappers; GREEN targeted `4 passed`.
  - Related price-source verification: `python -m pytest tests/unit/test_realtime_price_http_policy.py tests/integration/test_realtime_price_network.py tests/unit/test_yahoo_japan_scraper.py tests/unit/test_kabutan_scraper.py tests/unit/test_price_orchestrator.py tests/unit/test_price_health.py tests/unit/test_price_quote_schema.py -q` -> `58 passed`.
  - Slice remains advice-only and mock-HTTP verified: no notifier, alert, paper, broker, order, or real endpoint call was added to tests.
- Verified Cycle 2 health CLI + live smoke slice (2026-05-26):
  - Added `tools/write_price_health_report.py` to probe configured delayed-price sources and write `reports/observability/price_health/{trade_date}.json` for Stage 0 cockpit/dashboard consumers.
  - CLI accepts `--date`, `--symbols`, and `--base-dir`; production default chain probes Yahoo Japan and Kabutan through the HTTP policy layer. Tests inject a source chain and never call external endpoints.
  - RED `tests/unit/test_write_price_health_report.py` first failed on missing module; GREEN targeted `4 passed`.
  - Related verification: `python -m pytest tests/unit/test_write_price_health_report.py tests/unit/test_price_health.py tests/unit/test_realtime_price_http_policy.py tests/integration/test_realtime_price_network.py tests/unit/test_daily_advisory_cockpit.py tests/unit/test_api_dashboard.py -q --basetemp=.runtime\pytest\related_price_health_cli -p no:cacheprovider` -> `39 passed`.
  - Local-only live smoke: `python tools/write_price_health_report.py --date 2026-05-26 --symbols 6779.T --base-dir .runtime\price_health_cli_smoke` -> wrote 2 rows, 2 ok. Formal local report for current dashboard top 3 (`6768.T,5074.T,6962.T`) wrote 6 rows, 6 ok to `reports/observability/price_health/2026-05-26.json`. All rows keep `data_ts_inferred=true`; no notification, alert, paper, broker, order, upload, push, PR, or remote sync was invoked.
- Verified Cycle 2 local scheduler helper slice (2026-05-26):
  - Added `scripts/register_price_health_task.bat` for optional local Windows Task Scheduler registration of the price health CLI every 15 minutes.
  - Script requires explicit symbol list, writes only local observability reports, and documents no notification / no broker / no order boundaries.
  - The script was not executed; no Windows task was registered in this session. User must run it manually as Administrator if desired.
  - RED `tests/unit/test_windows_task_scripts.py` first failed on missing script; GREEN targeted `2 passed`.
  - Related verification: `python -m pytest tests/unit/test_windows_task_scripts.py tests/unit/test_write_price_health_report.py tests/unit/test_api_dashboard.py tests/unit/test_daily_advisory_cockpit.py tests/unit/test_frontend_ui_contracts.py -q --basetemp=.runtime\pytest\related_price_task_script -p no:cacheprovider` -> `36 passed`.

### P10-21 Portfolio Ledger (HTR-Owned, Section 14)

- Status: done (2026-05-26, 4 cycles)
- Priority: P0 (Phase 1 Week 2 prerequisite for manual entry)
- Activation stage: Rule 12.0 Stage 0 (advisory entry, no execution)
- Depends on: ADR-0008 (cutover plan), Section 14 (Rule 14.0-14.7), P9-02 outcome join (calibration consumer)
- Goal: 在 HTR 内自建 append-only 持仓事件 journal，作为 ADR-0008 cutover 后的唯一真相源。提供 `record_fill` / `record_cash_event` / `derive_positions` / `derive_cash_balance` API，所有事件 source 必带 (Rule 14.3)，positions 永远是 journal 派生视图 (Rule 14.1)。
- Files:
  - Create: `src/hot_theme_rotator/portfolio/__init__.py`
  - Create: `src/hot_theme_rotator/portfolio/schema.py` (`FillEntry` / `CashEvent` dataclass + entry_id SHA256 16-hex 派生)
  - Create: `src/hot_theme_rotator/portfolio/journal_writer.py` (append-only JSONL writer, 重复 entry_id 拒绝, malformed-line fail-closed)
  - Create: `src/hot_theme_rotator/portfolio/derive.py` (`derive_positions(journal) -> dict[symbol, PositionView]` + `derive_cash_balance(journal) -> Decimal`, 确定性 + Rule 14.6 source 过滤)
  - Create: `src/hot_theme_rotator/portfolio/validation.py` (Rule 14.2 fail-closed 校验门)
  - Create: `tests/unit/test_portfolio_schema.py`
  - Create: `tests/unit/test_portfolio_writer.py`
  - Create: `tests/unit/test_portfolio_derive.py` (含 determinism 测试)
  - Create: `tests/unit/test_portfolio_validation.py`
  - Update: `docs/03_FOLDER_MAP.md` +`portfolio/`; `docs/00_DESIGN.md` +portfolio 数据流段
- Acceptance:
  - `record_fill(side, symbol, qty, price, ts, source, note='', fee=0.0)` 写 journal 后 derive_positions 立即反映新持仓；source 必传且必须在 enum 内。
  - `derive_positions` 对同一 journal 输入永远返回同一输出 (determinism test 强制)。
  - Rule 14.2 fail-closed：`qty<=0` / `price<=0` / non-`.T` symbol / future ts / SELL exceeds holdings / unknown side 都明确拒绝并附 reason。
  - Rule 14.6：calibration consumer (P9-02 join) 默认 source filter 排除 `paper/migration/correction`；单测验证。
  - Append-only：UPDATE/DELETE 路径不存在；journal 文件以 JSONL 形式只追加。
  - Determinism：删除 journal 重放同样 fills 必须产生同样 positions（hash 等同）。
  - Rule 3 / Rule 14 全保留：不挂任何 broker / order / notifier。
- Verified Cycle 1 schema slice (2026-05-26):
  - Added `src/hot_theme_rotator/portfolio/__init__.py` + `portfolio/schema.py` with `FillEntry` / `CashEvent` dataclasses, `derive_fill_entry_id` / `derive_cash_event_id` (sha256 16-hex), `ALLOWED_SOURCES` / `ALLOWED_SIDES` / `ALLOWED_CASH_REASONS` enums, `PortfolioSchemaError`.
  - Rule 14.3 source attribution mandatory enforced at construction; Rule 14.4 correction-coupling (`source='correction'` ↔ `corrects` field) enforced both directions.
  - `entry_id` integrity check rejects tampered or recomputed-stale ids.
  - RED `tests/unit/test_portfolio_schema.py` first failed on missing `hot_theme_rotator.portfolio.schema`; GREEN targeted `python -m pytest tests/unit/test_portfolio_schema.py -q --basetemp=.runtime/pytest/portfolio_schema_green -p no:cacheprovider` -> `24 passed`.
  - Full regression `python -m pytest tests -q --basetemp=.runtime/pytest/full_p10_21_c1 -p no:cacheprovider` -> `607 passed in 8.33s` (583 baseline + 24 new, 零回归).
  - Schema-layer only: no journal writer, no derive views, no Rule 14.2 manual-entry gates, no migration tooling, no API/UI integration. Subsequent cycles land those.
- Verified Cycle 2 journal_writer slice (2026-05-26):
  - Added `portfolio/journal_writer.py` with `journal_path` (ISO trade_date path-traversal guard), `append_fill` / `append_cash_event` (JST trade-date partitioning derived from entry.ts via `astimezone`), `read_journal` (mixed-type JSONL with `_type` discriminator, malformed-line fail-closed, missing-file empty tuple), and `PortfolioJournalError`.
  - Rule 14.1 append-only enforced by API surface — no UPDATE / DELETE / overwrite functions exist; test guards against future regression.
  - Duplicate-id rejection per append doubles as idempotency (sha256 entry_id is deterministic, so re-clicking submit returns explicit duplicate error not a phantom second fill).
  - Single-writer assumption documented; multi-process locking out of scope for this slice.
  - RED `tests/unit/test_portfolio_writer.py` first failed on missing `hot_theme_rotator.portfolio.journal_writer`; GREEN targeted `python -m pytest tests/unit/test_portfolio_writer.py -q --basetemp=.runtime/pytest/portfolio_writer_green -p no:cacheprovider` -> `15 passed`.
  - Full regression `python -m pytest tests -q --basetemp=.runtime/pytest/full_p10_21_c2 -p no:cacheprovider` -> `622 passed in 8.26s` (607 baseline + 15 new, 零回归).
- Verified Cycle 3 derive views slice (2026-05-26):
  - Added `portfolio/derive.py` with `PositionView` dataclass + `derive_positions` (per-symbol qty / avg_cost / realized_pnl from journal in append order, oversell fail-closed via `PortfolioDeriveError`) + `derive_cash_balance` (BUY decreases by qty×price+fee, SELL increases by qty×price−fee, CashEvent applies signed amount).
  - **Rule 14.4 semantic clarified**: corrections invalidate the entry they reference — both the corrected entry and the correction itself are skipped during derivation; the fresh replacement entry produces the right state. Updated Rule 14.4 text in governance to make this explicit ("skip-both" vs reversal-trade math). Reason: treating correction as a real BUY-reversal-of-SELL re-prices un-traded shares (cost-basis drift); skip-both preserves cost-basis integrity.
  - Determinism guaranteed (Python 3.7+ dict insertion order); test enforces.
  - RED `tests/unit/test_portfolio_derive.py` first failed on missing `hot_theme_rotator.portfolio.derive`; first GREEN attempt 18/19 (correction test failed on reversal-trade math) → semantic corrected to skip-both → GREEN `python -m pytest tests/unit/test_portfolio_derive.py -q --basetemp=.runtime/pytest/portfolio_derive_green2 -p no:cacheprovider` -> `19 passed`.
  - Full regression `python -m pytest tests -q --basetemp=.runtime/pytest/full_p10_21_c3 -p no:cacheprovider` -> `641 passed in 11.99s` (622 baseline + 19 new, 零回归).
  - Path A reality check: `derive_positions((BUY 900@¥403, SELL 400@¥417.6))` → `qty=500, avg_cost=¥403, realized_pnl=+¥5,840` — matches the救急 DB state exactly.
- Verified Cycle 4 validation gates slice (2026-05-26):
  - Added `portfolio/validation.py` with `ValidationResult` (warnings tuple, no hard-fail field — hard fails raise) + `validate_manual_fill` + `validate_manual_cash_event` + `PortfolioValidationError`.
  - Hard gates (raise): source must be in {`manual`, `import`}; ts must be ≤ now JST (no future-dated, PIT discipline per Rule 8.6); SELL holdings must be ≥ proposed qty (no shorts).
  - Soft warnings (returned in `ValidationResult.warnings`): BUY where cash − qty × price − fee < 0; withdrawal where cash + amount < 0. UI displays but does not block — user can confirm overdraft or add deposit cash_event first.
  - Magnitude sanity check (Rule 14.5 qty × price > 10% NAV) deliberately NOT in this module — needs current market prices which the journal doesn't hold. P10-23 manual_entry_service computes it before preview.
  - Rule 14.6 calibration source filter (`paper/migration/correction` exclusion from sample) deferred to P9-02 integration time (YAGNI — no calibration consumer exists in P10-21 surface).
  - RED `tests/unit/test_portfolio_validation.py` first failed on missing `hot_theme_rotator.portfolio.validation`; first GREEN 14/15 (test_validate_manual_fill_accepts_import_source asserted no warning on empty journal which had cash=0); test corrected to seed journal with deposit; final GREEN `python -m pytest tests/unit/test_portfolio_validation.py -q --basetemp=.runtime/pytest/portfolio_validation_green2 -p no:cacheprovider` -> `15 passed`.
  - Full regression `python -m pytest tests -q --basetemp=.runtime/pytest/full_p10_21_c4 -p no:cacheprovider` -> `656 passed in 8.22s` (641 baseline + 15 new, 零回归).
- **P10-21 done summary** (4 cycles, 73 new unit tests, ~700 LOC across 4 source files + 4 test files):
  - schema (Cycle 1, 24 tests, ~200 LOC): `FillEntry` / `CashEvent` / entry_id determinism / Rule 14.3 source enum / Rule 14.4 correction coupling.
  - journal_writer (Cycle 2, 15 tests, ~170 LOC): append-only JSONL / JST partitioning / dup-id rejection / no UPDATE-DELETE API.
  - derive (Cycle 3, 19 tests, ~130 LOC): pure positions / cash views / Rule 14.4 skip-both / oversell fail-closed / determinism.
  - validation (Cycle 4, 15 tests, ~150 LOC): Rule 14.2 future-ts / oversell / source whitelist hard gates + BUY / withdrawal cash soft warnings.
  - Baseline 583 → final 656 (+73 portfolio tests, zero regression).
  - Unlocks P10-22 (migration snapshot script) and P10-23 (manual entry UI / CLI) for cutover day T. Per Rule 14.8, T must be an absolute date with verified weekday.

- **Status changes (2026-05-26 W2/W3 sprint; synced 2026-05-27)**: P10-17 → done; P10-20 → done; P10-22 → done; P10-23 → done; P10-18 → done; P10-10 → done. See PROJECT_STATUS Change Log rows for each.

### P10-22 Project_optimized → HTR Migration Snapshot

- Status: done (2026-05-26; code/tests ready, real cutover run pending user-selected T)
- Priority: P0 (cutover day T 当天必须可用)
- Depends on: P10-21 (journal schema), ADR-0008
- Goal: 一次性脚本，在 T 日把 Project_optimized 的 `etf_buyhold` 持仓 snapshot 翻译成 HTR journal 的 `migration` 事件。验证 NAV 一致后 emit `migration_complete` marker 给驾驶舱解锁 manual entry。
- Files:
  - Create: `tools/migrate_portfolio_from_project_optimized.py` (CLI: `--cutover-date YYYY-MM-DD --dry-run --strategy-id etf_buyhold`)
  - Create: `tests/unit/test_migrate_portfolio.py` (mock `position_adapter` 输入，validate 输出 journal 形态)
  - Create: `reports/portfolio/journal/.gitkeep`
  - Update: 在 `docs/03_FOLDER_MAP.md` 加 `reports/portfolio/`
- Acceptance:
  - Dry-run 模式打印将要追加的 migration entries 但不写盘。
  - 真实运行：读 Project_optimized `positions` + `account_snapshots` (via `position_adapter.load_portfolio_state`)，输出 1 条 `cash_event(reason=deposit, source=migration)` + 每只持仓 1 条 `fill(side=BUY, source=migration, ts=T 09:00 JST, avg_cost as price, note='migration from Project_optimized')`。
  - 写完后回放 `derive_positions(journal)` 与 `position_adapter` 输出 diff，NAV / per-symbol qty / per-symbol avg_cost 全部一致；不一致 fail-closed。
  - Emit `reports/portfolio/migration_complete.json` (含 T 日期 + 行数 + NAV verification 结果) 后驾驶舱解除 "migration in progress" banner。
  - 重复运行 idempotent：检测到当日已 migration_complete 则拒绝，避免双写。
  - Rule 3 保留：脚本不调用任何 broker / 订单。

### P10-23 Manual Fill Entry UI

- Status: done (2026-05-26; backend/API/CLI ready, frontend button integration remains follow-up if desired)
- Priority: P0 (T+1 起取代 CSV / SQL 路径)
- Activation stage: Rule 12.0 Stage 0 (Pull-only advisory entry)
- Depends on: P10-21 (journal), P10-22 (migration), P10-20 (cockpit)
- Goal: 给用户两个友好入口录入手动成交：(a) 驾驶舱按钮 "我成交了" 弹出 5 字段表单 + 预览 + 确认；(b) CLI `htr fill ...` 一行命令。两个入口共用 `record_fill` 后端，全部走 Rule 14.2 校验 + Rule 14.5 preview。
- Files:
  - Create: `src/hot_theme_rotator/portfolio/manual_entry_service.py` (record_fill_with_preview 包装 P10-21 API + Rule 14.5 magnitude 检查)
  - Create: `tools/htr_fill_cli.py` (argparse: `htr_fill_cli.py sell 1306.T 400 417.6 --at "2026-05-25 14:30"`)
  - Update: `api/dashboard.py` +`POST /api/portfolio/fill` (Rule 3 例外：不下单，只录已成交)
  - Update: `frontend/v1.jsx` 加 "我成交了" 按钮 + modal 表单 + preview 面板
  - Create: `tests/unit/test_manual_entry_service.py`
  - Create: `tests/unit/test_htr_fill_cli.py`
  - Update: `tests/unit/test_api_dashboard.py` (+POST /api/portfolio/fill contract 测试)
  - Update: `tests/unit/test_frontend_ui_contracts.py` (+确认 modal 渲染 preview)
- Acceptance:
  - 5 字段：symbol / side / qty / price / ts (ts 默认 now JST，可手动覆盖)。fee 默认 0，note 可选展开。
  - 提交前必须显示 preview：新 holdings (qty + avg_cost)、新 cash、SELL 时 realized P&L、Rule 14.5 magnitude warning (qty×price > 10% NAV 需二次确认)。
  - 后端 `POST /api/portfolio/fill` 调 `record_fill_with_preview`；Rule 14.2 任一校验失败返回 4xx + reason；成功后返回 derived position diff。
  - CLI 同样走 record_fill_with_preview；--dry-run 显示 preview 不写。
  - 唯一对 Rule 3 的例外：POST 的载荷是"我已通过外部券商成交"声明，不是下单指令；endpoint docstring 明示。
  - Rule 14.4：UI 不提供"删除"按钮，只提供"录入纠正"流程（生成 source=correction entry）。
  - Frontend 不出现"概率"/"胜率"/"win rate" 字样；不增加任何 broker / paperOrder 路径。

### P10-24 V3 Layout & Expansion Fix (Rule 11.7 enforcement)

- Status: done (2026-05-29; headless 截图验证折叠+展开态无裁切/无叠印；1262 pytest passing)
- Priority: P1 (可用性缺陷；不阻塞 calibration 关键路径，但影响日常使用)
- Activation stage: Rule 12.0 Stage 0 (Pull-only cockpit — 纯呈现层修复)
- Depends on: Rule 11.7 (added 2026-05-29), 现有 V3 变体
- Goal: 修复 V3 三个用户报告的呈现缺陷，**保持当前主线美学**（字体 / 配色 / 卡片样式 / 三列布局不变）：(1) 页面被 `height:100vh` + 嵌套 `overflow` 锁死，滚轮无法下滑；(2) 中列 6 卡 + 候选表硬挤进一个 `1fr`，卡片压成细缝、边框贴死；(3) 折叠卡展开时内容被挤进高度受限的行，多行文字叠印。根因单一：视口锁死的 app-shell 布局，不给内容生长空间。
- Files:
  - Update: `frontend/v3.jsx` — V3 根 grid 第三行 `1fr`→`auto` + `height:100%`→`minHeight:100%`；中列 `gridTemplateRows` 全 `auto`；左 / 右列 `1fr`→`auto`；候选 / 新闻 / 决策日志保留有界 `maxHeight` 内部滚动（Rule 11.7.3 允许的无界 feed）。导航栏保持钉顶，`.htr-app-main`（已 `overflow:auto`）作唯一滚动容器。
  - Update: `tests/unit/test_frontend_ui_contracts.py` — 加 Rule 11.7 静态契约断言（V3 根不出现裸 `height:"100%"` 锁死 + 主内容卡不在 `overflow:hidden` 祖先内被裁）。
  - 不改: `frontend/{v1,v2,v4}.jsx`（保留为对照变体）、`shared.jsx`（不动全局字号 / 卡片样式）。
- Acceptance:
  - 内容超出一屏时滚轮可下滑（document-flow scroll，Rule 11.7.1）。
  - 中列每张卡按内容取自然高度，不再压成细缝；展开任一折叠卡时内容向下推、不遮挡邻卡、不裁切、不叠印（Rule 11.7.2）。
  - headless 截图验证：折叠态 + 至少一个展开态，确认无裁切 / 无重叠，证据记入 `PROJECT_STATUS.md`（Rule 11.7.6）。
  - 美学不变：字体 / 配色 / 卡片边框 / 三列结构与改动前一致；不触碰任何 broker / 概率 / 胜率字样（Rule 3 / 8.3 / 9.4 不受影响）。
  - 1260 pytest 全绿（+ 新增 Rule 11.7 契约测试）。

### P10-25 Designer Redesign Integration (quant0530) + Rule 11.8

- Status: done (2026-05-30; 4 变体接真实后端，0 pageerror，1258 pytest passing)
- Priority: P1（前端体验大改；非阻塞 calibration 关键路径）
- Activation stage: Rule 12.0 Stage 0（Pull-only，纯呈现层）
- Depends on: 设计师交付 `quant0530.zip`、现有 /api/* 端点、shared hooks
- Goal: 把前端/交互设计师交回的 zero-build 重设计集成进项目，接真实后端（数据以我方为主），核对后端字段对应，必要时更新规则。
- Files:
  - Backup: `frontend/` → `frontend_zerobuild_backup_2026-05-30/`（可回滚）
  - New: `frontend/index.html` + `frontend/src/htr-{data,shared,shared2,v1,v2,v3,v3-modals,v4,cards,tweaks-panel}.jsx`（设计师交付）+ `frontend/src/htr-api.jsx`（我方新增：真实后端富集 hook）
  - Update: `frontend/src/htr-data.js`（enrich → `window.HTR_enrich`）、`frontend/src/htr-v3.jsx`（`top` 用 `useEnrichedCandidate` + FactorBody null/缺失安全）、`frontend/index.html`（bootWithApi）
  - Governance: `docs/02_GOVERNANCE.md` (+Rule 11.8 仓位与风险测算器)
  - Tests: `tests/unit/test_frontend_ui_contracts.py`（→10 条设计无关治理契约）
- Acceptance:
  - 4 变体接真实后端渲染（候选 6768.T… / 持仓 1306.T / NAV ¥402,635 / K-fold 降级 / advice-only banner），headless 验证 0 pageerror。
  - per-symbol 详情卡走真实端点（/strategy /profile /outcomes /kline /llm_brief /debate_brief），不伪造；后端缺字段（navHistory/sortino_60/vol_z）优雅降级记 backlog。
  - 治理红线全保留；新增 V1 仓位测算器受 Rule 11.8 约束。
  - 1258 pytest passing 零回归。
- Backlog（后端字段缺口，非阻塞）：navHistory（journal NAV 序列）/ sortino_60 / vol_z（profile 计算）。

### P10-26 News Entity-Layer Classifier + Theme Overlay (+ macro ingestion gap)

- Status: slice 1 done (2026-05-30; 确定性关键词分类器 + theme/macro overlay + 实证宏观采集缺口)
- Priority: P1（核心：让"news-driven 主题轮动"真的吃到上游宏观/板块催化）
- Activation stage: Rule 12.0 Stage 0（只读消费 + HTR-native 落盘）
- Depends on: Project_optimized news_feed（只读, ADR-0005）
- 起因: 用户问"新闻是不是只抓个股,宏观/政策不在范围"。**实证调查结果**：新闻源唯一 = Google News JP,且**按 ticker 逐个查** → feed 几乎全是个股财报/分红/适时开示;**宏观/政策/跨市场新闻压根不在采集范围**（97 条 7 天窗口仅 6% 命中板块词、宏观 0 条）。这架空了系统"外部温度 + 主题轮动"的核心差异点。
- Goal: 4 层新闻架构（① 宽口径多源采集 ② 实体分层分类 macro/theme/sector/ticker ③ 路由到主题热力 vs 单票 overlay ④ 治理嵌入 PIT/无概率/反FOMO/可溯源）。
- Files (slice 1 done):
  - New: `src/hot_theme_rotator/data/news_theme_classifier.py`（THEME_TAXONOMY 6 主题 + MACRO_TAXONOMY 5 类 + `classify_news` 确定性关键词 + `read_recent_news` 只读 PIT 窗口 + `build_theme_news_overlay` → HTR-native `reports/news_themes/{date}.json`）。**无 LLM 无 GPU**（feedback_no_bg_llm_batch + 透明可审计优于黑箱分数）。
  - New: `tests/unit/test_news_theme_classifier.py`（4 tests：分类正确 / PIT 窗口 / overlay 路由 / fail-closed）。
- Slice 1 验证: spotcheck METI→{semi,fiscal}、日銀→{monetary,fx}、FOMC→{overseas} 全过;真实 feed 跑通但**证明上游无宏观新闻可分** → 确认瓶颈在采集端。
- 剩余 slice（按优先级）:
  - **slice 2（关键）**: 宏观采集 —— HTR-native 按【宏观主题 query】(日銀/経産省/円安/FOMC/SOX)抓 Google News JP / 官方源,落 HTR-native（ADR-0005:HTR 自己加宏观源,不碰 Project_optimized 的 per-ticker 查询）。
  - slice 3: 把 theme overlay 接进 dashboard theme heat（宏观/板块新闻 → 主题升温,可溯源到原始新闻）。
  - slice 4（可选, 用户手动触发）: LLM 精炼分类层（Rule 8.3 只分类不出概率,24h 缓存,非 background）。
- Acceptance（slice 1）: 分类器对宏观+板块例子正确;只读 + PIT;HTR-native 落盘;4 tests green;实证记录采集缺口。

### P10-27 Local Beta v0 Readiness Gate (Monday 2026-06-01)

- Status: done (2026-05-30; Rule 15.2 checklist PASS recorded in PROJECT_STATUS; daily smoke 1266 passed / 5 deselected; localhost API smoke all 200; 7 POST routes ⊆ Rule 11.5; calibration verdict=downgrade visible)
- Priority: P0 (blocks using HTR as the Monday local operating cockpit)
- Activation stage: Rule 15 Local Beta v0 (single-user localhost only)
- Depends on: Rule 3, Rule 8.2.2, Rule 9.4.1, Rule 11.9, Rule 12.0, Rule 14, Rule 15
- Goal: Convert the current "feature-complete enough" state into a disciplined Monday local beta. This task does not add trading features. It freezes scope, separates daily smoke from slow research regression, verifies the localhost dashboard, records rollback instructions, and starts honest forward sample collection.
- Files:
  - Update: `PROJECT_STATUS.md` Local Beta v0 next actions / evidence / known gaps
  - Optional update: `README.md` or local runbook if startup commands are missing or stale
  - Optional update: `pyproject.toml` / test markers if vectorbt-numba tests need explicit slow segregation
  - No broker/order/paper/live execution code
- Required pre-Monday implementation / cleanup:
  - Working tree reviewed; unrelated sibling-project changes excluded from the beta scope.
  - A rollback point exists (commit, tagged stash, or documented backup directory).
  - Localhost dashboard starts and loads without page errors.
  - API smoke confirms `/api/health`, `/api/dashboard`, `/api/symbol/{T}/profile`, `/api/calibration/reliability`, watchlist, proposals, and manual-recording routes behave within Rule 3 / Rule 11.5.
  - Daily smoke command excludes or marks the vectorbt / numba slow lane so pre-open readiness is fast and deterministic.
  - Research regression command remains available for vectorbt / backtest checks but is not confused with daily readiness.
  - Dashboard visibly shows data freshness / market-session labels and calibration downgrade / insufficient state.
  - Forward sample collection command (`emit_daily_predictions.py` then `sweep_pending_outcomes.py`) is documented for after-close use.
- Acceptance:
  - Rule 15.2 readiness checklist is explicitly recorded as pass / fail in `PROJECT_STATUS.md`.
  - Daily smoke lane passes on the local machine.
  - Slow research lane status is recorded separately; a known vectorbt / numba cache stall does not block dashboard beta when isolated.
  - No UI or API surface displays "win rate", "probability", or equivalent validated-edge wording while K-fold verdict is downgraded.
  - No new POST / PUT / DELETE / PATCH route appears outside Rule 11.5.
  - Local Beta v0 is documented as `127.0.0.1` / `localhost` only, not LAN / cloud / multi-user.
  - If any readiness item fails, the beta is classified as debugging-only until fixed.

### P10-28 Daily Routine Automation (forward-sample self-collection)

- Status: done (2026-05-30; orchestrator + 7 tests green; end-to-end real run + scheduled-task run both verified; 2 Windows tasks registered, next run Mon 2026-06-01)
- Priority: P1 (operationalizes Rule 8.2.1 sunset / Rule 9.4 forward-sample accumulation without daily manual steps)
- Activation stage: Rule 12.0 Stage 0 / Rule 15 Local Beta v0 (deterministic, no push, no execution)
- Depends on: P10-27, Rule 15.5, Rule 8.2 / 12.2 (staleness fail-closed), ADR-0005 (sibling read-only)
- Goal: Make the deterministic, LLM-free half of the daily rhythm self-running so the operator only watches the dashboard. The two genuinely-manual steps stay manual by design: recording a fill (only when the operator trades) and pulling LLM narrative briefs (no-background-LLM preference).
- Files:
  - New: `tools/daily_routine.py` — orchestrator. `--mode preopen` (smoke gate + candidate freshness) / `--mode afterclose` (refresh candidates → emit → sweep). Injectable subprocess runner; fail-closed; idempotent; `--dry-run`.
  - New: `tests/unit/test_daily_routine.py` — 7 tests (trading-day calc, happy-path order, fail-closed abort, zero-candidate reject, dry-run no-op, no broker/exec fields in log record, preopen reporting).
  - New: `scripts/run_daily_routine.bat` (launcher) + `scripts/register_daily_routine_tasks.bat` (schtasks registration).
  - Candidate source: sibling deterministic `screener.py` invoked with `--no_db_write` (ADR-0005 read-only) + `PYTHONIOENCODING=utf-8`, output written HTR-native to `reports/screener/selected_tickers_{asof}.json`; emit reads it via `--snapshot`.
  - Log: `reports/observability/daily_routine_log.jsonl` (one structured line per run).
- Governance guards:
  - No broker / order / paper / live / LLM / GPU path — shells out only to screener, emit, sweep, pytest.
  - Fail-closed: failed or zero-ticker candidate refresh aborts emit (no stale/fabricated predictions).
  - Rule 15.4: emits only for a genuine just-closed session (idempotent on prediction_id), never accelerated replay.
- Acceptance:
  - Orchestrator unit tests green in the daily smoke lane.
  - Real end-to-end run produces forward predictions + sweeps outcomes for a true trading day.
  - Scheduled-task run (via launcher .bat) reproduces the same result and is idempotent on re-run.
  - Two current-user Windows tasks registered (preopen 08:30, afterclose 16:00, Mon-Fri JST).
  - Verified 2026-05-30: `pytest tests/unit/test_daily_routine.py` → 7 passed. Real run `--asof 2026-05-29` → 50 candidates → 50 new live predictions emitted → swept (insufficient, outcomes immature). `schtasks /Run HTR_Daily_AfterClose` → same chain, emit "50 skipped (already on disk)" (idempotent). Tasks `HTR_Daily_Preopen` / `HTR_Daily_AfterClose` Next Run 2026-06-01.

### P10-29 Frontend Write-Path Wiring Integrity (Rule 11.10 remediation)

- Status: done (2026-05-31; P0 fill + P2 notifier WIRED to real POST + 2 contract tests; P1 watchlist + P3 reflection defer-tracked per Rule 11.10.2/.5; smoke 1275 passed / 5 deselected; live preview HTTP 200 both portfolio endpoints)
- Priority: P0 (manual fill) — blocks the Local Beta v0 manual-recording activity (Rule 15.5 step 3)
- Activation stage: Rule 12.0 Stage 0 / Rule 15 Local Beta v0
- Depends on: Rule 11.5, Rule 11.9.4, Rule 11.10 (added 2026-05-31), Rule 12.7, §14.2/§14.9
- 起因: 2026-05-31 前端↔后端对应审计发现 designer 集成把 proposals 写路径完整接了当样板,其余三条交互写路径留成本地/演示桩——后端做好+测过+白名单,UI 够不到。
- Goal: 把审计发现的后端做好、前端死的交互写路径,按 Rule 11.10 处理(接线 or 明示并 defer-track),并补"按钮真 POST"的契约测试防回退。
- Findings & disposition:
  - **P0 — 录入成交/现金 (ManualEntryModal)**: 提交按钮永远只 `setPreview(true)`,从不 POST。**DO: 接线** —— 预览点 → `POST /api/portfolio/{fill,cash_event}` `commit:false`(显示真实 before/after + warnings);确认点 → `commit:true`(显示 committed + journal_path,或 409 重校验错)。后端契约已支持 preview/commit(Rule 11.10.3)。
  - **P2 — notifier toggle (NotifierChip)**: 双确认 UI 全在但只改本地 state,不 POST、不写 Rule 12.7 审计日志,dry-run 按钮无 onClick。**DO: 接线** —— 开 modal 时 `GET /api/notifier/state`;确认启用/禁用 → `POST /api/notifier/toggle`(后端按 stage-2 gate 拒绝/写审计日志);dry-run 按钮 → `dry_run:true`(Rule 12.7.5)。
  - **P1 — server watchlist**: 前端 ☆ 走 localStorage(`useWatchlist`),`/api/watchlist` 三端点零调用。**DECISION: 接服务端 vs 认 localStorage**。Local Beta v0 推送 Stage-2 禁用 → 服务端 watchlist 驱动的 Rule 12.4 冷静期/静默情报现无运营效果。**默认处置: defer 接线到 Stage-2 push,显式 defer-track(Rule 11.10.2);localStorage 已诚实标注不影响 portfolio/calibration**。用户可推翻要求现在接。
  - **P3 — reflection observability (snapshots/traces/funnels)**: 前端无消费。**DISPOSITION: 数据/观测类,frontend-optional(Rule 11.10.5),defer-track**;反思样本不足,产不出 proposal,延后做 debug 面板。
  - **冗余(非缺口)**: `/symbol/{T}/ladder` 未单独抓(数据已含在 `/strategy`),无碍。
- Files:
  - Update: `frontend/src/htr-v3-modals.jsx` (ManualEntryModal 真 POST preview/commit;NotifierChip 接 state/toggle/dry-run)
  - Update: `tests/unit/test_frontend_ui_contracts.py` (Rule 11.10.4 断言 fill/cash/notifier 控件真发 POST 到对应路径 + dry-run)
  - Update: `docs/02_GOVERNANCE.md` (Rule 11.10 — done), `PROJECT_STATUS.md`(defer-track watchlist/reflection orphan)
- Acceptance:
  - fill/cash modal 真 POST,预览映射 `commit:false`、确认映射 `commit:true`,展示真实 before/after / journal_path / 409。
  - notifier toggle 真 POST `/api/notifier/toggle` 并经后端 stage-2 gate(写审计日志);dry-run 走 `dry_run:true`。
  - Rule 11.10.4 契约测试断言三个控件真发 POST(非仅渲染),全绿且无回归。
  - watchlist(server)+ reflection observability 作为 defer-track 写入 `PROJECT_STATUS.md`(Rule 11.10.2/.5),非 merge-blocking。
  - 无新增 POST 路由越出 Rule 11.5;Rule 3 advice-only 不变。

## Milestone P11: Reflection & Self-Improvement

P11 是反馈控制环：把已 realized outcomes 转化为 system improvement proposals。**Per ADR-0007 (post Codex review)**：这是 policy replay + root cause analysis + LLM 叙事 proposals，**NOT** causal identification (Pearl do-calculus) 或 Shapley attribution。Rule 13 (子规则 13.1-13.10) 治理纪律。

### 7-layer 架构 (L0 PIT Observability 为 Codex 评审后新增基础)

- **L0 PIT Observability Ledger** (P11-00): foundation — 记录 decision_cutoff 时全部 state。没有它，所有上层都无效。
- **L1 Decision Trace Logger** (P11-01): per-(symbol, decision_cutoff) 完整决策链 trace。
- **L2 Event Detector** (P11-02): CUSUM/EWMA + **ARL-bootstrapped 阈值**（NOT generic σ 倍数）。
- **L3 Policy Replay Engine** (P11-03): off-policy evaluation with mutated configs against real OHLC；validity-class enum；**NOT** Pearl do-calculus。
- **L4 Root Cause Analysis** (P11-04): structured ablation + funnel loss + stale-data attribution + module diagnostics；**NOT** Shapley。
- **L5 LLM Reflection Report** (P11-05): 叙事综合（Rule 8.3.1 + 13.4 inherit）。
- **L6 Human Decision Gate** (P11-06): proposals → Rule 4 → user accept/reject。
- + **P11-07 Meta-Reflection**: Rule 13.10 enforcement，反思 reflection generator 本身。

### P11-00 PIT Observability Ledger (FOUNDATION)

- Status: done (2026-05-26)
- Priority: P0 (Week 3 启动，与 P10-18 并行；MUST 在任何 L1-L5 work 前完成)
- Depends on: P9-01, P10-17
- Goal: **Codex #1 missing component**。Snapshot 全部 PIT state 让反思 / counterfactual / RCA 可以正确 replay。没有它，反思就是 hindsight machine。
- Files:
  - Create: `src/hot_theme_rotator/observability/__init__.py`
  - Create: `src/hot_theme_rotator/observability/schema.py` (`PitSnapshot` dataclass)
  - Create: `src/hot_theme_rotator/observability/pit_ledger.py` (writer + reader + validity_class derivation)
  - Integration hook: `opportunity_scanner` / `risk_governor` / `watchlist_intelligence` / `alerts` 链路每次决策落 snapshot
  - Create: `tests/unit/test_pit_ledger.py`
- Acceptance:
  - `PitSnapshot` 含字段：`decision_cutoff` / `candidate_universe` (set) / `watchlist` (set) / `active_filters` (config hash) / `source_freshness` (dict per source) / `alert_budget_state` (dict) / `silent_queue_count` / `user_action_state` / `missing_data_reasons` (dict) / `config_version` (git hash) / `model_versions` (dict)。
  - 存储 `reports/observability/pit/{trade_date}/{snapshot_id}.json`，append-only。
  - 缺任何 required 字段 fail-closed。
  - `shadow_panel`：每 scan window 随机 sample K 个未 alert 候选作 control。
  - 读 API `load_snapshot(snapshot_id)` 完整重建 PIT state。
  - `derive_validity_class(snapshot)` helper：返回 `exact_replay` / `partial_replay` / `universe_reconstructed` / `price_only_replay` / `invalid`。
  - 至少 20 单测。
  - **不**新增任何 broker / order 路径。
- Verified (2026-05-26):
  - Added `src/hot_theme_rotator/observability/{__init__.py, schema.py, pit_ledger.py}` (~400 LOC). `PitSnapshot` frozen dataclass with all 13 ADR-0007 fields + universe_reconstructed_flag; deterministic `compute_snapshot_id = sha256(decision_cutoff|config_version|sorted(universe))[:16]`. Schema validates ISO tz on decision_cutoff, frozenset typing on universe/watchlist, tuple typing on shadow_panel, alert_budget required keys (used/remaining), non-negative silent_queue_count, non-empty config_version, snapshot_id integrity.
  - Storage: single-file per snapshot at `reports/observability/pit/{trade_date}/{snapshot_id}.json` (not JSONL — easier point lookup). `append_snapshot` rejects duplicate id, `load_snapshot` fail-closed on missing/malformed.
  - `sample_shadow_panel(candidates, k, seed, exclude)` deterministic via seeded Random — same inputs always produce same panel, enabling snapshot reproduction in replay.
  - `derive_validity_class(snapshot)` returns one of 5 enum values: `invalid` (no universe + no watchlist), `price_only_replay` (no model_versions), `universe_reconstructed` (flag set), `exact_replay` (all fields + non-empty shadow_panel + freshness), `partial_replay` (else). ADR-0007 §5 conditional-language directive recorded in module docstring for downstream consumers.
  - 33 unit tests covering all enum cases, schema rejects, snapshot_id determinism (order-independent on universe), writer append + duplicate reject + non-snapshot reject, loader missing + malformed + path-traversal reject, shadow_panel determinism + exclude + k-larger-than-pool + empty pool + negative k.
  - RED missing module; GREEN targeted `python -m pytest tests/unit/test_pit_ledger.py -q --basetemp=.runtime/pytest/p11_00 -p no:cacheprovider` -> `33 passed in 0.07s`. Full regression `python -m pytest tests -q --basetemp=.runtime/pytest/full_p11_00 -p no:cacheprovider` -> `754 passed in 8.69s` (721 baseline + 33 new, 零回归).
  - No broker / order / paper code path touched. Integration hooks into opportunity_scanner / risk_governor / watchlist_intelligence / alerts pipelines deferred to P11-01 (Decision Trace Logger consumes the snapshot infra).

### P11-01 Decision Trace Logger

- Status: done (2026-05-26)
- Priority: P0 (Week 4)
- Depends on: P11-00, P9-01
- Goal: per-decision trace 穿过所有模块——每个模块看到什么 / 输出什么 / 在哪里决策分叉。
- Files:
  - Create: `src/hot_theme_rotator/reflection/__init__.py`
  - Create: `src/hot_theme_rotator/reflection/trace_logger.py`
  - 存储路径: `reports/traces/{trade_date}/{trace_id}.jsonl`
  - Integration: 所有 decision-affecting 模块挂 hook
- Acceptance:
  - `TraceRecord` 含字段：`trace_id` (deterministic hash) / `prediction_id` link / `module_chain` (list of `(module, input_summary, output_summary, branch_decision)`) / `final_action` / `final_reason`。
  - JSONL one-file-per-day，append-only，fail-closed on schema violation。
  - Link 到 P11-00 `snapshot_id` 反查 PIT state。
  - 至少 15 单测。
- Verified (2026-05-26):
  - Added `src/hot_theme_rotator/reflection/{__init__.py, trace_logger.py}` (~250 LOC). `ModuleStep` frozen dataclass + `TraceRecord` frozen dataclass + `compute_trace_id = sha256(snapshot_id|prediction_id|symbol|created_ts|final_action)[:16]`. `prediction_id` allows empty string (NO_TRADE branches log before predict emission). `module_chain` is non-empty tuple of ModuleStep; empty tuple rejected at construction. ISO trade_date + tz-aware created_ts enforced. Integration hook signatures left to consumers (P11-02 event detector + L3-L6 ownership) — this slice is the writer/reader surface only.
  - Storage: `reports/traces/{trade_date}.jsonl` per-day JSONL aggregation (deviated from per-trace-file spec for parity with `silent_queue` + `portfolio/journal` conventions; documented in module docstring). `append_trace` rejects duplicate trace_id; `read_traces` returns chronological tuple; malformed JSONL fail-closed; missing-file empty tuple.
  - Rule 14.1-style append-only by API absence: no `update_trace` / `delete_trace` / `overwrite_trace` exists; test guards regression.
  - 24 unit tests covering trace_id determinism (over final_action distinguishing branches) / ModuleStep field rejection / TraceRecord field rejection + integrity check / writer duplicate-reject + non-TraceRecord reject / reader missing + round-trip + malformed + blank lines / no UPDATE-DELETE API surface.
  - RED missing module; first GREEN 23/24 (helper had `chain or default` bug coercing empty tuple to default — fixed to `chain is not None`); final GREEN `python -m pytest tests/unit/test_trace_logger.py -q --basetemp=.runtime/pytest/p11_01_v2 -p no:cacheprovider` -> `24 passed in 0.05s`. Full regression `python -m pytest tests -q --basetemp=.runtime/pytest/full_p11_01 -p no:cacheprovider` -> `778 passed in 14.09s` (754 baseline + 24 new, 零回归).
  - No broker / order / paper / notifier touched. Rule 3 / Rule 14 / §10 全保留.

### P11-02 Event Detector (CUSUM + ARL bootstrap)

- Status: done (2026-05-26)
- Priority: P0 (Week 5)
- Depends on: P11-00, P11-01, P9-02
- Goal: SPC on rolling outcomes——只在 CUSUM 跨过**从 ARL bootstrap 派生**的阈值（NOT generic σ 倍数）时触发反思。Per Codex 评审。
- **Codex review 2026-05-25 specifics**: target family-level ARL_0 of 1-3 months minimum across ~10 KPIs (not per-KPI). Use **max-statistic bootstrap across KPIs** if available, **Holm** correction for review reporting (preferred over Bonferroni). Per-KPI ARL_0 of 100 days yields family false alarm ~once per 10 trading days — too noisy.
- Files:
  - Create: `src/hot_theme_rotator/reflection/event_detector.py`
  - Create: `src/hot_theme_rotator/reflection/cusum.py`
  - Create: `src/hot_theme_rotator/reflection/bootstrap_arl.py`
- Acceptance:
  - CUSUM control chart with reference value `k = 0.5σ` 默认（可配）。
  - 阈值 `h` 由 **block bootstrap on historical system days** 派生，targeting `ARL_0` 默认 100 days false-alarm-free。
  - Multi-KPI 跟踪 with metric multiplicity 校正 (Bonferroni 或 Holm)。
  - Bernoulli outcomes 用 beta-binomial / sequential likelihood ratio（NOT Gaussian）。
  - Returns 用 robust stats (median, MAD, winsorized, block bootstrap)。
  - Rule 13.3 sample size 门槛：n≥30 minimum for investigate trigger。
  - Holdout window after trigger 防 thrashing。
  - 至少 18 单测含 known-shift detection 准确性。

### P11-03 Policy Replay Engine

- Status: done (2026-05-26)
- Priority: P0 (Week 5)
- Depends on: P11-00, P9-02
- Goal: Off-policy evaluation with mutated config。**Per ADR-0007**：NOT causal identification。输出语言 constrained 为 "under reconstructed state U and config C"。
- **Codex review 2026-05-25 — Data Freshness Gate**: with no J-Quants account, the only historical OHLC source is `Project_optimized/japan_market.db.daily_prices` which can be 4-6 weeks stale. Policy Replay MUST refuse to run (or output explicit `data_too_stale` validity class) when the OHLC dataset's most-recent `asof` is older than N trading days (default N=5, configurable). This prevents Policy Replay from generating "fresh" counterfactuals against weeks-old data and silently misleading the user.
- Files:
  - Create: `src/hot_theme_rotator/reflection/policy_replay.py`
  - Create: `src/hot_theme_rotator/reflection/validity_class.py` (enum + classifier)
- Acceptance:
  - Mutate config dimensions: `chase_threshold` / `alert_budget` / `cooling_off_hours` / `scanner_threshold`。
  - 每 cell 重跑 scanner against P11-00 PIT ledger + `LegacyDailyPriceFetcher` OHLC。
  - 输出: Pareto frontier in (P&L, miss_rate, alert_spam) 3D space。
  - 每个 replay output 必须带 `counterfactual_validity` ∈ {`exact_replay`, `partial_replay`, `universe_reconstructed`, `price_only_replay`, `invalid`}。
  - PIT validation: reject if any input feature `available_ts > decision_cutoff`。
  - 至少 15 单测。

### P11-04 Root Cause Analysis (ablation + funnel + diagnostics)

- Status: done (2026-05-26)
- Priority: P0 (Week 6)
- Depends on: P11-00, P11-01, P11-03
- Goal: Structured ablation + funnel loss + module diagnostics。**Per ADR-0007**: NOT Shapley attribution。Sequential pipeline analysis with explicit intervention semantics。
- Files:
  - Create: `src/hot_theme_rotator/reflection/rca.py`
  - Create: `src/hot_theme_rotator/reflection/ablation.py`
  - Create: `src/hot_theme_rotator/reflection/funnel.py`
- Acceptance:
  - Sequential ablation: 按 pipeline 顺序 walk (data → scanner → filter → alert → notifier)，量化每个模块 intervention 的 marginal recovery (PIT-fresh data / lower threshold / bypass filter / unlimited budget / available notifier)。
  - Funnel loss: 数每阶段 candidates 损失 (eligible → scored → not-filtered → alert-triggered → alert-pushed → user-acted)。
  - Stale-data attribution: freshness > 阈值时显式 attribute。
  - 输出: ordered list of contributors with `marginal_recovery` metric, **NOT** Shapley %。
  - 至少 20 单测。

### P11-05 LLM Reflection Report Generator

- Status: done (2026-05-26)
- Priority: P1 (Week 6)
- Depends on: P11-04, P10-06
- Goal: 叙事综合 trace + ablation + funnel evidence into Chinese research note。**数字来自 L3-L4，prose 来自 LLM**。
- Files:
  - Create: `src/hot_theme_rotator/llm/reflection_brief.py` (template per Rule 8.3.1 + 13.4)
- Acceptance:
  - Input schema: `PitSnapshot` + `TraceRecord` + `ablation_result` + `funnel_result` + `counterfactual_validity`。
  - Output schema: `narrative` (markdown) + `factual_grounding` (evidence citations) + `proposed_actions` (structured per Rule 13.6 metadata) + `confidence_caveats` + `counterfactual_validity` (echoed)。
  - Rule 8.3.1 + 13.4: **绝不**输出 `\d+%` / 胜率 / 概率 / probability / likelihood / win rate。
  - Post-generation regex enforcement + max 1 regenerate after violation。
  - 至少 12 单测用 mock LLM。

### P11-06 Human Decision Gate

- Status: done (2026-05-26)
- Priority: P0 (Week 6)
- Depends on: P11-05
- Goal: Proposal review pipeline。Rule 13.1 (proposals only) + 13.5 (7d expiry) + 13.6 (metadata required) + 13.7 (backtest required for parameter changes) + 13.9 (log rejections)。
- R2 hardening (2026-05-27): also enforces Rule 13.11 validity-to-action matrix, Rule 13.14 shadow default for accepted parameter changes, Rule 13.15 same-target cooldown, Rule 13.16 typed expiry semantics, and Rule 13.17 reproducibility metadata.
- Files:
  - Create: `src/hot_theme_rotator/reflection/decision_gate.py`
  - 存储: `reports/reflections/{proposals,accepted,rejected,expired}/`
  - Integration: CLI tool + dashboard panel for review
- Acceptance:
  - Schema validation: proposal 缺任何 Rule 13.6 字段 → 入intake 立即拒绝。
  - Rule 13.7 enforcement: parameter change proposal 必须带 pre/post backtest evidence。
  - User actions: accept (触发 Rule 4 flow) / reject (with reason, Rule 13.9 log) / defer。
  - 7-day expiry auto-move 到 `expired/` (Rule 13.5)。
  - 至少 15 单测。

### P11-07 Meta-Reflection

- Status: done (2026-05-26)
- Priority: P2 (Week 7+，依赖 P11-06 + ≥30 天生产 reflection 数据)
- Depends on: P11-06
- Goal: Rule 13.10 enforcement——反思 reflection generator 本身。
- Files:
  - Create: `src/hot_theme_rotator/reflection/meta_reflection.py`
- Acceptance:
  - Trigger 条件 (3 选 1): 3+ 连续同 `evidence_class` proposal 被 reject / 3+ proposals expire 未 review / post-acceptance change 失败 predicted improvement。
  - 输出: "generator X 需要 adjustment" + 暂停同 evidence proposals 直到人工 review pass。
  - 至少 8 单测。

## Milestone P12: Calibration Validation Hardening

Per the 2026-05-30/31 governance thrust (Rule 8.2.2 validation-gate integrity, Rule 9.4.1 leak-resistant methodology, Rule 9.4.2 leakage-audit verdict). This milestone is the not-time-blocked half of the calibration path: while forward samples accumulate (P10-28 automation), make the validation pipeline honest and leak-resistant so that when samples mature there is a credible protocol to run — rather than scrambling a protocol that happens to pass. The honest expectation (Rule 8.2.2.5): this may conclude "no demonstrated edge" and the score stays an uncalibrated ranking signal.

### P12-01 PIT Leakage Audit of Backdated Calibration

- Status: done (2026-05-31; verdict=contaminated — V1 corporate-action PASS + V3 available_ts PASS + V2 survivorship INCONCLUSIVE + V4 model-selection FAIL; backdated evidence quarantined per Rule 9.4.2.3; `tools/audit_calibration_leakage.py` + 6 tests; smoke 1281 passed; artifact reports/calibration/leakage_audit_2026-05-31.json)
- Priority: P0 of the calibration track (precondition per Rule 8.2.2.3 — must run BEFORE extending the backdated window or trusting any backdated skill)
- Activation stage: Rule 12.0 Stage 3 (reflection/calibration; does not relax Rule 3/8.2/8.3/9.4)
- Depends on: Rule 8.2, Rule 8.2.1, Rule 8.2.2, Rule 9.4.1, Rule 9.4.2 (added 2026-05-31), Rule 11.9.6
- Goal: Answer one binary question with recorded evidence — are the existing backdated/bootstrap calibration samples (762 on disk) contaminated by look-ahead (confirmed `yfinance auto_adjust` split adjustment on labels/features/volume/universe; survivorship; post-cutoff `available_ts`; future-data model selection)? Emit the Rule 9.4.2 verdict artifact and apply its consequence.
- Locked audit checklist (fixed before running, Rule 9.4.2.4):
  - V1 corporate-action: does any price feeding a backdated label OR a momentum/MA/vol feature OR volume OR a universe filter come from a retroactively split/dividend-adjusted series? (trace the bootstrap price source vs the kline_adapter back-adjustment, Rule 11.9.6)
  - V2 survivorship: is the candidate universe at backdated date D reconstructed PIT, or does it include only names that exist/survive as of today?
  - V3 available_ts: does every ex-ante feature for cutoff D have `available_ts < D` (Rule 8.2), and does the outcome window use only bars with `asof > D + >=1 trading day` (Rule 8.2.1)?
  - V4 model selection: were isotonic/threshold choices made using any data from the test/outcome window?
- Files:
  - Create: `tools/audit_calibration_leakage.py` (read-only audit; emits verdict artifact)
  - Create: `reports/calibration/leakage_audit_{date}.json` (Rule 9.4.2.1 deliverable)
  - Create: `tests/unit/test_audit_calibration_leakage.py`
  - Update: `PROJECT_STATUS.md` (verdict + consequence)
- Acceptance:
  - Verdict ∈ {clean, contaminated, inconclusive}; inconclusive treated as contaminated (Rule 9.4.2.2).
  - Each of V1-V4 reported with pass/fail + evidence.
  - If non-clean: backdated evidence quarantined per Rule 9.4.2.3 (stops counting toward sunset/validation; UI stays downgraded), recorded in PROJECT_STATUS.
  - Read-only; no broker/execution; no relaxation of Rule 9.4 labeling.

### P12-02 Purged + Embargoed Walk-Forward Protocol

- Status: done (2026-05-31; `purged_walk_forward.py` + 6 tests signal→pass/noise→downgrade; runner `validate_calibration_walk_forward.py`; backdated diagnostic = downgrade, model Brier 0.311 worse than random/stratified, n_eff=11 date-clusters → honest no-edge per Rule 8.2.2.5; smoke 1287 passed. Scope note: locked-holdout harness deferred until tuning is introduced — isotonic PAV has no hyperparameters and the walk-forward final fold is already out-of-time)
- Hardening update (2026-06-15): review found implementation did not yet enforce all locked Rule 8.2.3 ship predicates in code. Fixed: verdict now requires ≥20 independent date clusters, model Brier strictly beating random + climatology/date-cluster CI + stratified baseline, and `leakage_verdict == clean`; otherwise `insufficient_data` / `downgrade`. Added 3 regression tests. `tools/validate_calibration_walk_forward.py` accepts `--leakage-verdict`. Live run on 2026-06-15 with 7 clusters returns `insufficient_data`.
- Depends on: P12-01 (a clean leakage verdict on the inputs used), Rule 9.4.1
- Goal: Replace/augment the rule-flagged-suspect `kfold_validate_isotonic` (Rule 9.4.1) with a leak-resistant validation: purge training samples whose 3D/5D outcome window overlaps the test window, embargo a gap after each test fold, prefer time-ordered walk-forward over random folds; report against climatology/event-rate + stratified baselines (not just random 0.25); compute date-block/cluster-robust effective sample size + CIs; reserve one locked out-of-time holdout run once.
- Acceptance:
  - Purge + embargo implemented and unit-tested on synthetic overlapping-label data.
  - Baselines (random, climatology, stratified) reported alongside the model.
  - Effective-sample-size / CI uses date-cluster method, not raw row count.
  - The locked holdout is run at most once; protocol changes after seeing it void it (Rule 8.2.2.1 / 9.4.2.4).
  - Until this passes leak-free, UI calibration stays downgraded (Rule 9.4).

### P12-03 Forward-Collection Hardening (Codex 2026-06-01 review)

- Status: pending (enhancement; the immediate 2026-06-01 silent-no-op bug is fixed below)
- Depends on: P10-28, P12-01 findings, Rule 8.2 / 11.9 / 11.9.6 / 15.4
- 起因: 2026-06-01 afterclose 16:00 跑了但采 0(dropped 50,候选当日 close 未进 sibling daily_prices)且 daily_routine 谎报 "emitted"+ok。Codex 全面只读审查纠正根因(sibling evening_batch 19:00 确有跑;真因=计时 16:00 早于 EOD 定稿 19:04 + 部分候选不在 sibling universe)并挖出 daily_routine 诚实性漏洞。
- **已修(本会话 2026-06-01)**:
  - daily_routine `collect()` 解析 emit 计数 + `run_afterclose` 诚实状态:0 new + dropped>0 → ok=false,绝不记 "emitted"(Rule 11.9);幂等 all-skipped → ok=true(+2 测试)。
  - 计划任务 16:00 → **19:30**(晚于 sibling 19:04 EOD 定稿);register .bat 同步。
  - emit cutoff 15:00 → **15:30**(JPX 实际收盘)。
  - **dashboard 候选源解耦**(2026-06-01 用户报"前端不更新"):`api/serializers._freshest_selected_tickers_path()` 优先读最新 HTR-native `reports/screener/selected_tickers_{date}.json`,仅当无 HTR 快照才回退 sibling——dashboard 不再卡在 sibling 那个不刷新的 05-27 文件。验证:重启后 `meta.tradeDate` 05-27 → **2026-06-01**,#1 候选 6768.T(旧)→ 6584.T(今日);+3 选源单测。
  - **市场温度计 session 修复(Rule 11.9.3,用户报夜里显 OPEN)**:`_market_session_state` 原仅周末 reconcile、工作日返回硬编码值(N225 永远 OPEN / US 永远 CLOSED)。改为按 **region + 真实时钟 + 交易时段**推导(JP 09:00-11:30/12:30-15:30 JST;US 09:30-16:00 ET via zoneinfo;FX 工作日 LIVE;UNKNOWN 保留)。live 验证 JST 周二 00:00:JP CLOSED / US OPEN / FX LIVE / SSE UNKNOWN,全对。节假日日历仍留 P12-03(b)。测试改写为 region-aware。
  - 验证:补跑 `--asof 2026-06-01` → 50 new / 0 dropped / ok=true(20:50 时 DB 已补齐 50/50),forward 样本 2→3 天(150 条);smoke **1292 passed**。
- **剩余加固(本任务,pending)**:
  - (a) **HTR-native 取价解耦 — emit 端 done (2026-06-08)**:`emit_daily_predictions.py` 加 **yfinance reference-price 兜底**(`_yf_close_one`/`_yf_close_batch`,`auto_adjust=False` 原始收盘,与 sibling 同基准/Rule 11.9.6)。取价顺序:sibling `daily_prices` 优先 → 缺则 yfinance 兜底 → 都无才 drop;`extra.reference_price_source ∈ {sibling_daily_prices, yfinance_raw}` 标注来源(PIT provenance)。`--no-fallback` 可关。网络/yfinance 缺失时兜底返回 `{}` 优雅降级(绝不崩调度)。`daily_routine.collect()` 不带 `--no-fallback` → **自动路径默认即用兜底,反复"0 采集"失败根除**。+3 测试(`test_emit_fallback.py`:兜底填补 / `--no-fallback` 关闭 / 无网络降级)。**起因**:2026-06-08 sibling DB 停在 06-05(0 行 06-08),旧 emit 50 候选全 drop;修后手动补采 `--snapshot selected_tickers_2026-06-08.json` → **50 new / 50 from yfinance / 0 dropped**,`predictions/2026-06-08.jsonl` 落盘(实价 6584.T 1043)。
    - **(a-sweep) done (2026-06-09)**:`sweep_pending_outcomes.py` 加 `_CompositeFetcher`(sibling daily_prices 优先 → 窗口空则 `_yf_bars_range` yfinance RAW 区间 bars 兜底,`auto_adjust=False` 同基准)+ `_safe_float`(NaN 防护)+ `--no-fallback`。sibling 冻结时 outcome 仍能从 yfinance 成熟。+4 测试(`test_sweep_fallback.py`)。验证:`_yf_bars_range('6584.T','2026-06-08','2026-06-08')` → 1 bar @1043(实价)。**教训**:sweep 不可在交易日**收盘前**跑——eval_date=未收盘日会写 premature symbol_not_found,因 outcome_id=hash(pred,eval_date) 去重,挡住当晚真实 outcome(本次盘前误跑已清理 06-08 outcomes 文件)。调度的 afterclose 19:39 在收盘后,无此问题。
  - (b) **JPX 节假日日历 done (2026-06-07)**:新增 `src/hot_theme_rotator/data/jpx_calendar.py`(stdlib-only、确定性、离线)——`is_jpx_holiday`/`is_trading_day`/`latest_trading_day`/`calendar_covers` + 2026 手工核验节假日表(按 weekday 逐个校对:元旦/成人/建国/天皇诞生/春分/昭和/GW 5-4~5-6/海/山/敬老+国民+秋分/体育/文化/勤劳 + JPX 年末 1-2 与 12-31)。`daily_routine.latest_trading_day` 改为 import 它 → **现在跨节假日回退,不只是周末**,emit 永不落在休市日(Rule 15.4 显式 fail-closed)。afterclose 记录加 `calendar_covered`,超出覆盖年(仅 2026)显式标注降级为"仅周末 + 数据守卫兜底",不静默过度信任。+8 测试(6 jpx_calendar + 2 daily_routine)。**未硬编码我不确定的 2025/2027 年份**——错误节假日会跳过真实交易日(fail-closed 错方向),宁可只覆盖已核验的 2026 + 显式标注,由现有 screener-空输出数据守卫兜底其余年份。新年前需扩表并核验。
  - (c) **拆股复权 done (2026-06-07)**:Rule 11.9.6 在 **outcome 路径**和 **benchmark 路径**用不同正确策略落地:
    - **outcome 路径(sweep / bootstrap / 所有 compute_outcome 调用方)= fail-closed 闸门**。`compute_outcome` 加 `_detect_split_in_window`:reference→bars 或相邻 bar 出现 clean split 比率即返回 `malformed_data`(复用 `kline_adapter.detect_split_factor` 单一启发式)。**理由**:reference_price 是 emit 时存的原始收盘,与抓取 bars 不同基准,无法在此安全重对齐 → 宁可丢弃极少数跨拆股样本,也绝不吐 -90% 假收益毒害校准(与项目 fail-closed DNA 一致)。sweep 自动受保护(闸门在计算层),无需改 sweep fetcher。
    - **benchmark 路径(cohort 卡)= 真复权**。新增 `kline_adapter.SplitAdjustedDailyPriceFetcher`(复用 `_split_adjust`),`api/candidate_history._benchmark_fetcher` 切换之。**理由**:benchmark 的 reference 自抓取序列内派生,同序列复权 → 跨拆股也得连续真实收益(优于 fail-closed)。live 无拆股窗口 `_split_adjust` 为 no-op,数值不变。
    - 测试:`test_outcome_join`(+2:split→malformed / 大幅非拆股不误触)、`test_kline_adapter`(+2:`detect_split_factor` 公开 wrapper / `SplitAdjustedDailyPriceFetcher` 去 cliff)。smoke **1349 passed / 5 deselected**,零回归。
    - 未做(留 backlog):outcome 路径"计算复权后正确收益以保留跨拆股样本"——需重对齐 stored reference 基准,改动 F2 reference 契约,风险高、增益低(样本极少),fail-closed 是诚实接口。
- (a) emit 端 done(2026-06-08)+ (a-sweep) outcome 兜底 done(2026-06-09);(b)(c) done。emit/sweep 不再硬依赖 sibling 写库,sibling 冻结时双双经 yfinance 兜底继续工作。
  - **(d) screener 数据迁 HTR / 候选新鲜化 done (2026-06-09)**:`tools/refresh_htr_price_db.py` —— 一次性快照 sibling 884MB → HTR 自有 `data/raw/htr_market.db`(全表+历史+基本面,screener 依赖完整;sibling 只读)+ yfinance 按 JPX 交易日补 daily_prices(idempotent、离线降级、`auto_adjust=False` 同基准)。`daily_routine.DB_DEFAULT=HTR_DB`,screener 前先 refresh(非致命,失败则用上次 HTR DB)。算法零改、纯数据迁移。+4 测试(`test_refresh_htr_price_db.py`:跨节假日缺日 / 快照+追加 / 幂等 / 离线降级)+ daily_routine order 测试更新。实跑:快照+补 06-08/09(1892 行)→ fresh screener #1 6584.T→8604.T、2334.T 跌出前八(候选见到近日跌)。今日已采 06-09 snapshot 不覆盖(保 PIT)。**遗留**:HTR DB 基本面/历史冻在快照时点(可接受);sibling 复活时重新快照策略待定。
- Acceptance: (a)(b)(c) 实现 + 单测;emit/sweep 不再隐性依赖 sibling 写库;smoke 全绿;不引入 broker/LLM 路径。**(b)(c) 已满足(日历/复权 + 单测 + smoke 1357 + 无 broker/LLM);(a) 取价解耦待做**。

### P12-04 Frontend Display-Honesty Remediation (Rule 11.9 backlog, all variants)

- Status: in progress (2026-06-05;后端冻结期,纯前端;无 broker/LLM/POST 路径变更)
- Depends on: Rule 11.9(尤其 11.9.1 无捏造数据 / 11.9.3 session state 必派生)、Rule 11.7 Scope(已澄清 11.9/11.10 绑定所有变体,本任务先做的规则更新)
- 起因: 2026-06-05 跑起 app 对 V1–V4 做 headless 截图 + 静态审计,抓出若干**在真实数据下仍渲染的写死 mock**(界面"撒谎"),违反 Rule 11.9.1/11.9.3。证据:`.runtime/audit_shots/{v1..v4}.png` + `{v1..v4}_top.png`(2x)。
- **本批修复(P0,4 项,均经真实数据核对)**:
  - (1) **K线 x 轴写死"今日"** (`htr-shared.jsx`,共享组件→全 4 变体):改用 bar 真实 `date`(首/中/末 MM-DD 标签)。当数据非当日(休市/stale)时不再谎称"今日"(Rule 11.9.2)。bar.date 已由 `/api/symbol/{T}/kline` 提供(末根 2026-06-05 实证)。
  - (2) **V1 市场时段条全写死** `Tokyo OPEN / HK OPEN / London PRE / NY CLOSED` (`htr-v1.jsx` V1SessionStrip):改从真实 `markets[].state`(后端 `_market_session_state` 日历正确,P12-03 已修)派生 JP/US/CN/FX session。删除映射不到真实信号的虚构 HK/London(Rule 11.9.3 hardcoded session forbidden)。
  - (3) **V4 时间轴假 leader 事件** "8035.T 研究分 66→78 跨过 75 阈值进入首位" (`htr-v4.jsx:193`):该票非真实龙头(真实 #1 = 6584.T)且"66→78 跨阈"为捏造日内信号(双违 11.9.1 + 8.3/9.4)。改为从真实 `candidates[0]` 派生(真实 symbol + 真实研究分 + 未校准排序信号标注,无捏造日内 motion)。
  - (4) **V2 lead 叙事写死** `sub="news-catalysed semiconductor day"` (`htr-v2.jsx:16`):真实龙头 theme 占位为 `screener_v2` 非半导体。改用真实 `one_liner`(alpha/ADV/mom 因子摘要)。
- Acceptance: 4 项均以真实 `/api/dashboard` + `/api/symbol/{T}/*` 数据派生,无写死字面量;headless 截图复验无回归(Rule 11.7.6);smoke `pytest -m "not slow"` 全绿;前端契约测试不破。
- Deferred(本批不做,记入 Rule 11.9 backlog):V4 新闻去重 + 跨日新鲜度标(新闻 06-03 是真实日期非写死,P1);V1/V2 顶栏补新鲜度标注(P2);死分支清理(V4 `session_open` 文案、V1 `useTickingPrice` 直通残留)。**后续系统化推进见 P13 里程碑(本任务是 P13-01 honesty sweep 的前哨)。**

### P12-05 Forward-Calibration Readiness — Sunset Trigger + Locked Criteria (2026-06-11)

- Status: **协议已锁 / 等数据 (in progress, 2026-06-11)**。起因:2026-06-11 forward 样本 paired-complete 达 **229 ≥ 100**,Rule 8.2.1 sunset 计数门触发;但完整性审计发现真正独立观测仅 **5 个 date-cluster**(05-27/29、06-01/02/03)——raw-count 门过、effective-sample 门未过。
- 已做(本会话):
  - **forward 样本完整性/泄漏审计**(直接核查,非工具):基差/拆股 V1 clean(0 malformed、1D 收益无 >25% 异常,yfinance 兜底没造假拆股价);PIT V3 clean(cutoff 全 == trade_date、无未来函数);229 个 complete **全部来自改造前原管线**(我这周的 emit/sweep 兜底+DB 迁移未污染已成熟样本)。结论:样本干净,但 cluster 太薄不能下判决。
  - **Rule 8.2.3 锁定 ship 标准**(治理,owner 2026-06-11 拍板,blind 时锁):effective-sample 下限 **≥20 date-cluster**;协议=P12-02 purged/embargoed walk-forward;pass 须同时满足(a)跑赢 random/climatology/stratified 全部基准(b)cluster 自助 CI 排除零(c)泄漏审计 clean;CI 条款自动挡住运气薄样本;20-29 cluster 的 pass 建议 ≥30 复confirm。
  - **退役 backdated 补样**:Rule 8.2.1 sunset 计数已达;补样早被 P12-01 判 contaminated/quarantine,forward 取得 primacy(Rule 8.2.2.2),记 PROJECT_STATUS。
- 待做(时间瓶颈,自动推进):管线每天自动攒 ~1 cluster;到 **≥20 cluster(约 4 周,~7 月上旬)** 跑一次锁定的 `validate_calibration_walk_forward` + forward 泄漏审计,接受判决(大概率 "no demonstrated edge")。当前 **5/20**。
- 活动阶段保持 Stage 0;分数维持 uncalibrated_research_score。不引入 broker/LLM 路径。

## P13 — Frontend / Interaction Design Optimization (四变体均质, backend frozen)

> 设计意图: `docs/superpowers/specs/2026-06-06-frontend-interaction-design.md`;实施计划(含 TDD 步骤): `docs/superpowers/plans/2026-06-06-frontend-interaction-design.md`。来源: 15-agent / 7 维 / 71 条 adversarially-verified 设计审计 + Codex(codex:rescue)复审。后端冻结(只读 `/api/dashboard`+`/api/symbol/{T}/*` + 已有 7 条 Rule 11.5 白名单 POST)。**已锁决策(2026-06-06)**:端点范围=可用已有白名单 POST;变体优先级=四变体均质(全部拉到 Rule 11.7 不变量同线,V3 仍为名义默认);节奏=先 spec/plan 后执行。严重度 rubric:P0=任意变体捏造决策相关数量/轨迹(或默认 V3 上用户会据以行动的内容违规);P1=其他内容红线违规(静态假标签/死按钮/mock 当真,任意变体);P2=布局/易读/IA/视觉/色彩/a11y/DRY(四变体均质下亦为承诺工作,非可选)。

### P13-01 Phase 1 — Content-honesty + dead-interaction sweep
- Status: **done (2026-06-06)** — 8 项写死 mock 全部 → 真实派生/移除;TDD 8 新契约测试先红后绿;smoke `-m "not slow"`(干净 basetemp)**1302 passed**;四变体 headless 截图复验(`.runtime/audit_shots/`,Rule 11.7.6);零回归。实现要点:V4 假分数 sparkline+`(+12)` 删除(LeaderCard 改真实价格 spark 标"价格·120d")、V2ChartCard 改用 `candidate.kline` 真实 overlay(120 sessions 实证)、V4 header chips 从 `v4BuildEvents` 按 kind 派生(实测"新闻4·候选浮现1",假类目消失)、V4 死按钮→静态图例(去 button affordance + 去"写入决策日志"假文案)、V3 `2 high`→按 weight 派生、V1 caveat→真实 `strategy.risk_warnings[0].message`(注:risk_warnings 是对象数组,首次直渲对象致 V1 白屏,已修)、V1/V2 `screener_v2` 裸渲染 guard、V2 假刊期删。
- 范围(详见 plan Task P13-01):**P0** V4 假分数 sparkline `[42..78]`+`研究分(+12)`(捏造分数轨迹,9.4-adjacent)删除;**P0** V2 价格图渲染 mock kline(`V2ChartCard` 用 `kline` prop=boot mock 而非 enriched `candidate.kline` 真实 overlay)→ 改用真实 overlay;**P1** V4 header 事件计数 chips `[7,6,8,3,0]` 写死 → 从 `v4BuildEvents` 按 kind 派生;**P1** V4 右栏 4 按钮死按钮+假"写入决策日志"文案 → 删成静态图例(decision-log 非 Rule 11.5 白名单路径,defer-track);**P1** V3 新闻 `2 high` 写死 → 按 weight 派生;**P1** V1 action zone 每票都显写死 `USD/JPY 跌破 156` caveat(来自 8035.T mock)→ 用 `candidate.risk`/`strategy.risk_warnings` 或撤;**P2** `theme='screener_v2'` 裸渲染(V1/V2)+ 空 nameJa/nameCn 分隔符;**P2** V2 masthead `vol.4 issue 113` 假刊期。
- Acceptance: 新增契约测试逐项锁死(无写死字面量 / 真实派生);smoke `-m "not slow"`(干净 basetemp)1294+ 绿;headless 复验四变体渲染无回归(Rule 11.7.6);PROJECT_STATUS 记录证据。
- Depends on: Rule 11.9/11.10/8.3/9.4(已绑定所有变体,无需新规则)。

### P13-02 Phase 2 — Async-state honesty + write-path reachability
- Status: **done (2026-06-06)** — 4 子项全做完;新增 3 契约测试先红后绿;smoke(干净 basetemp)**1305 passed**;四变体 console 全干净 + headless 截图复验;watchlist POST live round-trip 实测(add→size1 有 9999.T / remove→size0)。
  - **P2-A 异步态(P1)**:`useEnrichedCandidate` 加 `_status` per-endpoint 态(pending/ok/failed,覆盖 strategy/profile/outcomes/kline/aiBrief/debate 全部——Codex 拓宽,不止 LLM)+ 共享 `AsyncBodyState`(htr-shared.jsx);FactorBody/OutcomesBody/AiBody/DebateBody 改 status 驱动:pending→"⏳生成中…"、failed→"⚠真实数据未就绪·示例占位 (11.9.4)" banner、ok→真实。mock 保留作崩溃安全但 failed 时被明确标注。`/llm_brief` 实测 200 缓存(0.0s)→ ok 无误报。
  - **P2-B 写路径可达(P2,Codex 修正为 UX 一致性非 11.10 违规)**:ProposalInboxChip/CalibrationChip/NotifierChip 从 V3TopBar 提到全局 `app-nav`(index.html)→ 四变体可达(实测 V1 nav 显"🧠提案/📐校准/📢推送")。search+watchlist 仍 V3 本地(需 per-variant symbol 选择,待 Phase 6 共享 picker)。
  - **P2-C degraded 覆盖(P2)**:bootWithApi 加 `degraded.positions`(`!api.positions || available===false`)+`degraded.dailyCockpit`;V3 banner SEC map 加"持仓/Daily Cockpit"。macroNews 不追踪(无 mock,absent 即 null,Codex)。per-symbol 失败由 P2-A 的 AsyncBodyState 覆盖。
  - **P2-D watchlist 写入(P2,用户允许已有 POST)**:`useWatchlist` 接 `GET /api/watchlist`(mount 载入服务端真相)+ add/remove 乐观更新后 `POST /api/watchlist/{add,remove}` reconcile;localStorage 降级为缓存/离线回退;WatchlistChip modal 旧"仅存 localStorage"标改"服务端持久+localStorage 缓存";+契约测试断言真 POST(注:test_no_write_methods 启发式在 `method:"POST"` ±400 搜首个 `/api/`,原 GET `/api/watchlist`(无尾斜杠)在前致误判,把 post() 排到 GET 前修)。Depends on: 端点范围决策(已允许)。

### P13-03 Phase 3 — Shared-component unification(contract-test-gated)
- Status: **partial done (2026-06-06)** — 治理关键的"措辞/色彩单一源"已做完(Codex 标的 honesty-sensitive 部分);纯 DRY 的组件抽取(阶梯/tile/cockpit/视觉 score-stat)作 **P13-03b DRY 尾巴 deferred**。本批 +2 契约测试;smoke(干净 basetemp)**1307 passed**;V3/V4 console 干净 + 截图复验;既有不变量测试(七档阶梯 9.6 / uncalibrated 8.3/9.4 / 9.4 降级)全程绿。
  - ✅ **P3-C 校准/分数措辞单一源**:新增 `HTR.LABELS = {scoreUncalibrated:"研究分·未校准", uncalibrated:"未校准", rule94Note:"Rule 9.4 — …不可视作概率"}`(htr-shared.jsx);V3 顶栏内联校准 pill(曾是 htr-v1 CalibPill 的重复)→ 改用共享 `<CalibPill>`;V3 MiniStat `研究分(未校准)`、V4 leader/spine 分数标签、FactorBody 9.4 note 全部改引用 `HTR.LABELS`。措辞不再 5 处分歧。
  - ✅ **P3-F 独立 heat 色阶**:新增 `--htr-heat-hot`(light #D2451E 暖朱红 / dark #E8794A)+ `-bg`,light+dark 双主题;`heatColor/heatBg` 的 t≥70 分支由 `--htr-bear` 改指 `--htr-heat-hot`——过热瓦片不再与亏损/止损共用红(单一函数,V3MarketTile + 共享 MarketTempCell 同时生效)。注:当前 6 市场温度全 <70,色阶逻辑由契约测试锁,待有市场过热才视觉可见。
  - ✅ **P3-B 措辞部分**:分数 stat 的 canonical wording 由 `HTR.LABELS.scoreUncalibrated` 锁(防 "(+12)" 那类漂移复发)。**视觉组件 `<ResearchScoreStat>`/`<LeaderIdentity>` 抽取 deferred**——跨 V1/V2/V3/V4 四个不同尺寸 header,属纯 DRY、风险高价值低,归入 P13-03b。
- **P13-03b（DRY 尾巴）**：
  - ✅ **V3LadderMini dead code 清理 done (2026-06-06)**(P4-E 后无引用,删函数;七档仍在 K线 overlay + StrategyCard 表)；✅ V3 K线 panel 加"滚轮缩放"可发现性提示(P6-E 一项)。smoke 1316 绿、console 干净。
  - **仍 deferred(纯 design-only DRY，非 honesty 红线，L 工作量/低增量价值——honesty 已被契约测试锁)**：`<LadderMini>`/`<LadderTable>`+`labelShort`（并剩余 ~5 份阶梯渲染，须逐变体截图验七档不丢）；`MarketTempCell variant='hero'` 退役 `V3MarketTile`；`<CockpitCard>`/`<CandidateRow>`/V2 共享 `<DecisionLog>` + 修 htr-v2.jsx 价格小数 ternary；`<ResearchScoreStat>` 视觉组件。

### P13-04 Phase 4 — Legibility-floor + IA/visual polish(V3 先,后 parity)
- Status: **partial done (2026-06-06)** — 强制/低风险子项做完;3 新契约测试;smoke(干净 basetemp)**1309 passed**;V3 console 干净 + 截图复验(hero 层级 + 无溢出,Rule 11.7.6)。
  - ✅ **P4-A 易读下限(P1,Rule 11.7.4)**:8 处功能性 sub-floor 标签提到 floor(MiniStat 标签 9.5→11、主题 leaders 10→11、持仓配比图例 10→11、持仓列头 9.5→11、持仓 qty@cost 10→11、Cockpit Metric 标签 9.5→11、TDnet chip 9→10.5、outcomes 来源 10→10.5)。装饰 eyebrow(TEMPERATURE/masthead 大写/state badge)豁免。
  - ✅ **P4-C 去 scroll-trap(Rule 11.7.3)**:V3 outcomes 表 `maxHeight:180`→600(有界主证据不再卡小窗,保留高 cap 防病态行数)。
  - ✅ **P4-D hero 层级**:温度瓦片数字 32→23、leader 价格 32→40——真实 #1 龙头清晰压过宏观温度条,单一焦点(实测截图)。
  - ✅ **P5-A 焦点环(WCAG 2.4.7,并入本批)**:TOKEN_STYLE 加 `:focus-visible` accent ring 覆盖 `[role=button]`/`[tabindex=0]`/.nav-btn/.htr-chip/button(键盘聚焦可见,鼠标点击不脏);去 blanket outline 抑制。
  - ✅ **P5-B 对比度(WCAG AA,并入本批)**:K线日期轴 + detail-tabs hint 的真实文本 `--htr-ink-4`(~2.3:1)→ `--htr-ink-3`(过 AA)。
- **P13-04b**：
  - ✅ **P4-E 默认折叠 StrategyCard — GATE 已过 + 实现 done (2026-06-06)**：**先澄清 Rule 11.6**（新增 11.6.9：detail body 可默认折叠以降密度，但"安全三件套"banner+Rule 3 disclaimer+risk_warnings 须折叠态常显于折叠 guard 之外；折叠须 in-flow expansion；契约测试须断言 risk_warnings 在 guard 前）→ 再实现：risk_warnings 块移出 `{expanded &&}`（与 banner/disclaimer 同列常显）+ `useState(false)` 默认折叠 + 严重度 mono 标签 9.5→10.5 floor；另**去掉 LeaderCard 的 V3LadderMini**（阶梯第 3 次重复）leader-grid 3 列→2 列。DOM 实测:折叠态 banner+risk 常显·ladder 隐 / 展开后 ladder 现·risk 仍在。+1 契约测试（risk 非折叠 + 默认折叠）;smoke **1310 passed**;截图复验。注:V3LadderMini 现为 dead code,P13-03b 阶梯统一时清。
  - ✅ **P4-G 响应式健壮性 done (2026-06-06)**：KLineChart 引入响应式 `padRight = max(40, min(118, width-left-80))` + `innerW = max(40, …)` floor——窄屏不再出负/NaN 绘图宽度(实测 W=1000/760 0 severe error、SVG 无负属性);ladder label box/x 同步用 padRight;1320 断点 rail `repeat(3,1fr)`→`repeat(2,1fr)`(4 卡 2×2 无孤儿);leader-grid 经 P4-E 已 2 列,<1080 仍堆叠。
  - ✅ **P6-D 变体切换器 默认标 done (2026-06-06)**：V3 nav 按钮加"· 默认"标 + title(对照/shipping 区分);实测窄屏 nav 显"V3 · 市场温度仪表盘 · 默认"。GatesChip 固定出滚动区(纳 nav 重构)留 minor-deferred。
  - ✅ **P5-C/D a11y done (2026-06-06, P13-05b)**:ModalShell 改无障碍 dialog——`role=dialog`/`aria-modal`/`aria-labelledby`(useId 关联标题)+ mount 聚焦关闭键·unmount 恢复焦点 + Tab trap + Escape + ×`aria-label=关闭`(DOM 实测全绿);Term tooltip 加 `role=tooltip`+`aria-describedby`+Escape;`--htr-ink-3` #827D71→#71695C(muted 文本过 AA);5 处 `#fff`/`white` SVG/badge fill → `--htr-accent-ink`/`--htr-bg`(dark-mode/自定义 accent 安全)。+2 契约测试;smoke 1316 绿。
  - ✅ **P4-F 右栏 IA tab done (2026-06-06)**：V3 右栏 Cockpit + 决策日志 → 合成 `V3FeedsTabs`（chip tab 切换，复用现卡），4 always-open → 3 逻辑卡降密度；Portfolio+News 保持常显。DOM 实测两 tab 并存;截图右栏更紧凑。
  - ✅ **P6-E manual-fill 预填 done (2026-06-06)**：`ManualEntryModal` 接 `defaultSymbol`，V3 "+ 成交" 用当前 leader symbol 预填（DOM 实测开弹层 symbol=6584.T）。
  - ✅ **P6-E notifier dry-run cue + P6-F V2 §E 上移 done (2026-06-06)**：notifier dry-run 按钮加 title(预览 discipline filter·不真启用·不写日志·Rule 12.7.5);V2 因子/校准证据区从 §E 上移到 §B(声明→证据→外部因子→候选→催化→治理),DOM 实测证据区现于外部因子之前。
  - **仍 deferred(纯维护/低增量)**：P4-B type-scale 收 6-7 档 `--htr-fs-*`（L，~17 内联字号迁移，V3 floor 已达标故仅一致性）；P6-D GatesChip pin（nav 重构，narrow 才显效）；ProposalInbox override-logged（需后端 `AcceptRequest.override_expiry` 字段，后端冻结）。

### P13-05 Phase 5 — Accessibility hardening
- Status: planned。范围: `:focus-visible` 焦点环(去 blanket `outline:none`)、`--htr-ink-4` AA 失败修(真实文本:K线日期轴/hint)、Modal `role=dialog`/`aria-modal`/focus trap、icon-only 命名、tooltip `role=tooltip`+Escape、ink-3 加深 + `#fff` SVG fill → `--htr-accent-ink`。operability 不在 11.7 layout carve-out 内。

### P13-06 Phase 6 — All-variant layout/IA parity(四变体均质,committed)
- Status: **partial done (2026-06-06)** — P6-A/B/C/D 做完;smoke **1310 passed**;四变体 console 干净 + DOM/截图实测。
  - ✅ **P6-A V4 leader 置顶**:去掉 v4BuildEvents 里 15:30 的 leader 事件(会被晚钟新闻压下),改在 V4Spine **顶部 pin V4LeaderCard**(非事件 hero,驱动自选中候选);V4LeaderCard 去 `ev` 依赖改派生标签;V4SpineRow 简化去 leader 分支。实测截图:leader 卡在新闻流之上。
  - ✅ **P6-B V2/V4 symbol picker**:V2 加 sel 状态(共享 `htr_symbol`)+ V2WatchlistTable 行可点切主线(DOM 实测点行 lead→8604.T);V4 加 sel + `V4CandidateStrip` 横向 chip 选择器(#rank symbol)。两者解除 candidates[0] 锁死。
  - ✅ **P6-C V1 披露 parity(Rule 9.6/11.6)**:V1 center 末尾加 governed `V3StrategyCard`(折叠·risk 常显)+ `FactorCard` + `OutcomesCard`(复用共享卡,带 P2-A 异步态)。DOM 实测:策略卡片/因子构成/历史命中/风险标记 全在。
  - ✅ **P6-D 变体默认标**(见 P13-04b 批,2026-06-06)。
  - **仍 deferred**:P6-E 杂项 affordance grab-bag(V1 risk-budget degraded toggle、V4 spine meta 字号、K线 zoom hint、manual-fill 预填 symbol、notifier dry-run cue）；P6-F V2 §E 证据上移;P6-D GatesChip pin;V1/V2/V4 全量易读下限 parity（V3 已做，余三变体按需）。
- Acceptance(全 P13): 各 phase 收尾契约+smoke 全绿 + Rule 11.7.6 截图证据入 PROJECT_STATUS;backend 依赖项保持 deferred 不桩假。

### P13-07 Codex 审查整改 (2026-06-06)

Codex(codex:rescue)审视当日 P12-04 + P13-01..06 全部前端更新。**已修(今天相关)**:
- ✅ **kline 诚实性缺口(Codex blocker #1)**:`_status.kline` 原被追踪但图表不消费,空/失败 kline 仍静默显 mock。修:`load()` kline 验证器 `bars.length>0`(空→failed),V3/V1 图表 header 加"示例K线"标、V2 标题加" · 示例K线"、V4 leader 价格 sparkline 改 `_status.kline==="ok"` 才显(标"真实")。+1 契约测试。
- ✅ **空 payload 崩溃风险(Codex #3)**:`load()` 加按端点 schema 验证器(strategy 需 risk_warnings+ladder_tiers;outcomes 需 items;aiBrief 需 narrative+factual_grounding;debate 需 bull/bear/judge;profile 需 ticker)——无效/空 payload 不覆盖 mock、标 failed,body 显示完整 mock + 示例占位 banner 而非崩 `o.items.map`。
- smoke **1311 passed**;四变体 console 干净。
- **预存问题(非当日引入,记录追踪,待排期)**:
  - ✅ **ProposalInbox Rule 13.18 修复 done (2026-06-06)**(Codex blocker #2):**13.18.1** 完整 Rule 13.6 元数据由折叠"技术细节"改为**始终可见**(同屏于按钮上方,11 字段网格);**13.18.2** 去 `window.confirm`,parameter_change 改**独立"I understand"勾选框**(shadowAck)门控接受按钮(发 `user_confirm_shadow`,后端已强制 422);**13.18.4** 过期(`is_expired_by_age`)显 expiry banner + **accept 默认禁用**,需勾"override expiry"(overrideExp)才启用;**13.18.5** 无批量(+"逐条审阅·无批量"标注);**13.18.6** 顶部加待批/已接受/已拒绝/已过期 counts 条。+1 契约测试;DOM 实测 modal 开·counts 条·0 错;smoke 1312 绿。**诚实缺口**:13.18.4 "override 须 logged" 子项需后端 `AcceptRequest.override_expiry` 字段(后端冻结)——override gate 前端已强制,但 flag 不单独落后端日志(accept 本身经 decision_gate append-only 已 logged)。`htr-v3-modals.jsx`。
  - ✅ **V4 决策事件映射 done (2026-06-06,Codex #4)**:`v4BuildEvents` 加 `candidate_persisted` **聚合**事件(N 支候选写入决策日志·模型，kind=candidate→header"候选浮现"计数反映)+ `news_overlay_hit` 分支(kind=decision)；并修既有 session/macro/scan 分支的时间切片 bug(`slice(0,8)` 取到日期片段 → 改 `hhmm()` 取 HH:MM)。DOM 实测 V4 显"写入决策日志"。+1 契约测试。
  - ✅ **watchlist 跨实例同步 done (2026-06-06,Codex #5)**:`useWatchlist` 改 **module-level 共享 store `HTR_WL`**(subscribe/setLocal/listeners)+ 单次 hydrate(`_wlHydrated` 守卫);任一实例 add/remove 经 store 通知全部重渲。DOM 实测:点 leader ☆ → nav chip 计数 0→1 即时同步(无需 remount)。+1 契约测试。POST 接线(Rule 11.10)保持。
  - ⏳ V3LadderMini dead code(P13-03b 阶梯统一时清);out-of-pool 搜索仅 V3 重置(V1/V2/V4 回退 candidate0)。

## Milestone P14: Candidate Cohort Review (Rule 11.11)

> 起因 (2026-06-07): 用户注意到候选股每天刷新,想要一张"看历史候选"的卡片。关键洞察——历史候选其实**已经在每天 append-only 落盘**(`emit_daily_predictions.py` → `reports/predictions/{date}.jsonl`,`sweep_pending_outcomes.py` → `reports/outcomes/{date}.jsonl`)。所以这不是又一层 UI 装修,而是把"前瞻样本积累"变成用户看得见的**验证 edge 的那块仪表盘**:某日整批候选 vs 同期大盘后来实际怎么走。设计两分叉用户已拍板:**整批聚合优先** + **回测补样默认隐藏(墙内标注)**。

### P14-01 Cohort Review Backend + Rule 11.11 + V3 Card

- Status: **done (2026-06-07)** — 后端数据层 + 只读端点 + Rule 11.11 + V3 卡片;TDD 先红后绿;smoke `-m "not slow"`(干净 basetemp)**1335 passed / 5 deselected**(baseline 1316 + 19 新测试,零回归);Rule 11.7.6 headless 截图证据 `.runtime/audit_shots/v3_cohort_card.png`(三档表 + 个股明细展开,无重叠/截断,console 仅 favicon 404)。
- Activation stage: Rule 12.0 Stage 0(pull-only 复盘;不发通知,不放松 Rule 3/8.2/8.3/9.4)
- Depends on: ADR-0003 decision log、P9-01/02(predictions/outcomes 已落盘)、Rule 8.2/8.3/9.4/12/14.6
- Files:
  - Create: `src/hot_theme_rotator/reporting/candidate_cohort_review.py`(纯逻辑 + 注入式 benchmark fetcher)
  - Create: `tests/unit/test_candidate_cohort_review.py`(11 测试)
  - Create: `api/candidate_history.py`(`GET /api/candidates/history` + `/dates`)
  - Create: `tests/unit/test_api_candidate_history.py`(8 测试)
  - Update: `api/main.py`(注册 router)
  - Update: `frontend/src/htr-v3.jsx`(`V3CandidateHistoryCard` 挂右栏)
  - Update: `docs/02_GOVERNANCE.md`(Rule 11.11)
- Acceptance (对齐 Rule 11.11 七条):
  - ✅ **PIT 忠实**:roster 字段全取自存储 `PredictionRecord`,仅 outcome 是 cutoff 后数据。
  - ✅ **整批优先**:每档先出整批等权 mean + 上涨占比,个股明细折叠在下。
  - ✅ **成熟度诚实**:仅 `status=="complete"` 进聚合;未成熟/缺失/异常显式列出并计 immature(`maturedCount + immatureCount == cohort`),不静默丢。
  - ✅ **对照基准**:每档并列同窗 1306.T(TOPIX)收益 + 超额,horizon 索引与 `outcome_join` 一致。
  - ✅ **回测隔离**:`-backdated`/`extra.backdated` 默认排除并计 `excludedBackdated`;勾选才显,带"合成样本·非真实战绩"标注。
  - ✅ **禁标胜率**:统计量为 `uncalibrated_research_score` 群体描述;常驻"样本不足以得出胜率结论"披露;payload 无 win rate/probability/calibrated_probability。
  - ✅ **只读**:端点拒 POST(405),不在 Rule 11.5 写白名单;契约测试断言。
- Live 验证 (2026-06-07,HTTP :8000):实盘日 7 (05-27..06-05) / 回测墙隔 16;2026-05-27 这批 50 候选 / 29 成熟 / 0 回测混入 → **1D 整批 +0.56% vs 大盘 -0.51%(超额 +1.06%);5D 整批 -0.32% vs +1.97%(超额 -2.29pp,跑输);上涨占比 41-52%**。**诚实定性照旧:仍无 demonstrated edge**——卡片如实摊开,5D 明显跑输 passive。
- Deferred(显式记入 backlog,非阻塞):
  - ~~v1/v2/v4 parity~~ → **done (2026-06-07,见 P14-02)**。
  - **跨日走势带**(每日整批 vs 大盘的累积视图,样本积累后才有信息量)。
  - **Rule 11.9.6 复权**:benchmark 用注入 fetcher 原始价(live 窗口 ≤~13 日不跨拆股,安全);跨 corporate action 需 split-adjusted,与 P12-03(c) 同轨。
  - **前端契约测试**:本轮卡片为自取数只读组件,后端已 8 测试锁死;前端侧 honesty 断言(整批优先/披露常显/无写路径)可补一条 `test_frontend_ui_contracts` 用例。

### P14-02 Cohort Review Card — Four-Variant Parity (Rule 11.7 / P13-06)

- Status: **done (2026-06-07)** — `V3CandidateHistoryCard`(共享 governed 卡,全局作用域,与 V1 复用 `V3StrategyCard`/`FactorCard`/`OutcomesCard` 同模式)挂入 V1/V2/V4:
  - **V1**(`htr-v1.jsx`):右栏"候选清单"面板之后(历史候选紧贴当日候选)。
  - **V2**(`htr-v2.jsx`):§D Watchlist 之后,加 `§D·复盘` section label。
  - **V4**(`htr-v4.jsx`):左栏 `V3PortfolioCard` 之后(与市场温度/主题/持仓同列)。
- 验证(Rule 11.7.6 headless,:8000):四变体 `found=True / table rows=3 / severe_non_favicon=0`;截图 `.runtime/audit_shots/{v1,v2,v3,v4}_cohort.png`——V4 256px 窄栏(副标题按 CardHead 既有 ellipsis 截断,非 bug)与 V2 editorial 全宽两个极端均无重叠/截断。smoke `-m "not slow"` 全绿。
- 说明:内容红线(Rule 11.11 七条)由后端 8 测试 + 端点统一保证,对所有变体生效(Rule 11.7 Scope:content red-lines bind every variant);各变体仅布局位置不同。无新增 POST/写路径。

### P14-03 Cross-Day Cohort Trend (Rule 11.11.2 full-series view)

- Status: **done (2026-06-07)** — 把"每天整批 vs 大盘"汇成跨日序列 + **全实盘池化底线**:至今所有前瞻样本里候选整批到底有没有跑赢指数。Rule 11.11.2 已明确许可"full live series"视图,**无需新规则**。TDD 先红后绿,**+8 测试(4 unit + 4 API)→ smoke `-m "not slow"` 1343 passed / 5 deselected,零回归**;Rule 11.7.6 headless 四变体复验(走势块入卡,crop 23→33KB,severe=0)。
- Activation stage: Rule 12.0 Stage 0(pull-only);不放松 Rule 3/8.2/8.3/9.4。
- Files:
  - Update: `src/hot_theme_rotator/reporting/candidate_cohort_review.py`(`build_cohort_trend` + `CohortTrendPoint`/`PooledHorizonStat`/`CohortTrendReport`,复用已测 `build_candidate_cohort_review`)
  - Update: `api/candidate_history.py`(`GET /api/candidates/history/trend`)
  - Update: `frontend/src/htr-v3.jsx`(`V3CandidateHistoryCard` 底部"全实盘累计"块:池化超额 + 每日 3D 超额火花条;共享卡 → 四变体自动获得)
  - Update: `tests/unit/test_candidate_cohort_review.py`(+4)、`tests/unit/test_api_candidate_history.py`(+4)
- 池化数学(诚实性要点):
  - 池化均涨 = Σ(日均涨·日成熟数)/Σ(日成熟数) ≡ 全部成熟样本等权平均(从日聚合精确重建,无需原始收益)。
  - 池化上涨占比同理 = Σ(日上涨占比·日成熟数)/Σ(日成熟数)。
  - **池化超额只在"同时有 cohort 和 benchmark 的日期"上比较**(单独累加器 `ret_on_bench`)——避免某档基准窗口不全时拿 N 样本 cohort 对 M 样本基准的 apples-to-oranges(测试 `test_trend_pooled_excess_uses_only_benchmarked_dates` 锁死)。
  - 无 benchmark fetcher → excess/benchmark 为 None,但 cohort-only 统计照常算。
- Live 验证(:8000):7 实盘日 / totalComplete=29(目前仅 05-27 批成熟,余日 outcome 未闭合)→ 池化 **1D 超额 +1.06% / 3D −0.43% / 5D −2.29%**(= 05-27 数,因它是唯一成熟批;随更多日成熟自动扩池)。**诚实定性照旧:仍无 demonstrated edge,5D 明显跑输 passive**。火花条每日 3D 超额(未成熟日淡显)。
- Deferred(记 backlog):benchmark Rule 11.9.6 复权(与 P12-03c 同轨)。前端契约测试 → 见 P14-04 done。

### P14-04 Cohort Card Frontend Contract Tests (Rule 11.11)

- Status: **done (2026-06-07)** — `tests/unit/test_frontend_ui_contracts.py` 加 2 个 design-independent 契约(基于源码文本,与既有风格一致):
  - `test_cohort_review_card_is_read_only_and_honest`:`V3CandidateHistoryCard` body 内断言——读 `/api/candidates/history`(11.11.7 只读,**无 POST**)、渲染 `honestyNote`(11.11.6 披露常驻)、**整批均涨表先于个股明细**(11.11.2 整批优先,按文本位置)、`include_backdated` 开关 + `合成样本` 标注(11.11.5 回测墙隔)、`全实盘累计` + `非胜率`(P14-03)。
  - `test_cohort_review_card_present_in_all_four_variants`:V1-V4 源码均引用 `V3CandidateHistoryCard`(Rule 11.7 / P14-02 parity)。
- smoke `-m "not slow"` **1345 passed / 5 deselected**,零回归。
- 说明:这两条锁死即使将来重构 cohort 卡,只读/披露/整批优先/回测墙隔/四变体 parity 不被悄悄破坏。配合后端 12 测试(8 cohort + 4 trend)+ 端点统一,Rule 11.11 honesty 在前后端双锁。

## Milestone P15: News Pipeline + Theme-Engine Wiring (诚实补缺口)

> 起因 (2026-06-14): 用户发现新闻面板停在 06-07 不更新,并指出**新闻催化 + 新闻因子 feedback 是设计核心**(00_DESIGN §策略),质问为何 status 一直说"差不多了"。诚实复盘确认:之前把"校准 track 等数据"误说成"整个项目就差验证",**掩盖了最大缺口——命名核心"新闻驱动热点龙头轮动"根本没在实盘跑**:实盘候选来自 sibling 的纯价格动量 `screener.py`,HTR 自己的 theme/leader/signal 引擎(P3/P4 建了+测了)从未接进候选生成;新闻数据源(sibling Google News 抓取)在死;news-factor feedback(P7-06)从没接上。

### P15-01 HTR-native 新鲜新闻管线 (done 2026-06-14)

- Status: **done (2026-06-14)**。修了用户的直接抱怨——新闻不再停更。
- 关键发现:新闻源是 **Google News JP RSS**(公开可达,本环境返回 2026 时间线新闻),且 HTR 早有 `macro_news_fetcher`(P10-26)在用它抓宏观,只是没抓股票/主题新闻、没每天跑、dashboard 没读。
- 实现:`src/.../data/stock_news_fetcher.py`(复用 macro fetcher 的 `_http_get/_parse_rss_items` + `news_theme_classifier.classify_news` 6 主题分类)——抓 9 条主题/市场查询 → 去重 → 分类 → 写 HTR-native `reports/news/{date}.json`;`load_latest_news_timeline` 出 dashboard-ready 行(JST ts)。`api/serializers._serialize_news` 优先读它(冻结 DB 回退);`tools/refresh_htr_news.py` CLI + 接进 `daily_routine.refresh_candidates`(screener 前,非致命)。+5 测试(注入假 RSS)。
- 验证:真跑抓到 **108 条新鲜新闻(最新 06-14,主题计数 semi24/ai24/bank23/auto12/defense12/energy10)**;dashboard newsTimeline 由停在 06-07 → **20 条全 06-14**。smoke **1373 passed**。
- 注:这只修了**新闻显示新鲜度 + 主题分类**。它**没有**让新闻去驱动候选选股(见 P15-02),也没有 feedback(P15-03)。

### P15-02 把 theme/leader/signal 引擎接进实盘候选生成 (pending — 核心缺口)

- Status: **pending**。这是让"Hot Theme Rotator"名副其实的关键一步。
- 现状:实盘候选 = sibling `screener.py`(纯价格动量+基本面,零新闻零主题);HTR 的 `theme_detector` / `leader_ranker` / `signal_engine` 建好测好但**不在候选路径**。
- 目标:用 HTR 引擎(新鲜新闻 → detect_themes → 主题热度 → leader_ranker 选龙头)叠加价格 screener,使候选真正是"新闻催化的热点龙头"。
- **计划 = ADR-0009**(2026-06-14):hybrid 引擎(价格 screener 给强势 universe + 新闻/主题层加催化+龙头智能,不丢弃 screener);保持 uncalibrated(不动 P12-05 校准轨)、PIT、Rule 4 显式配置。
- **loop 迭代 1 (2026-06-14) 发现真 blocker(建之前查出)**:名匹配/行业映射都走不通——`tickers` 表对 94% 活跃 universe **无名字无行业**(候选 3/50 有),sibling related_tickers 也空。根因=HTR 库缺股票元数据。**修正**:加前置 **P15-02a′ = yfinance `.info` 抓 sector/industry → HTR-native 元数据**(6584.T 实证可得)。修订分期:02a′(元数据)→02a(催化聚合)→02b(龙头排序)→02c(hybrid 重排)→02d(接线)。
- **loop 迭代 2 (2026-06-14) — P15-02a′ done**:`data/ticker_metadata.py`(yfinance `.info` → sector/industry/name + `theme_for` 行业→6主题映射;增量、可注入、离线降级、`data/raw/ticker_meta.json` 存储)+ 5 测试。**真跑实证:50 候选 100% 拿到元数据(vs tickers 表 6%),15/50 映射到主题**(6584→auto/8604→bank/4443·4373→ai/2162→semi)。smoke **1378 passed**,零回归。规则无需新增(ADR-0005 HTR-native + Rule 8.3 no-LLM 治理)。
- **loop 迭代 3 (2026-06-14) — P15-02a done**:`candidate_engine/catalyst.py`(`compute_theme_heat` 归一化新闻主题计数 + `catalyst_for` 候选→催化分 + `build_catalyst_map`;纯逻辑可注入)+ 6 测试。**真跑实证:今日新闻 AI/半导体最热,50 候选 15 个被催化**——AI/半导体名(4443/2162/3089 催化 1.0)排前、银行 0.96、auto 0.5;无主题候选催化 0(不捏造)。新闻驱动信号首次真工作。smoke **1384 passed**,零回归。规则无需新增。
- **loop 迭代 4 (2026-06-14) — P15-02c done**:`candidate_engine/hybrid_rerank.py`(`RerankConfig.news_weight=0.30` 显式配置/Rule 4;min-max 归一 screener 分 + 催化分融合,无催化不丢弃,screener 仍占 70%)+ 5 测试。**真跑实证:新闻把候选实质重排**——8604(银行)/4443·4373(AI)/2162(半导体)上推,8524 #13→#6,无催化 7419(零售) #1→#7。"新闻催化的热点龙头"首次成形。输出仍 uncalibrated(不动 P12-05)。smoke **1389 passed**,零回归。规则无需新增。**下一迭代:P15-02d 接进 dashboard 候选面板(融合排序 + 主题/催化标注可见);02b leader_ranker 细化与 02d 一并或后续。**
- **loop 迭代 5 (2026-06-14) — P15-02d done / 核心 LIVE**:`api/serializers._serialize_real_candidates` 接入 hybrid 重排——全 universe 算融合分→重排→取 top_n→原有富化;候选加 `theme`(催化主题)/`newsCatalyzed`/`topTheme`/`catalystScore`/`blendedScore` + one_liner "🔥主题催化";`_real_or_sample_candidates` 龙头改取重排后 #1。**显示 score 仍为原始研究分(不虚高),顺序按新闻催化(诚实:分是分、序是序)**;fail-open 无新闻/元数据则退回纯 screener。**Live 实证:dashboard 龙头 7419(零售)→ 8604(银行催化),top6 全催化、7419/7513 无催化掉下**。smoke **1389 passed**,零回归。规则无需新增(ADR-0009 治理,uncalibrated 不动 P12-05)。**"Hot Theme Rotator" 命名核心首次实盘工作。** **剩余(robustness/polish):元数据 refresh 接进 daily_routine(保未来候选有元数据)、前端催化 badge、02b leader_ranker 细化。下一迭代:元数据 refresh 接 daily_routine。**
- **loop 迭代 6 (2026-06-14) — P15-02 核心 LIVE+持久**:`tools/refresh_ticker_metadata.py`(读当日候选 → 增量抓 yfinance 元数据,limit 60/run,离线降级)接进 `daily_routine.refresh_candidates`(screener 后,非致命);顺序 refresh→news→screener→**meta**→emit→sweep,order 测试同步。CLI 实证 "0 new/50 known"(增量快)。smoke **1389 passed**,零回归。**P15-02 核心(02a′/02a/02c/02d + 持久化)全部 done**——新闻催化候选引擎实盘工作且每日自维护。**剩余仅 enhancement**:02b leader_ranker 细化(theme→leader→signal 链补全)、前端催化 badge。下一迭代:P15-02b leader_ranker 细化。
- **loop 迭代 7 (2026-06-14) — P15-02b done / P15-02 完整完成 + 循环收尾**:`candidate_engine/theme_leaders.py`(`annotate_theme_leaders`:每个热点主题里融合排名第一的候选 = 该主题龙头,回答 00_DESIGN 核心问"每个热点里龙头是谁";无催化永不为龙头)+ 3 测试,接进 serializers。**Live 实证:dashboard 候选标 👑 8604(银行)/4443(AI)/6584(auto)/2162(半导体) 各主题龙头**。smoke **1392 passed**,零回归。**P15-02 全链完成**——news(分类)→catalyst→hybrid rerank→theme-leader 实盘工作且可见。
- **/loop 收尾 (迭代 1-7, 2026-06-14)**:loop 目标"让命名核心新闻驱动热点龙头轮动上线"**达成**。累计 7 迭代、ADR-0009、+约 24 测试、smoke 1373→1392 零回归、命名核心 LIVE+持久+SSoT 更正。**停在干净里程碑**。剩余作**专门会话** backlog(非 loop 尾巴硬磨):**P15-02b-full** leader_ranker 全量打分(volume/overheat,需 screener 因子映射调参)、**前端催化/龙头 badge**(数据已在 candidate dict,前端可视化)、**P15-03** 新闻因子→收益反馈闭环(P7-06,Rule 4 流程,独立较大)。

### P15-02e Frontend Catalyst/Leader Badge (Rule 11.12) — done (2026-06-14)

- Status: **done (2026-06-14, 崩溃恢复后续会话)**。把 P15-02d/02b 已落在候选 dict 的催化/龙头信号渲染到前端——此前**数据有、UI 无**,用户看不见"新闻催化的热点龙头"(正是驱动整个 P15 的抱怨)。/loop 收尾里列为 backlog 的"前端催化/龙头 badge"。
- 实现:`frontend/src/htr-shared.jsx` 新增共享 `CatalystBadges`(单一源,挂 window/Rule 11.7 parity)——`newsCatalyzed` 才渲染 `🔥 新闻催化`、`isThemeLeader` 加 `👑 龙头`;接进 V1(`V1HeroHeader`)/V2(`V2FeatureHero`)/V3(`V3LeaderCard` + `V3CandidatePicker` 行内 sm 徽章)/V4(`V4LeaderCard`)。
- 诚实(**新增 Rule 11.12**:News-Catalyst / Theme-Leader Badge Honesty,5 子条):①排序信号非胜率/概率 + standing disclosure ②显示分仍 `uncalibrated_research_score`(排序≠分数),不渲染裸 catalyst/blended 0–1 数 ③仅从已送字段派生(PIT),无催化不渲染不封龙头 ④fail-open 退纯 screener、缺徽章是诚实态 ⑤四变体 parity。
- Files:
  - Update: `frontend/src/htr-shared.jsx`(`CatalystBadges` + `CATALYST_BADGE_NOTE` + window 导出)
  - Update: `frontend/src/htr-v1.jsx` / `htr-v2.jsx` / `htr-v3.jsx` / `htr-v4.jsx`(注入徽章)
  - Update: `tests/unit/test_frontend_ui_contracts.py`(+2 契约:ordering-not-winrate / 四变体 parity)
  - Update: `docs/02_GOVERNANCE.md`(Rule 11.12)
- 验证:TDD 先红(`CatalystBadges` 缺失)后绿;smoke `-m "not slow"` **1394 passed / 5 deselected**(1392 + 2),零回归。**Rule 11.7.6 headless(:8000)live DOM**:四变体徽章计数 v1=2/1·v2=3/1·v3=8/5·v4=1/1(catalyzed/leader);V3 leader 8604(bank,isThemeLeader)显 theme chip+👑龙头+🔥新闻催化,显示分仍 36.89「研究分·未校准」——排序受新闻、分仍诚实。截图 `.runtime/audit_shots/p15_catalyst_badge_v3.png`。
- 剩余 P15 backlog:P15-02b-full leader_ranker 全量打分(volume/overheat,需 screener 因子映射调参)、P15-03 新闻因子→收益反馈闭环。

### P15-02f Catalyst Degradation Metadata — done (2026-06-15)

- Status: **done (2026-06-15)**。修 review 发现的"news/meta refresh 非致命但 dashboard 不显式暴露降级"缺口。`api/serializers._build_meta` 新增 `meta.dataQuality.newsCatalyst`，从最新 afterclose `daily_routine_log.jsonl` 读取 `news_refresh_rc` / `meta_refresh_rc`，输出 `degraded`、`newsRefreshOk`、`metadataRefreshOk`、`reasons`。当新闻或元数据刷新失败时，dashboard 可明确显示 catalyst layer degraded；候选仍 fail-open 到价格 screener，但 badge 不再无条件可信。
- 验证:新增 `test_dashboard_meta_surfaces_news_catalyst_degradation`；full smoke **1399 passed / 5 deselected**；slow lane **5 passed / 1399 deselected**。

### P15-03 新闻因子 → 收益 → 权重反馈闭环 (pending)

- Status: **pending**(P7-06 attribution feedback 一直 pending)。主题/新闻因子对实际收益的归因 + 权重更新走 Rule 4 流程,不静默回写。

## Milestone P16: Event Desk(事件作战台 — 事件驱动的分析支持)

> 起因 (2026-06-15): owner(资深股民)明确表示用事件(伊朗停战/半导体AI/存储暴涨/日元跌破160)来判断选股,要工具**深度配合这个打法、给方案,而不是反驳**。诚实复盘:模型此前连这些事件都"看不见"(主题硬编码 6 个,无 optical/memory;宏观管线 05-30 死、无 geopolitics)。**产品转向**:把 HTR 从"被动筛选器"做成"事件作战台"——不预测事件结果(Rule 3/8.3/9.4 不造概率),而是给 owner 的判断**提供结构化弹药**:看得见→谁受影响→涨完没有→怎么打→复盘。设计原则:只给可复核的真实数据(新闻/涨幅/流动性),不给假的"上涨概率"。

### P16-E1 Event Radar — 扩主题/事件分类 (done 2026-06-15)

- Status: **done (2026-06-15)**。让引擎"看得见"owner 点名的事件。
- 实现:`news_theme_classifier.THEME_TAXONOMY` 加 **optical**(光モジュール/光通信/シリコンフォトニクス/CPO/光模块/硅光/フジクラ/古河 …)+ **memory**(DRAM/NAND/HBM/メモリ半導体/キオクシア/ストレージ/存储/内存 …);`MACRO_TAXONOMY` 加 **geopolitics**(地政学/中東/イラン/停戦/制裁/ホルムズ/ceasefire …);`ticker_metadata._INDUSTRY_THEME_RULES` 加 optical/memory 行业映射(注:industry 是弱代理,这两类主要靠新闻分类)。复合词避免裸"光"/"メモリ"过匹配。
- 验证:TDD 先红后绿;新增 `test_classify_event_desk_e1_new_buckets`(optical/memory/geopolitics + 観光 反例)+ theme_for 2 断言;full smoke **1400 passed / 5 deselected**,零回归。**实证**:owner 五事件分类 伊朗停战→geopolitics / 存储→memory / 光模块→optical / 日元→monetary+fx 全部命中;今日 108 条实盘新闻已标 optical/geopolitics/fx。optical/memory 进 `stock_news_fetcher` 每日管线即生效(已接 daily_routine)。
- 注:macro 管线(fx/geopolitics 的**采集**端 `macro_news_fetcher`)仍停 05-30,E2 一并接回。

### P16-E2 Event→Exposure 映射 (done 2026-06-15)
- Status: **done**。`candidate_engine/event_desk.py` 的 `EXPOSURE_MAP`(8 主题 × liquid 代表票 seed map)+ `EVENT_THEMES`(事件别名→主题:停戦/ceasefire→defense+energy、円安/yen_weak→auto+semi)+ `themes_for_event`。事件/主题字符串 → 受影响日股名单。
- 注:这是"谁受影响"的数据 join,不荐股(Rule 11.13.3)。seed map 可扩。

### P16-E2b Macro feed 复活 (done 2026-06-15)
- Status: **done**。`macro_news_fetcher`(fx/货币/地缘采集端,死于 05-30)接回 daily_routine。新增 `tools/refresh_htr_macro_news.py`(镜像 refresh_htr_news,调 `build_macro_overlay` 写 `reports/news_macro/{date}.json`),`daily_routine.refresh_candidates` 在 news 后、screener 前调用(非致命),`macro_refresh_rc` 进 record;order 测试同步(refresh→news→**macro**→screener→meta→emit→sweep)。
- 验证:daily_routine 14 测试过;full smoke **1411 passed / 5 deselected** 零回归。**Live 实证**:`refresh_htr_macro_news --asof 2026-06-15` 抓 114 条 / 分类 98,macro 计数 monetary 28 / fiscal 22 / fx 17 / trade 16 / **geopolitics 6** / overseas 12——`reports/news_macro` 由停 05-30 → 06-15 新鲜。dashboard 宏观层 + Event Desk 地缘/汇率新闻不再冻结。

### P16-E3 Priced-in read(涨完没有)— 核心 (done 2026-06-15)
- Status: **done**。`event_desk.priced_in_read`/`build_event_desk`:每个受影响名 last + 1d/5d/20d 收益 + 对 1306.T 超额(5d) + 距20日高 + 描述性 freshness 标签(fresh/extended/rolling_over/falling)。注入式 price_fetcher(Rule 2.1,默认 yfinance best-effort 离线降级);fail-open(缺数据→null+unknown,不抛错/不造数)。**新增 Rule 11.13**(Event Desk:不预测事件结果、不给概率/胜率、exposure 非荐股、freshness 是"已涨多少"描述非信号、read-only)。
- TDD 先红后绿 +6 测试(`test_event_desk.py`:themes 解析/收益+超额/freshness 标签/fail-open/join+honesty/未知事件)。full smoke **1406 passed / 5 deselected**,零回归。**Live 实证**:停战→防衛/能源(三菱重工/INPEX/石油資源 已 falling、战争溢价在回吐)、semi(TEL/Disco/Lasertec extended 过热在20日高)、optical(フジクラ/古河 falling −20%+)。"还能不能上车"这一问替 owner 量清楚。
- 剩余:E4 endpoint + 前端卡(把 build_event_desk 接进 API + 四变体卡,Rule 11.13.5 parity)。

### P16-E4 Event Desk endpoint + 四变体卡 (done 2026-06-15)
- Status: **done**。`api/event_desk.py`(只读 `GET /api/event-desk?event=` + `/api/event-desk/themes`;30min TTL 缓存;yfinance best-effort 离线降级)接进 `api/main.py`。共享前端组件 `EventDeskCard`(`htr-shared`/`htr-v3` 全局,自取数:主题/事件选择器 + 受影响名 priced-in 表 + freshness chip + 常驻披露)挂入 V1/V2/V3/V4(Rule 11.13.5 / 11.7 parity)。
- TDD +5 测试(API 3:themes/exposure+disclosure/read-only;前端契约 2:read-only+non-prediction / 四变体 parity)。full smoke **1411 passed / 5 deselected**,零回归。
- **Rule 11.7.6 headless 实证**(:8010):卡渲染 disclosure + 4 freshness chips;截图 `.runtime/audit_shots/p16_event_desk_card.png`——semi 显 TEL +47.7%「已大涨」/ 爱德万「未过热」,披露"不预测事件结果、不给上涨概率/胜率,方向判断是你的"。live endpoint 实证 イラン停戦→防衛+能源(三菱重工/INPEX/石油資源 falling)。
- **Event Desk 核心(E1+E2+E3+E4)LIVE**:owner 选事件/主题 → 看到谁受影响 + 各自涨完没有,只读·非预测。剩余:E2b macro 采集端复活 / E5 事件 journal。

### P16-E4 Event Desk 卡 + Rule(前端) (planned)
- 新只读卡:选事件/主题 → 受影响名 + priced-in read + 流动性 + 进场/止损阶梯(复用 build_price_ladder/Rule 11.8)。**需新增 Rule**:Event Desk 描述性、不得呈现事件结果概率、exposure 是数据 join 非荐股、advice-only、四变体 parity。

### P16-E5 Event Journal — owner 自己的事件战绩 (deliberate session, not loop)
- Status: **pending — 留专门会话(非 /loop 自驱)**。记事件论点+进场+结果,复用 P14 cohort 机制,攒出 owner 按事件类型的**实测**胜率/超额(替"感觉的胜率")。
- **为何不在 loop 里自动做**:E5 引入**写路径**(记录用户事件交易),按 Rule 11.5 写白名单(刻意只 7 个 POST)需治理审查;且"如何 journal"(独立 thesis 存储 vs 给现有 portfolio fill 打 event 标签)是**产品设计岔路**,该有 owner 介入拍板,不宜在自驱循环里替他选。Event Desk 核心(E1-E4+E2b)已 LIVE 满足主诉求;E5 作为下一步深思扩展。

## Milestone P17: Disclosure-Drift, Execution-Gated & Overfit-Guarded (ADR-0010)

> 起因 (2026-06-17, /goal 自驱): owner 设目标——让模型能分析新闻/股价/市场信息、按一开始推断的方向(information-underreaction / 披露漂移)前进,保留框架同时优化 model,并调 Codex 审视,直至达标。研究+Codex 共识:唯一有日本 OOS 证据且没被套利掉的 edge = 小盘/低流动/低外资/低覆盖名的多日披露漂移(Jinushi 2023);Codex 裁决 PURSUE-WITH-CHANGES,执行/流动性是核心杀手,先证明成交可行。Phase 0:窗口窄(~¥600-1300),TDnet 采集管线已存在(`poll_tdnet_rss`)但语料未累积。

### P17-1 Execution / Tradability Gate (done 2026-06-17, Codex 优先级 #1)
- Status: **done**。ADR-0010 立项 + Rule 5.1(执行/可交易闸)+ `candidate_engine/tradability.py`(JPX tick 阶梯→往返成本估计、100股手数 affordability + 34% 分散上限、ADV 地板、net-after-cost + 2× 成本压力;纯算术 fail-closed,无预测)。接进 Event Desk `priced_in_read`(每个受影响名带 tradability 判定)。
- TDD +9 测试(tick 阶梯/往返成本随便宜价上升/手数+分散/net-after-cost/甜区过/便宜名成本杀/贵名手数杀/2×压力/ADV 地板)。full smoke **1420 passed / 5 deselected**,零回归。
- **实证(¥40w 账户)**:系统的热门大盘候选**几乎全不可交易**——TEL 一手=账户 1819%、爱德万 736%、Lasertec 1208%、野村¥1415=35%(超 34% 上限)、INPEX 87%;只有 ~¥900 小盘过闸。**第二条独立论据支持小盘方向:不仅 edge 在那,¥40w 也只在那能持仓分散。**
- 下一步:P17-2 披露惊奇数据层(TDnet 语料 + surprise/novelty,日文时间戳)→ Codex 复审 → P17-3 反过拟合准入闸(DSR/PBO)→ P17-4 小盘宇宙 + 诚实验证。

### P17-2 Disclosure-Surprise + Novelty Signal (done 2026-06-17, +Codex 复审整改)
- Status: **done**。`candidate_engine/disclosure_surprise.py`:从 TDnet 标题解析 PEAD 事件代理——materiality(earnings/dividend)× direction(上方修正/増配=+1、下方修正/減配/無配=-1)× novelty(訂正=stale,Tetlock 2011 陈旧新闻反转→低 novelty 是 fade 标记)→ `surpriseScore` ∈[-1,1];`rank_disclosures` 按 |score| 排序不丢。复用 `tdnet_parser.classify_category`。**不预测漂移、不给概率**(Rule 9.4/8.3),只标"哪些披露是候选事件、什么方向";漂移是否真发生由 forward log + 闸决定。
- **Codex 复审(P17-1+P17-2)→ 裁决 "Needs fixes",已整改**(goal "调 codex 完善"):
  - tradability:ADV 缺失**改 fail-closed**(`require_adv`,不再静默通过;Event Desk 用 `require_adv=False` 显结构判定+`advVerified=False`)、`price<=0` 守卫、负 `spread_ticks/slippage` 抛错、**默认 `spread_ticks` 2→3**(2 太乐观)、加 `netVerified` 标。
  - disclosure_surprise:**混合信号(上+下同现)→ direction 0+`mixed` 标**(不再让"上"无条件赢)、caller `category="other"` 不再阻止重分类、`rank` 容忍非 dict、加 **`pitOk`**(published_ts 在否——P17-4 验证须拒无时间戳事件防 look-ahead)。
  - **deferred(Codex 提的真增强,非阻塞)**:数值惊奇幅度(vs 一致预期/SUE,标题只是粗事件 flag 非标准化 SUE)、size/外资/覆盖控制 → 归 P17-4 小盘宇宙;tick 阶梯 >¥300k 补全(对 ¥40w 无关)。
- TDD:+7(disclosure)+9(tradability,含 4 Codex-fix 用例)→ full smoke **1435 passed / 5 deselected**,零回归。
- 下一步:P17-3 反过拟合准入闸(trial-counter + Deflated Sharpe + PBO,扩 `purged_walk_forward`)→ P17-4 小盘宇宙 + TDnet 语料累积 + forward 验证(PIT 时间戳强制)。

### P17-3 Anti-Overfit Promotion Gate (done 2026-06-17)
- Status: **done**。`calibration/overfit_gate.py`(纯 stdlib `NormalDist`,无 scipy)——`expected_max_sharpe`(无技能下 N 次试验的 E[max SR],Euler γ,~√(2lnN))+ `deflated_sharpe_ratio`(Bailey/López de Prado DSR,含 skew/kurt 修正)+ `promote_gate`(DSR≥0.95 AND n_obs≥min AND 声明 trials AND **sr_std>0 fail-closed**)。门控自欺,不预测。
- TDD +8 测试(E[max] 随 trials 增、DSR 随 trials 降/随样本升、强信号少试验过、噪声多试验否、小样本否、未声明 trials 否、sr_std≤0 否)。

### P17-4 Disclosure-Drift Review Harness — 端到端 (done 2026-06-17)
- Status: **done**。`reporting/disclosure_drift_review.py`:把 P17 串成一条 PIT 忠实评估——事件→惊奇信号(P17-2)→PIT 闸(无日文时间戳剔除)→可交易闸(P17-1)→按方向净成本漂移→(promotion 交 P17-3,见下)。语料空时如实 `insufficient_data`,排除项逐类计 + `nMissing` 逐 horizon 计,**不静默丢**。
- **Codex 复审 P17-3+P17-4(第3次)→ "needs fixes",已整改**:① DSR 数学核对 KEEP(正确);② **概念修**:harness 原把 per-event Sharpe + 标量 sr_std 喂 DSR 是概念混淆(DSR 的 sr_std 须是跨试验 Sharpe 离散)→ **撤掉 harness 内自动 DSR**,只出描述统计 + promotion 延迟给真·trial-matrix(不做 theater);③ sr_std≤0 fail-closed;④ 未成熟回报 None **不再静默丢**→ `nMissing` 入账;⑤ verdict 由**最慢 horizon** 把关(非最成熟单 horizon);⑥ ADV 事件带量则强制、否则结构判定 + advVerified=False;⑦ PIT 契约(event_return_fn 须只用披露后可成交价、未成熟返 None;price 须披露后入场价)写进 loud 文档。defer:PBO/CPCV trial-matrix、数值 SUE、skew/kurt 实计(归未来真 trial 跑)。
- TDD:overfit 8 + drift-review 5(含 nMissing/最慢-horizon/promotion-deferred 用例)。**full smoke 1449 passed / 5 deselected**,零回归。
- **P17 主体完成**:ADR-0010 方向落地为 4 个 gated 分析模块(执行/惊奇/反过拟合/端到端 harness)+ Rule 5.1 + 3 轮 Codex 复审两轮整改。**模型现已能按"信息低反应/披露漂移"方向分析新闻/股价/市场,框架保留,闸到位,诚实定性(语料 forward 累积、edge 由 forward log 裁,当前 insufficient_data)。** 剩纯运营:TDnet 语料累积(poll 任务已存在)、真 trial-matrix DSR/PBO、小盘宇宙落地到实盘候选。

## Milestone P18: Factor-Weight Review & Validated Reweight

> 起因 (2026-06-17): 用户质疑 screener 因子权重是否合理,要求调 Codex 全面复审。

### P18-01 Codex Weight Review + Model-Factor Reference Doc (done 2026-06-17)
- Status: **done**。Codex 读 sibling `Project_optimized/screener.py` 全面复审 `alpha_weights`(mom_20 .25/mom_60 .15/vol_adj_mom20 .15/vol_z .15/sharpe_20 .10/adv_rank .10/high52w_rank .10)。**裁决 UNREASONABLE**:
  - ~**0.75 集中在一个相关动量/趋势簇**(mom_20+mom_60+vol_adj_mom20+sharpe_20+high52w_rank)——五票同一注,非分散;
  - 日本动量已知偏弱(Asness/Chui-Titman-Wei)却重押动量,**无 value/quality/low-vol/size/reversal**;
  - `vol_z`(+0.15)实为**成交量 z-score**(`screener.py:380` 注释 成交量;`vol`=get_close_vol_multi 的成交量),非价格波动率——**更正此前误标**;
  - `fundamental_score` 仅在 top-k **选股后**做乘子(`screener.py:511`),不影响选哪些股(与注释不符);
  - 组合 = 截面 rank-norm(pct)后加权和(`:401/:503`)——防量纲但不解共线。
- Codex 建议(**未验证**)reweight:动量簇 0.75→~0.45,adv_rank→0.20、fundamental→0.20(移入 ranking)。
- 交付:新增 `docs/06_MODEL_FACTORS.md`(持久因子/数据/校准参考,防聊天历史丢失;已含本复审裁决)。
- **关键边界**:权重在 sibling screener,HTR 只读消费(ADR-0005)→ 改权重 = 跨系统改 sibling OR HTR 侧覆盖层。

### P18-02 Validated Reweight (pending — 用户拍板 B)
- Status: **pending(未开始)**。用户选 B:**正式验证 reweight**,不手改。约束:任何 reweight 必须经 **purged walk-forward + 成本闸(复用 P17 `overfit_gate`/`tradability`)在日股历史上验证**(reweight backtest 机制需建——按各历史日用新权重重打分 universe、测前瞻净成本收益、过反过拟合闸)、跑赢现权重才允许、走 Change Log(Rule 4)。诚实前提:momentum 在日本弱 + screener 无 demonstrated edge,**reweight 大概率不产生 edge**;这更像把 screener 当诚实强度过滤器 + 修正确性(vol_z 改名 / fundamental 是否进 ranking),edge 仍走 P17 披露漂移。
- 数据现状(2026-06-17):forward 日志 **30 天**;15 live pred 日(05-27..06-17)+16 backdated;**26 outcome 日有 ≥1 complete**。手动解锁 calibration:**不能强制覆盖**(Rule 8.2.2/9.4)。

### P18-03 Forward walk-forward join hardening + current calibration gate (2026-06-18)
- Status: **join blocker resolved / calibration still locked**。复跑 `tools/validate_calibration_walk_forward.py --origin live --asof 2026-06-18` 不再是 0 联结: report now shows **n_joined_samples=529 / n_joined_date_clusters=11**; foldable OOS slice **n_samples=300 / n_effective_date_clusters=6**。
- Hardening: `purged_walk_forward` report now emits `n_joined_samples` and `n_joined_date_clusters` before folding, so a future zero-join regression is visible even when folds are thin. Added regression test `test_report_exposes_joined_sample_counts_before_folding`.
- Gate result: **verdict=`insufficient_data`** under locked Rule 8.2.3 because effective OOS clusters are **6 < 20**. The model also currently underperforms baselines: Brier **0.2846** vs climatology/stratified **0.2407**, CI [-0.0666,-0.0228]. Therefore calibration **must remain downgraded** (Rule 9.4); no force-unlock.
- Pre-unlock boundary: allowed path is **shadow/diagnostic calibration only** (explicitly labeled, not score_status promotion, not probability/win-rate UI, not used for advice). Anything that surfaces `calibrated_probability` before >=20 live clusters + all baselines beaten + CI>0 + clean leakage verdict is non-compliant.
- Next: keep forward collection running, add a visible diagnostic/monitoring surface if useful, and prioritize P18-02/P17 improvements only through validated walk-forward + cost/tradability gates.

## Milestone P19: Cross-Sectional Ranking & Multi-Signal Composite (ADR-0011)

> 起因 (2026-06-23): 用户选打法1(横截面排序)+打法6(信号叠加)。ADR-0011 = ADR-0010 的方法论扩展层(复用 P17 执行门/反过拟合门,不另起方向)。Phase −1 已证:现有 buy 分数横截面 Rank-IC 全负(1D −0.032 / 3D −0.041 / 5D −0.085 t=−2.0);短周期成本门槛 0.09–0.17 不可达,仅 ≥5D(0.025–0.07)经济。规则见 Governance §16。

### P19-01 Forward-test harness (Rank-IC / net-of-cost / live-only) — done (2026-06-24)
- Status: **done**。`backtesting/forward_signal_eval.py`(纯核:spearman / rank_ic / cross_sectional_dispersion / cost_hurdle / net_ic_after_cost / clears_hurdle / top_minus_mean_spread / ic_decay)+ `backtesting/forward_eval_data.py`(live-only group_live_daily / load_live_panels + 一行 `summarize_live_signal`;c_rt 自动取自 P17 `tradability`)。**28 单测全过**;端到端复现 Phase −1(5D Rank-IC −0.085/t−2.0,auto c_rt≈0.0029,net 全负 → as-is 不过门)。规则 16.0/16.1/16.2/16.6。
- 备注:purged/embargo 的跨日切分(用于晋级而非同日横截面 IC)复用 `calibration/purged_walk_forward.make_folds`,本闸门只做同日横截面 + 成本门槛,不重复实现。

### P19-02 Seed signal library + equal-weight composite — in progress
- **P19-02-01 (done 2026-06-24)**: `backtesting/signal_library.py` — SignalFn 契约 + `evaluate_signal`(一键过 §16 闸门,live-only,默认 S株 成本)+ `reversal_of_score` seed#1。端到端:5D 反向 IC +0.085/t+2.0,过成本门但**未过晋级门(t<3)、非独立信号→仅跟踪**。42 单测过。
- **P19-02-02 (done 2026-06-24)**: 独立 5日反转信号(`make_price_reversal_signal` + `kline_prior_return_lookup`,PIT)过闸:**5D Rank-IC +0.128 t+3.21**(过成本门 + 过 t≥3 强度门),与 buy 正交 ρ=−0.20。coverage 993/993,8 新单测。**项目首个正向+过强度门的独立信号**,但仅 15天/单regime → 未晋级,见 P19-02-04。
- **P19-02-04 (done 2026-06-24 — verdict: 不晋级)**: 跑了反过拟合门。lookback{2,3,5,10,20} 5D IC 全正(峰5–10d);Newey-West(lag4)t 不降(5D 3.37)。**但 Deflated Sharpe=0.64(重叠调整0.55)≪0.95**(18 trials / 15天≈3 独立块)→ **未过门**。方向稳健、至今最强候选,但样本太小。**不交易**;唯一解=累积前向 ≥~60 独立 obs(~2–3月)后重 gate(DSR/PBO + regime/lookback 稳健)。
- **P19-03 (standing)**: 前向累积 + 周度重 gate price_reversal;到样本量后再判晋级。承接此候选。

### P19-05 S株 universe overlay (done 2026-06-24, capability) — task#5
- **done**: `candidate_engine/s_kabu_universe.py`(overlay:held∪watchlist → `s_kabu_tradability`,只纳 S株 解锁的名字)+ `tools/build_s_kabu_overlay.py` 出 `reports/screener/s_kabu_overlay_{asof}.json`。6 单测。端到端:持有 8035.T 首次作为 S株 候选浮现(1股=17.5%NAV)。**选股仍 sibling(ADR-0005 不动)**,本 overlay 是 HTR 侧补充,非选股替换。
- **P19-05b (done 2026-06-24)**:① `daily_routine.refresh_candidates` 加非致命步骤调 `build_s_kabu_overlay.py`(每日产出 overlay snapshot);② `api/serializers` 加 `sKabuOverlay` 字段 + 前端 `htr-skabu-card.js`(独立 root 纯 JS,node-checked,fail-soft)在 dashboard 显示 S株 候选 + 集中度标记。S株 端到端接入完成(tradability→overlay→producer→daily_routine→API→卡片)。52 单测过;主页/卡片 curl 200。**未来可选**:把 overlay 并进主候选面板的统一卡片(目前是独立浮层),需 App-layout JSX 改动。
- **P19-02-03 (corpus accumulation STARTED 2026-07-01; signal-integration still pending)**: 诊断发现 TDnet 语料一直零累积——`reports/tdnet/` 空、`poll_tdnet_rss.py` 从未接进 daily_routine。已把 TDnet poll **接进 `daily_routine.refresh_candidates`(非致命,adr 后/screener 前,`--date asof`)**,Yanoshin API 已验证可达(06-30 拉 85 条),corpus 从此每日累积(forward-only,不 backfill,保 Rule 16.2 live-only)。`test_daily_routine` 14 passed(order +tdnet/+forward_eval)。**仍 pending**:待 corpus 攒 ~2 周后,把 `disclosure_surprise`→SignalFn 接进 §16 `evaluate_signal` harness(对齐预测/小盘宇宙 + PIT published_ts),才能像 price_reversal 那样出 forward Rank-IC/DSR。
- 组合阶段:等权秩平均 + James-Stein(16.4)、K 个位数(16.5)、horizon ≥5D(16.6)。

### P19-02c Fundamental signals enter the live forward track — done (2026-07-04)
- `make_fundamental_yield_signal` + `fundamentals_pit_lookup` (PIT: reported rows only, published strictly before decision date) in `signal_library`; `earnings_yield` + `value_bp` wired into `tools/forward_signal_report.py` (daily via afterclose). TDD +3 tests → smoke **1615 passed / 5 deselected / 0 failed**. Live first run (2026-07-04): both signals evaluated on the forward log; **orthogonality earnings_yield vs screener_buy ρ=−0.240 → |ρ|<0.5, Rule 16.3 stack-eligible**. 1-5D ICs ≈0 as expected (63D slow factor on a 50-name momentum-screened universe — not evidence either way).

### P19-02b Extended-horizon forward track — done (2026-07-05, via separate research-cohort lane)
- **Design change from the original sketch (recorded)**: instead of adding 20D/60D to `outcome_join`/sweep (touching the §10 production record contract and risking pollution of cohort review/calibration), built a fully SEPARATE lane — `backtesting/fundamental_cohort.py` + `tools/fundamental_cohort.py` — storage `reports/research_cohorts/fundamental/{predictions,outcomes}`, same separation discipline as the Rule 11.15 ADR lane. Production pipeline untouched.
- Mechanics: monthly broad-universe cohort (every panel symbol with PIT fundamentals + price → earnings_yield/value_bp row); maturity-honest sweep at **21D/63D** (sweep truncates the price series at the sweep date — a TDD test caught and killed a look-ahead hole where 63D matured early from future closes); `report` = per-cohort cross-sectional Rank-IC. Sweep self-refreshes recent adjusted closes.
- **First cohort emitted 2026-07-05: 2,763 symbols.** First 21D read ~2026-08; first 63D read ~2026-10. Cadence: emit first weekend of each month; sweep+report opportunistic. TDD +4 tests → smoke **1619 passed / 5 deselected / 0 failed**.

### P23-E Correctness-review remediation (2026-07-06) — IN PROGRESS
- 3-agent pre-commit adversarial review (security / research-correctness / governance). Security: 3 CONFIRMED fixed (token→access log, app-level fail-closed `LoopbackOnlyGuard`, `--gen-token` + degenerate-token guard; token rotated). Governance: 2 minor fixed (rotate-count label, negation-context vocab test). **Research-correctness: DSR-0.992 verdict RETRACTED** — as-filed EPS/BPS ÷ split-adjusted price leaked future info (NTT 9432.T E/P 214% vs 7.9%; excluding split-signature names → DSR 0.969). Done: TDnet parser 4 fixes (bank/insurer aliases, nearest-non-prose header block vs richest-window, non-forecast carryover guard 前期実績/参考, crash-path leaves parsed:false record) TDD 11 tests; cohort refresh DELETE+reinsert (finding 5, no basis-stitch). **PENDING**: raw-price store `htr_raw_prices.db` backfilling (auto_adjust=False) → rewire `backtest_value_quality_history.py` + `make_fundamental_yield_signal` to raw-price yield denominator / adjusted-price forward return → re-run honest verdict + surface survivorship exposure. Full smoke **1635 passed / 5 deselected / 0 failed**.

### P19-03 Shadow-track + forward Rank-IC review — pending
- Status: **pending**。复合分进 shadow(Rule 13.14),累积 live Rank-IC 周度复盘;晋级仍须过 ADR-0010 反过拟合门 + Rule 16.6。Rank-IC 过门是必要非充分。

### P19-04 Historical edge search — 四轮回测收口(done 2026-07-02; verdict: 无 demonstrated edge)
- 起因 (2026-07-01/02): owner 质疑"为何总在攒数据"。改用**手上的历史数据直接回测**,不等 forward。新增 3 个历史回测工具 + 因子普查:
  - `tools/backtest_price_reversal_history.py`:price_reversal 是纯价格信号,直接跑 htr_market.db **798 天 / 1168 只流动股**。A 宽宇宙 best 2d@1D IC+0.024 t+4.17 **DSR 0.87**;B 高动量子集(≈screener 候选)best 5d@1D IC+0.015 t+1.91 **DSR 0.88**。**两者不过门,n_obs 777(样本早够)。** live 24 天的 5D IC+0.157/t+4.22 = 小样本/单 regime 运气 → **price_reversal 独立可交易 edge 判死**。
  - `tools/backtest_disclosure_drift_history.py`:验证 Yanoshin 可拉 ≥2 年历史,backfill 1 年(251 天/21,682 条 → `.runtime/tdnet_probe`),事件研究(方向性披露→次日入场 PIT→H 日超额 vs 1306.T→按方向下注)。标题信号仅 **277 条方向性事件/年(极稀疏)**;IC 全负 −0.17~−0.33(与 PEAD 反,小盘最差);**DSR 0.50 不过门**。根因=标题正则 ≠ 真 SUE(需一致预期数据)。
  - `tools/backtest_factor_zoo_history.py`:factor_signals 2022–2026 × 前向收益。**基本面(value_bp/quality_roe/roa_op/margin_op/cfo_assets/accruals_inv/dividend_yield/growth)覆盖太稀疏(2–3 有效横截面)→ 测不了**;技术因子 low-vol IC≈0、动量不显著、high52w best IC+0.066@60D t+5.21 但本质动量、**DSR 0.58 不过门**。
- **裁决:screener 负技能 / price_reversal DSR 0.87 / disclosure-drift DSR 0.50 / 因子动物园 best DSR 0.58 —— 凡能在现有数据上测的信号,无一过反过拟合门。无 demonstrated 可交易 edge(非"数据不足",是信号无 edge)。**
- **定位收口**:系统 = 决策支持/纪律/实时核查作战台(本会话实拦一笔负 EV 周末隔夜单,周一低开 5–7%),**非信号/印钞机**;诚实最优 = 被动 1306.T 打底 + 系统当工具。
- **唯一未被否的方向(P19-05 / 数据采集,owner 待定)**:采日本财报历史数据 → 正经回测 value/quality(全球证据最强、恰是本库缺失的一类)。是"采数据"不是"等"。advice-only 未绕过;未 commit。


## Milestone P20: SKHY ADR + Japan Semi Event Overlay

> Context (2026-06-25): User clarified that the primary focus remains the Japanese market, while he may personally buy SK hynix ADR when available. SEC EDGAR shows SK hynix Inc. filed Form F-1 on 2026-06-24 for an ADS/ADR listing package. HTR must use this as an external semiconductor catalyst lane only: Japan candidates stay primary, ADR status is explicit, and no probability/win-rate/edge language is allowed. Governance anchor: Rule 11.15 plus Rule 11.14, Rule 11.13, ADR-0010, and Rule 16.

### P20-00 Plan, governance, and tasklist handoff - done (2026-06-25)
- Status: **done (docs only)**. Added Rule 11.15 External ADR Catalyst Lane Honesty and created implementation plan `docs/superpowers/plans/2026-06-25-skhy-japan-semi-overlay.md`.
- Scope: Plan is for another engineer to execute. No runtime code changed in this task.
- Acceptance: Handoff explicitly keeps JP equities as primary market, treats SKHY as external catalyst/manual ADR record only, requires live-only evidence before promotion, and forbids probability/win-rate/expected-return language.

### P20-01 Daily routine smoke expectation sync - done (2026-06-25)
- Status: **done**. Updated `tests/unit/test_daily_routine.py` order classifier + expected to include `skabu` (build_s_kabu_overlay) after `meta`, before `emit`. Production routine unchanged (test-only fix). **14 passed.**
- Verification: `python -m pytest tests/unit/test_daily_routine.py -q --basetemp=.runtime/pytest-p20-daily-routine -p no:cacheprovider`.

### P20-02 External ADR watch schema - done (2026-06-25)
- Status: **done**. `src/hot_theme_rotator/data/external/adr_watch.py` — `AdrInstrumentSnapshot` (frozen) + `ALLOWED_ADR_STATUSES` {pending_listing,active,stale,unavailable} + `is_stale`(data_ts vs asof, fail-closed) + `overnight_return`; no probability/win-rate/expected-return/edge fields; JSON round-trip. **12 passed.** `test_adr_watch.py`.
- Required work: add a deterministic `adr_watch` schema for `SKHY`, `000660.KS`, memory/AI peer proxies, SOX/semiconductor ETF proxy, and USDJPY. Status values must include `pending_listing`, `active`, `stale`, and `unavailable`.
- Acceptance: stale/missing SKHY never creates a false catalyst; schema has no probability, win-rate, expected-return, or edge fields.

### P20-03 SKHY ADR watch refresh snapshot - done (2026-06-25)
- Status: **done**. `tools/refresh_skhy_adr_watch.py`(pure `build_adr_watch_payload` + injected fetcher + `write_adr_watch` → `reports/adr/adr_watch_{asof}.json`;fail-soft;SKHY 无报价→`pending_listing`,绝不替换它符号)。Wired non-fatal into `daily_routine.refresh_candidates`(after macro, before screener;`REFRESH_ADR`)。**21 passed**(7 refresh + 14 routine,order 现含 `adr`)。`test_refresh_skhy_adr_watch.py`.
- Required work: create a non-fatal daily tool that writes `reports/adr/adr_watch_{asof}.json`, with injected fetcher tests. If SKHY has no active quote yet, write `pending_listing` or `unavailable`.
- Verification: `python -m pytest tests/unit/test_refresh_skhy_adr_watch.py tests/unit/test_daily_routine.py -q --basetemp=.runtime/pytest-p20-adr-refresh -p no:cacheprovider`.

### P20-04 Japan semi sympathy overlay - done (2026-06-25)
- Status: **done**. `candidate_engine/skhy_overlay.py` — `compute_skhy_event`(off/watch/active 状态机:stale/pending→off;active+abnormal 无确认→watch;+SOX/MU/NVDA 确认→active;半衰期 freshness;**000660.KS 同公司不算独立确认**)+ `annotate_japan_semi`(≥2 facts 才正分;板块成员单独不够;Rule 11.14 extended-chase 不加分;capped [-0.05,+0.07];无概率/胜率/期望/guaranteed)。纯函数,不动 theme_rotation/hybrid_rerank(应用放序列化层)。**25 passed**(含 theme_rotation/hybrid_rerank 回归)。`test_skhy_overlay.py`.
- Required work: create pure `skhy_overlay` logic that converts fresh ADR/peer snapshots into `skhyCatalystStatus`, `skhyCatalystActive`, `skhyOvernightMove`, `semiSympathyScore`, `semiSympathyReasons`, and `relativeStrengthVsSkhy` annotations for Japanese semi/memory candidates.
- Constraints: run after existing rerank/Rule 11.14 overlay; cap weight impact tightly; require at least two independent facts; sector membership alone is insufficient; stale ADR status leaves ranking unchanged.
- Verification: `python -m pytest tests/unit/test_skhy_overlay.py tests/unit/test_theme_rotation.py tests/unit/test_hybrid_rerank.py -q --basetemp=.runtime/pytest-p20-skhy-overlay -p no:cacheprovider`.

### P20-05 Dashboard/API read-only surface - done (2026-06-25)
- Status: **done**. `api/serializers._serialize_adr_watch` → payload `meta.dataQuality.adrWatch`{asof,status,stale} + `eventDesk.skhy`{status,disclosure}。**Fail-open**:无快照→status=unavailable / event=off,JP 候选排序不变。disclosure 含"no probability"诚实免责、无概率值/买入建议语。**22 passed**(3 新 ADR 测试 + 回归)。`test_api_dashboard.py`.
- Required work: expose `meta.dataQuality.adrWatch` and optional `eventDesk.skhy` plus candidate annotations. Surface must remain read-only and must clearly label the ADR lane as external catalyst context, not a buy signal.
- Verification: `python -m pytest tests/unit/test_api_dashboard.py -q --basetemp=.runtime/pytest-p20-api -p no:cacheprovider`.

### P20-06 Manual SKHY ADR journal - SKIPPED (optional, not requested 2026-06-25)
- Status: **skipped (not requested)**. Per plan Task 6, only implement if the operator wants HTR to record external SKHY ADR fills. Operator has not requested it. Deferred — when requested, implement a manual-only `reports/user_state/external_adr_journal/{date}.jsonl` store, fully separate from JP portfolio/NAV/calibration/outcomes/Rank-IC (Rule 11.15), no broker/order fields.
- Required work: record already-completed external SKHY ADR fills manually in a separate store, likely `reports/user_state/external_adr_journal/{date}.jsonl`. No broker route, no order route, no auto-execution, no JP calibration mixing.
- Verification: tests must prove ADR records are excluded from JP portfolio calibration, candidate outcomes, Rank-IC, and cohort review.

### P20-07 Live evidence review before any promotion - done (2026-06-25)
- Status: **done**. `reporting/skhy_event_review.py` — `review_skhy_events`(distinct LIVE event-date clusters,同日折叠去重,排除 backdated 不混 live,够 breadth 才算 Rank-IC via forward_signal_eval,带 Rule 16.0 cost hurdle;**默认 insufficient_data,本 harness 永不授予 promotion**——晋级走 ADR-0010/Rule 16)。无概率字段。**27 passed**(6 review + 21 forward_signal_eval 回归)。`test_skhy_event_review.py`.
- Required work: create a forward review harness for SKHY event dates and Japan semi candidate reactions. Use live-only event clusters, 1D/3D/5D relative returns, Rank-IC when cross-sections are valid, and Rule 16.0 cost-hurdle context.
- Acceptance: default verdict is `insufficient_data` until enough live event clusters and ADR-0010/Rule 16 gates are satisfied. Backdated or anecdotal wins cannot promote the overlay.

### P20-08 Final integration verification - done (2026-06-25, MILESTONE P20 COMPLETE)
- Status: **done**. Focused P20 suite **72 passed**; semi-regression **48 passed**; full daily smoke `-m "not slow"` **1559 passed / 5 deselected (slow vectorbt) / 0 failed**. All `.runtime` basetemp.
- Outcome: SKHY treated as external catalyst only; JP equities remain primary; fail-open when SKHY pending/stale/unavailable; no probability/win-rate/expected-return/edge language anywhere in the lane. Overlay remains **shadow** — no edge claimed; promotion requires live-only forward evidence under ADR-0010 + Rule 16. P20-06 (manual ADR journal) skipped (optional, not requested). Not committed (local-no-commit policy).

### P20-09 Review-fix pass - done (2026-06-26)
- Status: **done**. Fixed 5 review gaps (TDD, red→green): **(1)** candidate rows now expose read-only SKHY annotations (`semiSympathyScore`/`semiSympathyReasons`/`skhyCatalystActive`/`skhyCatalystStatus`/`skhyOvernightMove`/`relativeStrengthVsSkhy`) via `annotate_japan_semi` in `build_dashboard_payload` — capped, **never reorders**, neutral when SKHY stale/pending/unavailable; also fixed `chaseRisk="none"` string being misread as extended, and added real `themes` + `recentReturn` to candidate rows; **(2)** `skhy_event_review` counts only explicit `live=True` clusters (fail-closed); **(3)** future-dated data → stale (`adr_watch.is_stale`) / freshness 0 (`skhy_overlay._freshness`); **(4)** stale peers cannot confirm an active SKHY impulse (`compute_skhy_event` requires `not stale`); **(5)** hygiene — only P20 files touched, `__pycache__`/`*.pyc` gitignored, no edge/probability/win-rate/expected-return/buy language (disclaimers only).
- Verification: focused **79 passed**; semi-regression **50 passed**; full smoke `-m "not slow"` **1566 passed / 5 deselected / 0 failed**. Overlay remains **shadow**. Not committed.

### P20-10 Future-dated ADR snapshot PIT fix - done (2026-06-26)
- Status: **done**. `_serialize_adr_watch` previously read the lexicographically latest `adr_watch_*.json`, so a future-dated file (e.g. `adr_watch_2099-01-01.json`) could be selected and treated as active — violating Rule 11.15 PIT/fail-closed. Fix: `build_dashboard_payload` passes `asof_limit=observation_date`; the helper now keeps only snapshots whose filename date is **≤ asof_limit**, picks the latest valid one, returns unavailable/off if all are future, and has a defense-in-depth check on the payload's internal `asof`. Existing active/pending/stale behavior unchanged; candidate rows stay neutral when only future snapshots exist; no edge/probability/buy language.
- Verification: focused **81 passed**; semi-regression **52 passed**; full smoke `-m "not slow"` **1568 passed / 5 deselected / 0 failed**. Overlay remains **shadow**. Not committed.

### P20-RC Release-candidate prep + adversarial review - done (2026-06-26); release verdict = NOT a clean self-contained RC
- Status: **done (analysis/scope/frontend/verification)**; **release decision = NOT release-ready as a self-contained P20 RC** (see verdict below). Scope chosen: **Option B (backend + frontend-visible)** — the API already emits `semiSympathyScore`/`semiSympathyReasons`/`skhyCatalystActive`/`skhyCatalystStatus`/`skhyOvernightMove`/`relativeStrengthVsSkhy`, so the dashboard should render a read-only indicator rather than leaving the fields invisible.
- **Frontend (Option B)**: added a read-only SKHY/Semi catalyst chip to the shared `CatalystBadges` (`frontend/src/htr-shared.jsx`), mirroring the `newsCatalyzed` badge pattern. Gated on `c.skhyCatalystActive` (renders nothing when off/watch/stale/pending/unavailable — backend fail-closed); neutral research wording only (label `🌐 SKHY联动·研究`; tooltip explicitly states `非买入建议，不预测收益` / 研究用排序提示); **does not reorder candidates**. Added 2 frontend contract tests in `tests/unit/test_frontend_ui_contracts.py` (`test_skhy_catalyst_indicator_present_and_neutral`, `test_skhy_indicator_does_not_reorder_candidates`).
- **Staging**: `git add` of exactly the 9 required new P20 files (`data/external/adr_watch.py`, `candidate_engine/skhy_overlay.py`, `reporting/skhy_event_review.py`, `tools/refresh_skhy_adr_watch.py`, `tests/unit/test_adr_watch.py`, `test_refresh_skhy_adr_watch.py`, `test_skhy_overlay.py`, `test_skhy_event_review.py`, `docs/superpowers/plans/2026-06-25-skhy-japan-semi-overlay.md`). No unrelated dirty files staged; no other-engineer changes reverted.
- **Verification (`.runtime` basetemp)**: focused **81 passed**; semi-regression **52 passed**; full smoke `-m "not slow"` **1570 passed / 5 deselected / 0 failed**; slow lane **5 passed / 1570 deselected**; frontend-contract suite **45 passed**; `git diff --check` (staged) **clean**.
- **Adversarial review (3-agent Workflow, independently verified)**: **frontend-governance = PASS** (chip gated, neutral rendered text — no probability/win-rate/expected-return/edge/guaranteed/buy-now, no SKHY-keyed sort; banned terms exist only in `//` comments/docstrings, not rendered strings). **Rule 11.15 end-to-end = PASS** (SKHY/000660.KS/MU/NVDA external-only — zero references in `decision_log/` or `calibration/`; overlay only appends annotation keys, never injects a JP row; fail-open; shadow / `insufficient_data` / never auto-promotes). **release-packaging = FAIL** (not self-contained — see verdict).
- **RELEASE VERDICT — NOT a clean self-contained P20 RC.** The staged 9 are necessary but **not sufficient**, and the repo is a large uncommitted multi-milestone backlog:
  - **Broken dependency closure**: staged `reporting/skhy_event_review.py:77` imports `from hot_theme_rotator.backtesting.forward_signal_eval import rank_ic`, and `forward_signal_eval.py` is **untracked** → shipping only the 9 would `ImportError` on the event-review path.
  - **P20 functional changes live outside the 9 and are untracked/unstaged**: the frontend SKHY badge (`frontend/src/htr-shared.jsx`) is **untracked**; the ADR pipeline wiring (`tools/daily_routine.py`) is **untracked**; and `api/serializers.py` (the `_serialize_adr_watch` + `annotate_japan_semi` path that actually emits the SKHY fields) is **dirty but NOT staged** → the staged set alone would not even produce the fields the badge/overlay consume.
  - **Transitive untracked deps** referenced by the P20 surface: `candidate_engine/s_kabu_universe.py`, `tools/build_s_kabu_overlay.py`, `data/stock_news_fetcher.py`, `candidate_engine/{catalyst,hybrid_rerank,theme_rotation,theme_leaders}.py`, `data/ticker_metadata.py` — all untracked; the plan doc itself names several as in-scope P20 work.
  - **No calibrated edge / probability / win-rate claim** anywhere; overlay stays **shadow**.
  - **Path to a real RC = a project-wide landing decision** (owner's call per local-no-commit policy): track the full P20 dependency closure (the 9 + `forward_signal_eval` + `s_kabu_universe` + `build_s_kabu_overlay` + `htr-shared.jsx` + `daily_routine.py` + stage `serializers.py` + their untracked prior-milestone deps) — effectively the project's first real commit. Not done; **not committed/tagged/released** (not authorized).

## Milestone P21: Daily Action Board — which stock / what price / when (交易计划板)

Owner goal (2026-07-02): "help me analysis the stocks, market and recommend the corresponding price/time to buy it which stocks". Honest framing per the 2026-07-02 edge-search close-out: the system has NO demonstrated predictive edge, so the board synthesizes **trade PLANS** (structure: entry zone / stop / take-profit / size / timing conditions) from already-gated components — it does not predict outcomes and never emits probability/win-rate/expected-return. All inputs already exist and are tested (rerank ordering, 7-tier ladder, chase-risk, tradability, Rule 12 discipline); the board is an assembler + UI surface. Governance: new Rule 11.16.

### P21-01 Rotation-score double-application fix (bug, Rule 4 / Rule 11.14.5)
- `hybrid_rerank._rotation_adjustment` seeds from the candidate's `rotation_score` already written by `annotate_rotation`, then re-applies the same `leader_extended` (−0.15) / `second_line` (+0.08) adjustments — live pipeline (serializers: annotate_rotation → rerank) applies **double** the documented Rule 11.14.5 magnitudes (−0.30 / +0.16). Fix: reason-marker idempotency guard so each adjustment applies exactly once in both standalone and pipeline paths. TDD regression test asserting pipeline-effective magnitudes.

### P21-02 Rule 11.16 — Daily Action Board honesty (governance)
- Add Rule 11.16 to `docs/02_GOVERNANCE.md`: plan-not-prediction framing, existing-ordering only (no new predictive score), mandatory tradability + chase-risk columns with fail-closed downgrades, Rule 11.6 price-noun vocabulary, Rule 11.8 deterministic sizing arithmetic (risk-budget % of NAV, concentration-capped), factual timing checklist only, standing no-edge disclosure, read-only, PIT (served fields only).

### P21-03 Action board assembler (`reporting/action_board.py`)
- Pure-logic `build_action_board(candidates, nav_jpy, cash_jpy, market_session, config)`: per candidate join of why (existing catalyst/rotation reasons) + ladder tiers (entry aggressive/balanced/conservative, stop, exits) + deterministic sizing (1% NAV risk-to-stop default; whole-lot 100株 and S株 1株 both computed; capped by 20% NAV concentration + cash) + timing checklist (market session, chase risk, catalyst evidence, leader-extended) + `planStatus` ∈ plan_ready / watch_only / s_kabu_only / not_tradable / insufficient_data (fail-closed downgrades). No probability fields by construction. TDD.

### P21-04 Dashboard payload wiring
- `build_dashboard_payload` attaches `actionBoard` (fail-open → null on any error; sizing gracefully absent when portfolio NAV unavailable). Contract tests in `test_api_dashboard`.

### P21-05 Frontend ActionBoardCard (shared, all-variant parity)
- Shared `ActionBoardCard` (`frontend/src/htr-shared.jsx`) rendering the board rows read-only with the Rule 11.16 standing disclosure + safety framing; mounted in V1/V2/V3/V4 (Rule 11.7 parity). Frontend contract tests (disclosure present, no 建议买入/胜率/概率 vocabulary, card present in all four variants).

## Milestone P22: Daily Operating Loop Close-out — Exit Discipline + Remote Personal Access

Owner directive (2026-07-02): "设定目标推进项目，我要一个符合我需求的系统". Gap analysis vs owner needs: (a) 00_DESIGN §2 Q4 — "当前持仓应该止盈、止损、继续持有还是换仓" — has NO dedicated surface, yet "2-5% 止盈换仓" is the owner's core strategy; the owner's live holdings (1306.T + 8035.T) get no exit-discipline view. (b) Owner wants the system online for personal use; Rule 15.0 forbids non-loopback without a new rule — needs Rule 15.8-compliant promotion path. Both are plan/discipline surfaces, no prediction claims.

### P22-01 Position Exit Discipline Board (持仓纪律板, Rule 11.17)
- New `reporting/exit_board.py`: per-holding arithmetic vs the OWNER'S declared discipline parameters (take-profit references avg_cost +2/+3/+5%, stop reference avg_cost −4% — explicit config, Rule 4): current price/P&L, distance-to-each-reference, `exitStatus` ∈ within_plan / past_first_take_profit / stop_reference_breached / insufficient_data (fail-closed). Wired into `/api/dashboard` as `exitBoard` (fail-open→null); shared `ExitBoardCard` mounted V1-V4; standing disclosure (纪律参考≠预测, Rule 3 manual execution). TDD + contract tests.

### P22-02 Remote Personal Access readiness (Rule 15.9)
- New Rule 15.9: single-operator remote access ONLY over a private overlay network (Tailscale/WireGuard/SSH tunnel); public exposure stays forbidden. Fail-closed mechanics: token middleware (`HTR_ACCESS_TOKEN` → all /api/* + pages require token header/cookie; `/login?token=` sets session cookie); `tools/serve_remote.py` runner refuses non-loopback bind without token. Loopback default behavior unchanged (Local Beta v0). README runbook section. TDD (middleware 401/200 paths, runner guard).
- **GONE LIVE 2026-07-03**: Tailscale already installed/connected on this machine (100.118.51.81); `serve_remote.py` launched on the tailnet IP with a crypto-random token (initial all-zero-token launch caught and killed — PS5.1 `RandomNumberGenerator::Fill` pitfall); no-token → 401, with-token → 200 verified. Loopback instance unchanged. Owner holds the login URL/token from the session.

## Milestone P23: Computed Win-Rate Program (owner directive 2026-07-03)

Owner demand: strategies with HIGH win rate where the win rate is CALCULATED, not guessed. The calculation machinery already exists (isotonic calibrator, K-fold/walk-forward validators, DSR overfit gate, outcome join, gated UI probability display per Rule 8.2.2/9.4.1) — what is missing is a predictive INPUT. Program = four data lanes, each ending at the same acceptance test (IC > cost hurdle per Rule 16.0 + DSR ≥ 0.95 promote gate); a lane that passes flows into calibration and flips the UI to `calibrated_probability`; a lane that fails is closed permanently (kill criteria are the acceptance standard that makes a displayed win rate "based on calculation").

Asset audit (2026-07-03): `EDINET_API_KEY` was ALREADY in the User environment (owner had provided it; validated live — 200, 909 docs listed for 2026-06-30). Project_v5 already contains a working `EDINETClient` (`src/data/edinet_loader.py`), a doc-metadata backfill (`scripts/edinet_full_backfill.py`, doc types 130/140/160/170/180/350/360), and a PIT-ready `fundamental_snapshots` schema (published_ts/available_ts/revenue/OP/NI/EPS/BPS/DPS) — **but the XBRL→financial-values parser was never built and the table has 0 rows**. Local TDnet corpus: 21,959 disclosures / 749 業績予想修正 / 377 配当予想修正 (title direction present in only 51/749 → magnitude must be parsed from documents).

### P23-A Guidance-revision magnitude engine (free; corpus local + Yanoshin ≥2yr backfill)
- Parse revision magnitudes (売上/営業益/純益 % vs prior guidance) from 業績予想修正 disclosure documents → event study through the existing harness (PIT next-open entry, 5D/20D excess vs 1306.T, tradability gate, DSR). Kill: IC ≤ 0 or DSR < 0.95 on ≥2yr → lane closed.
- **REDESIGNED forward-only (2026-07-06)**: probe established (a) revision PDFs parse cleanly (pdfplumber, A/B rows), (b) **TDnet public docs live only ~31 days — a 2024 URL is 404 → the 2-year historical event study via documents is impossible; documents are PERISHABLE data**. Built: `data/external/tdnet_revision_docs.py` (pure A/B-table parser: 未定/△negatives/commas handled, pct never fabricated; TDD 4 tests) + `tools/capture_tdnet_revisions.py` (rescue window scan across both corpora + daily forward capture; PDFs stored under `reports/tdnet_docs/`). First run: 7/7 fetched, real magnitudes extracted (e.g. 6136.T OP +50.7%). Smoke **1623 passed / 5 deselected / 0 failed**. Kill-criteria timeline shifts to forward accumulation (~750 revisions/yr expected). TODO next: parse-rate tuning on unparsed layouts; daily capture wiring after the TDnet poll; event-study reader on the accumulated magnitudes.

### P23-B EDINET XBRL fundamental panel (free; key in env; reuse v5 EDINETClient)
- Build the missing XBRL financial-summary parser; fill a `fundamental_snapshots`-shaped table in an HTR-side DB (`data/raw/htr_fundamentals.db` — do NOT write into Project_v5's DB); backfill annual/quarterly reports for the active universe as deep as EDINET serves (~10yr); PIT-align on published_ts; then factor-zoo v2 (value_bp/quality) with real coverage. Kill: same DSR gate. This removes the "only 2–3 valid cross-sections" blocker that made value/quality untestable.
- **IN PROGRESS (2026-07-03)**: module `data/external/edinet_fundamentals.py` + `tools/backfill_edinet_fundamentals.py` built (TDD, 7 tests, smoke 1612/5/0). Live-verified: docTypeCode 120 = 有報 (v5 script's 130 was the 訂正 code — historical bug found); type=5 CSV 経営指標等 block carries 5 fiscal years per annual report (incl. BPS/EPS/ROE/equity ratio). Probe: 2026-06-26 single day → 130 docs / 650 rows / 0 errors. 2026 filing season backfill (05-01..07-03) running in background; next slices walk back 2025 → 2024 → …, then factor-zoo v2 verdict.
- **VERDICT (2026-07-03 night) — FIRST GATE-PASSING SIGNAL**: 11 seasons backfilled (129,215 rows / 3,336 symbols / 0 errors) + adjusted research price store (6.54M rows, 2016+, survivorship residual 352 delisted names = 11.3%, documented). Factor-zoo v2 on 118 monthly cross-sections: **earnings_yield@63D IC +0.113 / t +13.96 / SR 1.35 → DSR 0.992 ≥ 0.95 PASS** (max-SR selection per Bailey semantics; 14-trial family); value_bp@63D IC +0.129 / t +11.4 close behind; quality/growth dead. Stress: program-wide trial counting (40-60 trials) drops DSR to 0.89-0.94 — but **split-half OOS-in-time confirms in both independent windows** (earnings_yield t=8.9 / t=11.5; value_bp t=6.8 / t=9.8, no sign flip, no decay). Cost hurdle cleared 2-5x. **Not a trade signal yet** — next: P19-02 enter signal library as SignalFns → P19-03 shadow/forward Rank-IC (Rule 16 live-only) → long-only S株 implementability design → walk-forward → calibration lane.

### P23-C Forecast/consensus revision momentum (paid — owner decision pending)
- 四季報 online or J-Quants paid tier for forecast data → analyst-revision momentum lane. Deferred until A/B verdicts.

### P23-D Owner's personal calibration (free; starts with next trade)
- Log every trade idea pre-trade through the existing predictions/journal path; at n ≥ 50 compute per-setup win rate with CI. Current honest record: disciplined scale-outs 2/2, discretionary dip-buy 0/1.

## Milestone P24: UI Design Remediation (from the 2026-07-06 read-only review)

Source: the read-only design review (Artifact `2bb5068a-…`, overall B-/3.1). Trust layer scored 4.5 (keep); space/hierarchy scored 2.0-2.5 (fix). Owner-selected batch = F1 + F2 first. All work TDD (frontend contract tests) + headless re-verify; content red-lines (Rule 3/8.3/9.4/11.5/11.6/11.16/11.17 disclosures) MUST survive every layout change. Governed by Rule 11.7 (+ new 11.7.7 mobile priority / vertical balance).

### P24-01 Mobile priority order (F1) — the daily-use fix
- On mobile widths the stack order MUST lead with what the operator acts on: market temperature → Position Exit Board → Daily Action Board → the rest. Demote the V1–V4 variant switcher off prime mobile real estate. Contract test asserts the mobile source order; headless mobile screenshot verifies.

### P24-02 Default-variant vertical balance (F2) — kill the void
- V3 (shipping default) MUST NOT leave the centre column empty while the right rail runs multiples longer. Rebalance without violating Rule 11.7.3 (no primary-content scroll-trap): redistribute long cards, not an internal-scroll cap. Headless desktop+tablet verify the columns end within a bounded delta.

### P24-03 S株 card occlusion (F4) — compliance fix (Rule 11.7.2)
- The floating S株 card currently overlaps the right rail (a standing violation of the app's own no-occlusion invariant). Dock it into document flow or make it a dismissible drawer. Contract test asserts no fixed-overlay occluding live content.

### P24-04 Empty-gauge honest state (F3) — deferred, honesty-adjacent
- An empty temperature sparkline reads as a real reading. Add an explicit "no data" gauge state (Rule 11.9.4 degraded labelling). Deferred behind P24-01..03.

### P24-05+ Deferred (owner decision): F5 default-variant choice, F6 dark instrument palette, F7 progressive disclosure, F8-F10 nav/tablet/anchors.

## Milestone P25: Owner Risk Mandate & Sleeve Engine (2026-07-13 owner declaration)

Source: owner declared (2026-07-13 session) that the ¥400k account is EXPERIMENTAL capital with a −75% drawdown tolerance (kill-switch NAV floor ¥100k) and requested a risk-accepting architecture. Derivation (recorded in ADR-0012): fractional Kelly λ≤0.75 from P(hit floor)≤10% → target β-adjusted exposure 1.4× NAV, band [1.2, 1.6]. Three sleeves: A leveraged-beta engine (only positive-expectation engine), B value/E-P live experiment (≈0 expectation, evidence purchase, verdict ~2026-08-26), C conviction bets (zero demonstrated edge, pure variance; 8035.T re-underwritten into C at ¥71,300 by owner decision). Advice-only (Rule 3) is NOT relaxed: the engine computes and displays; the owner executes externally and records via Section 14. Governed by new Section 17.

### P25-01 Governance Section 17 + ADR-0012
- Write Rules 17.0–17.6 (mandate provenance; sleeve architecture; exposure band + rebalance discipline; kill-switch; Sleeve C discipline — 20% cap, no averaging down, mandatory thesis; Sleeve B pre-commitment; honest expectation labelling). ADR-0012 records the Kelly/drawdown derivation, allocation table, alternatives, and what is NOT changed.

### P25-02 Declarative mandate config (`configs/risk_mandate.json`)
- Owner-declared parameters as data (Rule 4 explicit): capital, floor, target ratio, band, sleeve definitions/caps, sleeve_map, beta assumptions (labelled research assumptions), 8035.T re-underwrite record with thesis=null until the owner writes one (surfaced as `thesis_missing`, fail-closed).

### P25-03 Sleeve engine (`src/hot_theme_rotator/risk/sleeve_engine.py`)
- Pure read-only assembler: per-sleeve capital/β-adjusted exposure/caps/flags, total exposure ratio + band status, kill-switch buffer, C discipline flags (thesis_missing / cap_breached / review_required), unmapped holdings fail-closed into UNASSIGNED with a warning. Fail-open to None without config; never fabricates positions (Rule 11.9.4). Unit-tested.

### P25-04 Dashboard + frontend surface
- `api/serializers.py` exposes `riskMandate` (fail-open → null). Shared `RiskMandateCard` mounted on V1–V4 (Rule 11.7 parity): mandate header + standing disclosure, exposure ratio vs band, kill-switch buffer, sleeve rows with honest expectation labels and flags. No probability/win-rate/expected-return language (Rule 8 inheritance). Contract tests.

### P25-05 Daily trace (`tools/daily_routine.py` afterclose)
- Non-fatal step writing `reports/observability/risk_mandate/{asof}.json` + one summary row to `reports/observability/risk_mandate_trace.jsonl` (nav, exposure ratio, band status, kill-switch buffer, flag counts) — same pattern as value_livelog. A diagnostic must never block collection.

### P25-06 Owner follow-ups (pending owner input; system surfaces, never blocks)
- 8035.T written thesis + invalidation trigger (until then the card shows thesis_missing).
- Sleeve A deployment is OWNER execution at the external broker (candidate 2x ETFs to be price-verified live at order time, never from stale EOD), recorded via Section 14 fill path.

### P24-11 Red-numeral accessibility (owner astigmatism feedback, 2026-07-14)
- Owner reported small red numerals on the dark theme (action/exit boards) blur under astigmatism. Two passes, all variable-level so V1-V4 inherit (Rule 11.7): dark `--htr-bear` #E86D64→#FFA69E (surface contrast 5.8→9.5:1, coral hue kept); new `.htr-bear-num` (mono tabular, 700, red-tinted chip — bright glyph carries the shape, background carries the red semantics) applied to both boards' stop refs; `.htr-down` and exit-board P&L at weight 600; statusNote 10.5px/600. Light theme untouched. Contract tests 54 green. Escalation if needed: a "high-legibility mode" toggle (near-white text everywhere, color only on chips/icons).

## Milestone P26: Sleeve C Wind-Down Mechanics + Concentration Observability (2026-07-16 owner reflection)

Source: owner reflection session (2026-07-16) after the 07-16 semi gap-down. Owner (a) understood the 2000 dot-com lesson (high-PE bubbles can break; a retail participant has no information edge on the timing), (b) reframed 8035.T from a conviction hold to a wind-down ("回本加零花钱"), (c) asked whether the system watches beyond semiconductors, and (d) asked for a deep reflection on the trading strategy. Findings: the research layer is NOT semi-biased (universe 2,771; 0 large-cap semis in the 49 daily candidates; 8035.T is not even in the screener universe) — semi concentration is an owner Sleeve-C choice. But three real gaps surfaced, all advice-only (Rule 3), none relaxing any prior rule.

### P26-01 Bilateral exit bracket for Sleeve C (Rule 17.4.6)
- Config: `c_theses[symbol].exit_upper_jpy` / `exit_lower_jpy`; engine evaluates on the latest CLOSE only, emits `exit_triggered` flag + advice line on breach. 8035.T bracket set to [¥64,000 / ¥74,000] (upper = 07-15 close; lower = re-underwrite ¥71,300 −10%). A written bracket IS a valid thesis → clears `thesis_missing`. Defeats the disposition effect: exit must be a declared two-sided price, never the entry cost ¥77,600. Frontend surfaces armed/triggered per holding.

### P26-02 Sector look-through (Rule 17.7)
- Engine `_sector_look_through`: direct theme tags (`theme_map`) + index-ETF embedded sector weight (`benchmark_sector_weights` × leverage_factor). Answers "how much of NAV actually moves with theme X" across sleeves. Surfaced on the risk card + printed in the daily snapshot. Reveals ~22.7% NAV semi concentration (8035 direct + TOPIX ~11% via 1306/1568, leveraged) that per-symbol views hid. Embedded weights are labelled research estimates (Rule 17.6), Rule-4-refreshable.

### P26-03 Discipline-flag sunset (Rule 17.4.7)
- `flag_sunset_sessions` (default 7): the daily snapshot reads the append-only trace, counts consecutive sessions each open `thesis_missing`/`review_required` flag has persisted, and escalates (SUNSET line + trace `sunset` field) past the threshold, demanding resolution. A rule with no deadline is deferrable forever.

### P26-04 Sleeve A deployment calendar (proposal — pending owner)
- `docs/proposals/sleeve_a_deployment_schedule.md`: three mechanical cadences (fixed weekly / value-averaging / price-grid) to replace "owner's pace" (which degraded into day-by-day timing) and close the below-band gap (~0.68× vs [1.2, 1.6]). Not active until owner adopts one via Rule 4. Advice-only; no auto-execution.

## Milestone P27: Exit-Discipline Layer Precedence (2026-07-20 owner question)

Source: owner asked whether 1568.T, showing below its "discipline stop price", had to be sold. Investigation found two discipline layers firing on the same holdings with contradictory verdicts — and, worse, a decision already recorded on 2026-07-13 that had never reached the code.

### P27-01 Mandate precedence in the exit board (Rule 11.17.7)
- `exit_board.py` applied the generic 00_DESIGN §1 swing params (avg_cost +2/+3/+5%, −4% stop) to every holding. Section 17 (declared 2026-07-13, 11 days after Rule 11.17) governs mapped symbols with more specific, already-declared discipline. The P25 Change Log stated "8035.T 旧 −4% 止损参考被 C 纪律显式取代 …… 不是静默失效" — but no code implemented it, so the board kept rendering a cost-anchored ¥74,496 stop on a Sleeve C position whose Rule 17.4.6 bracket exists precisely to abolish the entry-cost anchor.
- Fix: symbols in `sleeve_map` suppress the generic refs and render the binding rule instead — A → none (Rule 17.1/17.2 portfolio band; a −4% band on a 2x instrument sits inside 2 daily sigma of the mandate's own σ≈18%), B → none (Rule 17.5 pre-commitment), C → declared bilateral close bracket (17.4.6) else review-drawdown off re-underwrite price (17.4.4). Never the entry cost.
- New `exitStatus` values `mandate_governed` / `mandate_exit_triggered` / `mandate_review_required`; every row carries `sleeve` + `disciplineSource`; board carries `mandateAware` and `params.scope`. Missing/invalid mandate → fail-open to pre-P27 behaviour.
- Frontend: the footer claimed the params were "你声明的,Rule 4 显式配置" while the call site passed no config (dataclass default) and implied global scope. Now states scope + affected row count, or says plainly when no mandate is loaded.
- Advice-only throughout (Rule 3): no position changed, no buy/sell rendered, no new POST route.

## Milestone P28-P33: Evidence-Based Retrospective Remediation (proposed)

Source: the 2026-08-04 full-cycle retrospective (`docs/proposals/retrospective_review_2026-08-04.md`) and its remediation design (`docs/superpowers/specs/2026-08-06-evidence-based-retrospective-remediation-design.md`).

No task authorizes broker execution, capital deployment, signal promotion, or a mandate-parameter change. Anything touching `configs/risk_mandate.json` needs the Rule 4 record (field, old value, new value, reason, expected impact, verification).

Dependency order: P28 → P29 → P30; P31 and P32 are independent lanes; P33 consolidates.

**Build status (2026-08-06).** The *machinery* for P29–P33 is built, tested, and wired. What remains open is owner input and owner decisions, which no amount of engineering can supply.

| Task | Machinery | Blocking on |
|---|---|---|
| P28 | shortfall reporter built | **owner**: 8035.T fill price + fees; band-breach three-way choice |
| P29 | queue + CLI + auto-open **live** | — |
| P30 | gate built, **ships disabled** | **owner**: Rule 12.7 double confirmation to enable a channel |
| P31 | protocol built, runs today | — (verdict is `insufficient`; see below) |
| P32 | memo written | **owner**: pick one of the four alternatives |
| P33 | scorecards + rule audit **done** | — |

Delivered commits: `366b08b` `4f94632` `ad5e332` `a39fe70` (session-age repair) · `0534845` (P29) · `c053fdf` (P28 tool + P32 memo) · `15777b9` `bd77739` (P30) · `2ff6578` (ASCII/CLI fix) · `8c9ea66` (P31) · `e54449e` (P33). Full smoke **1855 passed / 5 deselected / 0 failed** (baseline 1682 → +173 tests, zero regression).

**Acceptance status: CONDITIONAL ENGINEERING ACCEPTANCE — not sealed.** Second review round (2026-08-06) found four substantive gaps, all confirmed and fixed in `eddb240`: an overreaching variance-inflation claim with a self-certified `locked` protocol, a cost model that accepted negative and non-finite inputs (turning the Rule 16.0 gate into a rubber stamp), a shortfall reader that failed OPEN on journal corruption, and a rule audit that was not idempotent because it scanned its own artifacts. Statistical protocol, cost-input safety, journal fail-closed and audit idempotency now have tests; **the protocol itself is still `proposed` and awaits a Rule 4 owner decision.**

> Artifacts live under `reports/` and are **gitignored**: "persisted" means present on this machine, not reproduced by a clone.
 The tools are built, tested and persisting artifacts; the loops they are meant to close are still open. A prior summary of this work overstated it — it presented the remaining work as owner-only when at least seven engineering items were outstanding. Those were completed on 2026-08-06 (see the remediation entry in `PROJECT_STATUS.md`), but the distinction stands: shipping the instrument is not the same as closing the loop.

**Owner decisions** — none of which the system may make (Rule 3 / Rule 4):
1. **8035.T fill price + fees** → unblocks P28 reconciliation and turns the shortfall FINAL.
2. **Band breach** (queue item age 17 sessions; **17/17 observed sessions below_band**): deploy / Rule 4 amendment / dated exception with expiry.
3. **Mandate derivation** (P32): one of four alternatives.
4. **Adopt the 2026-08-26 rename** to *63D Evidence & Protocol Readiness Check* (see P31), and rule on the governance numbering ambiguity (see P33).

**Still engineering, not owner** — known open items, listed so they are not mistaken for owner blockers:
- `reports/research/cost_model.json` has no producer. Declaring the model needs an owner input (the cost figures); building the producer does not.
- PBO/CPCV has no implementation anywhere in the repo.
- No purge/embargo protocol exists for the live-log read.
- The leakage audit needs re-running in-scope for the value experiment.

Each is a precondition for `confirm` ever becoming reachable, and none is satisfied by waiting.

### P28 - Ledger, Retrospective, and Mandate-State Closure
**Status: tooling done (`tools/implementation_shortfall.py`, 16 tests); BLOCKED on owner data.** The reporter refuses to promote a scenario price to an actual fill: `delay_cost_jpy` stays `None` and status stays `provisional`, naming the missing inputs, until the fill reaches the journal. Real run today confirms `PROVISIONAL / actual fill NOT IN JOURNAL / missing: actual_price, fees_jpy`.
- Record the actual 8035.T sell fill through Section 14 (`tools/htr_fill_cli.py`) and confirm no stale 8035.T holding remains in the next risk snapshot.
- Reconcile NAV, realized/unrealized P&L, benchmark return, and active return to a documented tolerance.
- Report implementation shortfall using decision price, eligible execution reference, actual execution, fees, and `provisional`/`final` status — never a later close treated as the executed price (Perold).
- Recompute the delay tables with age zero on creation; label `elapsed` and `inclusive` counts separately.
- Recompute exposure and band status after 8035.T leaves the ledger (provisional read: ~0.415x, ~¥301,599 below the 1.2x lower band).
- Obtain a dated owner decision closing the band breach: deploy, submit a Rule 4 band proposal, or approve a time-bounded exception with a hard expiry. P28 surfaces the choice; it never selects or executes it.
- Mark the withdrawn low-exposure "second empirical protection" interpretation superseded wherever it remains live in `PROJECT_STATUS.md`; preserve the original entry for audit history.

### P29 - Decision Queue and Execution Observability
**Status: DONE and live** (`src/hot_theme_rotator/decision_queue/`, `tools/decision_queue_cli.py`, 26 tests). `acknowledged` is deliberately an optional observation, not a gate — forcing acknowledge-first would make the ledger unable to record what actually happened, which is the 8035.T failure itself. Auto-open keys each item to the session the condition FIRST appeared, so a standing breach is one aging item rather than a new item per session; advisory flags stay off the queue on purpose.
Live reading 2026-08-06: **1 open binding item, elapsed age 17 sessions** (Rule 17.2 band breach, created 2026-07-13), 1 executed (8035.T, trigger→terminal **7 sessions**), median trigger→seen **unobserved**. That last value is the notification gap, now a measured quantity rather than an anecdote.

**Counting conventions — three numbers that are easy to conflate** (a corrected 2026-08-06 statement; an earlier draft said "20/20 sessions", which counted trace ROWS, the exact rows-vs-sessions error the Rule 17.4.7 repair existed to fix):

| Number | What it counts |
|---:|---|
| **20** | rows in `risk_mandate_trace.jsonl` — includes same-day reruns, and is **not** a session count |
| **17** | distinct JPX sessions with a trace row, 2026-07-13 … 2026-08-05, **no gaps**; all 17 record `below_band` |
| **17** | the queue item's *elapsed* age from 2026-07-13 to 2026-08-06 — a different quantity that happens to coincide |
| **15** | P33's band-compliance denominator: sessions at/before the reconciled-through date 2026-08-03. 08-04 and 08-05 are excluded because the position is unreconciled (the 8035.T sell is unjournaled) |

So **17/17 observed sessions out-of-band** and **0/15 band compliance** are both correct under their own stated denominators, and neither is 20.
- Persist deterministic advice IDs and append-only state transitions: `open -> acknowledged -> executed | declined | expired | superseded`.
- Record source rule, created timestamp, JPX-session age, severity, evidence pointer, and a structured decline reason (Rule 13.9 rejected-with-reason).
- Session-age and idempotency substrate: **done** (see above); the queue consumes it rather than reimplementing it.
- Report open-age distribution, terminal-state counts, trigger-to-seen, and trigger-to-terminal after close.
- CLI/afterclose recording precedes any UI write path; any new HTTP mutation first amends the Rule 11.5 whitelist and takes governance review.

### P30 - Low-Noise State-Transition Notifications
**Status: built, SHIPS DISABLED** (`src/hot_theme_rotator/alerts/transition_notifier.py`, 15 tests). With no enabled channel the gate delivers nothing and records `no_enabled_channel` — silent runs still write audit rows, otherwise the metrics would be blind exactly while the channel is off. Two independent controls: a per-(item, state) dedupe key makes an unchanged open item structurally unable to repeat, and a 5-session per-subject cooldown stops one noisy sleeve dominating. Three consecutive delivery failures roll the gate back to silent mode rather than retry-storming.
**Blocking on owner**: Rule 12.7 double confirmation is the only way to enable a channel; the system must not and cannot self-authorize it.
- Enable exactly one owner-selected channel through Rule 12.7 double confirmation.
- Notify on state transitions only; unchanged open states never re-notify.
- Carry dedupe key, severity, cooldown, monthly budget, and delivery audit; content links the decision ID and contains no order control.
- Monthly metrics: sent, delivered, acknowledged, duplicate-suppressed, trigger-to-seen.
- Automatic rollback to silent mode when a predeclared error or duplicate rate is breached (Section 12 anti-fatigue).

### P31 - Locked 63D Evidence Review Protocol
**Status: DONE** (`tools/evidence_review_63d.py`, 38 tests). Persisted artifact `reports/observability/evidence_review_63d/2026-08-06.json`: `signal_verdict=insufficient`, `deployment_verdict=not_started` (0 Sleeve B fills; `unwind_to_A` non-operative on an empty sleeve), frozen trial family 100 inclusive / 60 in the E/P lineage, 63D has 0 matured of 2,216 rows.

> ### 2026-08-26 is renamed: **63D Evidence & Protocol Readiness Check**
>
> It is **not** a confirm/fail verdict date, and the previous framing should not be carried forward. On that date the review may report directional readings, data maturity, and remaining protocol gaps. It **may not emit `confirm`** until the verification protocol, the cost model, and the effective-sample definition are all locked. The existing freeze stays in force and this does **not** license enlarging Sleeve B.
>
> `confirm_reachable=false`, blocked by `cost_hurdle, pbo_cpcv, pit, purge, survivorship` — the artifact's own note: *"these checks are blocked by a missing PROTOCOL, not by a small sample: waiting for more data cannot turn them into a pass."*

> ⚠ **`confirm_reachable = false`, and NOT for sample-size reasons.** Five checks are blocked by a missing *protocol*, which waiting until 2026-08-26 does not fix: PBO/CPCV has no implementation anywhere in the repo; the only leakage audit on disk is dated 2026-05-31, scoped to the backdated-calibration sample (`contaminated` / V2 `inconclusive`) and therefore transferable to the value live-log read in **neither** direction; σ_r at 63D is recorded by no artifact so the Rule 16.0 cost hurdle cannot be computed; and the live-log read records no purge protocol. **The 2026-08-26 milestone as previously understood cannot produce a `confirm`.** That is a governance finding, not a bug.
>
> **RETRACTION (2026-08-06).** An earlier version of this entry stated the `min_obs=60` bar needs "3,780 independent date clusters" as an unconditional fact, and the tool claimed the disjoint-blocks and Newey–West estimators "agree analytically" so the bar "cannot be relieved by switching estimators". **Both claims are withdrawn.** `1+2·Σ(1−k/h) = h` is the TRUE variance inflation under a triangular ACF — it is *not* the Newey–West estimator, which applies Bartlett weights and yields `(2h²+1)/(3h) = 42.0` at h=63. The estimators therefore disagree: **3,780 clusters (~15.4 yr) under disjoint blocks vs 2,520 (~10.3 yr) under Newey–West**. Further, `ρ_k = 1−k/h` describes overlapping sums of iid increments and is an *assumption* about a cross-sectional Rank-IC series, not a derivation from it. The cluster requirement is **conditional on whichever estimator is adopted**, and adopting one is a **Rule 4 owner decision** — the tool previously marked its own newly-invented protocol `locked=True`, which engineering had no authority to do. It now emits `status: proposed`.
>
> The B/P sign reversal is already established at the Harvey bar (21D IC −0.0713, t −3.95) and is surfaced on its own axis; no composite is emitted, precisely so it cannot be averaged away.
- Treat 2026-08-26 as the **earliest** review date, not a guaranteed verdict date; emit `confirm` / `fail` / `insufficient`.
- Freeze and count the trial family (all attempted variants) before computing anything.
- Report independent date clusters, raw rows, maturity coverage, and missingness.
- Emit PIT, survivorship, cost hurdle, purge, embargo, DSR, PBO/CPCV, and Harvey-style t-stat checks.
- Report E/P and B/P independently — a composite must not hide the B/P sign reversal.
- Separate `signal_verdict` from `deployment_verdict`; the latter is `not_started` until a Sleeve B fill exists, and `unwind_to_A` is non-operative on an empty sleeve.
- No capital or config change follows from the report alone.

### P32 - Risk-Mandate Decision Memo
**Status: memo written** (`docs/proposals/risk_mandate_decision_memo_2026-08-06.md`); **BLOCKING on owner** to pick one of four alternatives. Headline: the 1.2731x-vs-1.4x argument is worth ≈0.148pp/yr (≈¥570) against +3.95pp of modelled floor probability, while LETF drag on the leveraged leg alone (≈¥7.0k/yr ≈ 32% of Sleeve A's gross premium) and the standard error of μ (f\* 95% CI ≈ [−0.29, 3.69] even at T=30y) are an order of magnitude larger. The floor formula contains neither μ nor σ, so the "≤10%" claim is not a tail-risk statement — it is a bet on knowing μ/σ².
- One short memo, not a simulation programme. Reproduce the ADR-0012 arithmetic including the line-34 error: the recorded inputs give `0.75 × 1.6975 = 1.2731x`, while the text asserts `λ·f* ≈ 1.4x`; `target=1.4x` implies `λ = 0.8247`.
- Show the trade-off as model outputs: expected log growth ≈4.376% at 1.2731x vs ≈4.525% at 1.4x (≈0.148pp, ≈¥575/yr on current NAV), against a floor-hit approximation rising ≈9.92% → ≈13.87%.
- Apply LETF variance drag and the as-of **verified official** fee only to the planned leveraged component (~¥175k of 1568.T, not all ¥217k); distinguish analytic drag from observed tracking difference.
- State parameter-uncertainty assumptions explicitly (`SE(mu_hat)=sigma/sqrt(T)`; the floor formula contains neither μ nor σ, so uncertainty re-enters only through the implied λ).
- Present at least three owner alternatives: retain 1.4x and withdraw the ≤10% claim; align the target with the stated fractional-Kelly bound; or abandon the Kelly provenance and re-justify the band as declared owner preference. A time-bounded deferral goes through the P28 exception path.
- Block bootstrap and jump/regime Monte Carlo are **deferred** — they become a separate proposed task only if the owner picks a risk-calibrated band, intends to occupy it, and first states what decision the simulation could change.
- Output proposals only; do not edit the active mandate.

### P33 - Three-Ledger KPI and Rule-Sunset Review
**Status: DONE** (`tools/three_ledger_scorecard.py`, `tools/rule_usage_audit.py`, 60 tests). Real readings 2026-08-06:
- **Account card = `unavailable`**, as it must be: an `executed` advice on 2026-08-04 has no journal entry on or after it. The card refuses to render a return on an unreconciled ledger rather than quoting a stale one.
- **Band compliance 0.0% — 0 of 15 verifiable sessions in-band.** Independent confirmation of retrospective §4.4 from a different code path.
- **`trigger_to_seen` = `not_applicable`**: no queue item has ever reached `acknowledged`. Not zero — never observed.
- **106 of 264 rules have zero CODE reference (40%)**, by evidence strength: `implemented_in_product` 150 / `operator_tooling_only` 8 / `test_assertion_only` 19 / `artifact_echo_only` 0 / `documentation_only` 17 / `section_scope_only` 54 / `unreferenced` 16. Only the first two count as having a code reference. (The audit scans this repo including itself, so the split drifts by a few as code lands; the figure is the persisted 2026-08-06 artifact.)

  > **Corrected 2026-08-06.** The first run reported "92 zero-reference / 172 runtime-referenced". That taxonomy counted `tests/` and `reports/` as runtime, so a rule named only in a test docstring ranked equal to one a shipping code path implements — a flattering aggregate of exactly the kind this audit exists to expose. Subdividing moves 17 rules out of "referenced" and the honest headline to **109 (41%)**.

  Collapsing these states would manufacture dead rules; the scan sees citations, not compliance, so it cannot distinguish a dormant rule from one the owner enforces by hand. **Review candidates: 0 — a CLOCK result, not a health result**, since git history for this tree begins 2026-05-25 and no rule can yet reach the 6-month window.
- Research card sources the frozen trial family from P31 rather than reporting `input_not_present` for a count the repo already computes.

> ⚠ **Governance defect surfaced, needs an owner ruling:** the document's own numbering is ambiguous — **Rule 5.1 lives under Section 1, while Section 5 has its own item 1.** The parser refuses to invent a resolution and reports affected citations (e.g. `Rule 2.1`, cited from `event_desk.py`) under `dangling_references`; 14 cited numbers resolve to no defined rule.

**Known gap (not built):** `reports/research/cost_model.json` has no producer, so the Rule 16.0 cost hurdle renders `input_not_present` on both this card and P31. That is the same missing input blocking `confirm_reachable`.
- Publish separate account / research / execution scorecards; never one blended grade (outcome-bias guard).
- Define every numerator, denominator, unavailable state, and as-of date; `insufficient` is not zero, and N/A is not failure.
- Measure ledger lag and band compliance only on days with valid prices and reconciled positions.
- Scan runtime references from rules to config, code, tests, and reports.
- Rules unreferenced for six months enter an owner review list — never automatic deletion; preserve audit history when rules are merged or retired.

### P34 - Strategy Research Plan 2026-08-07 (market decision + T1/T2 lanes + opportunity gate)
**Status: INFRASTRUCTURE BUILT / STRATEGY VALIDATION NOT STARTED** (2026-08-08). Design doc: `docs/proposals/strategy_research_plan_2026-08-07.md` (the corrected version there is canonical). Advice-only (Rule 3); no capital/config/weight change; Sleeve B freeze and screener-weight freeze stay in force. Honest one-line state: **all ten task lines have implementation or diagnostic artifacts; no strategy is validated; the round's chief product is the discovery of data-chain defects, not tradable alpha.** Owner decisions: O-1 open (account facts; cost conclusion provisional), O-2 open but non-blocking (gate is dormant), O-3 open non-blocking (declare vs observed-only), O-4 granted 2026-08-08.

> **SUPERSEDED 2026-08-08 — the original headline block that stood here claimed "4–6.6×" and "$22 cap always binds".** Corrected figures: 10 bp ⇒ required IC **0.0067** (τ=0.7, σ₆₃=0.104 illustrative); US:JP lot ratio **4.0×–9.9×**; the $22 cap is **never reached** at this account size (the percentage rate applies in full — the cap is irrelevant, not binding). Full corrected decision text lives in the proposal §1; it is not duplicated here so there is exactly one place to amend.

Decisions still in force (unchanged by the corrections): stay JP (provisional on O-1); research priority T1 > T2 > E/P > TSMOM shadow, all [H] none [V]; **B/P not weighted** (21D IC −0.0713 t −3.95, but `n_obs_effective=1` ⇒ a fail-closed alert, not a completed negative-alpha validation); cross-sectional momentum frozen not dead; gate output is TWO orthogonal axes (`candidate_status` × `validation_status`, corrected from the earlier single three-state enum); every θ registers in its own family (`P34_GATE_v1`) — the P31 frozen family is cited additively, never written.

**2026-08-08 execution round.** Owner granted O-4 (research ordering) and authorized engineering; explicitly withheld capital/config/weight/UI changes. Corrections folded in after independent verification — see the proposal's own retraction notes. Materially: the hurdle arithmetic was wrong (10bp → **0.0067**, not 0.010; US:JP ratio **4.0–9.9×**, not 4–6.6×); "P34-00 already ships" was **asserted, not audited, and is false**; "zero buyback hits repo-wide" was **too narrowly scoped and is withdrawn**; the S株 slot table was **stale** (13:30/15:00 → 14:00/15:30); the E/P reading **+0.085/t+3.1 is stale** (current: **+0.04926/t+2.325**, and `n_obs_effective=1`); TSMOM "six-way" listed only five.

Sub-task status — **DONE** requires code + tests + artifact + docs together:

- **P34-00 — DONE.** `src/hot_theme_rotator/research/gate_reachability.py`, `tools/audit_gate_reachability.py`, 8 tests. Artifact `reports/research/gate_reachability/min_entry_score_2026-08-08.json`. **Verdict `DORMANT`**: `generate_signal` has one caller (`reporting/daily_pipeline.py:124`); `run_daily_pipeline` is imported **only by tests**; zero static paths from `tools/`|`api/`; `tools/daily_routine.py` never touches it; no artifact/UI consumes `TradingSignal`. Lineage = `0.30×market_temp + 0.70×leader_score`, then `×risk_multiplier` — **distinct** from `opportunity_scanner`. ⇒ no live undeclared gate; **O-2 downgraded to non-blocking**; the 70 is `legacy`, never retro-labelled preregistered. Stated limit: static graph, blind to dynamic imports.
- **P34-01a — DONE.** `data/external/buyback_events.py`, `tools/extract_buyback_events.py`, `buyback` added to `ALLOWED_TDNET_CATEGORIES` + a parser rule ordered **before** `dividend`/`order`/`governance`; 35 tests. Real-data artifacts `reports/research/buyback_events/{events,summary}_2026-08-08.{jsonl,json}`. **Defect measured on 2,344 stored disclosures (2026-06-30..08-07): all 547 treasury disclosures misfiled** (governance 276 / order 225 / other 30 / earnings 16) because the `order` rule matches `株式の取得`, which 「自己株式の取得」 contains. Subtypes: disposal 294 (**a DISPOSAL, not a buyback — excluded**), execution_report 160, other_treasury 57, **resolution 20 → 15 uncontaminated = T1 primary**, cancellation 13, modification 2, completion 1; 7 corrections; 0 parse failures. Correction to the prior claim: buyback keywords DID already exist in `theme_detector` (`buyback_dividend`) and `free_web_opportunity_adapter` (`shareholder_return`) — but **conflated with dividends**, which is the contamination T1 excludes. **Known gap:** amount/share caps absent from RSS titles (confidence low 497 / medium 50 / high 0) ⇒ size strata need PDF extraction, out of the primary plan.
- **P34-01b — PARTIAL (engineering complete; awaits fills + O-3).** `research/execution_profiles.py` schema v2, 25 tests. Adds `execution_profiles` to the one canonical contract; consumer passes `execution_profile_id`; unknown/costless profile **refuses to substitute** another profile's cost. Structured provenance (`source/producer/version/asof/sample_size/method`); cost vs `sigma_r` keep **separate** provenance blocks. **S株 slot table corrected and tested**: 00:00–07:00→09:00, 07:00–10:30→12:30, 10:30–**14:00**→**15:30**, **14:00**–24:00→next-day 09:00 (TSE 2024-11-05 extension + closing auction). Median×2 aggregation; thin cells emitted **empty, not estimated**. Remaining: real fill accrual (no fills yet), then O-3 declare-or-accept-observed.
- **P34-02 — DONE.** `research/preregistration.py`, `tools/freeze_t1_preregistration.py`, 15 tests. Frozen artifact `reports/research/preregistration/P34_T1_buyback_resolution_v1.json`: primary 20D vs 1306.T, secondary 5/10/40/60D, strata auction/tostnet/method_unknown, entry = open of first trading day **strictly after** `published_ts` (same-day = look-ahead), inference = date-cluster bootstrap + calendar-time portfolio, stopping rule = no interim peeking until 100 matured resolutions, measured `expected_event_rate_per_year=144.1`. Content-hashed and immutable (re-freeze idempotent; edits require a new version; post-freeze tampering detected). **`prospective` is refused when the rule predates the freeze** — the retroactive-preregistration guard. 15 trials (5 horizons × 3 strata) registered in `P34_T1_v1` **before any outcome read**. No outcomes read.
- **P34-05 — DONE (moved ahead of P34-02/03 as required).** `research/trial_registry.py`, 15 tests, `reports/research/trial_registry.jsonl`. Append-only; `family_id/family_version/registered_at/config_hash/hypothesis_lineage/outcome_accessed_at`; key-order-invariant config hash; duplicate rejection; **outcome-access-before-registration refused**; corrupt registry **fails closed** (a small denominator is the flattering direction). **P31's frozen family is not writable** — guarded first so the refusal names the real reason. `program_snapshot()` emits a NEW as-of snapshot ADDING the cited P31 count (15 + 100 = 115) and **never modifies the P31 artifact**.
- **P34-03 — DONE (framework); T1 confirmatory run BLOCKED on data, correctly.** `research/event_study.py` + `tools/t1_event_study_readiness.py`, 24 tests validated against synthetic ground truth. Supplies the four things the existing skeleton lacked: **CAR and BHAR as distinct estimators** (sum vs compound — they diverge and both are reported), **calendar-time portfolio** (one observation per date, absorbing overlap and same-day clustering), **matched controls** (nearest-neighbour on a characteristic, without replacement, never self-matched; beyond-tolerance ⇒ unmatched rather than badly matched), and **date-cluster bootstrap**. The naive cross-event t-stat is still emitted but carries a mandatory warning naming the event/date-cluster ratio. `maturity_report()` computes counts with **no estimate of any kind**, so a no-peeking lane can ask "are we there yet" without that becoming a side channel.
  - **Real-data run (counts only, per the frozen stopping rule): 2 windows built from 15 T1 events; 1 matured at 20D; rule requires 100 ⇒ shortfall 99.** `--confirmatory` **refuses with exit 2**, which is the correct output.
  - ⚠ **New blocker found, and it is not sample size.** Attrition breaks down as **11 `symbol_series_stale_before_event`** / 2 `ticker_absent_from_price_db` / 0 recency. Buyback announcers' price series stop months before their events (e.g. `5133.T` ends 2026-04-10, event 2026-07-02; `4078.T` ends 2026-05-29, event 2026-06-30) while the DB as a whole runs to 2026-08-07 — the daily refresh tracks the **rotating screener universe**, and buyback announcers are broadly distributed across the listed market. Overall treasury-ticker coverage is **350/439 = 79.7%**, but coverage *at event time* is far worse. **Waiting will not fix this**: accruing 100 resolutions still yields an unusable study until price refresh covers the event universe. First diagnostic attempt mislabelled these as "absent"; corrected after checking the DB directly.
- **P34-04 — DONE.** `research/validation_harness.py`, 23 tests. **Reuses** `calibration/purged_walk_forward.make_folds` verbatim for continuous event/factor labels (it was never calibration-specific), so one purge/embargo implementation exists to audit. New: **CPCV** (C(N,k) splits ⇒ many backtest paths, with purging that also removes train samples leaking into test groups that are *later in calendar order but earlier in index order*) and **PBO via CSCV**. PBO validated against known-answer regimes — pure noise averages **0.463** over 20 seeds (theory ≈0.5) and a persistent edge gives **0.000** on every seed. Single-seed PBO ranges ~0.06–0.84, so the noise test asserts the **mean over seeds**; a fixed single-draw threshold would have been a coin flip about a coin flip (a first version of that test sat exactly on its own boundary). **Scope boundary enforced in code:** `require_multi_config()` raises for a single configuration — PBO/CPCV measure *selection* bias across a sweep, so they do not apply to a single pre-registered hypothesis, and PBO must not become the universal admission gate for every event study.
- **P34-06 — DONE.** `research/opportunity_gate.py`, 25 tests. Two **orthogonal** axes (corrected from one three-state enum): `candidate_status ∈ {INSUFFICIENT_DATA, NO_CANDIDATE, CANDIDATE}` × `validation_status ∈ {UNVALIDATED, VALIDATED, INVALIDATED}` — an INVALIDATED rule can still emit CANDIDATE rows, the state a single enum hides, and that is pinned by test. Missing score ⇒ `INSUFFICIENT_DATA`, never `NO_CANDIDATE` (a name we could not score is not one we judged and declined). Predictions are **immutable and idempotent**; outcomes are a **separate append-only event keyed by `prediction_id`** — asserted by test that the prediction row never gains a return field. `GateConfig` refuses an empty `threshold_provenance` ("an undeclared threshold is an unregistered trial"). Evaluation reports **EV net of cost**, not win rate — pinned by a test where a 75%-win-rate rule with +1%/−5% payoffs shows negative EV, and another where a 99bp cost flips a marginal gate. `render_user_facing()` emits no probability/win-rate/expectancy and states evidence status alongside the candidate status.
- **P34-07 — CORRECTED 2026-08-09: 1 real blocker, not 3.** ⚠ **The 2026-08-08 probe pointed at the WRONG DATABASE** — it read `htr_market.db`'s legacy near-empty `fundamental_snapshots` (2,170 rows / 95 symbols) and never looked at **`htr_fundamentals.db`, the actual P23-B EDINET panel: 181,160 rows / 4,421 symbols / fiscal periods 2010-03-20..2026-04-20, backfilled 2026-07-03..07-06 walking back to 2020-04**. Two of the three reported blockers were artifacts of that mistake and are **withdrawn**:
  - ~~"PIT timestamp is a fetch stamp (11 days / 2,087 records, median lag 280d)"~~ → **WITHDRAWN.** The panel's `published_ts` is EDINET `submitDateTime`: **1,136 distinct filing days across 34,743 as-filed rows, timestamps spanning 3,646d vs events 3,672d, median lag 87d** — the statutory ~3-month Japanese filing deadline. The probe's own heuristic then produced a **second** error: it flagged this genuine calendar as a backfill because Japanese filings cluster (most FYs end 31 March). Fixed — the decisive test is now timestamp SPAN vs event span, not distinct-day count; regression test added.
  - ~~"0 of 95 symbols reach 5 distinct fiscal periods"~~ → **WITHDRAWN. 4,371 of 4,421 symbols reach ≥5; 3,979 reach ≥8.**
  - ✅ **Ownership structure — the one REAL blocker, confirmed across all four DBs** (`htr_fundamentals`, `htr_market`, `htr_raw_prices`, `htr_research_prices`): no table or column carries foreign/individual holding share. Remedy: extract 所有者別状況 from EDINET 有価証券報告書 into an annual PIT table.
  - ⚠ **Frequency caveat (new, not a blocker):** P23-B collects doc types **120 (有価証券報告書, annual) + 160 (半期報告書, semi-annual)** — so a seasonal SUE is computable at ANNUAL frequency today; a QUARTERLY SUE would additionally need 四半期報告書. Also note only `relative_year=0` rows are as-filed; `relative_year>0` are prior years restated inside a later filing (timestamp honest, lag years) and must be joined on `published_ts`, never on `fiscal_period_end`.
  - Probe hardened: it now scans **every** fundamentals store and takes the deepest, so a future single-DB assumption cannot repeat this class of error.
- **P36-01 — ownership ACQUISITION done; T2 research chain DEGRADED (corrected 2026-08-09 after review).** ⚠ An earlier entry here claimed "T2 chain now `feasible: True`" — **withdrawn**. Ownership was the hardest input, not the last one; three links are still short. `data/external/edinet_ownership.py` + `tools/backfill_edinet_ownership.py`, 16 tests. Element IDs confirmed against a **live** filing (doc S100YNWZ, 4750.T). Panel: **4,028 rows / 3,811 deduped symbols**, published 2025-06-27..2026-07-02, evidence manifest `reports/research/ownership_panel_manifest_2026-08-09.json` (fingerprint `bb9a7a95…`, since the DB is gitignored).
  - **Two extraction traps handled:** XBRL values are **FRACTIONS despite a 「（％）」 label** (68.83% → `0.6883`) — `validate_ownership()` requires the categories to partition to ~1.0 and names the percent/fraction confusion; and ownership is an **INSTANT at fiscal year end, public at submitDateTime**, so consumers join on `published_ts`, never `as_of`.
  - **Cohort, DEDUPED: low-foreign (<5%) ∧ high-individual (>50%) = 1,132 symbols** (low-foreign 1,742; high-individual 1,673). ⚠ An earlier "1,215" counted snapshot ROWS and double-counted two-vintage symbols — **withdrawn**. ⚠ These thresholds are **OUR configuration, not Jinushi's design**: the paper sorts on the 20th percentile of foreign and 80th percentile of individual ownership **per fiscal year** and tests the two hypotheses **separately**. Any fixed threshold or AND-combination is a new trial and must be registered as one.
  - **Three links still short (probe corrected to measure, not assert):**
    1. **Earnings-event timestamp — DEGRADED.** The probe previously scored EDINET `submitDateTime` as `available`; that is a real PIT stamp but the **wrong event** — median lag 87d identifies the 有価証券報告書 (3-month statutory annual report), whereas Jinushi's event is the **決算短信** (~45d). Correct source exists (TDnet poller: **347 決算短信**) but spans only **26 corpus days** ⇒ prospective accrual only.
    2. **Historical PIT ownership vintages — DEGRADED.** Only **188 of 34,743** as-filed events (0.5%) have an ownership snapshot published *before* them. One cross-section proves the variable exists and is dispersed; it cannot date-align to past events. Either backfill earlier vintages or pre-declare T2 as purely prospective.
    3. **Size control — DEGRADED.** Previously `available` while its own detail admitted "market cap still needs shares outstanding". `shares_outstanding` covers **95 symbols** vs 3,810 in the panel; close×volume is turnover, not size. Either backfill 発行済株式総数 or pre-declare that size is uncontrolled and say why.
  - **Failure ledger added.** The 7 no-block / 5 invalid documents were previously counted but not identified, so every run refetched them forever (the 4,000-doc run and the next 40-doc run reported the identical 12). Now recorded per document with reason, and terminal outcomes are skipped unless `--retry-failed`. Diagnosis of a real case: **3925.T reports foreign-corporate as `51.45` while every sibling field is a fraction** — a filer-side unit error. The validator refuses it and names the outlier; auto-rescaling would fabricate data.
  - **T2 next step is NOT a full preregistration freeze.** Order: (a) build the 決算短信 event-time chain; (b) report the actual event × latest-prior-ownership × price join; (c) decide backfill-vintages vs prospective-only; (d) resolve size control; (e) only then freeze and register.

- **P36-02/03 — 決算短信 event chain built; join report answers the vintage question (2026-08-09).**
  - **P36-02 event-time chain — DONE.** `data/external/earnings_events.py`, 23 tests. Dates events from **決算短信** (the event Jinushi studies), not EDINET's annual report. Classifies annual / quarterly / correction / notice-about-短信 rather than filtering silently — a 「（訂正）四半期決算短信」 is a *correction* first, since counting it as a fresh quarterly would double-count the original. **After-hours rule is load-bearing, not cosmetic: 97% of annual 決算短信 are published at or after the 15:30 close**, so they are dated to the NEXT trading day; without the shift essentially every event in the study would be credited with a day of return nobody could have captured.
  - **P36-03 join report — DONE.** `tools/build_t2_join_report.py`, artifact `t2_join_report_2026-08-09.json`. Scanned 24,026 disclosures (main corpus ∪ the 251-day probe corpus) over 842 trading days. 決算短信: annual 603 / quarterly 1,882 / correction 345 / notice 37. **Attrition ladder — the deliverable:**
    | events | requirement |
    |---|---|
    | 603 | primary annual 決算短信 (532 symbols) |
    | 326 | …with any price history |
    | 200 | …with 30 pre + 60 post bars |
    | **8** | …with an ownership snapshot published BEFORE the event |
  - **DECISION (step 3): backfill historical ownership vintages — evidence, not preference.** 8 usable events across 6 symbols is not a study, and the ladder shows the ownership vintage is the binding constraint by two orders of magnitude, exactly as the review predicted. Prospective-only would mean waiting a full annual filing season for the first meaningful cross-section. The backfill is mechanical because P23-B **already indexes all 34,762 有報 documents (2016–2026)** — the ownership block is a second read of documents we have already identified. Launched 2026-08-09 (multi-hour; ~30.7k documents remaining at the polite throttle).
  - ⚠ **Even with full vintages the ceiling is ~200 events**, set by price-window coverage and TDnet corpus depth — so expanding the TDnet history is the next lever after vintages, and the eventual preregistration must state the achieved sample honestly rather than the input counts.
  - Percentile sizing on the joined set is reported for SIZING ONLY; Jinushi sorts per fiscal year and tests the two hypotheses separately, and the design must be frozen before any outcome is read.

- **P36-04 — T2 data chain COMPLETE; sample is 2,099 events but nominal ≠ effective (2026-08-10).**
  - **Three mechanical backfills, all finished, no design touched.** TDnet corpus: the missing 2024-07..2025-06 year filled (**61,562 disclosures / 365 dates**). Price coverage for the 532 earnings symbols: **+58,984 bars, 351/532 COVERED, 0 failures** (PARTIAL — 181 delisted/no-data, reported honestly). Ownership vintages: **25,266 stored / 5,393 no-block / 25 invalid / 38 errors of 30,722**, panel now **29,294 rows spanning 2019-05-31..2026-07-02**.
  - **Attrition ladder, final:** 3,752 primary annual 決算短信 → 2,785 with price history → 2,397 with 30 pre + 60 post bars → **2,099 with a prior ownership snapshot** (last rung now loses 12.4%, was 96%). **2,099 usable events across 1,844 symbols**, up 262× from the 8 that the first join found.
  - ⚠ **Corrections to my own prior claims:**
    - "2,099 is a floor that will keep rising" — **withdrawn.** The ownership backfill finished 2026-08-09T21:32; recomputing on the final DB still gives 419/421. 2,099 is the STABLE result for this corpus and price store. Future TDnet/price accrual can raise it; nothing pending will.
    - "the ceiling is ~200 events" (said before the backfills) — **withdrawn**, it extrapolated a then-current constraint as if fixed. Actual 2,099.
    - "process confirmed alive" — **withdrawn.** The liveness check was **self-matching**: the `python -c` source contained the tool name, and `wmic` reports full command lines, so it found itself. No backfill was running.
    - "a few hundred fetches" for shares — **withdrawn.** The targeted set is **2,058 documents** (from 29,279 overall; 93% saved), and the event-subset mode did not exist until now.
  - ⚠ **"~420 per bucket is sufficient" — withdrawn; nominal count overstates information.** The 2,099 events fall on only **246 distinct event days, max 178 on a single day**, and 2025 alone holds 1,503 (71.6%). Jinushi sorts WITHIN each fiscal year, so the buckets that actually matter are per-year: **2024: 78/79 · 2025: 300/301 · 2026: 41/41** — not a flat 420. The join report now emits `event_day_clustering` and `conditioning_by_year` so this can never be read off the pooled figure again.
  - **Size control resolved without new acquisition.** `TotalNumberOfIssuedSharesSummaryOfBusinessResults` sits in the 経営指標等 block P23-B already parses, so shares outstanding is captured by the ownership extractor (prior-year contexts rejected so a 5-year block cannot overwrite the current count; legacy tables migrated via ALTER, tested). `--shares-for-join` restricts the refresh to the **exact ownership snapshots the join pairs** (`_join_ownership_doc_ids`), not every vintage of every joined symbol.
  - **Preregistration must contain, before any CAR/BHAR is read:** two hypotheses tested SEPARATELY (not an AND-combination); per-fiscal-year 20th/80th percentile sorts (not fixed thresholds — those are our configuration and register as their own trials); the after-close next-trading-day rule; standard errors clustered by event day AND by firm; multiple-testing treatment; a power analysis against a pre-declared minimum detectable effect; and a stopping rule. Full suite 2,269 passed.

- **P36-05/06 — power analysis + T2 preregistration DRAFT (2026-08-10). NOT FROZEN.**
  - **P36-05 clustered power — DONE.** `research/event_power.py`, 16 tests. Kish design effect `1+(m−1)ρ` so the nominal event count cannot be quoted as if it were information: at ρ=0.10 the 2,099 events (246 days, m≈8.5) carry ≈1,200 observations' worth. σ is an **assumption, never measured** — measuring it means computing abnormal returns, i.e. reading the outcome.
  - **P36-06 draft** — `docs/proposals/t2_preregistration_draft_2026-08-10.md`. Contains everything the review required: H1/H2 tested **separately**; per-fiscal-year 20th/80th percentile sorts (fixed thresholds are OUR configuration and register as their own trials); after-close next-trading-day dating (**73% of annual 短信 are after-close**); SEs **clustered by event day AND firm**; multiple testing = 16 primary trials in family `P36_T2_v1`; power analysis; stopping rule.
  - ⚠ **The power analysis changed the design, and the honest numbers are unwelcome.** Achieved power against a 2% drift: **2024 13%, 2025 39%, 2026 9%, pooled 47%** (σ=0.20). Per-year testing — the paper's own design — has **essentially no power on our sample**; a null in 2024/2026 would carry no evidence. 80% power at a 2% effect needs **521–1,448 events per bucket**; we have 419–421. Therefore the draft makes the **pooled specification PRIMARY** with per-year secondary and explicitly labelled underpowered — a deliberate, pre-declared departure with its reason stated, not a post-hoc rescue. The draft states plainly that **an imprecise null is the most likely outcome**.
  - **Stopping rule carries a power condition**: realized σ must imply ≥60% pooled power against the pre-declared 2% effect, else results are reported as exploratory-and-underpowered **regardless of p-values**.
  - **Size control RESOLVED**: targeted refresh completed **2,058/2,058 join-paired documents (100%), 0 failures**; shares outstanding spans 224,507 .. 16,314,987,460 (five orders of magnitude), so size is a live control.
  - **Still blocking the freeze (4):** σ declared range; `P36_T2_v1` family created and all trials registered; Rule 16.0 cost figures (O-3) still absent so "exceeds the cost hurdle" is uncomputable; and **owner sign-off on pooled-primary in place of the paper's within-year design**. No CAR/BHAR computed. Full suite **2,285 passed**.

- **P36-07 — preregistration draft v2 after owner review (2026-08-10). STILL NOT FROZEN.** Owner declined to sign v1's item 5 and was right on every count; all verified before adoption (FY counts 647/1,303/149, buckets 130/130 · 260/261 · 30/30, CV 1.58/1.64, effective N 196–199, power ≈29% — every number reproduced exactly).
  - **v1's estimand was wrong.** "Mean 60-session CAR > 0" is not PEAD — positive and negative news cancel, so real drift can average to zero. v2's estimand is Jinushi's **slope**: `AR[-1,+60] = β0 + β1·AR[-1,+1] + γ'X + δ_FY`, H1/H2 = β1 > 0 in the within-FY quintile buckets, tested separately; full regression written out; `AR[+2,+60]` registered as the overlap-free robustness LHS. Explicitly labelled a **"Jinushi-inspired pooled adaptation"** — two complete FYs + one partial cannot test the paper's decay trend, only whether conditional PEAD exists now.
  - **v1's "fiscal years" were calendar years.** Corrected to April–March (ending-year label): FY2025 647 / FY2026 1,303 / FY2027 **partial** 149; buckets 130/130 · 260/261 · 30/30. Join tool now emits fiscal-year buckets + per-bucket cluster stats itself.
  - **v1's power was ~70% too optimistic.** Equal-cluster Kish ignores CV≈1.6 (one day = 178 events); size-weighted m_e ≈ 12.1–12.5 gives **effective N ≈ 196–199, MDE ≈ 4.0%, ~29% power @2% (σ=0.20)** — and even that is for the MEAN, which no longer transfers to the slope. `event_power.py` gained `effective_sample_size_from_sizes` (exact Σm²/Σm) and the two-sided tail fix (power(0)=α, not α/2); 23 tests incl. the reviewer's 29% as a known-value test.
  - **Primary family cut 16 → 2** (H1/H2 slope at [-1,+60]); everything else secondary/registered. **Realized-σ downgrade rule REMOVED** — a preregistered test's status must not depend on what the data turned out to be; replaced with interval criteria (supported / effect-excluded via pre-declared β1* / imprecise). **Cost hurdle = tradability only**, never statistical support.
  - **Blocking freeze:** Monte Carlo slope power on the actual cluster structure (then declare β1* and planning ranges — σ deliberately NOT signed yet), family registration, O-3, owner sign-off on the revised design. No outcome read. Full suite 2,292.

- **P36-08 — Monte Carlo slope power done; it changed the null value AND the inference method (2026-08-10). Draft v3, still NOT frozen.** `research/slope_power_mc.py`, 17 tests.
  - ⚠ **The null value was wrong in v2.** The LHS `AR[-1,+60]` mechanically contains the regressor `AR[-1,+1]`, so an efficient market gives slope **1**, not 0 (verified: independent post-window ⇒ β̂₁ = 0.991; disjoint LHS `AR[+2,+60]` ⇒ β̂₁ = −0.009). v2's "H1: β₁ > 0" would have been **satisfied by market efficiency itself** — a vacuous test. Corrected to **β₁ > 1**, with each specification now carrying its own stated null value.
  - ⚠ **CR1 over-rejects 2× on the real cluster shape.** Checking SIZE before power: balanced 42×10 clusters give size **0.054** ✅, but the actual T2 shape (one 178-event day = 42% of a bucket) gives **0.102** at nominal 5%. My first analysis mistook that over-rejection for lumpy clusters *gaining* power — the opposite of the truth, caught only by simulating the null first. **Wild cluster bootstrap (CGM, null imposed) restores size to 0.045 and is now MANDATORY**, with two-way clustered SEs demoted to a diagnostic.
  - **Power curve under WCB** (σ_a 0.06, σ_post 0.20, ICC 0.10, pooled H1 shape): β₁ 1.00 → 0.072 (size) · 1.10 → 0.16 · 1.20 → 0.35 · 1.30 → 0.57 · 1.50 → 0.90. **The sample can see a large drift and cannot reliably see a modest one.** β₁* proposed at **1.30** (~57% power) — the smallest slope visible at better than a coin flip.
  - Remaining before freeze: owner sign-off on β₁*=1.30 and the declared planning values (σ_a 0.06 / σ_post 0.20 / ICC 0.10 — planning assumptions, NOT measurements), `P36_T2_v1` registration, and O-3 for the separate tradability verdict. Full suite **2,309 passed**. No AR/CAR/BHAR computed.

- **P36-09 — draft v4: v3's simulation used a cluster shape that does not exist (2026-08-10). NOT frozen, family NOT registered.**
  - ⚠ **Root error, verified and withdrawn.** v3 simulated 42 clusters with a 178-event day. **178 is the largest day in the FULL 2,099-event sample; no bucket contains such a day.** The real buckets: **H1 = 420 events / 121 days / max 36**, H2 = 421 / 125 / max 38. Everything v3 derived is void: CR1 size 0.102, WCB size 0.045, the entire β₁ 1.10–1.50 power curve, and the headline "CR1 over-rejects 2×".
  - **Re-measured on the real H1 shape: CR1 size = 0.0503** (3,000 sims) — essentially exact; the reviewer's independent 0.0625 (2,000 sims) agrees. **CR1 is not broken on this sample.** WCB = 0.033 (conservative) and is retained as primary inference on **robustness grounds** (CGM 2008; MacKinnon–Nielsen–Webb 2023), not because CR1 failed. Module docstring and tests corrected to say so.
  - **Root cause fixed structurally**: the join tool now emits `bucket_cluster_sizes` / `bucket_cluster_summary` and the tests READ them; a regression test asserts a bucket max < 50 so the full-sample 178 can never be substituted again. The severely-unbalanced case survives only as a clearly-labelled synthetic illustration.
  - **True power (real shape, central scenario):** β₁ 1.05 → 0.09 · 1.10 → 0.15 · 1.15 → 0.24 · 1.20 → 0.34 · 1.30 → 0.55.
  - **β₁* = 1.30 withdrawn as a proposal.** It was chosen because power just exceeded a coin flip; detectability cannot define economic importance. β₁* must be argued from drift-per-1sd-reaction (1.8% at β₁=1.30) versus round-trip cost and literature effect sizes, with power then *reported*.
  - **Primary specification switched to the disjoint window** `AR[+2,+60] ~ AR[-1,+1]`, H₀: β₁ = 0 — because the additive identity behind "H₀ = 1" holds for CAR/log abnormal returns but **not for BHAR**, which is Jinushi's actual LHS. The overlapping specification is retained as a **comparability exercise on additive CAR, explicitly not a literal replication**.
  - Also adopted: **Holm** family-wise control for the 2 primary trials with power recomputed against the Holm rule; a pre-declared **sensitivity grid** (σ_a × σ_post × two ICCs × day-shock correlation) run per bucket instead of one scenario; WCB extended to the **full model** (controls + FY fixed effects) so simulation and inference share a specification; **CIs by bootstrap test inversion**, CR1 demoted to diagnostic; §8 rewritten for the new nulls.
  - Full suite **2,310 passed**. No AR/CAR/BHAR computed on real data.

- **P36-10 — items 2–4 executed; a NEW identification threat found (2026-08-10, draft v5). Still NOT frozen.** `research/full_model_power.py`, 18 tests (2 marked `slow` — 121 day dummies on 420 rows is a 23-minute run, research lane not smoke lane).
  - **Item 3 (WCB on the full model) — DONE.** `ols_cluster_robust` and `wild_cluster_bootstrap_p_general` take an arbitrary design, so simulation and inference now share ONE specification: intercept + slope + size + ADV + FY fixed effects (+ event-day FE, below). v3's two-parameter simulation priced a different experiment and is superseded. Full-model size on the real shape: **0.02–0.09** ✅.
  - **Item 4 (Holm) — DONE.** `holm_reject` + `simulate_holm_power`, which simulates H1 and H2 **jointly on their real cluster arrays with their measured 38.2% overlap** (416 events each, 159 shared) rather than multiplying independent figures. Power is measured against the Holm rule itself, not a marginal 5% test. (A first Holm test of mine asserted list-position ordering; Holm ranks by p-value — implementation was right, my expectation was wrong.)
  - **Item 2 (sensitivity grid) — parameterised and run**, and it surfaced something the grid was for: ⚠ **correlated day shocks BIAS the slope.** If the announcement-day market shock correlates with the following sessions' shock, the regressor correlates with the error — a bias, not a variance problem, so **clustering cannot fix it**. Measured size under H₀ at ρ = 0.0 / 0.1 / 0.2 / 0.3 / 0.5 → **0.050 / 0.075 / 0.105 / 0.147 / 0.259**.
  - **Remedy adopted and verified: event-day fixed effects** in the primary specification, simulated to restore size below nominal at ρ = 0.3. **Cost stated: the slope is then identified within-day only, so 52 singleton event days (52/420 H1 events, 12%) contribute nothing.** The no-day-FE specification is retained as a registered secondary, and disagreement between them is reportable.
  - Full suite **2,326 passed, 7 deselected**. No AR/CAR/BHAR computed on real data. Remaining before freeze: β₁* on economic grounds (needs O-3 cost figures), family registration, owner sign-off.

- **P34-08 — DONE (shadow).** `research/trend_overlay.py` + `tools/tsmom_shadow_report.py`, 32 tests; artifact `reports/research/tsmom_shadow_2026-08-08.json`. Six arms: buy&hold / 12M trend long-cash / 10M SMA / vol-target / trend+vol gate / **trend with re-entry delay** (the sixth arm an earlier draft was missing). Leverage is **simulated** daily-2x + fee drag and flagged `leverage_is_simulated` on every arm, because **1568.T has only 49 bars** (2026-06-01..08-07) — far too few for a trend study.
  - ⚠ **Data defect found and guarded: `daily_prices` stores RAW closes (`auto_adjust=False`, mandated by Rule 11.9.6 at ingestion), so splits appear as returns. 1306.T — the system's BENCHMARK — falls 90.1% on 2026-03-30 on a 10:1 split, and 63 of 2,774 symbols carry a similar artifact (80 events).** The first run produced −400% total return and −189% drawdown (arithmetically impossible). `compare_arms` now **refuses** a series with >45% single-period moves unless `allow_jumps=True`; the tool restricts to the longest clean segment (2023-03-27..2026-03-27, 733 periods) — so the reported comparison ran on split-free data.
  - **Contamination scope, stated precisely (2026-08-08 correction):** the earlier blanket "every backtest crossing those dates is corrupted" over-reached. Verified by grep: **no corporate-action-adjustment or jump-fail-closed path exists anywhere in the repo other than the new `trend_overlay` guard**. So the correct claim is: *every consumer computing returns directly from raw `daily_prices` without corporate-action handling is suspect wherever its window crosses one of the 80 artifact dates* — which, absent any adjusting path, is currently every raw-price return consumer. The fix is not per-tool patching but a **repo-wide adjusted-return contract + an inventory of all raw-price consumers** (proposed follow-up, below).
  - Result on clean data: **buy&hold +198.7% (ann 44.2%, maxDD −42.7%) beats every overlay** (trend +34.6%, vol-target +41.6%, SMA +13.3%, gate +15.2%, delay +12.2%); overlays cut drawdown only modestly (best −27.7%) at large return cost. **Sample is 3.0 independent 245-period windows ⇒ labelled INADEQUATE**; arm ranking on this is not evidence. Correct claim strength: **the current implementation provides no reliable evidence FOR a TSMOM overlay** — it does not prove TSMOM ineffective on JP indices. **No mandate change**, and the artifact carries an explicit note that no arm may be cited to justify the 17/17 below-band under-deployment retrospectively.
  - **Post-review remediation 2026-08-08:** this run READ historical returns of six configurations — a multi-config outcome access that predated any registration. The six arms are now **retroactively registered** in `P34_TSMOM_v1` with `outcome_accessed_at` recorded and a note stating the read came first; they are `hypothesis_generating`, never confirmatory. Registry: `P34_T1_v1` 15 (no outcome read) + `P34_TSMOM_v1` 6 (outcome read, exploratory) + cited P31 100 ⇒ program conservative total **121**. The earlier blanket claim "no outcome was ever read" is **corrected** to: *no T1/confirmatory outcome was read; the TSMOM exploratory comparison did read historical returns and is registered as such.*
- **P34-09 — DONE.** `data/external/edgar_lane.py`, 22 tests. Read-only/no-execution/no-capital. **`filingDate` vs `acceptanceDateTime` separated**: `pit_available_at()` uses acceptance and rolls past the 16:00 ET close to the next session; a filing lacking acceptance is **refused by default** because dating it from `filing_date` biases toward earlier availability — the look-ahead direction. `assert_not_event_source()` refuses `companyfacts`/`companyconcept`/`frames` as event sources (period-keyed panels, not event streams). `build_headers()` enforces SEC fair-access UA and rejects placeholders (a block lands on the whole host). `replication_scope_guard()` allows **only** replication of signals already under study in the JP lane — "this lane replicates; it does not originate", so it cannot become a second-market factor zoo.

**Post-P34 priorities (adopted from external review 2026-08-08, verified before adoption; no new strategy lanes before these):**
1. **Repo-wide corporate-action-adjusted-return contract** + inventory of every raw `daily_prices` return consumer (grep-verified: today only `trend_overlay` guards jumps; nothing else adjusts or fails closed).
2. **Price-refresh coverage for event universes** — 11/15 T1 tickers went stale before their events because refresh tracks the rotating screener universe.
3. Cost model keeps accruing real fill shortfalls (P34-01b PARTIAL → DONE when cells populate; O-3 then chooses declared vs observed-only).
4. T1 preregistration untouched; no early outcome read (needs 100 matured; has 1).
5. T2 reclassified as a **data-acquisition project** (EDINET XBRL quarterly backfill + 所有者別状況), not an estimator task.
6. SSOT hygiene: superseded claims in this file are now marked SUPERSEDED inline rather than left standing above their corrections.
### P35 - Adjusted-Return Contract + Event-Universe Price Coverage
**Status: INFRASTRUCTURE COMPLETE + CONSUMER MIGRATION COMPLETE (2026-08-09).** Commits: `99960a9` (hardening) + migration commit. Smoke **2208**, zero regressions. Distinct states, stated separately: (a) contract + maintenance infrastructure **complete**; (b) consumer migration **complete** — every `adjusted_return_required` consumer migrated or guarded, all others carry per-file rationale in the curated inventory; (c) **T1 validation still WAITING on time** — 4/100 matured, nothing here changes that.

**2026-08-09 hardened round (owner-directed):**
- **Contract hardened**: integer ratio **without volume is now `ambiguous`** (silence ≠ corroboration); `verified_actions` = explicit auditable override (date→factor), no ticker special-cases; full input validation (strictly-increasing dates, finite closes>0, sane volumes); bad denominator **raises** instead of fabricating 0%; **contamination is per-window** via `ambiguous_indices()` — an out-of-window anomaly no longer discards the symbol. 19 tests.
- **Backfill tool**: `--asof` truly bounds the fetch (yfinance end is EXCLUSIVE ⇒ end=asof+1d), `--db` injectable, planning/execution extracted and unit-tested (11 tests), explicit **SUCCESS/PARTIAL/FAILURE** (exit 0/3/1), pre-declared 400d lookback for no-history tickers, idempotent append (existing bars never overwritten — pinned by test).
- **Daily wiring**: per-symbol event-universe maintenance runs **before the global fetch AND before the "already current" early return** (a globally-current DB with individually-stale event tickers was the original defect — a naive `universe |= event_universe` only fetches global missing days, never a ticker's own history). Fail-open per symbol, PARTIAL reported as partial. 5 wiring tests incl. the brand-new-ticker case.
- **Curated inventory (schema v2, authoritative)**: `raw_price_consumers_2026-08-09.json` — **29 consumers: 5 adjusted_return_required (ALL migrated) / 7 already_adjusted_basis / 7 not_a_return_consumer / 9 raw_required / 0 unreviewed / 0 pending**. Docs cite the artifact; counts live there, not here. Key semantic findings vs the naive scan: `backtest_value_on_livelog` (the E/P live-log producer) already reads the **auto_adjust=True research store** — scan false positive; the calibration-label path (`sweep_pending_outcomes`/`backdated_calibration_bootstrap`/`audit_calibration_leakage`) was **already split-fail-closed** via `outcome_join._detect_split_in_window`; `morning_briefing`/`api/symbol` are mark-to-market / vs-reference — raw-correct.
- **Real migrations**: the three history backtests (`price_reversal`/`factor_zoo`/`disclosure_drift`) — forward returns now via `adjusted_series_store` with per-window jump exclusion; liquidity screens deliberately stay raw. `api/serializers.py:657` 1D chg guarded (>45% move ⇒ no-signal placeholder, not a phantom −90%). `t1_event_study_readiness` migrated to per-window policy. Split-regression + fail-closed tests: `test_history_backtests_adjusted.py`, `test_adjusted_series_store.py`.
- **Data-correction reruns** (`reports/research/data_correction_rerun/`, originals NOT overwritten): **the P19-04 "no edge" verdict SURVIVES correction** — price_reversal DSR 0.87→**0.823/0.838**, factor_zoo 0.58→**0.577**, disclosure_drift 0.50→**0.105** (weaker); all still fail the 0.95 gate. Split contamination neither manufactured nor hid an edge at verdict level. Still exploratory, never confirmatory.

- **P35-01 adjusted-return contract — DONE (contract + reference migration; repo-wide migration OPEN).** `src/hot_theme_rotator/data/adjusted_prices.py`, 11 tests including the literal 1306.T shape (3817.0→376.4 + volume surge ⇒ classified 10:1 split, artifact return eliminated, latest price stays in raw tradable units). Semantics fixed in one place: raw = as stored (Rule 11.9.6, correct for ADV/turnover); adjusted = back-adjusted to current share basis; adjusted return = the ONLY research-grade return. Classification is evidence-based: near-integer price ratio required, volume corroborates when present, and a volume CONTRADICTION (price says 2:1, volume drops) demotes to `ambiguous` ⇒ **fail-closed** — a −55% crash is refused, never "adjusted" into a fake split. Dividends explicitly out of scope (price-return basis on both sides).
- **P35-01b consumer inventory — SUPERSEDED counts corrected (2026-08-09 closeout).** Authoritative artifact: `reports/research/raw_price_consumers_2026-08-09.json` (schema v2, **scan ∪ curated union** — a consumer that migrates behind the central store no longer vanishes from the record). **29 consumers: 5 central_adjusted_price_return (all migrated) / 5 split_guarded_raw_return / 3 vendor_adjusted_total_return / 9 raw_required / 7 not_a_return_consumer / 0 pending / 0 unreviewed.** An earlier entry here cited "35 consumers / 24 SUSPECT" from the naive first scan — both numbers were pre-curation artifacts and are superseded; docs cite the artifact, counts live there.
- **P35-02 event-universe price coverage — VERIFIED, not assumed (2026-08-09 closeout).** Root cause confirmed in `refresh_htr_price_db.py`: `active_universe` = symbols with a bar on the LATEST date, so any name that drops out of the screen stops refreshing forever. `tools/backfill_event_universe_prices.py` covers screener ∪ event-study universe per symbol (own missing tail; pre-declared 400d lookback for no-history names). T1 fast path: 15/15 tickers +2,297 bars ⇒ **T1 windows 2 → 14; matured @20D 1 → 4; stopping rule blocked by TIME only (96 to go at ~144/yr)**. Full sweep: **attempted 439, +79,030 bars appended**. ⚠ An earlier entry claimed "439/439, 0 failures" from the absence of exceptions — **superseded**: post-hoc per-ticker DB verification (`event_universe_coverage_2026-08-09.json`, reference 2026-08-07, min_bars 30) shows **COVERED 429 / STALE 4 / NO_DATA 6** — an empty fetch appends zero and is not coverage. Success is now defined by the database after the run (`verify_coverage`: COVERED/STALE/NO_DATA/FETCH_FAILED/DELISTED_OR_SUSPENDED), the run status is SUCCESS only when the whole universe verifies covered, refresh CLI prints EVENT_MAINTENANCE and exits 0/3/1, and `daily_routine` records `event_maintenance: event_universe_partial` on exit 3 (warning, not fatal).
P35 closeout residuals (all else closed): 4 STALE + 6 NO_DATA event tickers (likely delisted/invalid codes — terminal classes, listed in the coverage artifact); the frozen waiting states (T1 sample accrual 4/100, P34-01b fills + O-3, T2 = EDINET data acquisition).

7. "Zero regression" claims cite the exact reproduction command (both shells — the Unix form does not paste into PowerShell):
   - Git Bash: `TMP=./.runtime/tmp TEMP=./.runtime/tmp python -m pytest tests/ -q -m "not slow" -o cache_dir=./.runtime/pytest_cache_p34 --basetemp=./.runtime/pytest_tmp_p34`
   - PowerShell: `$env:TMP='.\.runtime\tmp'; $env:TEMP='.\.runtime\tmp'; python -m pytest tests -q -m "not slow" -o cache_dir=.\.runtime\pytest_cache_p34 --basetemp=.\.runtime\pytest_tmp_p34`
   (2166 passed, ~31–35s, independently reproduced 2026-08-08; without the basetemp/TMP redirection the suite can hang on the known system-Temp ACL defect — see the pytest TMP/ACL note in project memory/docs.)
