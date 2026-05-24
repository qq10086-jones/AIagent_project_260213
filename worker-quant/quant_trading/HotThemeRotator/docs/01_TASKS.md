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

- Status: pending
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

### P8-19 Morning Briefing CLI

- Status: pending
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
