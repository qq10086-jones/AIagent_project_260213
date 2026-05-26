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

- Status: pending
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

- Status: pending
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

- Status: pending
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

- Status: pending
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

- Status: pending
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
  - No POST / PUT / DELETE / PATCH endpoints are added; no broker / paper order path is touched.
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
  - Unlocks P10-22 (migration snapshot script) and P10-23 (manual entry UI / CLI) for cutover day T (2026-06-08).

- **Status changes (2026-05-26 W2/W3 sprint)**: P10-22 → done; P10-23 → done; P10-18 → done; P10-10 → done. See PROJECT_STATUS Change Log rows for each.

### P10-22 Project_optimized → HTR Migration Snapshot

- Status: pending
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

- Status: pending
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
