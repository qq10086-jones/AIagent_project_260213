# Folder Map

## Root

```text
HotThemeRotator/
  README.md
  PROJECT_STATUS.md
  configs/
  data/
  docs/
  notebooks/
  reports/
  src/
  tests/
  tools/
```

## Root Files

- `README.md`: 项目入口和定位。
- `PROJECT_STATUS.md`: 唯一项目更新文件。

## configs

放置配置样例和未来实盘/回测配置。禁止提交真实 API key、券商 token、账户信息。

Planned files:

- `strategy.example.yaml`
- `data_sources.example.yaml`
- `risk.example.yaml`

## data

只放本项目专用数据或缓存。旧项目数据库保持在 `../Project_optimized`，通过 adapter 读取。

- `data/raw`: 原始下载数据，不人工修改。
- `data/interim`: 清洗中间结果。
- `data/processed`: 可直接进入信号引擎的数据。
- `data/external`: 外部市场温度缓存，例如美股、A股、汇率、指数。

## docs

项目文档。

- `00_DESIGN.md`: 设计。
- `01_TASKS.md`: 任务。
- `02_GOVERNANCE.md`: 治理。
- `03_FOLDER_MAP.md`: 目录职责。
- `04_DATA_AND_OPEN_SOURCE.md`: 数据源和开源项目。
- `adr/`: 重大架构决策。

## notebooks

临时研究 notebook。任何 notebook 结论如果要进入项目事实，必须整理进正式报告或任务，不允许 notebook 成为事实来源。

## reports

机器或人工生成的输出。

- `reports/daily`: 开盘前和盘中简报。
- `reports/backtests`: 回测结果。
- `reports/paper`: paper 信号、复盘和模拟表现。
- `reports/predictions`: §8.6 / §10 gate 3 decision log JSONL 持久化（P9-01 起）。每个交易日一个 JSONL 文件，附 `prediction_id`、`input_snapshot_id`、`model_version`、`decision_cutoff`。
- `reports/outcomes`: §10 gate 4 outcome join JSONL 持久化（P9-02 起）。每个交易日一个 JSONL 文件，附 `outcome_id`、`prediction_id`、`evaluated_as_of`、`status`（complete / insufficient_data / symbol_not_found / future_cutoff / malformed_data）、`realized_returns`（1D/3D/5D）、`ladder_touches`（七档触达事件）。
- `reports/tdnet`: TDnet 適時開示 disclosures JSONL 持久化（P10-14 起）。每个交易日一个 JSONL 文件，附 `disclosure_id` (deterministic SHA-256 hash) / `ticker` / `company_name` / `published_ts` / `collected_ts` / `title` / `category` (earnings/order/tob/dividend/split/suspension/governance/other) / `url` / `summary` / `raw`。**HTR-native storage** per ADR-0005 read-only 契约 — HTR 不写回 Project_optimized DB。
- `reports/observability/pit/`: P11-00 PIT Observability Ledger（ADR-0007）。每个交易日一个子目录，每个 decision_cutoff 一个 `{snapshot_id}.json` 含完整 PIT state (universe / watchlist / filters / freshness / budget / silent_queue / user state / missing-data reasons / config version / model versions)。反思系统的真正基础。
- `reports/traces/`: P11-01 Decision Trace Logger。每个交易日一个子目录，每个 (symbol, decision_cutoff) 一个 `{trace_id}.jsonl` 含完整决策链 trace (module_chain / branch_decision / final_action / final_reason + link to PIT snapshot_id)。
- `reports/reflections/`: P11-06 Human Decision Gate 持久化。子目录 `proposals/` (active) + `accepted/` (Rule 4 触发后) + `rejected/{date}/{id}.json` (Rule 13.9 含 reason) + `expired/` (Rule 13.5 7-day auto-expire)。
- `reports/meta_strategy_journal.jsonl`: Rule 8.9 + Rule 12.6 共享 — 跨策略 advisory 决策记录。每行含 timestamp / source_strategy / target_strategy / symbol / proposed_delta / reason / user_decision。

- `reports/observability/price_health/`: P10-19 Cycle 2 delayed-price source health snapshots. One JSON per trade date, produced by `data/external/realtime_price/health.py` and the local CLI `tools/write_price_health_report.py`; rows include source, symbol, ok/fail, checked_ts, price, data_ts, wall_ts, inferred timestamp caveat, uncertainty, and fail_reason.
- `reports/observability/silent_queue/`: P10-17 Stage 1 silent watchlist event queue. One append-only JSONL per trade date; entries are dashboard-visible only and must carry `push_allowed=false`.

## src

正式代码。

```text
src/hot_theme_rotator/
  attribution/        # §8 universal attribution, 跨标的归因与决策评分
  calibration/        # §10 gate 5 calibration engine: Brier / log loss / reliability bins
  common/             # 跨模块 schema 与公共工具
  data/               # 数据接入与标准化 — P8-10..P8-15 起包含 Project_optimized 适配器
                      #   legacy_project_adapter.py / position_adapter.py /
                      #   kline_adapter.py / market_temp_adapter.py /
                      #   theme_heat_adapter.py / news_adapter.py / universe_adapter.py
                      #   见 ADR-0005
                      # data/external/    — P10-14 起。外部数据源 adapter（TDnet RSS / 未来 Yahoo JP scraper /
                      #   J-Quants live bridge）。HTR-native 存储到 `reports/tdnet/` 等，不写回 Project_optimized。
                      #   tdnet_schema.py / tdnet_storage.py / tdnet_parser.py / tdnet_rss_adapter.py (Cycle 2)
                      # data/external/realtime_price/  — P10-19 起。Best-effort delayed price orchestrator
                      #   (Codex 评审后 NOT "real-time")。yahoo_japan / kabutan / twelvedata / stooq /
                      #   orchestrator (fallback chain + cache + conditional consensus for high-salience) /
                      #   health.py (source health checks + reports/observability/price_health snapshots) /
                      #   http_policy.py (rate limit + User-Agent rotation + robots policy + Cloudflare detection)
  observability/      # P11-00 done (ADR-0007 §1 — Codex 评审 #1 missing component)。PIT Observability Ledger。
                      #   schema.py (PitSnapshot frozen dataclass + compute_snapshot_id + VALIDITY_CLASSES enum)
                      #   pit_ledger.py (append_snapshot / load_snapshot / sample_shadow_panel / derive_validity_class)
                      #   存储 reports/observability/pit/{trade_date}/{snapshot_id}.json
                      #   integration hooks into scanner / risk / watchlist / alerts → P11-01 ownership
  reflection/         # P11 全 7 layers done (ADR-0007). L0 在 observability/.
                      #   trace_logger.py (L1, P11-01 done)
                      #   cusum.py + bootstrap_arl.py + event_detector.py (L2, P11-02 done — Codex 评审 #3 全部)
                      #   policy_replay.py + validity_class.py (L3 NOT Pearl, P11-03 done — Data Freshness Gate)
                      #   rca.py + ablation.py + funnel.py (L4 NOT Shapley, P11-04 done — marginal_recovery)
                      #   decision_gate.py (L6, P11-06 done — Rule 13.5-13.9 enforced)
                      #   meta_reflection.py (L7, P11-07 done — Rule 13.10 trigger detection)
                      #   存储 reports/{traces,reflections/{proposals,accepted,rejected,expired}}/
  llm/                # P10-06 pending + P11-05 done (gemma4:e4b default), Rule 8.3.1 + 13.4 enforce.
                      #   reflection_brief.py (P11-05 done — narrative + Rule 8.3 regex + conditional language)
                      #   per_ticker_brief.py / ollama_client.py 真实集成归 P10-06 后续 cycle
  alerts/notifiers/   # P10-10 done。GuardedAlert envelope + 三个 channel (desktop/email/telegram) stub。
                      #   notifications_enabled 默认 False；只接受 push_allowed=True 的 GuardedAlert (Rule 12 type-enforced)。
                      #   alerts/discipline.py holds P10-18 Rule 12.1-12.6 evaluator + cross-strategy journal writer (Rule 12.6 / 8.9).
  watchlist_intelligence/  # P10-17 起。User watchlist + 持续 scan + event detection。
                      #   monitor.py / event_detector.py (不同于 reflection/event_detector.py)
                      #   silent_queue.py writes reports/observability/silent_queue/{trade_date}.jsonl
  user_state/         # P10-12 / P10-17 起 (Rule 11.3)。User-state store (watchlist / notes)，
                      #   严格隔离于 system-state (reports/, japan_market.db)。绝不进 decision_log。
  decision_log/       # §8.6 / §10 gate 3 + gate 4: predictions / outcomes JSONL 持久化与 outcome join
  execution_advice/   # 人工执行建议
  leader_ranking/     # 主题内龙头排序
  market_temperature/ # 日股市场温度 + 外部温度
  opportunity/        # §9 realtime opportunity 扫描与价格阶梯
  portfolio/          # P10-21..23 done (Section 14, ADR-0008)。HTR-owned append-only event journal + manual entry surface。
                      #   schema.py / journal_writer.py / derive.py / validation.py (P10-21)
                      #   manual_entry_service.py (P10-23, preview/commit + Rule 14.5 magnitude check)
                      #   migration.py (P10-22, Project_optimized → HTR cutover one-shot + NAV verify + idempotent marker)
                      #   存储 reports/portfolio/journal/{trade_date}.jsonl + reports/portfolio/migration_complete_{date}.json
  reporting/          # 报告与复盘
                      #   daily_advisory_cockpit.py (P10-20 Stage 0 pull-only cockpit payload)
  risk/               # 风控
  signal_engine/      # 信号生成
  theme_detection/    # 主题识别
  ui/                 # Streamlit 等用户界面层（legacy fallback）
```

## api

P8-09 起新增。FastAPI 只读 JSON 层，作为 Python 数据层与 frontend / Streamlit / 任何 HTTP 客户端的契约（ADR-0004）。

```text
api/
  __init__.py
  main.py              # FastAPI app + CORS + 生产模式 frontend/ 静态挂载 + 挂载 dashboard / symbol router
  dashboard.py         # GET /api/dashboard — V3 完整数据形状
  serializers.py       # Python 对象 -> V3 JSON（gates / markets / themes / candidates / news / log）
  symbol.py            # P8-18: GET /api/symbol/{ticker}/{kline,profile,ladder} 探索端点（Rule 11 + §6.11.1）
```

不暴露任何 POST/PUT/DELETE，不触发任何执行通路（Rule 3 在 API 层锁死）。

## frontend

P8-09 起新增。React 18 V3 市场温度仪表盘（ADR-0004，Phase 1 = zero-build）。

```text
frontend/
  index.html           # 入口：CDN 加载 React + Babel；fetch /api/dashboard 后 mount 选择的 variant
  shared.jsx           # 设计 tokens + 共享组件 + GLOSSARY + P8-18 交互 hook（useSelectedSymbol /
                       #   useSymbolKline / useSymbolProfile，localStorage user-state）
  v1.jsx               # V1 三栏专业终端（含 V1KLineLadderPanel 实时随 selected symbol 切换）
  v2.jsx               # V2 研究备忘录
  v3.jsx               # V3 市场温度仪表盘（含候选清单 onClick 切换 leader card）
  v4.jsx               # V4 决策日志为脊
  design-canvas.jsx    # 设计师 side-by-side 画布（产品 nav 不挂载，离线参考）
  tweaks-panel.jsx     # 颜色/字体/密度实时调节面板
  data.js              # mock 数据，仅作为 Python 层尚未供给字段的 fallback
```

Phase 1 全部为静态文本文件，无 `node_modules/` 无 `dist/`。FastAPI 在 `api/main.py` 中直接挂 `frontend/` 于 `/`。Phase 2（Vite 迁移）参见 ADR-0004。

## tests

测试。

- `tests/unit`: 单元测试。
- `tests/integration`: 需要本地数据库或多模块协作的测试。
- `tests/fixtures`: 小型样例数据。

## tools

一次性或命令行工具。工具不能绕过 `src` 中的正式模块直接写策略逻辑。

- `realtime_opportunity_demo.py`: P8-03 CLI demo，渲染机会面板与七档阶梯。
- `streamlit_opportunity_app.py`: P8-06/07/08 Streamlit fallback，端口 8501。
- `morning_briefing.py`: P8-19 开市前 CLI，`--watchlist X,Y --source db|yfinance`，输出 §9.4 banner + 持仓 marked-to-latest + watchlist 七档阶梯。Rule 11 read-only 用户态工具，不写 decision_log。
- `write_price_health_report.py`: P10-19 Cycle 2 delayed-price health CLI，按 `--symbols` 探测 Yahoo Japan / Kabutan 等本地配置源并写入 `reports/observability/price_health/{date}.json`。观测用途 only；不发送通知，不写订单。

## scripts

本地运维辅助脚本。除非用户明确要求，Codex 不运行会注册系统任务或改系统状态的脚本。

- `register_tdnet_poll_task.bat`: P10-14 TDnet RSS polling 的 Windows Task Scheduler 注册脚本。
- `register_price_health_task.bat`: P10-19 delayed-price health report 的 Windows Task Scheduler 注册脚本；需要显式传入 symbol 列表，只调用 `tools/write_price_health_report.py` 写本地观测报告。
