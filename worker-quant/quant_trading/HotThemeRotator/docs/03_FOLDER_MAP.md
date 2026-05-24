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
- `reports/outcomes`: §10 gate 4 outcome join JSONL 持久化（P9-02 起）。每个交易日一个 JSONL 文件，附 `outcome_id`、`prediction_id`、`evaluated_as_of`、`status`（complete / insufficient_data / symbol_not_found / future_cutoff）、`realized_returns`（1D/3D/5D）、`ladder_touches`（七档触达事件）。

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
  decision_log/       # §8.6 / §10 gate 3 + gate 4: predictions / outcomes JSONL 持久化与 outcome join
  execution_advice/   # 人工执行建议
  leader_ranking/     # 主题内龙头排序
  market_temperature/ # 日股市场温度 + 外部温度
  opportunity/        # §9 realtime opportunity 扫描与价格阶梯
  portfolio/          # 组合状态与换仓
  reporting/          # 报告与复盘
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
