# HotThemeRotator

## Local User Interface

Run the local dashboard:

```powershell
cd E:\AIagent_project_260213\worker-quant\quant_trading\HotThemeRotator
streamlit run .\tools\streamlit_opportunity_app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

The first screen is `今日机会中心`: it shows the top candidate, staged buy zone, stop price, staged sell zone, plain-language reasons, and visible risk warnings. It uses sample data by default. Switch to `免费行情 yfinance` in the sidebar to try quote-only free web data. All output is research-only and uncalibrated until feedback calibration is implemented.

日股为主、A股和美股为外部温度因子的热点龙头轮动工具。

本项目不是高频交易系统，也不是自动实盘下单系统。第一阶段目标是把主观的“市场温度、新闻热度、板块龙头、2%-5%止盈换仓”流程固化成可解释、可回测、可复盘的量化决策工具。

## 项目定位

- 主市场：日本股票。
- 外部温度：A股、美股、汇率、指数、半导体/AI/汽车/机械等跨市场链条。
- 核心策略：新闻催化 + 市场温度 + 热点主题 + 龙头强度 + 严格退出。
- 执行模式：只生成交易建议和风控提示，人工确认后执行。
- 继承资产：参考 `../Project_optimized` 的日股数据库、新闻 overlay、候选排序、盘中建议和报告经验；参考 `../Project_v5` 的 V7 News-Driven Hot Theme Hunter 设计。

## 入口文档

- `PROJECT_STATUS.md`：唯一项目更新文件，所有进度、决策状态、下一步都只更新这里。
- `docs/00_DESIGN.md`：产品和系统设计。
- `docs/01_TASKS.md`：任务清单。
- `docs/02_GOVERNANCE.md`：治理规则和修改规则。
- `docs/03_FOLDER_MAP.md`：目录职责。
- `docs/04_DATA_AND_OPEN_SOURCE.md`：数据源和开源项目选型。

## 唯一推进规则

任何改动必须先对应到 `docs/01_TASKS.md` 中的任务，并在 `PROJECT_STATUS.md` 记录状态。禁止新增散落的 `PROGRESS_*.md`、临时总结文档或口头状态作为项目事实来源。
