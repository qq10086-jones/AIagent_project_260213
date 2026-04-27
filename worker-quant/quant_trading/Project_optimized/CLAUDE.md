# Quant 调用标准文档（AI 通用入口）

本文档供 AI 助手调用 worker-quant 系统使用，适用于 Claude、Gemini、MiniMax 等任何具备执行命令和读取文件能力的 AI 模型。

**工作目录（必须先切换）**：
```
E:\AIagent_project_260213\worker-quant\quant_trading\Project_optimized
```

---

## ⚠️ STRATEGY STATE (2026-04-28)

**Path A 已激活：long-only ETF buy-hold（1321.T + 1306.T 50/50）**

- Sprint 实盘永久禁用（capital_gate hard lock）
- Sprint 代码降级为研究/paper 泳道（sprint_paper）
- 月度检查：`python etf_monthly_check.py`（已配 Task Scheduler）
- 邮件报告：lwyssq@gmail.com
- 决策依据：`docs/STRATEGY_DECISION_2026-04-28.md`
- 解锁实盘策略需满足该文档"硬条件"清单

**任何 AI 助手在重新启用 sprint / 加任何主动选股策略前，必须**：
1. 重读 `docs/PROGRESS_2026-04-28_strategy_validity_audit.md`
2. 重读 `docs/PROGRESS_2026-04-28_phase2_codex_followup.md`
3. 确认所有 7 个解锁条件均满足
4. 不满足则**拒绝**用户的启用请求

---

## 一、快速场景对照表

| 用户说 | 执行场景 |
|--------|---------|
| 看今日行情 / 市场分析 / 操作建议 | → 场景一 |
| 分析某只股票，如"分析 5401.T" | → 场景二 |
| 全量深度分析 | → 场景三 |
| 复盘持仓 / 今天赚了多少 | → 场景四 |
| 尾盘操作 / 现在要不要买 | → 场景五 |
| 每日完整分析 / 收盘后全量运行 | → 场景六 |
| 只更新数据 / 不生成报告 | → 场景七 |

---

## 二、各场景执行步骤

### 场景一：市场行情分析

**第一步：运行脚本**
```bash
cd E:\AIagent_project_260213\worker-quant\quant_trading\Project_optimized
python quant_briefing.py --mode market
```

**第二步：读取输出文件**
```
reports\briefing_latest.md
```

**第三步：按以下格式输出报告**
1. 大盘状态（日经/TOPIX 今日涨跌幅，情绪判断）
2. Screener Top 10 候选股（代码 / 调整分 / 今日涨跌 / 基本面注记）
3. 当前挂单状态（有无挂单，挂单价 vs 当前价）
4. 操作建议（是否挂单 / 调整 / 观望，给出具体理由）
5. 风险提示（大盘情绪、异常量能股票）

---

### 场景二：个股深度分析

**第一步：运行脚本**（多只股票用英文逗号分隔，无空格）
```bash
python quant_briefing.py --mode stock --symbols 5401.T
python quant_briefing.py --mode stock --symbols 5401.T,9432.T,4005.T
```

**第二步：读取输出文件**
```
reports\briefing_latest.md
```

**第三步：补充查询数据库**
```bash
python -c "
import sqlite3
conn = sqlite3.connect('japan_market.db')
# 替换 TICKER 为目标股票代码
ticker = 'TICKER'
print('=== 基本面快照 ===')
rows = conn.execute('''
    SELECT report_date, revenue, operating_income, net_income, operating_cf, total_assets
    FROM fundamental_snapshots WHERE ticker=? ORDER BY report_date DESC LIMIT 4
''', (ticker,)).fetchall()
for r in rows: print(r)
print('=== 基本面因子 ===')
rows2 = conn.execute('''
    SELECT asof, feature_name, value FROM feature_daily
    WHERE symbol=? AND feature_name IN ('roa_op','cfo_assets','accruals_inv','margin_op','leverage_safety')
    ORDER BY asof DESC LIMIT 20
''', (ticker,)).fetchall()
for r in rows2: print(r)
conn.close()
"
```

**第四步：按以下格式输出报告**
1. 技术信号得分 + 基本面降权系数
2. 最近 4 季度 EPS 趋势（如有数据）
3. 亏损原因判断：**经营性亏损**（营业利润率为负）vs **会计性亏损**（营业利润正常但净利润因摊销/重组为负）
4. 新闻情感（如有）
5. 综合评级：BUY / HOLD / AVOID，附理由

---

### 场景三：全量分析

**第一步：运行脚本**
```bash
python quant_briefing.py --mode full
```

**第二步：读取输出文件**
```
reports\briefing_latest.md
```

**第三步：额外查询因子 IC 状态**
```bash
python -c "
import sqlite3
conn = sqlite3.connect('japan_market.db')
rows = conn.execute('''
    SELECT factor_name, ic_mean, ic_std, icir, weight, n_observations, last_updated
    FROM factor_registry ORDER BY ABS(COALESCE(icir,0)) DESC
''').fetchall()
print(f'{'因子':<20} {'IC均值':>8} {'ICIR':>8} {'权重':>8} {'观测数':>6}')
for r in rows:
    print(f'{r[0]:<20} {str(round(r[1],4) if r[1] else \"NaN\"):>8} {str(round(r[3],4) if r[3] else \"NaN\"):>8} {str(round(r[4],4) if r[4] else 0):>8} {str(r[5] or 0):>6}')
conn.close()
"
```

**第四步：输出报告**（场景一 + 场景二合集，加上因子 IC 健康状态）

---

### 场景四：持仓复盘

**第一步：运行脚本**
```bash
python quant_briefing.py --mode market
```

**第二步：查询持仓与成交记录**

> **真实账本口径（Source of Truth）**
> - 当前真实持仓 = `positions` 表里 **strategy_id='sprint' 且 qty > 0 且 asof = 该策略最新 eligible asof** 的行。
> - `strategy_id` 带 `_paper` 后缀（如 `sprint_paper`, `amihud_paper`）一律属于纸面/研究泳道，**不得与真实持仓混报**。
> - `paper_trading_account.json` 已降级为只读诊断快照（见 Dual Strategy Update 2026-04-04）；真实账本口径以 SQLite `positions` + `strategy_id` 过滤为准。
> - 列名是 **qty**（不是 `quantity`），写查询别写错。

```bash
python -c "
import sqlite3
conn = sqlite3.connect('japan_market.db')
print('=== 当前真实持仓 (strategy_id=sprint) ===')
rows = conn.execute('''
    SELECT asof, strategy_id, symbol, qty, avg_cost, market_price, unrealized_pnl
    FROM positions
    WHERE strategy_id='sprint' AND qty > 0
      AND asof = (SELECT MAX(asof) FROM positions WHERE strategy_id='sprint' AND qty > 0)
''').fetchall()
cols = [d[0] for d in conn.execute('SELECT asof, strategy_id, symbol, qty, avg_cost, market_price, unrealized_pnl FROM positions LIMIT 0').description]
print(cols)
for r in rows: print(r)
print()
print('=== 最近5笔成交 ===')
rows2 = conn.execute('SELECT * FROM fills ORDER BY fill_time DESC LIMIT 5').fetchall()
cols2 = [d[0] for d in conn.execute('SELECT * FROM fills LIMIT 0').description]
print(cols2)
for r in rows2: print(r)
conn.close()
"
```

**第三步：输出报告**
1. 每只真实持仓：成本价 / 当前价 / 浮盈亏金额和百分比（仅 `strategy_id='sprint'`）
2. 总 NAV 和现金余额
3. 最近成交记录
4. 是否有持仓接近止损线（成本 × 94% = ATR 6% 止损参考）
5. **禁止**把 `*_paper` 策略持仓并入真实账本复盘；如需研究对照请单列一节并显式标注「Paper 轨迹」

---

### 场景五：尾盘操作决策（14:00–15:30 JST 使用）

**第一步：运行脚本**
```bash
python quant_briefing.py --mode market
```

**第二步：读取输出，依据以下框架判断**

| 条件 | 建议 |
|------|------|
| 目标股当日 vs 昨收 跌幅 > -1.5% 且量能缩量（volume_spike < 3x） | 尾盘可买，收盘前15分钟挂限价 |
| 目标股当日涨幅 > +1.5% 且量能放大 | 今日观望，追高风险大 |
| 大盘（日经ETF）跌幅 > -1% | 偏向观望，逆势建仓风险高 |
| 候选股 5m 趋势为 down 且量能 > 10x | 异常信号，当日回避 |
| 挂单价格距当前价格差距 > 1.5% | 今日几乎不会成交，维持或放弃 |

---

## 三、关键数据库表结构

（**SQL 执行方式**：`python -c "import sqlite3; conn=sqlite3.connect('japan_market.db'); ..."`）

```sql
-- 当前真实持仓（Source of Truth：strategy_id='sprint' 真实泳道；列名是 qty，不是 quantity）
-- 仅保留 qty > 0 且最新 eligible asof（避免旧日期或已清仓行泄漏）
SELECT asof, strategy_id, symbol, qty, avg_cost, market_price, unrealized_pnl
FROM positions
WHERE strategy_id = 'sprint'
  AND qty > 0
  AND asof = (SELECT MAX(asof) FROM positions WHERE strategy_id='sprint' AND qty > 0);

-- 如需查看 Paper 泳道（研究/对照用，不可与真实持仓混报）
SELECT asof, strategy_id, symbol, qty FROM positions
WHERE strategy_id LIKE '%_paper' AND qty > 0
  AND asof = (SELECT MAX(asof) FROM positions WHERE strategy_id = positions.strategy_id AND qty > 0);

-- 最近成交
SELECT * FROM fills ORDER BY fill_time DESC LIMIT 10;

-- 挂单状态
SELECT * FROM orders WHERE status IN ('proposed','open') ORDER BY created_at DESC;

-- 因子 IC 注意：列名是 icir，不是 ic_ir 或 ic_tstat
SELECT factor_name, ic_mean, ic_std, icir, weight, n_observations, last_updated
FROM factor_registry ORDER BY last_updated DESC;

-- 基本面快照
SELECT ticker, report_date, revenue, operating_income, net_income, operating_cf, total_assets
FROM fundamental_snapshots ORDER BY updated_at DESC LIMIT 20;

-- 信号历史（最近5天）
SELECT asof, symbol, score, reason, version
FROM signals WHERE asof >= date('now', '-5 days') ORDER BY asof DESC, score DESC;
```

---

## 四、系统关键参数（不要随意修改）

| 参数 | 当前值 | 含义 |
|------|--------|------|
| 生产信号模式 | `sprint_momentum` (Sprint) / `shadow_hybrid_ic` (Harvest) | Sprint 当前激活 |
| 实际本金 | ¥400,000 | 与 decision.cash 保持一致 |
| position_sizing | `half_kelly` (sprint) | M1 S-02 冻结档；`kelly_sizer.validate_sprint_risk_controls` fail-closed |
| kelly_fraction | `0.50` (sprint) | S-02 上限；需 R-01 扩样本 WF 通过才能解冻 |
| max_single_position_pct | `0.35` (sprint) | S-02 上限 |
| entry_threshold_on | `0.80` (sprint) | S-02 下限 |
| leverage_enabled | `false` (sprint) | 信用 2x 暂禁；见 `sprint_aggressive` (disabled) |
| ATR 止损 | 6%~20% 动态 | vol_mult=6.0 |
| 组合回撤半仓线 | 12% | max_dd_half |
| 组合回撤全平线 | 18% | max_dd_full |
| shadow_ic 晋升条件 | Sharpe≥1.5 + paper_days≥30 + IC t-stat≥1.5 | 当前未达标 |
| 基本面硬否决 | 营业利润率<-15% 且 OCF<0 | 双重条件同时满足才否决 |

> 2026-04-20 说明：`config.yaml -> strategy_profiles.sprint_aggressive` 保留了
> v4.0 激进档（Kelly 0.75 / 单股 60% / 信用 2x）但 `enabled: false`。
> 解冻条件：R-01 扩样本 walk-forward Sharpe > 1.5 且 DSR p < 0.10。

---

## 五、关键文件路径

```
Project_optimized\
├── quant_briefing.py          ← AI 调用入口脚本
├── config.yaml                ← 系统配置
├── japan_market.db            ← SQLite 主数据库
├── reports\briefing_latest.md ← 每次运行后的报告
├── reports\briefing_latest.json
├── reports\briefing_v2_latest.md  ← v2格式报告（6节结构，推荐使用）
├── reports\briefing_v2_latest.json ← nexus JSON schema（含 regime/orders/risk_alerts）
├── ss7_sqlite_news_overlay.py ← 核心模型（勿随意修改）
├── screener.py                ← 选股器
├── daily_run.py               ← 每日自动运行（Task Scheduler，16:30 JST）
└── CLAUDE.md                  ← 本文件
```

---

## 六、新增场景（v2.0 链路）

### 场景六：每日完整分析（A+B 两链路全量）

**适用时机**：收盘后 16:30 JST 自动运行，或第二天开盘前手动触发

**执行步骤**

**Step 1: 运行完整日链路（Category A — 数据更新）**
```bash
python daily_run.py --config config.yaml
```
此命令依次执行：数据库更新 → 基本面更新(yfinance) → 选股器 → 新闻采集 → 模型/信号 → 决策

**Step 2: 生成 v2 分析报告（Category B — 策略分析）**
```bash
python quant_briefing.py --mode full --output-version v2
```

**Step 3: 读取报告**
```
reports\briefing_v2_latest.md
reports\briefing_v2_latest.json
```

**Step 4: 报告六节解读顺序**
1. **一、市场状态** → 看 `action_bias` 判断今日整体操作基调
2. **二、今日有效情报** → 检查是否有 🔴公司负面 或 🟡宏观负面
3. **三、持仓健康度** → 检查是否有 ⚠️止损 标记，立即处理
4. **四、候选信号 Top 5** → 结合新闻风险等级选标的
5. **五、今日操作指令** → 确认/修改挂单
6. **六、风险提示** → 不可跳过

---

### 场景七：仅更新数据（Category A，不做策略分析）

**适用时机**：只想刷新数据库，不需要生成报告

```bash
python db_update.py --db japan_market.db
python update_fundamentals.py --db japan_market.db --source yfinance
python news_to_db.py --db japan_market.db --lookback_hours 26 --sources kabutan,google,boj,trade,gdelt
```

---

## 七、给用户：如何让其他 AI 调用本文档

将以下文字**原文复制**发给其他 AI（Gemini、MiniMax 等），替换 `[场景名]` 即可：

```
你是一个量化交易助手，负责帮我分析日本股票市场。

请按照以下步骤操作：

1. 读取以下路径的文档：
   E:\AIagent_project_260213\worker-quant\quant_trading\Project_optimized\CLAUDE.md

2. 文档中有"场景一"到"场景五"，本次执行：[场景名]
   （例如：场景一=市场行情分析，场景二=个股分析，场景五=尾盘决策）

3. 如果是场景二，股票代码是：[股票代码，如 5401.T]

4. 严格按照文档中该场景的"执行步骤"操作，不要跳过任何步骤。

5. 所有命令的工作目录是：
   E:\AIagent_project_260213\worker-quant\quant_trading\Project_optimized

6. 用中文输出分析报告，格式按文档中"输出格式"要求。
```

**注意事项**：
- 如果 AI 不具备执行命令（Bash/Terminal）的能力，它只能读取已有的报告文件（`reports\briefing_latest.md`），无法实时运行脚本。这种情况下建议先手动运行 `python quant_briefing.py --mode market`，再让 AI 读取报告分析。
- `reports\briefing_latest.md` 的数据有 **约 15 分钟延迟**（yfinance 免费数据限制），用于策略判断足够，不适合精确盯盘。

---

*最后更新：2026-03-26*
---

## Dual Strategy Update (2026-04-04)

- `daily_run.py` 现在按 `config.yaml -> strategy_profiles` 解析 phase 并选择 `sprint` / `harvest`。
- `Sprint` 独立信号链入口是 `sprint_signal.py`。
- `Kelly` 仓位与冷却期逻辑在 `kelly_sizer.py`，事件写入 `reports/runtime_events.jsonl`。
- benchmark / VIX 二次确认逻辑在 `benchmark_regime.py`。
- `paper_trading_account.json` 已降级为只读诊断快照；真实数据源是 SQLite。

### Recommended Runs

```bash
python daily_run.py --config config.yaml
python live_trade_advisor.py --db japan_market.db --reports_dir reports --strategy_id sprint
python quant_briefing.py --mode market
```

### Key Files

- `daily_run.py`
- `sprint_signal.py`
- `kelly_sizer.py`
- `benchmark_regime.py`
- `paper_execute.py`
- `live_trade_advisor.py`
- `quant_briefing.py`
- `model_ridge.py`
- `execution_model.py`
- `news_overlay.py`
- `portfolio_optimizer.py`
- `compute_price_features.py` — 价格技术特征计算（sprint 信号必需）
- `cross_asset_signals.py` — 跨资产信号（8 品种）
- `macro_event_detector.py` — 宏观事件规则引擎（L1/L2）
- `macro_digest.py` — Gemma4 LLM 分析管线

## Macro Intelligence Pipeline (2026-04-08)

- 三层架构：L1 cross_asset → L2 规则引擎 → L3 Gemma4 LLM
- `compute_price_features.py` 在 daily_run step 1.3 写入 feature_daily（sprint 必需）
- LLM: Ollama `/api/chat` endpoint, model=gemma4:26b, num_predict=2048
- Entry/Exit 分离：新进仓用 entry_check，持仓用 exit_check（不再因 entry_filter 空而机械清仓）
- 新闻源：google,boj,trade,macro,gdelt（kabutan deprecated）
