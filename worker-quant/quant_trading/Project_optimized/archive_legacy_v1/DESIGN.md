# Japan Quant Strategy — 系统设计文档

## 2026-03-10 PM Update

### Product Position

This project should now be treated as a research-and-operations platform, not a retail-ready autopilot product.

- It is already strong in pipeline completeness: data update, screener, backtest, risk control, decision package, and learning tables all exist.
- It is not yet a beginner-safe product. The main missing layer is productization: safer defaults, clearer mode selection, paper-trading workflow, and simplified operator UX.

### Current Quant Conclusion

Recent production-side diagnostics changed the project direction materially.

- The old assumption "improve screener sample size and Ridge will pass Learning M1" was wrong.
- After aligning `compute_ic.py` to the actual screened universe and logging production `factor_signals` directly from `ss7_sqlite_news_overlay.py`, the data shows:
- Several raw factors are genuinely useful on the production universe: `mom_consist`, `rsi14`, `vol_adj_mom20`.
- The current Ridge composite `pred_return` is weaker than these factors and can even point in the wrong direction.
- Two simple shadow composites (`shadow_eq`, `shadow_ic`) materially outperform the current Ridge mode in backtest under the same execution/risk model.

### Strategic Decision

The project should move from "Ridge-first" to "signal-composite-first".

- Keep Ridge as a benchmark, not as the unquestioned production default.
- Use shadow comparison as the standard promotion path for any new signal constructor.
- Only promote a signal mode after it wins on:
- production-universe IC / t-stat
- same-cost backtest
- paper-trading stability

### Target Product States

The project should be managed in three explicit product states:

1. Research state
   Used by quant researcher / PM to test factors, screeners, and signal constructors.
2. Operator state
   Used by a knowledgeable operator to run daily pipeline, inspect outputs, and approve trades.
3. Beginner-safe state
   Used by a non-expert with minimal parameter exposure, strong guardrails, and paper-trading-first workflow.

The codebase is currently between state 1 and state 2.

### Updated Roadmap Priority

#### P0: Promote the right primary signal

- Add config-level `signal_mode` selection to the pipeline entrypoint.
- Compare `ridge`, `shadow_eq`, and `shadow_ic` on every run.
- Run at least one full validation cycle with `shadow_ic` as the primary candidate.

#### P1: Make Learning trustworthy end-to-end

- Keep production-time `factor_signals` as the preferred source of truth.
- Use recomputed features only as historical backfill when production logs are missing.
- Add diagnostics for `pred_return` itself, not just raw factors.

#### P2: Paper-trading productization

- Build a one-command paper-trading mode.
- Auto-save decision snapshots, target weights, and account state for each cycle.
- Add beginner-safe summaries: current mode, expected turnover, max drawdown state, and stop-loss state.

#### P3: Beginner-safe operating layer

- Reduce visible parameters to a small approved set.
- Add fail-closed checks before trade generation.
- Add warnings for dangerous modes or missing data.

### PM View

From a product-management perspective, the project should not optimize for "more features" right now.

The next success criterion is:

"Can we safely operate the best currently-known signal mode repeatedly, explain its behavior, and validate it in paper trading?"

Until that is true, adding more model complexity is lower priority than improving clarity, defaults, and promotion criteria.

## 2026-03-10 Nexus Compliance Update

### Nexus Role

Inside Nexus v4, quant is not a standalone product domain with autonomous workflow authority.

It should be treated as:

- a terminal execution worker
- a domain evidence producer
- a quant-local state owner

It should not be treated as:

- a top-level task router
- an approval authority
- an unrestricted autonomous trading agent

Reference contract:

- [NEXUS_WORKER_CONTRACT.md](C:/Users/linweiye/AIagent_project_260213/worker-quant/quant_trading/Project_optimized/NEXUS_WORKER_CONTRACT.md)

### Required Nexus Alignment

This design document is now a domain design document plus a worker-integration design document.

To be Nexus-compliant, the quant subsystem must always describe:

- capability surface
- tool-to-script mapping
- required artifacts
- fail-closed / degrade-safe behavior
- queryable evidence sources

### Updated Design Rule

Every future quant feature should be accepted only if it answers both questions:

1. Is it a better quant decision mechanism?
2. Can Nexus route it, approve it, audit it, and roll it back safely?

If the second answer is unclear, the feature is not ready for integration even if the quant result looks good.

> **目标**：用 100 万日元本金，通过系统化量化交易在东京证券交易所（TSE）实现 20%+ 年化收益，
> Sharpe Ratio > 1.0，最大回撤控制在 20% 以内。

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [数据层](#2-数据层)
3. [Alpha 模型（ss7）](#3-alpha-模型ss7)
4. [新闻情绪层](#4-新闻情绪层)
5. [风险管理框架](#5-风险管理框架)
6. [执行模型](#6-执行模型)
7. [Learning 机制](#7-learning-机制)
8. [运营流水线](#8-运营流水线)
9. [资金规模约束与现实预期](#9-资金规模约束与现实预期)
10. [改进路线图](#10-改进路线图)

---

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        每日运营周期                               │
│                                                                   │
│  [数据入库]  →  [选股筛选]  →  [Alpha模型]  →  [新闻门控]        │
│  db_update     screener      ss7 backtest    news overlay          │
│                                                                   │
│       →  [风险管理]  →  [订单生成]  →  [手动执行]  →  [盘后处理] │
│          stop-loss     make_decision   SBI证券       post_trade    │
│          max-drawdown                                             │
│                                                                   │
│       →  [Learning 反馈]  ←  [实盘成交数据]                      │
│          IC追踪 / 因子调权   import_fills                         │
└─────────────────────────────────────────────────────────────────┘

基础设施：
- SQLite (japan_market.db)  ←→  MarketDB / trade_schema
- Redis Streams             ←→  worker.py（分布式任务调度）
- Obsidian Vault            ←→  report_obsidian.py（知识库归档）
```

### 核心文件

| 文件 | 职责 | 调用方式 |
|------|------|---------|
| `ss7_sqlite_news_overlay.py` | Alpha 模型 + 新闻门控 + 回测引擎 | `run_pipeline.py` via env |
| `db_update.py` | 数据入库（~100 只 TSE 股票）| `run_pipeline.py` |
| `screener.py` | 选股过滤（ADV、波动率、1手成本）| `run_pipeline.py` |
| `make_decision.py` | 订单生成 + 回撤门控 | 手动 / daily_run.py |
| `run_pipeline.py` | 完整流水线入口 | cron / 手动 |
| `daily_run.py` | 盘后定时任务调度 | Windows Task Scheduler |
| `worker.py` | 分布式工具调度（Redis Streams）| Docker |

---

## 2. 数据层

### 2.1 价格数据库

- **存储**：SQLite `japan_market.db`，通过 `MarketDB`（`market_db_v2.py`）访问
- **来源**：yfinance `auto_adjust=True`（复权价，消除分红/送股影响）
- **覆盖**：~100 只 TSE Prime 流动性股票 + 基准 ETF（1321.T、1306.T）
- **更新频率**：每日盘后增量更新（仅拉取 `last_date + 1` 以后的数据）

### 2.2 股票宇宙（db_update.py）

覆盖 12 个行业板块，约 100 只股票：

| 板块 | 代表股票 | 选入理由 |
|------|---------|---------|
| 半导体/AI | 8035.T TEL、6857.T Advantest | AI 主线，高 beta |
| 商社贸易 | 8058.T 三菱商事、8001.T 伊藤忠 | 巴菲特概念，高股息 |
| 金融保险 | 8306.T MUFG、8766.T 东京海上 | 加息受益，稳定分红 |
| 汽车 | 7203.T 丰田、7267.T 本田 | 汇率敏感，大市值 |
| 消费/零售 | 9983.T 优衣库、7974.T 任天堂 | 内需防御 |
| 医疗/制药 | 4502.T 武田、4568.T 第一三共 | 防御性，低相关 |
| 重工/国防 | 7011.T 三菱重工、7012.T 川崎重工 | 地缘政治对冲 |
| 电信/公用 | 9432.T NTT、9501.T 东电 | 高股息，低波动 |
| 能源/化工 | 5020.T ENEOS、1605.T Inpex | 大宗商品对冲 |
| 航运/运输 | 9101.T 日邮船、9020.T JR 东日本 | 周期性，高分红 |
| 互联网/平台 | 4755.T 乐天、6098.T Recruit | 成长性 |
| 钢铁/材料 | 5401.T 新日铁、5713.T 住友金属矿山 | 原材料周期 |

### 2.3 选股筛选（screener.py）

**硬过滤条件（全部满足才入选）：**

```python
ADV (20日均成交额) > 20,000,000 JPY        # 流动性门槛
缺失率 < 2%                                 # 数据质量
日波动率 in [0.5%, 6%]                      # 排除僵尸股和极端波动股
1手成本 (= 收盘价 × 100) < 150,000 JPY     # ★ 关键：确保100万日元可持有≥6只
```

**软排名（通过后按分数排序，取前 top_k 只）：**
```python
score = log(ADV) - |log(vol) - log(0.02)| - missing_rate × 50
```
偏好：高流动性、日波动率约 2%、数据完整。

---

## 3. Alpha 模型（ss7）

### 3.1 特征体系（当前版本）

| 特征 | 计算方式 | 捕捉信号 |
|------|---------|---------|
| ret1/5/20/60 | 多周期动量收益率 | 短/中/长期动量 |
| vol20/vol60 | 滚动收益率标准差 | 波动率状态 |
| ma_gap | MA50/MA200 偏离度 | 趋势偏离 |
| z_20 | 20日 Z-score | 均值回归 |
| rsi14 | RSI(14) / 100 | 超买超卖 |
| slope60 | 60日对数价格线性斜率 | 趋势动量 |

**已知局限**（P2 改进方向）：
- 9 个特征全为价格衍生，因子间相关度高（动量类 IC 相关 > 0.7）
- 缺乏基本面因子（PBR、ROE、股息率）——日本市场 Fama-French 效应显著
- 缺乏跨品种相对动量（vs 行业均值的超额收益）

### 3.2 预测模型

**PanelRidge（横截面 Ridge 回归）：**
```python
target = forward_return(H=20天) / vol_20    # 风险调整后的前瞻收益
model = Ridge(alpha=10) with z-score standardization
training = walk-forward, 252日滑动窗口, 无前瞻偏差
```

**前瞻偏差修复（关键）：**
- 训练标签 t 需要 t+H 日价格，因此训练集最新日期为 `decision_date - H`
- 预测用当日特征（已知），不泄漏未来信息

### 3.3 组合优化器

**长只均值方差 + 换手惩罚：**
```
min_w  -μ'w + λ·w'Σw + γ·‖w - w_prev‖²
s.t.   w ≥ 0,  Σwᵢ = 1
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| λ (lam) | 2.0 | 风险厌恶系数 |
| γ (gamma) | 10.0 | 换手惩罚强度 |
| shrink_delta | 0.5 | Ledoit-Wolf 协方差收缩比 |

**风险熄火规则（Risk-Off）：**
当基准 1321.T 收盘价 < 60 日均线时，目标权重全部置 0，持有现金。

---

## 4. 新闻情绪层

### 4.1 设计哲学

> **新闻不直接进入 Alpha 预测器**。它作为独立的"风险/曝险门控层"，
> 在市场情绪混乱或过热时降低持仓暴露，而不是追逐热点。

### 4.2 数据格式

```csv
date,ticker,sent,weight,conf
2026-01-05,7203.T,0.8,1.5,0.9
2026-01-05,8058.T,-0.3,1.0,0.7
```

| 字段 | 范围 | 含义 |
|------|------|------|
| sent | [-1, 1] | 情绪方向（正=看涨，负=看跌）|
| weight | ≥ 0 | 新闻重要性权重 |
| conf | [0, 1] | 情绪置信度 |

### 4.3 三因子计算（指数衰减，默认半衰期 3 天）

```
F (Directional Sentiment) = Σ(weight × conf × sent) / Σ(weight × conf)
A (Attention/Intensity)   = Σ(weight × conf)          ← 市场关注度，过高则过热
U (Disagreement)          = 1 - |F| / Σ|weight×conf×sent|  ← 信号分歧度
```

衰减公式：`w_day = exp(-λ × days_ago)`，λ = ln(2) / half_life

### 4.4 门控逻辑

```python
# 硬门：极端情绪时直接降至最低曝险
if A >= A_max or U >= U_high:
    g = g_min  # 默认 0.15（降至 15% 仓位）

# 软门：sigmoid 打分，综合三因子
score = k_absF × (|F| - absF_min) - k_U × U - k_A × A
g_soft = g_min + (1 - g_min) × sigmoid(score)
g_final = min(g_hard, g_soft)

# 应用：对每只股票分别门控，重新归一化
w_new[i] = w_target[i] × g_final[i]
w_new = w_new / sum(w_new)
```

### 4.5 当前状态与接入方式

目前新闻数据尚未自动化采集，需手动提供 CSV：
```bash
SS6_NEWS_ON=1 SS6_NEWS_CSV=./news.csv python ss7_sqlite_news_overlay.py
```

`worker.py` 中已有 `news.preclose_brief_jp` 和 `news.tdnet_close_flash` 工具，
**待办**：将这两个工具的输出自动格式化为新闻 CSV，接入 ss7 门控层。

---

## 5. 风险管理框架

### 5.1 单只股票止损（Stop-Loss）

```python
# 回测和实盘均适用
entry_px[ticker] = 买入时的加权平均成本价
if (current_price - entry_px[ticker]) / entry_px[ticker] < -0.08:
    w_target[ticker] = 0.0    # 当日目标权重置 0，下次调仓时执行清仓
```

**触发条件**：任意持仓的当前价格较成本价下跌超过 **8%**

### 5.2 组合最大回撤控制

```python
peak_equity = max(peak_equity, current_nav)
drawdown = (current_nav - peak_equity) / peak_equity

if drawdown < -0.18:   # 超过 18%
    dd_scale = 0.0     # 全部平仓，退出市场
elif drawdown < -0.12: # 超过 12%
    dd_scale = 0.5     # 降至半仓，保留现金
else:
    dd_scale = 1.0     # 正常持仓

w_target = w_target × dd_scale
```

### 5.3 make_decision.py 实盘回撤检查

```bash
python make_decision.py \
  --cash 1000000 \
  --lot 100 \
  --peak_nav 1050000 \   # 历史最高 NAV，手动输入或从 DB 读取
  --dd_half 0.12 \
  --dd_full 0.18
```

当前 NAV 低于峰值 12%：BUY 订单数量减半；低于 18%：取消所有 BUY 订单。

### 5.4 执行成本模型（回测与实盘一致）

| 成本项 | 参数 | 当前设置 | 说明 |
|--------|------|---------|------|
| 手续费 | fee_bps | 5.0 bps | SBI 证券零售端实际费率 |
| 滑点 | slippage_bps | 5.0 bps | 开收盘价差，日本市场约 3-8 bps |
| 市场冲击 | impact_k | 0.5 | `k × √(trade/ADV) × trade_notional` |
| 手数约束 | lot_size | 100 股 | 日本标准手数，向下取整 |

---

## 6. 执行模型

### 6.1 资金约束（ExecConfig）

```python
initial_capital = 1,000,000 JPY   # 100 万日元本金
lot_size_default = 100            # 1 手 = 100 股
```

**1手成本限制的必要性**（核心约束）：

| 股票 | 股价 | 1手成本 | 100万可买 | 持仓集中度 |
|------|------|---------|---------|----------|
| screener 过滤后（目标）| < ¥1,500 | < ¥150,000 | ≥ 6只 | < 17% 单股 |
| 丰田 7203.T | ¥3,200 | ¥320,000 | 3只 | 33%（过高）|
| 东京电子 8035.T | ¥24,000 | ¥2,400,000 | **买不起** | N/A |

**结论**：screener 的 `max_cost_per_lot=150,000 JPY` 是资金管理的第一道防线。

### 6.2 订单生成流程（make_decision.py）

```
1. 读取 ss7 输出的 target_weights.csv
2. 查询当前持仓（SQLite positions 表）
3. 估算当前 NAV（现金 + 持仓市值）
4. 检查最大回撤门控（可选）
5. 计算每只股票的目标股数 = ⌊(NAV × weight / price)⌋ 向下取整到手
6. diff = target_qty - current_qty
7. 过滤微小交易（< 5,000 JPY）
8. 输出 orders_proposal.csv，SELL 优先排序
```

---

## 7. Learning 机制

> **设计来源**：本节 learning 架构源自 Nexus AI Orchestrator 的学习闭环设计
> （`docs/01_design/learning/`），并做了**量化领域自适应（domain adaptation）**。
>
> **核心洞察**：Nexus 的 trace → feedback → guardrail → mem/rule 架构，
> 与量化因子研究的 signal → realized_IC → quality_gate → factor_registry 在数学上同构，
> 都是**带质量守卫的贝叶斯先验更新系统**。
>
> **关键差异**：量化版的反馈信号是**客观数值**（IC 可精确计算），
> 无需人类主观判断，可以完全自动化。

### 7.1 系统同构性：Nexus → Quant 的映射

| Nexus 概念 | 量化对应 | 统计学本质 |
|-----------|---------|-----------|
| `traces`（对话记录）| `factor_signals` 表（因子值+预测）| 观测样本 |
| `user 👍/👎`（主观）| `realized_IC`（客观，corr(预测,实际)）| 似然函数 |
| `mem_items`（正向记忆）| `factor_registry`（高IC因子增权）| 正向先验更新 |
| `rules`（负向约束）| `risk_rules`（低IC因子降权/排除）| 负向约束注入 |
| `quality_score ≥ 0.6` | `t-stat ≥ 1.5 且 ICIR ≥ 0.3` | 显著性检验替代主观评分 |
| 指纹去重 SHA256 | 因子相关性检测（\|corr\| > 0.7 → 视为重复）| 防止多重共线性 / 样本重复 |
| `decay half_life=30d` | EWMA(IC, halflife=60交易日) | 非平稳系统的遗忘机制 |
| `shadow_mode` | `paper_trading_mode` | A/B 测试 / 纸面交易验证 |
| 规则冲突管理 | 因子正交化（残差化去除已有因子成分）| PCA / Gram-Schmidt |
| `learning_events` 审计表 | `learning_audit` 表 | 可复现性 / 可回滚性 |

### 7.2 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Quant Learning Loop                           │
│                                                                  │
│  ss7 信号生成                                                    │
│    factor_values + predicted_IC                                  │
│       │  写入 factor_signals 表 (trace 类比)                     │
│       ▼                                                          │
│  实盘执行 (H=20天后)                                             │
│    realized_return = 实际持仓期收益率                            │
│       │  post_trade.py 自动计算                                  │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────┐        │
│  │          Learning Guardrails（质量守卫）             │        │
│  │                                                     │        │
│  │  Ingress Guard:  t-stat ≥ 1.5, 样本量 ≥ 20        │        │
│  │  Memory Guard:   ICIR ≥ 0.3, |corr_existing| < 0.7│        │
│  │  Rule Guard:     连续N期负IC → 生成排除规则          │        │
│  │  Observability:  accepted/rejected/conflict 计数   │        │
│  └──────────────┬────────────────┬────────────────────┘        │
│                 │ 通过            │ 拒绝                         │
│                 ▼                ▼                               │
│  factor_registry 更新    learning_audit 记录（含拒绝原因）        │
│  (EWMA权重调整)                                                  │
│       │                                                          │
│       ▼                                                          │
│  Decay Cron (每日凌晨)                                           │
│    EWMA衰减、过期规则归档、超上限清理                             │
│       │                                                          │
│       ▼                                                          │
│  ss7 下次运行时读取 factor_registry → IC加权因子合成             │
│                                                                  │
│  Shadow Mode: 更新权重但不执行订单（paper trading验证期）         │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 数据库扩展（SQLite）

在现有 `japan_market.db` 中新增以下表：

**A. `factor_signals`（对应 Nexus `traces`）**
```sql
CREATE TABLE factor_signals (
    signal_id    TEXT PRIMARY KEY,   -- sha256(date|ticker|factor)
    date         TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    factor_name  TEXT NOT NULL,
    factor_value REAL,               -- z-score 标准化后的因子值
    pred_return  REAL,               -- ss7 对该ticker的预测收益
    created_at   TEXT DEFAULT (datetime('now'))
);
```

**B. `factor_registry`（对应 Nexus `mem_items`，正向记忆）**
```sql
CREATE TABLE factor_registry (
    factor_name   TEXT PRIMARY KEY,
    ewma_ic       REAL DEFAULT 0,    -- 指数加权 IC（主权重信号）
    icir          REAL DEFAULT 0,    -- IC信息比 = mean(IC)/std(IC)
    t_stat        REAL DEFAULT 0,    -- IC序列的t统计量
    obs_count     INT DEFAULT 0,     -- 累计观测期数
    weight        REAL DEFAULT 1.0,  -- 当前在合成分数中的权重
    status        TEXT DEFAULT 'active',  -- active/suppressed/archived
    last_hit_at   TEXT,
    hit_count     INT DEFAULT 0,
    updated_at    TEXT DEFAULT (datetime('now'))
);
```

**C. `risk_rules`（对应 Nexus `rules`，负向约束）**
```sql
CREATE TABLE risk_rules (
    rule_id      TEXT PRIMARY KEY,
    rule_type    TEXT,    -- 'factor_exclusion'/'sector_cap'/'regime_filter'
    condition    TEXT,    -- e.g. "factor=ret1 AND regime=high_vol"
    action       TEXT,    -- e.g. "max_weight=0 OR scale=0.5"
    priority     INT DEFAULT 50,
    confidence   REAL DEFAULT 0.5,
    status       TEXT DEFAULT 'active',
    conflict_with TEXT,
    fingerprint  TEXT UNIQUE,
    last_hit_at  TEXT,
    hit_count    INT DEFAULT 0,
    expires_at   TEXT,   -- 规则有效期，NULL=永久
    created_at   TEXT DEFAULT (datetime('now'))
);
```

**D. `learning_audit`（对应 Nexus `learning_events`，审计日志）**
```sql
CREATE TABLE learning_audit (
    event_id    TEXT PRIMARY KEY,
    event_type  TEXT,  -- 'mem_accept'/'mem_reject'/'rule_accept'/'rule_conflict'
    factor_name TEXT,
    realized_ic REAL,
    t_stat      REAL,
    reason      TEXT,  -- 通过/拒绝原因
    shadow_mode INT DEFAULT 0,
    created_at  TEXT DEFAULT (datetime('now'))
);
```

### 7.4 反馈计算：realized IC（客观信号）

```python
# post_trade.py 中，每个调仓周期结束后执行
def compute_factor_ic(db_path: str, rebalance_date: str, H: int = 20) -> dict:
    """
    对 H 天前记录的因子值，与实际 H 期收益率计算 Spearman IC。
    Spearman（秩相关）比 Pearson 对极值更鲁棒，是量化因子研究的标准选择。
    """
    signal_date = trading_days_ago(rebalance_date, H)  # H 个交易日前

    # 读取 H 天前的信号
    signals = load_factor_signals(db_path, signal_date)

    # 读取实际 H 期收益
    realized = load_realized_returns(db_path, signal_date, H)

    ic_by_factor = {}
    for factor in signals.columns:
        valid = signals[factor].dropna().index.intersection(realized.dropna().index)
        if len(valid) < 10:  # 样本量不足，跳过
            continue
        ic = spearmanr(signals.loc[valid, factor], realized.loc[valid]).correlation
        ic_by_factor[factor] = ic

    return ic_by_factor
```

### 7.5 Learning Guardrails（质量守卫）

对应 Nexus 的三层守卫，量化版本：

**Ingress Guard（入口守卫）**
```python
def ingress_guard(ic_value: float, sample_size: int) -> tuple[bool, str]:
    # IC 的标准误 ≈ 1/√N，t-stat = IC / SE
    se = 1.0 / math.sqrt(max(sample_size, 1))
    t_stat = ic_value / se

    if sample_size < 20:
        return False, f"样本量不足({sample_size}<20)"
    if abs(t_stat) < 1.5:
        return False, f"IC统计不显著(t={t_stat:.2f}<1.5)"
    return True, "passed"
```

**Memory Guard（记忆守卫）：防止因子冗余**
```python
def memory_guard(factor_name: str, ic: float, registry: pd.DataFrame) -> tuple[bool, str]:
    # 检查是否与已有高权重因子高度相关（因子共线性）
    existing_factors = registry[registry['status'] == 'active']
    for _, row in existing_factors.iterrows():
        hist_corr = compute_factor_correlation(factor_name, row['factor_name'])
        if abs(hist_corr) > 0.7:
            return False, f"与{row['factor_name']}高度相关(corr={hist_corr:.2f})"

    # ICIR 门槛（对应 quality_score ≥ 0.6）
    icir = compute_icir(factor_name)
    if icir < 0.3:
        return False, f"ICIR不足({icir:.2f}<0.3)"
    return True, "passed"
```

**Rule Guard（规则守卫）：冲突管理**
```python
def rule_guard(new_rule: dict, existing_rules: list) -> tuple[bool, str]:
    # 同域规则冲突检测（对应 Nexus 的 conflict_with）
    for rule in existing_rules:
        if rule['rule_type'] == new_rule['rule_type'] and \
           rule['condition'] == new_rule['condition']:
            if new_rule['confidence'] > rule['confidence']:
                suppress_rule(rule['rule_id'])  # 高置信新规则取代旧规则
                return True, f"suppressed_old:{rule['rule_id']}"
            else:
                return False, f"低优先级冲突(existing={rule['rule_id']})"
    return True, "no_conflict"
```

### 7.6 EWMA 权重更新（衰减机制）

对应 Nexus 的 `decay half_life = 30天`：

```python
# 每次新 IC 观测后，更新 factor_registry
def update_factor_weight(factor_name: str, new_ic: float, half_life_days: int = 60):
    """
    指数加权移动平均（EWMA）更新因子权重。
    half_life = 60 交易日 → α = 1 - exp(-ln2/60) ≈ 0.0114
    即约 60 个交易日前的 IC 贡献权重减半。
    """
    alpha = 1 - math.exp(-math.log(2) / half_life_days)

    registry = load_factor_registry(factor_name)
    old_ic = registry['ewma_ic']

    # EWMA 更新
    new_ewma_ic = alpha * new_ic + (1 - alpha) * old_ic

    # 滚动 ICIR（使用最近 N 期 IC 序列）
    ic_history = load_ic_history(factor_name, lookback=60)
    new_icir = np.mean(ic_history) / (np.std(ic_history) + 1e-8)

    # 权重正比于 ICIR（负 ICIR 自动降至接近 0）
    new_weight = max(new_icir, 0.0)  # 负权重 floor 到 0

    save_factor_registry(factor_name, ewma_ic=new_ewma_ic, icir=new_icir, weight=new_weight)
```

**Decay Cron（每日凌晨，对应 Nexus 的 `0 3 * * *`）**：
- 长期未命中因子（90 天无 IC 更新）→ `status=archived`
- `risk_rules` 过期归档（`expires_at < now`）
- 项目内 `active` 因子超上限（>50）时，清理最低 ICIR 的条目

### 7.7 IC 加权因子合成（替代 Ridge μ）

```python
# 在 ss7 的 make_features 之后，替代 PanelRidge.predict() 的 μ 输出
def ic_weighted_composite(
    feats_today: pd.DataFrame,   # 当日各因子值 (ticker × factor)
    registry: pd.DataFrame       # factor_registry 中 active 因子的权重
) -> pd.Series:
    """
    IC加权合成分数：比 Ridge 更稳健，更可解释。

    composite_score[ticker] = Σᵢ (weight_i × zscore(factor_i[ticker]))

    其中 weight_i = ICIR_i（信息比，自动反映因子近期稳定性）
    """
    active = registry[registry['status'] == 'active']
    scores = pd.Series(0.0, index=feats_today.index)

    for _, row in active.iterrows():
        fname = row['factor_name']
        w = float(row['weight'])       # = ICIR，负IC因子权重=0
        if fname not in feats_today.columns or w <= 0:
            continue
        # z-score 标准化（截面内标准化，消除量纲差异）
        f = feats_today[fname]
        f_z = (f - f.mean()) / (f.std() + 1e-8)
        scores += w * f_z

    return scores  # 作为 μ 输入 solve_long_only_meanvar()
```

**切换策略**：先并行运行 Ridge 预测和 IC 合成，对比两者的近期样本外 IC，
当 IC 合成持续优于 Ridge 超过 60 天时，才切换为主信号（对应 `shadow_mode` → 生产）。

### 7.8 新闻门控参数自学习

```python
# 新闻门控触发后，记录事后收益用于参数反馈
def record_news_gate_outcome(
    ticker: str, gate_date: str, gate_value: float,
    realized_5d: float, realized_20d: float
):
    """
    评估新闻门控是否有效避险：
    - 门控触发（gate < 0.5）后，实际收益为负 → 门控有效
    - 门控触发后，实际收益为正 → 门控过于保守（误杀）

    基于历史门控效果，自动调整 NewsConfig 参数：
    - 误杀率高 → 放宽 A_max 或 U_high
    - 漏杀率高 → 收紧 A_max 或 U_high
    """
    # 写入 learning_audit 表，供后续统计分析
    ...

# 定期（每季度）运行参数搜索
def tune_news_config(audit_history: pd.DataFrame) -> NewsConfig:
    # 网格搜索：最大化门控有效率（避险成功 / 总触发次数）
    ...
```

### 7.9 Shadow Mode 灰度部署

对应 Nexus 的 `LEARNING_SHADOW_MODE=true`：

```yaml
# config.yaml 新增
learning:
  enabled: true
  shadow_mode: true          # 灰度：计算但不更新 factor_registry
  min_icir: 0.3              # 对应 LEARNING_MIN_MEM_SCORE=0.6
  min_t_stat: 1.5
  factor_max_active: 50      # 对应 LEARNING_MEM_MAX_PER_PROJECT=500
  decay_half_life_days: 60   # 对应 LEARNING_DECAY_HALF_LIFE_DAYS=30（交易日x2）
  ic_correlation_dedup: 0.7  # 对应指纹去重
```

**灰度流程**（与 Nexus 完全对应）：
1. `shadow_mode: true`：系统计算 IC、运行守卫逻辑、记录 `learning_audit`，但不修改 `factor_registry`
2. 观察 2-3 个月：核查 accepted/rejected 比例、守卫误杀率
3. 确认无异常后：`shadow_mode: false`，开始真正更新权重
4. 回滚：`enabled: false` 即时恢复原始 Ridge 预测，不丢失历史数据

### 7.10 与 worker.py 的集成点

`worker.py` 已有 `quant.compute_news_risk_factor` 和 `quant.deep_analysis` 工具，
learning 模块的触发可以通过 Redis stream 任务接入：

```python
# worker.py 中新增工具（建议）
"quant.learning_cycle": handle_learning_cycle,
# 每调仓周期结束后，通过 stream:task 触发：
# 1. compute realized IC for all factors
# 2. run guardrails
# 3. update factor_registry (if not shadow_mode)
# 4. generate learning health report → Obsidian

"quant.factor_health_report": handle_factor_health_report,
# 定期（每月）报告因子ICIR趋势、规则命中率、learning_audit统计
```

### 7.11 实现路线

| 阶段 | 内容 | 预计工作量 |
|------|------|-----------|
| **M1** | 数据库建表（factor_signals、learning_audit）+ post_trade.py 写入原始 IC | 1天 |
| **M2** | Ingress Guard（t-stat检验）+ learning_audit 记录 + shadow_mode | 1-2天 |
| **M3** | Memory Guard（ICIR、去重）+ factor_registry EWMA更新 + Decay cron | 1-2天 |
| **M4** | Rule Guard + risk_rules 冲突管理 + factor_registry → ss7 集成 | 2天 |
| **M5** | IC加权合成替代Ridge μ（并行对比60天后切换）| 1周 |
| **M6** | 新闻门控参数自学习 + worker.py 集成 | 1周 |

---

## 8. 运营流水线

### 8.1 每日流程（15:30 JST 收盘后）

```bash
# daily_run.py 或 Windows Task Scheduler 触发
python run_pipeline.py --config config.yaml

# 流程：
# 1) db_update.py         → 更新约100只股票价格（增量）
# 2) screener.py          → 筛选候选股（含1手成本过滤）
# 3) ss7_sqlite_news_overlay.py → 走前向回测，输出 target_weights.csv
# 4) report_obsidian.py   → 发布报告到 Obsidian vault
```

### 8.2 决策日（每 20 个交易日，约每月一次）

```bash
# 手动执行（或 make_decision 嵌入 daily_run）
python make_decision.py \
  --db japan_market.db \
  --cash 1000000 \
  --lot 100 \
  --peak_nav <历史最高NAV>

# 输出：artifacts/decision/YYYY-MM-DD/run_id/
#   orders_proposal.csv    ← 发给 SBI 证券执行
#   decision_snapshot.json ← 完整审计记录
```

### 8.3 成交录入（执行后）

```bash
# 从 SBI 证券导出成交单 → 填写 fills_YYYY-MM-DD.csv
python import_fills.py --fills fills_2026-03-10.csv --db japan_market.db
python build_positions.py --db japan_market.db
python build_account_snapshot.py --db japan_market.db
python post_trade.py  # 盘后处理 + （未来）IC 计算
```

### 8.4 新闻系统接入（半自动）

```bash
# worker.py 提供的工具（通过 Redis 任务触发）：
# - news.preclose_brief_jp  : 15:35 JST 盘前简报（日本市场）
# - news.tdnet_close_flash  : TDnet 收盘后公告解读
# - news.daily_report       : 日报汇总

# 待实现：自动将 worker.py 新闻工具输出格式化为 news.csv
# 接入 ss7：SS6_NEWS_ON=1 SS6_NEWS_CSV=./news_today.csv
```

---

## 9. 资金规模约束与现实预期

### 9.1 100 万日元的结构性限制

| 维度 | 问题 | 解决方案 |
|------|------|---------|
| 分散度 | 1手成本高 → 只能持3-5只 | screener 过滤 < 15万/手 |
| 交易成本 | 手续费占比高（小单 bps 更高）| 降低换手率（gamma=10），每月调仓 |
| 市场冲击 | 小市值股票流动性差 | ADV > 2000万，impact_k=0.5 |
| 基准超越 | 大盘蓝筹 β≈1，很难 α | 选低价多元化股票，含中型股 |

### 9.2 收益预期（现实校准）

| 情景 | 年化收益 | 说明 |
|------|---------|------|
| 被动持有 1321.T ETF | 15-20% | 2023/2024 日本牛市背景 |
| 60% ETF + 40% 本策略 | 16-22% | 降低主动选股风险 |
| 本策略优化后（乐观） | 18-25% | 需要 Sharpe > 1.0 |
| 本策略当前状态 | 待回测验证 | 资金参数已修复，可开始实测 |

**真实建议**：在正式全仓运行前，先用 3 个月**纸面交易**验证信号 → 成交 → 报告全流程，观察实盘与回测的偏差。

---

## 10. 改进路线图

### P0 — 已完成（2026-03-10）

- [x] 修复 `initial_capital=1M`、`lot_size=100`
- [x] 启用 `slippage_bps=5.0`、`fee_bps=5.0`、`impact_k=0.5`
- [x] screener 加入 1手成本硬过滤（< 15万 JPY）
- [x] 股票宇宙扩展至约 100 只 TSE Prime 股票
- [x] ss7 加入单股止损（8%）
- [x] ss7 加入组合最大回撤控制（12%/18%）
- [x] make_decision.py 加入实盘回撤门控
- [x] 删除旧版本代码（Project/、models/、searching/、*.zip）

### P1 — 下一阶段（数据验证 + Learning M1/M2）

- [ ] **回测验证**：在 2020-2024 历史数据上对比修复前/后净值曲线，与 1321.T 基准对比
- [ ] **纸面交易**：运行 2-3 个月，对比信号与实际可执行价格的偏差
- [ ] **行业集中度约束**：同一行业不超过 35%（修改组合优化器约束）
- [ ] **新闻自动接入**：将 `worker.py` 的 `news.preclose_brief_jp` 输出格式化为 `news.csv`，接入 ss7 门控
- [ ] **Learning M1**：建表（factor_signals、learning_audit）+ post_trade.py 写入原始 IC
- [ ] **Learning M2**：Ingress Guard（t-stat 检验）+ shadow_mode + 审计日志

### P2 — 中期优化（因子扩展 + Learning M3/M4）

- [ ] **基本面因子**：接入 J-Quants API 补充 PBR、ROE、股息率、盈利修正因子
  ```python
  factor_pbr    = 1 / PBR                    # 价值（Fama-French，日本有效）
  factor_roe    = ROE                         # 盈利质量（Novy-Marx）
  factor_div    = 年度股息 / 股价              # 日本高股息溢价
  factor_relret = ret20 - industry_avg_ret20  # 跨品种相对动量
  factor_vol_z  = volume_z_score_20           # 成交量异常（大量+上涨=强信号）
  ```
- [ ] **Learning M3**：Memory Guard（ICIR去重）+ factor_registry EWMA更新 + Decay cron
- [ ] **Learning M4**：Rule Guard + risk_rules 冲突管理 + factor_registry → ss7 集成
- [ ] **LightGBM 对比实验**：与 Ridge 并行运行，对比样本外 IC，择优

### P3 — 长期愿景（Learning 闭环完整 + 高级优化）

- [ ] **Learning M5**：IC 加权因子合成替代 Ridge μ（shadow 60天验证后切换）
- [ ] **Learning M6**：新闻门控参数自学习（基于历史门控效果反馈调整 NewsConfig）
- [ ] **worker.py 集成**：`quant.learning_cycle` + `quant.factor_health_report` 工具
- [ ] **Beta 中性化**：优化器加入 `β_target ≈ 1.0` 约束，避免对日经方向过度押注
- [ ] **波动率目标化**：组合年化波动率控制在 12-18% 区间（动态调整仓位规模）

---

## 附录：关键环境变量

ss7 全部参数可通过环境变量覆盖（`run_pipeline.py` 自动传递 config.yaml 值）：

```bash
SS6_INITIAL_CAPITAL=1000000
SS6_LOT_SIZE_DEFAULT=100
SS6_FEE_BPS=5.0
SS6_SLIPPAGE_BPS=5.0
SS6_IMPACT_K=0.5
SS6_STOP_LOSS_PCT=0.08
SS6_MAX_DD_HALF=0.12
SS6_MAX_DD_FULL=0.18
SS6_NEWS_ON=1           # 启用新闻门控
SS6_NEWS_CSV=news.csv   # 新闻数据文件路径
SS6_NEWS_HALF_LIFE_DAYS=3
SS6_DB_PATH=japan_market.db
SS6_TICKERS=9432.T,9433.T,...   # 由 screener 输出自动填充
SS6_BENCHMARK=1321.T
SS6_START=2020-01-01
SS6_H=20
SS6_REBALANCE_EVERY=20
SS6_OUTPUT_DIR=reports
```
