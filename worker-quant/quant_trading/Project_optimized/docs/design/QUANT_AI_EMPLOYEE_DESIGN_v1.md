# Worker-Quant — 专属量化AI员工 设计文档 v1.0

**作者**: PM + Quant Architect
**创建日期**: 2026-03-22
**状态**: DRAFT — 待评审
**对应任务清单**: `QUANT_AI_EMPLOYEE_TASKS_v1.md`

---

## 0. 愿景声明

> 将 worker-quant 打造成一名精通统计学与数学的资深量化交易员 AI 员工：
> 能够实时感知国际市场新闻、主动复盘持仓状况、基于严谨的统计模型提出可执行的操作建议，
> 并在每个交易日的关键时间节点主动推送情报，形成"发现→分析→建仓→持仓监控→退出"的完整闭环。

---

## 1. 现状差距分析（Gap Analysis）

### 1.1 已实现（骨架层，~25% 完成度）

| 模块 | 已有能力 | 主要工具 |
|---|---|---|
| 价格/行情 | yfinance实时报价，历史数据1年 | `quant.fetch_price`, `_fetch_quote_facts` |
| 技术指标 | RSI14, SMA20/60, Z20, ATR, 年化波动率, 5/20/60日收益率 | `_compute_quant_metrics` |
| 信号生成 | 规则打分系统 → Overweight/Neutral/Underweight | `_compute_quant_metrics` |
| 执行建议 | 三档限价(aggressive/balanced/patient), lot sizing | `_calculate_limit_prices` |
| 新闻获取 | GDELT, Google RSS, Yahoo Japan页面, yfinance news, OpenBB | `_merge_recent_news` |
| 新闻情绪 | Sentiment + Risk_Z因子 | `compute_news_risk_factor` |
| 持仓记录 | SQLite fills表, account_state表 | `portfolio_record_fill` |
| 持仓读取 | 从fills计算净持仓 | `_get_current_positions_from_fills` |
| 发现工作流 | 多标的筛选, 资金约束, 学习机制 | `discovery_workflow` |
| 报告生成 | HTML报告, PNG摘要卡, MinIO存储 | `news_daily_report` |
| 盘前/盘后 | preclose_brief_jp, tdnet_close_flash | 已有但数据质量受限 |

### 1.2 核心缺口（未实现）

**A. 统计/数学深度不足**
- 信号是规则加法，无统计显著性检验（无IC/ICIR）
- 无协方差矩阵（组合相关性盲区）
- 无组合优化（MVO/风险平价）
- 无回测框架（历史信号有效性未验证）
- 无VaR/CVaR（尾部风险完全未量化）

**B. 新闻数据源质量低**
- GDELT延迟1-2天，对隔日交易意义有限
- TDnet适时开示（日本最强alpha源：増配、自社株買い、業績修正）未接入
- 新闻情绪分数无历史校准，预测力未知
- 新闻与持仓完全脱钩

**C. 持仓管理能力缺失**
- 无浮动盈亏计算（成本价 vs 当前价）
- 无止损/止盈线跟踪
- 无组合层面P&L归因
- 无仓位调整建议（加仓/减仓/止损触发）

**D. 无主动决策闭环**
- 各工具孤立，无法回答："我持有X，今天发生了Y，我应该怎么办？"
- 无定时主动推送（依赖用户手动触发）
- 无跨交易日的学习与偏好记忆

---

## 2. 目标架构

### 2.1 系统层次图

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 0: 触发层                           │
│  Discord指令 | Cron定时 | 价格预警 | 新闻事件               │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Layer 1: 决策层 (新增)                    │
│  portfolio.morning_brief  — 早间综合情报                     │
│  portfolio.position_review — 持仓主动复盘                   │
│  portfolio.rebalance_suggest — 仓位再平衡建议               │
│  quant.event_alert — 新闻事件对持仓的影响评估               │
└───────┬──────────────────────────┬──────────────────────────┘
        │                          │
┌───────▼─────────┐    ┌──────────▼───────────────────────────┐
│  Layer 2: 分析层 │    │         Layer 2: 信号层              │
│  (已有+增强)     │    │         (需重构)                     │
│  deep_analysis  │    │  quant.factor_score  — 多因子打分    │
│  news_risk      │    │  quant.signal_backtest — 信号回测    │
│  tdnet_flash    │    │  quant.portfolio_risk — 组合风险     │
└───────┬─────────┘    └──────────┬───────────────────────────┘
        │                          │
┌───────▼──────────────────────────▼──────────────────────────┐
│                    Layer 3: 数据层                           │
│  yfinance(价格) | TDnet RSS(公告) | Google RSS(新闻)        │
│  japan_market.db(fills/account) | Redis(任务队列)           │
│  MinIO(报告存储) | LLM(Dashscope/Qwen)                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心数据流：每日决策闭环

```
08:30 JST  TDnet昨日公告抓取
    ↓
08:45 JST  morning_brief: 宏观新闻 + 公告 + 当前持仓状态
    ↓
09:00 JST  市场开盘
    ↓
实时       event_alert: 持仓相关新闻触发 → 影响评估
    ↓
12:00 JST  持仓浮动盈亏快报（午间）
    ↓
15:15 JST  preclose_brief: 盘尾推演 + 收盘前可执行清单
    ↓
15:35 JST  post_close: 当日复盘 + 明日计划
    ↓
用户执行   record_fill → 更新持仓
    ↓
循环
```

---

## 3. 模块详细设计

### 3.1 模块 A: 统计信号引擎重构

**目标**: 将规则打分替换为统计验证的多因子模型

#### A1. 因子定义（第一期）

| 因子 | 计算方式 | 类别 |
|---|---|---|
| `momentum_20` | 20日超额收益（相对N225） | 动量 |
| `momentum_60` | 60日超额收益 | 动量 |
| `reversal_5` | 5日收益率（短期反转） | 反转 |
| `rsi_norm` | (RSI14 - 50) / 50，归一化 | 技术 |
| `vol_regime` | 20日波动率 vs 60日均值（高波/低波） | 风险 |
| `z_score` | (Price - SMA20) / std20 | 均值回归 |
| `rel_strength` | 20日超额收益分位数（全watchlist内） | 横截面 |

#### A2. IC（信息系数）追踪

每次信号发出后，记录到 `signal_log` 表：
```sql
CREATE TABLE signal_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asof        TEXT NOT NULL,          -- 信号日期
    symbol      TEXT NOT NULL,
    signal      TEXT NOT NULL,          -- Overweight/Neutral/Underweight
    alpha_score REAL,
    price_at_signal REAL,
    price_5d    REAL,                   -- 5日后复权价（填充）
    price_20d   REAL,                   -- 20日后复权价（填充）
    ret_5d      REAL,                   -- 实际收益（信号发出后5日）
    ret_20d     REAL,                   -- 实际收益（信号发出后20日）
    ic_contrib  REAL,                   -- 本条信号IC贡献
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

每周运行一次 `quant.signal_backfill` 填充 `ret_5d / ret_20d`，计算滚动IC：

```
IC = correlation(alpha_score, ret_20d)  -- 过去N条信号
ICIR = IC.mean() / IC.std()             -- 信息比率
```

**准入门槛**: `|IC| > 0.05` 且 `|ICIR| > 0.3` 才认为信号有效

#### A3. 信号合成（加权，而非加法）

```python
# 替换现有的 score += 1.0 if ret_20d > 0
composite_score = (
    w_momentum * normalize(momentum_20) +
    w_reversal * normalize(reversal_5) +
    w_rsi      * normalize(rsi_norm) +
    w_vol      * normalize(vol_regime) +
    w_cross    * normalize(rel_strength)
)
# 权重初始均等，后续由IC反馈动态调整
```

#### A4. 组合风险度量

新增 `quant.portfolio_risk` 工具：
```
输入: 当前持仓列表
输出:
  - 持仓相关矩阵（pairwise correlation，60日日收益）
  - 组合年化波动率（加权）
  - 单仓集中度告警（单仓 > 40% 预警）
  - 概念重叠度（同行业/同板块持仓比例）
```

---

### 3.2 模块 B: 新闻数据源升级

**目标**: 从"有新闻"升级到"有alpha价值的新闻"

#### B1. TDnet 适时开示接入（最高优先级）

TDnet RSS端点（免费，无需API Key）：
```
https://www.release.tdnet.info/inbs/I_main_00.html  (HTML页面)
RSS镜像: 通过日経/Bloomberg等二次聚合获取
```

公告分类与alpha优先级：

| 公告类型 | 关键词 | Alpha强度 | 典型影响 |
|---|---|---|---|
| 業績修正（上方） | 業績予想の修正, 上方修正 | ★★★★★ | +3~8%, 1-3日 |
| 増配 / 特別配当 | 配当予想の修正, 増配 | ★★★★☆ | +2~5%, 当日 |
| 自社株買い | 自己株式の取得 | ★★★★☆ | +1~4%, 当日 |
| 業績修正（下方） | 下方修正 | ★★★★★ | -5~15%, 당일 |
| 決算発表 | 決算短信, 業績 | ★★★☆☆ | ±5%, 当日 |
| M&A / 業務提携 | 子会社, 資本業務提携 | ★★★☆☆ | ±3~10% |
| 行政処分 / 不祥事 | 行政処分, 不正 | ★★★★☆ | -5~20% |

实现：`tdnet_announcement_fetch(symbols: list) -> list[dict]`

#### B2. 新闻与持仓的绑定查询

新增索引结构：
```python
# 每条公告绑定到watchlist中的ticker
def _match_announcement_to_ticker(announcement: dict, watchlist: dict) -> list[str]:
    # 1. 証券コード（4桁）直接匹配
    # 2. 企業名称 alias 匹配（包含正式名、略称、英文名）
    # 3. 子会社名称 → 母公司ticker映射
```

#### B3. 新闻情绪量化校准

建立情绪标签数据集（手动标注50-100条历史公告）：
```
公告 → 实际3日收益 → 反推情绪基准
用于校准 compute_news_risk_factor 的 Sentiment 数值含义
```

---

### 3.3 模块 C: 持仓管理与复盘

**目标**: 从"记录工具"升级到"持仓管家"

#### C1. 数据库扩展

新增 `positions_snapshot` 表（每日定时快照）：
```sql
CREATE TABLE positions_snapshot (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asof        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    net_qty     REAL NOT NULL,
    avg_cost    REAL,          -- 加权平均成本
    current_price REAL,
    unrealized_pnl REAL,       -- (current_price - avg_cost) * net_qty
    unrealized_pnl_pct REAL,   -- unrealized_pnl / (avg_cost * net_qty)
    stop_loss   REAL,          -- 止损价（avg_cost * (1 - stop_pct)）
    take_profit REAL,          -- 止盈价
    days_held   INTEGER,       -- 持仓天数
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

成本价计算（加权平均，FIFO可选）：
```python
def _calc_avg_cost(symbol: str) -> float:
    # SELECT side, qty, price FROM fills WHERE symbol=?
    # BUY时累计成本，SELL时按FIFO或加权平均减少
```

#### C2. 新工具：`portfolio.position_review`

```
触发: 手动 / 每日12:00 / 15:35 定时
输入: 无（自动读取）
输出:
  持仓汇总表:
    symbol | 数量 | 成本价 | 现价 | 浮盈% | 持仓天数 | 止损价 | 信号
  组合统计:
    总市值 | 总浮盈 | 总浮盈% | 可用现金 | 仓位使用率
  风险提示:
    - 触发止损线的仓位（止损价 = 成本价 × 92%，可配置）
    - 持仓超过目标天数的仓位
    - 信号已转为 Underweight 的仓位
  操作建议:
    - 每个仓位的建议（持有/减仓/止损/加仓）
```

#### C3. 新工具：`quant.event_alert`

```
触发: 新闻摘取后自动调用（事件驱动）
逻辑:
  1. 取最新TDnet公告 + 重要新闻
  2. 与当前持仓交叉比对
  3. 对命中持仓生成影响评估
输出示例:
  ⚠️ 5020.T 持仓预警
  事件: 原油价格大幅下跌（Brent -8%）
  影响评估: 能源板块利空，ENEOS 历史敏感系数 β_oil=0.62
  建议: 关注开盘价，若低于止损线1,300 考虑止损
  当前浮盈: +1.3%（成本1,350），止损距离: -4.9%
```

---

### 3.4 模块 D: 主动推送体系

**目标**: 从"被动查询"升级到"主动情报员"

#### D1. 每日推送节点

| 时间（JST） | 工具 | 内容 |
|---|---|---|
| 08:30 | `news.tdnet_announcement_fetch` | 昨日TDnet公告抓取 |
| 08:45 | `portfolio.morning_brief` | 宏观摘要+公告+持仓状态 |
| 12:00 | `portfolio.midday_pnl` | 午间浮动盈亏快报 |
| 15:15 | `news.preclose_brief_jp` | 盘尾推演+可执行清单 |
| 15:35 | `portfolio.post_close` | 当日复盘+明日计划 |

#### D2. `portfolio.morning_brief` 设计

```
输出结构:
  【今日市场环境】
    昨日N225: ±X%  | Topix: ±X%  | VIX: XX
    期货: N225先物 XX (+/-X%)
    宏观: [TOP3宏观新闻，1行/条]

  【TDnet昨日公告（持仓相关）】
    ✅ 5020.T: 自社株買い発表（上限50億円）→ 预期催化剂
    ❌ 无持仓相关公告

  【持仓状态】
    5020.T  100股  成本1350  现价1367  +1.3%  ↑ 信号:Overweight
    现金: 263,250 / 400,000 JPY（65.8% 剩余）

  【今日关注】
    1. 9:00 开盘后观察5020.T成交量（公告后首日）
    2. 美联储本周讲话（周四），注意日元汇率波动
```

---

### 3.5 模块 E: 记忆与个性化

**目标**: 记住用户偏好，跨会话一致性

#### E1. 持久化配置文件

```json
// quant_trading/Project_optimized/user_profile.json
{
  "capital_base_jpy": 400000,
  "max_position_pct": 0.35,
  "stop_loss_pct": 0.08,
  "take_profit_pct": 0.20,
  "max_positions": 5,
  "horizon_days": 20,
  "risk_profile": "medium",
  "preferred_markets": ["JP"],
  "avoid_sectors": [],
  "preferred_sectors": ["energy", "telecom"],
  "notify_on_stop_loss": true,
  "notify_on_catalyst": true,
  "language": "zh"
}
```

#### E2. 学习记录扩展

现有 `discovery_learning_store` 扩展为多维度学习：
- 哪类信号历史IC最高（定期更新因子权重）
- 哪类公告对哪类板块影响最显著（事件研究积累）
- 用户实际采纳了哪些建议（用户反馈闭环）

---

## 4. 非功能性要求

### 4.1 延迟要求

| 场景 | 目标延迟 |
|---|---|
| TDnet公告 → Discord推送 | < 5分钟 |
| deep_analysis 单股 | < 30秒 |
| portfolio.position_review | < 15秒 |
| morning_brief 完整报告 | < 60秒 |

### 4.2 数据质量

- 技术指标：使用复权价（adjusted close），防止除权日失真
- 新闻：严格时效性过滤，6小时内视为"盘前有效"
- 持仓：每次 record_fill 后立即重算快照，不依赖定时任务

### 4.3 降级策略

- TDnet不可达 → 降级到 Google RSS 日文关键词
- yfinance超时 → 使用上一次缓存价格 + 标注"价格可能延迟"
- LLM超时 → 返回纯数据摘要（无LLM叙事），不阻塞

---

## 5. 技术债务说明

现有代码中已知需要重构的部分：

| 位置 | 问题 | 优先级 |
|---|---|---|
| `_compute_quant_metrics` | 信号打分是规则加法，应改为加权合成 | P1 |
| `news_daily_report` | GDELT为主数据源，延迟大，应降级为fallback | P0 |
| `deep_analysis` | 不调用 `compute_news_risk_factor`，与新闻脱钩 | P1 |
| `tdnet_close_flash` | 名为TDnet但实际用GDELT，名实不符 | P1 |
| `portfolio_record_fill` | 无成本价计算，`account_state` 只扣现金不算盈亏 | P0 |
| `_load_watchlists` | alias map不完整，日文公司名无法匹配新闻 | P2 |

---

## 6. 成功指标（KPI）

| 指标 | 当前值 | 目标值 | 时间窗口 |
|---|---|---|---|
| 信号IC（20日） | 未测量 | \|IC\| > 0.05 | 3个月滚动 |
| 信号ICIR | 未测量 | > 0.3 | 3个月滚动 |
| TDnet公告覆盖率 | 0% | > 90%（watchlist内） | - |
| 持仓P&L计算准确率 | N/A | 100% | - |
| morning_brief准时率 | 0%（无定时） | > 95% | 每交易日 |
| 用户采纳建议后实际盈利率 | 未跟踪 | > 55%（胜率） | 6个月 |

---

## 7. 版本规划

### v1.1 — 持仓管理基础（预计2周）
- 成本价计算 + 浮动盈亏
- `portfolio.position_review` 工具
- 止损线跟踪

### v1.2 — 新闻数据源升级（预计1周）
- TDnet RSS 接入
- 公告分类 + ticker映射
- `quant.event_alert` 工具

### v1.3 — 主动推送（预计1周）
- Cron触发 morning_brief / preclose / post_close
- Discord推送集成

### v1.4 — 统计信号重构（预计3-4周）
- signal_log 表 + IC追踪
- 多因子加权合成
- `quant.portfolio_risk` 工具

### v2.0 — 完整量化员工（预计8周累计）
- 因子IC反馈循环（自动调权重）
- 组合优化（风险平价或简单MVO）
- 用户偏好持久化 + 学习记录完善

---

*文档版本: v1.0 | 2026-03-22 | 待 PM + Quant 联合评审后冻结*
