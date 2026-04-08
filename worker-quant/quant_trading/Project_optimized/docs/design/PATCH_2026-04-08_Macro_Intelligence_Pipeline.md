# PATCH: 宏观情报管线 (Macro Intelligence Pipeline)

**日期**: 2026-04-08
**触发事件**: 伊美停战 → 日经 +5%，系统完全未检测到原因
**目标**: 补全 L1 宏观事件盲区，让 regime 能响应离散事件跳变

---

## 一、问题审计

### 1.1 今日复盘 (2026-04-08)

```
实际因果链:
  伊美停战(巴基斯坦宣布) → WTI原油 -10% → 地缘风险溢价消退
  → CME日经期货 +4.25% → 东京开盘日经 +2800円(+5%)

系统看到了什么:
  ✓ NK期货 +4.25% (cross_asset_signals.py)
  ✗ 原油暴跌 (无数据源)
  ✗ 停战消息 (新闻关键词未覆盖)
  ✗ 事件分类 (无 L1/L2 框架)

系统的 regime 判断:
  cross_asset_score = 0.40 (中性) ← 错误！应该是强 risk-on
  原因: VIX +6.66% 和 USDJPY -0.80% 被当作利空，
        但它们是停战后的正常二阶反应，不应抵消 NK期货的强信号
```

### 1.2 根因分析

| 缺陷 | 描述 | 严重度 |
|------|------|--------|
| 无大宗商品数据 | 原油/黄金/铜完全空白 | **Critical** |
| 无宏观事件检测 | L1 事件（停战/央行/关税）无覆盖 | **Critical** |
| 跨资产信号无因果推理 | 4 个信号独立加权，互相矛盾时取平均 | High |
| regime 纯回顾性 | MA20/60 需要 5-10 天才能反映今天的事件 | High |
| 新闻只覆盖个股 L3 | 决算短信≠停战协议，但系统同等对待 | Medium |
| 无盘前决策窗口 | 8:00 已知 NK+4.25%，但要等 16:30 才处理 | Medium |

---

## 二、架构设计

### 2.1 三层架构

```
┌─────────────────────────────────────────────────┐
│ 层3: LLM 宏观分析 (Gemma 4 27B, 家里的机器)     │
│   仅在层2触发 L1/L2 时调用                       │
│   输入: 跨资产数据 + 新闻标题                     │
│   输出: 事件分类 + 因果链 + regime_boost JSON     │
│   故障时: 退化到层2，不影响生产                   │
└──────────────────────┬──────────────────────────┘
                       │ HTTP (Ollama API via Tailscale)
┌──────────────────────▼──────────────────────────┐
│ 层2: 规则引擎 (macro_event_detector.py, 本机)    │
│   输入: 扩展版跨资产数据 (7+ 品种)               │
│   规则: NK期货 gap > ±3%? 原油 > ±5%? etc.      │
│   输出: alert_level (L1/L2/L3/none)             │
│   无外部依赖，100% 可靠                          │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│ 层1: 扩展跨资产采集 (cross_asset_signals.py)     │
│   现有: SP500, USDJPY, VIX, NK期货              │
│   新增: WTI原油, 黄金, 铜, SOX半导体指数         │
│   数据源: yfinance (零成本)                      │
└─────────────────────────────────────────────────┘
```

### 2.2 与 quant 主链路的联动

```
morning_briefing.bat / daily_run.py

  Step 0:   cross_asset_signals.py    ← 扩展版 (加原油/黄金/铜/SOX)
  Step 0.5: macro_event_detector.py   ← 新增: 规则引擎
  Step 0.7: macro_digest.py           ← 新增: LLM 分析 (可选)
  
  ...existing pipeline...
  
  Step 3:   sprint_signal.py          ← 修改: 读取 macro_event
            generate_sprint_artifacts()
              └── regime_score_v2     ← 修改: + event_boost
              └── regime_final = clamp(regime_v2 + event_boost, 0, 1)
```

**关键集成点**: `benchmark_regime.py` 的 `compute_regime_score_v2()` 输出 regime_score 后，
在 `sprint_signal.py` 中叠加 event_boost，得到最终 regime。

### 2.3 事件驱动 regime 覆盖层

```python
# regime 事件覆盖公式
regime_final = clamp(regime_score_v2 + event_boost * decay_factor, 0.0, 1.0)

# decay_factor = max(0, 1 - days_since_event / duration_days)
# 例: 停战当天 boost=+0.30, 第2天 +0.20, 第3天 +0.10, 第4天消失

# event_boost 范围约束: [-0.30, +0.30]
# 防止事件覆盖完全压过技术面 regime
```

今日场景模拟:
```
regime_score_v2 = 0.214 (MA 弱势)
event_boost = +0.30 (停战 L1, day 0, decay=1.0)
regime_final = clamp(0.214 + 0.30, 0, 1) = 0.514
position_scale = scale(0.514) ≈ 0.66

vs 当前: regime = 0.214, scale = 0.116 (过度保守)
```

### 2.4 扩展跨资产品种

| 品种 | yfinance ticker | 影响机制 | 权重建议 |
|------|----------------|----------|----------|
| WTI原油 | CL=F | 进口成本→制造业利润 | 0.15 |
| 黄金 | GC=F | 避险情绪反向指标 | 0.05 |
| 铜 | HG=F | 全球制造业景气 | 0.05 |
| SOX半导体 | ^SOX | 日经权重股直接映射 | 0.10 |

新增权重来源：从现有品种等比缩减。

调整前: sp500=0.35, usdjpy=0.20, vix=0.20, nk_futures=0.25
调整后: sp500=0.20, usdjpy=0.15, vix=0.10, nk_futures=0.15,
        crude_oil=0.15, gold=0.05, copper=0.05, sox=0.10

> 注: 以上权重为初始值，需用历史数据跑回归确定最优权重。

### 2.5 规则引擎: 事件检测阈值

```python
MACRO_EVENT_RULES = {
    # 规则名: (字段, 方向, 阈值, 事件级别, 默认 boost)
    "nk_futures_surge":    ("nk_futures_gap_pct",    "abs", 3.0, "L1", 0.25),
    "nk_futures_moderate": ("nk_futures_gap_pct",    "abs", 1.5, "L2", 0.10),
    "oil_crash":           ("crude_oil_change_pct",  "<",  -5.0, "L1", 0.20),
    "oil_spike":           ("crude_oil_change_pct",  ">",   5.0, "L1",-0.20),
    "oil_moderate":        ("crude_oil_change_pct",  "abs", 3.0, "L2", 0.10),
    "vix_spike":           ("vix_change_pct",        ">",  15.0, "L1",-0.15),
    "sox_surge":           ("sox_change_pct",        ">",   3.0, "L2", 0.10),
    "sox_crash":           ("sox_change_pct",        "<",  -3.0, "L2",-0.10),
    "yen_shock":           ("usdjpy_change_pct",     "abs", 2.0, "L2", 0.10),
}
```

多规则同时触发时: `event_boost = clamp(sum(all_boosts), -0.30, +0.30)`

### 2.6 LLM 管线 (Gemma 4 27B)

**触发条件**: 层2 alert_level == L1 or L2

**输入 prompt**:
```
你是日本股市宏观分析师。以下是今日盘前市场数据:

CME日经期货: {nk_gap}% | WTI原油: {oil_chg}% | VIX: {vix_chg}%
USD/JPY: {uj_chg}% | S&P500: {sp_chg}% | SOX: {sox_chg}%
黄金: {gold_chg}% | 铜: {copper_chg}%

规则引擎判定: alert_level={level}, 触发规则: {rules}

相关新闻标题:
{headlines}

任务:
1. 判断最可能的事件原因（一句话，20字以内）
2. 事件类型: geopolitical / monetary_policy / trade_policy / commodity_shock / earnings_macro / other
3. 对日经的影响: positive / negative / mixed
4. regime_boost 建议: -0.30 到 +0.30 的浮点数
5. 影响持续天数: 1-5 的整数
6. 受益板块和受损板块

只输出 JSON:
{
  "event_summary": "",
  "event_type": "",
  "impact_direction": "",
  "regime_boost": 0.0,
  "duration_days": 3,
  "sectors_positive": [],
  "sectors_negative": [],
  "confidence": 0.0
}
```

**LLM 输出约束**:
- regime_boost 被 clamp 到 [-0.30, +0.30]
- confidence < 0.5 时, boost 减半
- LLM 超时 (30秒) 或返回非法 JSON → 退化到层2规则引擎的 boost 值

### 2.7 新闻标题采集

LLM 不联网，新闻标题由 `macro_digest.py` 预采集:

```python
# 方案1: Google News RSS (免费，延迟 30-60 分钟)
feeds = [
    "https://news.google.com/rss/search?q=日経平均&hl=ja",
    "https://news.google.com/rss/search?q=Japan+stock+market&hl=en",
    "https://news.google.com/rss/search?q=oil+price+crude&hl=en",
]

# 方案2: GDELT API (免费，延迟数小时，覆盖全球)
# 方案3: 日経電子版 RSS (需订阅)
```

采集后写入 `macro_headlines` 表，供 LLM 读取。

### 2.8 DB schema 新增

```sql
-- 扩展 cross_asset_snapshots 表（加列）
ALTER TABLE cross_asset_snapshots ADD COLUMN crude_oil REAL;
ALTER TABLE cross_asset_snapshots ADD COLUMN crude_oil_change_pct REAL;
ALTER TABLE cross_asset_snapshots ADD COLUMN gold REAL;
ALTER TABLE cross_asset_snapshots ADD COLUMN gold_change_pct REAL;
ALTER TABLE cross_asset_snapshots ADD COLUMN copper REAL;
ALTER TABLE cross_asset_snapshots ADD COLUMN copper_change_pct REAL;
ALTER TABLE cross_asset_snapshots ADD COLUMN sox REAL;
ALTER TABLE cross_asset_snapshots ADD COLUMN sox_change_pct REAL;

-- 新表: 宏观事件记录
CREATE TABLE IF NOT EXISTS macro_events (
    asof TEXT NOT NULL,
    ts TEXT NOT NULL,
    alert_level TEXT NOT NULL,          -- L1/L2/L3/none
    triggered_rules TEXT,               -- JSON list of triggered rules
    rule_boost REAL DEFAULT 0.0,        -- 规则引擎的 boost
    llm_boost REAL,                     -- LLM 建议的 boost (可 NULL)
    final_boost REAL NOT NULL,          -- 最终 boost (合并后)
    duration_days INTEGER DEFAULT 3,
    event_summary TEXT,
    event_type TEXT,
    llm_raw_json TEXT,                  -- LLM 原始输出 (审计用)
    source TEXT DEFAULT 'rule_engine',  -- 'rule_engine' / 'llm' / 'manual'
    PRIMARY KEY (asof)
);

-- 新表: 宏观新闻标题 (喂给 LLM)
CREATE TABLE IF NOT EXISTS macro_headlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asof TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT,
    url TEXT,
    fetched_ts TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 2.9 盘前告警

当 alert_level == L1:
```
Discord webhook 推送:
  ⚠️ [L1 宏观事件] 2026-04-08
  NK期货: +4.25% | 原油: -10.2% | VIX: +6.7%
  事件: 伊美即时停战 (Gemma 分析)
  regime_boost: +0.30 → regime 0.214 → 0.514
  建议: 提高暴露至 66%，关注半导体/制造业
```

---

## 三、风险和约束

| 风险 | 应对 |
|------|------|
| Gemma 幻觉（无事件时编造事件） | 仅 L1/L2 触发时调用；confidence < 0.5 时 boost 减半 |
| 网络不通（Tailscale 断连） | 退化到层2规则引擎，不阻塞 |
| 新闻标题采集失败 | LLM 仍可根据纯数值推理；降低 confidence |
| event_boost 过大导致追高 | clamp [-0.30, +0.30]，且有 duration 衰减 |
| yfinance 大宗商品数据延迟 | 可接受：盘前采集，不需实时 |
| 历史回测不够（event 稀疏） | 先用保守参数上线，积累数据后调优 |

---

## 四、配置扩展 (config.yaml)

```yaml
cross_asset:
  enabled: true
  shadow_only: false              # 已上线
  tickers:
    sp500: "^GSPC"
    usdjpy: "USDJPY=X"
    vix: "^VIX"
    nk_futures: "NKD=F"
    crude_oil: "CL=F"            # 新增
    gold: "GC=F"                 # 新增
    copper: "HG=F"               # 新增
    sox: "^SOX"                  # 新增
  weights:
    sp500: 0.20
    usdjpy: 0.15
    vix: 0.10
    nk_futures: 0.15
    crude_oil: 0.15
    gold: 0.05
    copper: 0.05
    sox: 0.10

macro_events:
  enabled: true
  llm:
    enabled: true
    provider: "ollama"
    endpoint: "http://gemma-host:11434"  # Tailscale IP 或 hostname
    model: "gemma4:27b"
    timeout_seconds: 60
    fallback_to_rules: true       # LLM 不可用时退化到规则引擎
  rules:
    nk_futures_l1_threshold: 3.0
    crude_oil_l1_threshold: 5.0
    vix_spike_threshold: 15.0
    sox_l2_threshold: 3.0
  boost:
    max_abs: 0.30
    default_duration_days: 3
    decay_mode: "linear"          # linear / exponential
  alerts:
    discord_on_l1: true
    discord_on_l2: false
```
