---
title: Path A 迁移操作手册 — 3041.T → ETF
date: 2026-04-28
status: ACTIVE
---

# 🎯 这份文档解决什么问题

3041.T OCO 触发后（卖出价 ¥600 或 ¥558），你需要把约 ¥390k 现金转换成 1321.T + 1306.T 50/50 持仓。这份文档给你**逐步操作清单**。

---

## ⏱️ 阶段 1：等待 OCO 触发

**当前状态**（2026-04-28）：
- SBI OCO 单已挂：¥600 限价止盈 / ¥558 触价止损
- 持仓 3041.T 400 股，成本 ¥585
- 现金 ¥166,545

**触发后你会看到**（SBI 邮件 / app 通知）：
- 注文約定通知（注文番号 13 一部約定 or 全部約定）

**两种触发结果**：

| 触发价 | 注文额（含税前） | 触发后总现金（约） |
|---|---|---|
| ¥600（止盈） | ¥240,000 | **¥406,545** |
| ¥558（止损） | ¥223,200 | **¥389,745** |

**期间禁止做的事**：
- ❌ 取消 OCO 单
- ❌ 改 OCO 价格
- ❌ 加挂任何买单
- ❌ 做 T 或日内交易

---

## ⏱️ 阶段 2：OCO 触发当日（不要立刻建仓）

**触发当天什么都不要做**。理由：
- 触发当天市场情绪通常异常（特别是止损触发，往往伴随 panic）
- 立刻接入新仓位 = 在情绪高点买入
- **隔一个交易日**让市场恢复理性

**唯一动作**：在系统记录 sprint 平仓状态。
```bash
cd E:\AIagent_project_260213\worker-quant\quant_trading\Project_optimized
python import_fills.py --strategy_id sprint --asof <触发日> \
    --csv "<触发日>: SELL,3041.T,400,<触发价>"
# 这一步把 SBI 的成交录入 DB sprint lane，让 sprint 持仓正式归 0
# 详细参数见 import_fills.py --help
```

---

## ⏱️ 阶段 3：T+1 日建 ETF 仓位（核心步骤）

**择日**：触发后第 1 个交易日，盘前 09:00 JST 之前在 SBI app 完成下单。

### Step 1：计算建仓数量

假设你触发后总现金 = `C`（例：¥390,000）。

```
1321.T 配额 = C × 0.50 = ¥195,000
1306.T 配额 = C × 0.50 = ¥195,000
```

**查 1321.T 和 1306.T 当前价**（SBI app 或 yfinance），然后：

```
1321.T 目标股数 = floor(195000 / 1321.T 价格 / 100) × 100
1306.T 目标股数 = floor(195000 / 1306.T 价格 / 100) × 100
```

例（参考价）：
- 1321.T ≈ ¥62,400 → 不够 100 股（¥6.24M）→ **改买 10 股最小单位（如有 OK）或减少配额**
- 1306.T ≈ ¥3,200 → 195000 / 3200 ≈ 60 → 但最小单位 100 股，所以 **0 股**

⚠️ **重要规模问题**：1321.T 单位价格高（约 ¥6万/单位 100 股 = ¥600万），你 ¥400k 资金**买不起整数 100 股 1321.T**。

**修正方案**（基于实际价格）：
- 改用 **NF 日経 ETF (1346.T)** 替代 1321.T —— 价格约 ¥3万
- 或改用 **MAXIS 日経225 (1346.T)** —— 类似
- 或改用 **iFreeETF日経225 (2526.T)** —— 单位较小
- TOPIX 用 **NF TOPIX ETF (1306.T)** 价格约 ¥3000，可以买

**推荐改配置**（更适合 ¥400k 规模）：
- **1346.T**（NF 日経 225 ETF，约 ¥30,000/100 股）50%
- **1306.T**（NF TOPIX ETF，约 ¥3,000/100 股）50%

> ⚠️ 在执行前先查 SBI 实时价格，**手动验算每只股数 × 当前价 ≤ 配额**。

### Step 2：在 SBI 下单

```
3041.T → 取消所有挂单（确认 OCO 已成交，无残留）
1346.T BUY 成行 N 股（按上一步计算）
1306.T BUY 成行 M 股（按上一步计算）
```

**全部用「成行」**（市场价），不要用限价——避免错失。

### Step 3：在系统录入新持仓

```bash
cd E:\AIagent_project_260213\worker-quant\quant_trading\Project_optimized
python build_account_snapshot.py \
    --strategy_id etf_buyhold \
    --asof <T+1 日> \
    --bootstrap \
    --initial_cash <剩余现金>
```

如果 `build_account_snapshot.py` 没有 `--bootstrap` 参数，先用 SQL 手动插入：

```python
import sqlite3
conn = sqlite3.connect("japan_market.db")
asof = "2026-XX-XX"  # T+1 日
# 持仓
conn.execute("""INSERT INTO positions(asof, strategy_id, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl)
    VALUES(?, 'etf_buyhold', '1346.T', N股, 价格, 价格, N*价格, 0)""", (asof,))
conn.execute("""INSERT INTO positions(asof, strategy_id, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl)
    VALUES(?, 'etf_buyhold', '1306.T', M股, 价格, 价格, M*价格, 0)""", (asof,))
# 账户快照
conn.execute("""INSERT INTO account_snapshots(asof, strategy_id, ts, run_id, cash, positions_value, nav)
    VALUES(?, 'etf_buyhold', ?, 'bootstrap', 剩余现金, N*价格+M*价格, 总NAV)""",
    (asof, f"{asof} 16:00:00"))
conn.commit()
```

---

## ⏱️ 阶段 4：日常维护

**月度（1 日 09:30 自动）**：
- Task Scheduler 跑 `etf_monthly_check.py`
- 邮件发到 lwyssq@gmail.com
- 报告内容：NAV / 月度收益率 / vs 1321 / drift / 心理检查

**季度再平衡**（如 drift > 5%）：
- 邮件会 flag `⚠️ REBALANCE`
- 你手动在 SBI 卖超配的、买不足的
- 录入新 fills

**不做**：
- ❌ 看盘
- ❌ 主动卖出非再平衡的仓位
- ❌ 加新标的

---

## 🚨 紧急情况处理

### 情况 A：OCO 单不存在 / 被取消
**立刻**重新挂逆指値 ¥558 + 限价 ¥600，期間「今週中」。

### 情况 B：3041.T 跌破 ¥558 但 OCO 没触发
检查 SBI 注文照会。如果状态异常，**立刻成行卖出 400 股**。

### 情况 C：触发后忘记建 ETF 仓位
**最长拖延 5 个交易日**。超过 5 天的现金闲置 = 错过潜在 +0.5% 收益。设手机日历提醒。

### 情况 D：1346/1306 当天异动 > ±3%
延后 1 天再建仓。**不要追高，不要抄底**。

---

## 📞 你随时可以问 AI 的事

```
"我今天 OCO 触发了 ¥XXX，现金有 ¥XXX，怎么建 1346 + 1306 仓位？"
"1346.T 现在 ¥XXXXX，我应该买几股？"
"我建仓了，怎么录入 DB？"
"月报里说要再平衡，具体怎么操作？"
```

把数字告诉 AI，AI 会算给你具体股数。

---

## ✅ 完成 Path A 迁移的标志

- [x] 决策文档已签 (STRATEGY_DECISION_2026-04-28.md)
- [x] sprint hard lock 已激活 (capital_gate_config.yaml)
- [x] etf_buyhold 已注册 (strategy_registry.py)
- [x] daily_run 跳过 passive 链路 (代码已改)
- [x] 月度脚本已就绪 (etf_monthly_check.py)
- [ ] 3041.T OCO 触发并平仓
- [ ] 1346.T + 1306.T 实际建仓
- [ ] etf_buyhold 在 DB 里有 account_snapshot 记录
- [ ] Task Scheduler 注册成功（运行 register_etf_monthly_check.bat）
- [ ] 第一次月度邮件成功收到（5/1 09:30）
