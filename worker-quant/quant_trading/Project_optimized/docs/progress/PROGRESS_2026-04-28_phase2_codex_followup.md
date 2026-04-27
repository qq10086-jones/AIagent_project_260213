---
title: Strategy Validity Audit — Phase 2 (Codex Follow-up)
date: 2026-04-28
auditor: Claude (sonnet 4.6) + Codex review
verdict: ❌ FAIL CONFIRMED across all sprint variants. Path A recommendation strengthened.
---

# Phase 2 任务（应 Codex review 补做）

| 任务 | 状态 |
|---|---|
| A1: 修复 `_current_paper_position` strategy_id 泳道 bug | ✅ 完成 |
| A2: 加 `tests/test_paper_execute_inventory_guard.py` (5 tests) | ✅ 5/5 pass |
| C1: 用 N_trials=59 重测 DSR 验证稳健性 | ✅ 完成 |
| D1: 跑 sprint_aggressive 扩样本 (top_k=2 / top_k=1) | ✅ 完成 |

---

## 一、Codex 抓到的代码 bug 修复

### 原 bug
```python
simulate_fills(strategy_id=source_strategy)  # source = "sprint"
  → _current_paper_position(strategy_id=source_strategy)  # 读 sprint 真实泳道
  → 但 fills 实际写入 sprint_paper（不同泳道）
```
**后果**：sprint 真实有 400 股 → paper 也能继续"卖"400 股 → phantom 复现

### 修复
- 加新参数 `inventory_strategy_id` 到 `simulate_fills`
- main() 调用时传 `inventory_strategy_id=paper_strategy`
- 库存校验现在读正确泳道

### 验证（5 个单元测试全通过）
```
test_default_inventory_strategy_falls_back_to_source  ✓
test_guard_reads_paper_lane_not_source_lane          ✓ (Codex catch)
test_intra_run_position_delta                        ✓
test_sell_exceeding_inventory_rejected               ✓
test_sell_within_inventory_passes                    ✓
```

---

## 二、Sprint 变体扩样本结果（3 年, N=33 月）

| 变体 | Sharpe | MaxDD | 累计 | PSR p | DSR p (N=10) | DSR p (N=59) | IR_a | p_IR |
|---|---|---|---|---|---|---|---|---|
| sprint top_k=5（保守档代理） | 1.00 | -19.2% | +56.5% | 0.060 | 0.507 | 0.783 | -0.01 | 0.98 |
| aggressive top_k=2 | 0.75 | **-34.8%** | +53.9% | 0.143 | 0.694 | 0.899 | +0.05 | 0.94 |
| aggressive top_k=1 | 0.90 | -22.2% | +77.8% | **0.042** | 0.439 | 0.729 | **+0.27** | 0.66 |
| **1321.T 基准** | 1.02 | -10.6% | +58.2% | — | — | — | — | — |
| **等权基准** | **1.69** | **-7.8%** | +61.5% | — | — | — | — | — |

### 关键发现

**1. 所有 sprint 变体 DSR 全部 FAIL**
- N_trials=10：所有 p ∈ [0.44, 0.69]
- N_trials=59：所有 p ∈ [0.73, 0.90]
- **结论稳健性极强**，N_trials 怎么取都 FAIL

**2. 激进档（top_k=2）实测最差**
- Sharpe **0.75**（最低），MaxDD **-34.8%**（最差）
- 你账户 ¥393,745，按这个回撤 = **-¥137,000 潜在损失**
- 双股集中 = "如果两个都错"风险翻倍

**3. 极致集中（top_k=1）出现 weak signal**
- IR vs EW = +0.27（唯一正值）
- 但 t = 0.44，p = 0.66，**统计上无显著证据**
- 33 月样本要 detect IR=0.27 需要 ~85+ 月样本（power calc）
- **可能是 noise，也可能是 weak signal — 不足以下结论**

**4. 等权基准依然碾压所有变体**
- Sharpe 1.69 vs 最高 sprint 变体 1.00
- MaxDD -7.8% vs 最低 sprint 变体 -19.2%
- **每个维度都更好**

---

## 三、统计 power 修正声明（应 Codex review）

### N=33 月样本的 detection limit

| 真实 IR | N=33 detect 概率 (95%) | 需要 N |
|---|---|---|
| 0.50 | 87% | 28 |
| 0.30 | 41% | 76 |
| 0.20 | 19% | 170 |
| 0.10 | 7% | 670 |

**含义**：N=33 只够 detect IR ≥ 0.50 的策略。对中等 alpha（IR 0.20-0.30），统计上无能力。

**这意味着**：
- 我们的 FAIL 判决对"明显有 alpha"的策略稳健
- 但对"weak alpha"策略**力不从心**
- top_k=1 的 IR=0.27 落在这个盲区

---

## 四、Equal-Weight 基准 selection bias 声明（应 Codex review）

EW 基准用的是 `_aligned_universe_ew(...liq_mask=liq)`，**与策略共享同一个筛选后 universe**。

含义：
- IR vs EW 是 **相对 hurdle**（你必须比这个 universe 里随机选股的等权更好）
- 不是 **absolute alpha test**（vs 全市场）
- 但仍然是合理的"超过被动配置"基准

**修正表述**：
> 策略不能在 universe 内提供 selection alpha（无法选出比等权更好的子集）。  
> 这不排除"universe 选择本身有 alpha" — 但那是数据预处理层的功劳，不是策略层的。

---

## 五、修正后最终判决

### 不变的硬结论
1. ❌ Sprint 保守档无 selection alpha（DSR fail x2 + IR p=0.98）
2. ❌ Sprint 激进档无 selection alpha（DSR fail x2 + 风险更大）
3. ❌ 任何 sprint 变体都跑输等权基准
4. ✅ 不能冻结，不能上自动化

### 新增 nuance
1. ⚠️ top_k=1 极致集中**有 weak signal**（IR=0.27, p=0.66），但 33 月样本无法证实
2. ⚠️ 路径 A（清仓转 ETF）依然 dominant，但**100% market beta** 风险应明确
3. ⚠️ 现有 33 月样本对中等 alpha 无 detection power

---

## 六、推荐路径（修正后）

### 🟢 路径 A（首选）：清仓转 long-only ETF 等权
**为什么依然首选**：
- 等权基准 Sharpe 1.69，碾压任何 sprint 变体
- 零工程成本
- 你 3 年扩样本累计 +56-78%，等权同期 +61.5%（且回撤更小）

**风险量化**（Codex 要求）：
- 100% market beta，日经 -20% = **-¥78,748**
- 但比 sprint 任何变体都好（aggressive top_k=2 最大回撤 -¥137,000）

### 🟡 路径 B（折中）：3041.T 走完观察期 + 转部分 ETF
**新设计**（Codex 要求中间方案）：
- 3041.T OCO ¥558 走完
- 平仓后 50% (~¥197k) 立刻转 1321.T+1306.T 等权（市场 beta）
- 50% (~¥197k) 留作研究资金，sprint_paper-only（修复后的模拟器）跑 4-8 周
- 8 周后看 paper 是否能产生 IR > 0 vs EW

### 🔴 路径 C（不推荐）：彻底重设计 sprint
依然不推荐。现有证据显示 sprint 框架本身没有 selection alpha。

---

## 七、技术债（已识别，未做）

1. cash_ledger 跨策略污染审计（4/16+ 期间）— 已加入跟进清单
2. 滑点对 BUY/SELL 对称性验证 — `_fill_price` 实现层
3. 同日同订单 duplicate run_id 问题（4/23 出现两次） — 治理层
4. capital_gate 决策依据是否过激进 — 治理层
5. Newey-West 校正 IR test — 严谨性（结论稳健，仅形式）

---

## 八、最终签名

**Codex 抓到的 6 个有效问题中**：
- 1 个真正 P0 bug（`_current_paper_position` 泳道）✅ 已修
- 1 个不一致（N_trials=10 vs 59）✅ 已重测，结论稳健
- 1 个缺口（aggressive 未扩样本测）✅ 已补，结论更严
- 1 个披露缺失（EW selection bias）✅ 本报告补充
- 1 个建议（中间路径）✅ 路径 B 已重设计
- 1 个 power 限制声明 ✅ 本报告补充

**修正后的硬结论**：
- 所有 sprint 变体在 3 年扩样本 + 多重检验校正下 FAIL
- 等权基准 Sharpe 1.69 是真正的 winner
- 路径 A（清仓转 ETF）依然推荐，路径 B 作为更保守的折中

**给你的硬话**（赌徒心理对策）：

激进档实测**比保守档更差**。如果你的本能是"亏了应该上激进档赚回来"，统计学说"激进档亏更多"。MaxDD -34.8% 在 ¥400k 账户上 = **-¥137,000 潜在损失** —— 那是你现在亏损 (-¥10,000) 的 13 倍。

**Sharpe 1.69 的等权基准摆在那里**。3 年累计 +61.5%，最大回撤 -7.8%。**不需要任何策略**。
