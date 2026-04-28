---
title: Strategy Decision — Path A (Long-Only ETF Buy-Hold)
date: 2026-04-28
decided_by: User
based_on:
  - PROGRESS_2026-04-28_strategy_validity_audit.md
  - PROGRESS_2026-04-28_phase2_codex_followup.md
status: ACTIVE
---

# 决定

**放弃 sprint 主动选股策略。资金转入 long-only ETF 等权配置长期持有。**

**2026-04-28 配置更正**：1321.T 单位 ¥6.2M 超账户规模，改用 **1346.T（NF Nikkei 225 ETF, ¥30k/100sh）+ 1306.T（NF TOPIX ETF, ¥3k/100sh）50/50 配置**。两者跟踪同样的日経/TOPIX 大盘指数，仅最小交易单位价格不同。

# 为什么（一句话）

3 年扩样本 + 多重检验校正后，sprint 任何档位都没有 selection alpha；等权基准 Sharpe 1.69 / MaxDD -7.8% 在每个维度都碾压策略 (Sharpe 0.75-1.00 / MaxDD -19% to -35%)。**最优策略是不要策略**。

# 数据依据

| 变体 | Sharpe | MaxDD | DSR p (N=59) | IR vs EW p |
|---|---|---|---|---|
| sprint top_k=5 | 1.00 | -19% | 0.78 ❌ | 0.98 ❌ |
| aggressive top_k=2 | 0.75 | -35% | 0.90 ❌ | 0.94 ❌ |
| aggressive top_k=1 | 0.90 | -22% | 0.73 ❌ | 0.66 ❌ |
| **等权基准** | **1.69** | **-7.8%** | — | — |

# 执行阶段

## Phase 1：3041.T 退出（已就绪）
- SBI OCO 单已挂：¥600 限价止盈 / ¥558 触价止损
- **不取消 OCO**，让规则自然结清
- 期间不再有任何 sprint 交易决策

## Phase 2：ETF 部署（3041.T 平仓后立即）
- 全部现金（约 ¥390k 平仓后）转入 ETF 等权配置
- 入场方式：成行单，分 1-2 天内完成

## Phase 3：每月最小化维护
- 月度 Discord/邮件提醒查看 NAV
- 季度再平衡（如选多 ETF 配置）
- **不再有日度决策**

## Phase 4：技术层面 sprint 退役
- `capital_gate` 永久锁定 sprint 为 `paused_sunk_only`
- sprint 代码降级为研究/学术泳道（sprint_paper）
- daily_run 继续跑但不输出实盘指令
- 新建 `etf_buyhold` strategy_id 追踪 ETF 仓位

# 不变的事

- worker-quant 项目继续维护（研究价值）
- sprint_paper 在修复后的模拟器跑（看是否能产出有 alpha 的变体）
- 任何研究突破必须通过 DSR p<0.05 + IR vs EW p<0.10 + 24+ 月 OOS 验证才能升级回实盘
- 升级实盘需要重新走治理流程

# 心理学约束（写给未来的自己）

未来 3-6 个月你会**反复**想：
- "市场涨了一点，我不应该错过这个 alpha"
- "我看到一个好策略，能不能小试一下"
- "我朋友说他靠 XXX 赚了"
- "这次一定不一样"

**这份文档的存在就是为了告诉那时的你**：
1. 你 2026-04 已经投了 3 个月时间和 ¥10,000+ 试错
2. 数学已经证明 sprint 框架没有真 alpha
3. 等权 ETF 在每个统计维度都赢
4. **任何"小试一下"的冲动都是赌徒心理的回归**
5. 想做研究：用 paper 泳道，用 0 真金
6. 想加仓真金：先 24 月新策略 OOS + DSR 通过，没有捷径

# 解锁实盘策略的硬条件

任何想从 ETF buy-hold 升级回实盘策略的尝试，必须**全部满足**：

1. ✅ 24+ 月真正的 OOS 数据（不是 in-sample 调过的）
2. ✅ DSR p < 0.05（多重检验校正后）
3. ✅ IR vs EW p < 0.05（绝对而非边缘显著）
4. ✅ Bootstrap Sharpe 95% CI 不含 0
5. ✅ 至少包含一次 ≥ -10% 的 bear-market 样本
6. ✅ paper 泳道连续 8 周实测 IR > 0
7. ✅ 上述全部通过后，**初始资金 ≤ NAV 的 10%**，逐步放大

任何一项不满足 → 留在 ETF buy-hold。
