# Quant 系统改造计划 v3

**日期**: 2026-04-13 起草，2026-04-14 重大修订（v3）
**立场修正**：v2 过度偏向"治理/防御"，把用户原始诉求"买卖策略优化"搁在一边。v3 把重心扳回**策略进攻**，防御线只保留必要最小集。

## 版本迭代历史

- **v1 (2026-04-13 初稿)**：对 v1 晋升 / paper_days=30 规则的批判 + Phase 0-3 治理大纲
- **v2 (2026-04-13 下午)**：codex review 后补入 walk-forward 嵌入、多重检验、Phase -1 数据卫生、Research/Live 两层 gate
- **v3 (2026-04-14)**：**重要转向** —— 结合项目实用性评分（3.9/10 对标 Qlib），把主力放到 Alpha 因子扩展 + walk-forward runner。回滚 Paper 双步闸门（方向错）。引入 intraday_decision 产线

---

## 项目当前状态（2026-04-14 评分）

**总分 ≈ 3.9 / 10**（加权 4.5/10）对标 Microsoft Qlib / QuantConnect LEAN / Alphalens。

| 维度 | 分 | 问题 |
|---|---|---|
| 数据覆盖 | 5/10 | 无 survivorship / 无 TR / 复权未验证 |
| **因子库广度** | **3/10** | 25 个 vs Alpha158/360；**P0 改进空间** |
| 信号模型多样性 | 4/10 | Momentum+Ridge+ICIR 基础可用 |
| **回测严谨性** | **2/10** | 无 walk-forward / 无 purge / bps 平均 slippage；**P0 改进空间** |
| 统计验证 | 2/10 | 无 DSR / 无 FDR |
| 组合构造 | 3/10 | 无 Barra / 无 Ledoit-Wolf |
| 执行建模 | 3/10 | 无 T+1 限价簿 / slippage 常数 |
| **实盘集成** | **6/10** | SBI+paper 双源 + Discord = **相对优势** |
| 治理 / 可复现 | 5/10 | experiment_log / PIT audit 昨日补上 |
| 运维成熟 | 6/10 | 164 pytest / runtime_events / heartbeat |

**目标**：10 周内从 3.9/10 → 5.5-6/10（对标 Qlib 2020 年水平）。

---

## 用户诉求（必须对齐）

1. **"年收益 600% 以上"** —— 此目标单靠信号优化无法达成，需**杠杆 + 高胜率 + 集中持仓 + 低回撤**四件事同时发生。信号优化通常上限 50-100% 年化。本计划先把信号做到真实 IR > 0.5 且超额 > TOPIX，**600% 目标的路径单独讨论**（见附录 A）
2. **"盘中出信号，实时推送，我手动跟 SBI"** —— intraday_decision.py 已完成（2026-04-14），成为日常交付形态
3. **"模型已多轮认证，不要质疑"** —— 本计划不再要求作废信号链或冻结实盘；只加强**可测量性**

---

## v2 → v3 重大修订

### 保留
- Phase -1 数据卫生 8 项（D-1 ~ D-8）—— survivorship / 复权 / lot_size / 停牌 / available_at / 时区 / PIT / 1321.T TR
- experiment_log 预注册协议
- reality_check 四基准对比
- PIT 审计 + 不变性测试
- 多重检验 (FDR / Deflated Sharpe)
- Walk-forward 嵌入式决策（仍然是地基）

### 新增（v3 核心）
- **Phase A 改为 Alpha 因子扩展优先**（原 v2 Phase A 是 walk-forward pipeline，现拆分）
- **Phase B 独立成 walk-forward runner**
- **Phase C 新增 intraday 生产线**（已部分落地）
- **Phase D Barra-lite 风险模型**（v2 里混在组合构造里，拎出来）

### 回滚 / 降级
- ❌ **Paper 双步闸门（T0.1）** —— 方向错误，默认已关；代码保留但不再是 P0
- ❌ **冻结实盘建议** —— codex 的建议在用户场景下不成立
- ⚠️ **LLM 宏观层**：降级为**纯风控硬过滤**（regime_score 阈值降仓），**不进选股权重**。现状保持

### 放弃（明确砍掉）
- ❌ 不再扩新数据源（先把现有 25 因子用透）
- ❌ 不做 intraday 策略（只做 intraday 交付，不改信号频率）
- ❌ 不做衍生品 / 期权（杠杆问题走另一条路径讨论）
- ❌ 不做 tick 级仿真

---

## 新 Phase 结构

```
Phase -1 (数据卫生) ─────► 阻塞所有研究类改动
          │
          ▼
Phase A (Alpha 因子扩展)    ← v3 主战场
          │
          ▼
Phase B (Walk-Forward Runner) ← 回测严谨性从 2/10 → 6/10
          │
          ▼
Phase C (Intraday 生产线) ← 与 Phase A/B 并行
          │
          ▼
Phase D (Barra-lite + DSR/FDR) ← 风险 / 统计 从 2-3/10 → 6/10
          │
          ▼
     Research-Framework Gate (10 周)
          │
          ▼
     Live-Promotion Gate (≥6 个月)
```

---

### Phase -1 数据卫生（继续，2-3 周）

**不变**。D-1 ~ D-8 逐项完成。D-8 已完成（PIT 审计 + 3 P0 修复 + 不变性测试 + 实证无污染）。

**剩余任务**（按 ROI 排序）：
- [ ] **D-3** 1321.T total-return 序列（工作量最小、收益立刻可见）
- [ ] **D-6** 基本面 `available_at` vs `report_date` 审计
- [ ] **D-1** Survivorship / 退市票（工作量最大）
- [ ] D-2 / D-4 / D-5 / D-7（剩余项穿插做）

**Exit**：D-x 全部通过单元测试；任给 `asof`，查询结果 ≡ 当时可见数据。

---

### Phase A — Alpha 因子扩展（v3 新主力，3-4 周）

**目标**：从 25 因子 → 80+ 因子覆盖；对标 Qlib Alpha158 子集。**不发明因子，照搬学术已验证的**。

**任务**
- [ ] **A-1** Alpha158 子集移植（不要全部 158 个，选可解释 + 低相关的 60 个）
  - 价格动量：涵盖 Alpha158 的 ret/volatility/price-volume 关系公式
  - 基本面：ROE、P/B、P/E、earnings surprise
  - 反转：1 日 / 5 日反转因子
- [ ] **A-2** Cross-sectional rank 化（每个因子按日 cross-section 排名到 [0,1]）
- [ ] **A-3** 行业中性化（每因子减去同业中位数）
- [ ] **A-4** Winsorize + z-score 标准化
- [ ] **A-5** 因子相关性矩阵 + Gram-Schmidt 正交化
- [ ] **A-6** Marginal IC（逐个因子加入后的 IC 增量）

**关键约束**
- **不要自己发明因子**。发明等于加自由度，加自由度必须走 experiment_log 预注册
- **所有因子入库前必须有学术引用**（写到 factor_definitions.md）
- 每个因子的 IC 诊断走 Phase B walk-forward（不准在全样本上筛）

**Exit**：80+ 因子 + 每个带文献引用 + 正交化后 IC t-stat > 3 的 top-5 入白名单。

---

### Phase B — Walk-Forward Runner（3 周，回测严谨性质变）

**目标**：**单一这一个任务就能把 #4 回测严谨性从 2/10 → 6/10**。对标 Qlib `qrun` 工作流。

**管道**
```
for each (train_window, validation_window) in rolling_splits:
    # 训练窗（3 年） — 只看 train_window 数据
    1. Alpha 因子计算
    2. 正交化 + marginal IC
    3. 权重 = max-IR with Ledoit-Wolf shrinkage
    4. 组合构造（lot_size + cash + 行业 + 单票约束）
    5. 冻结所有参数

    # 验证窗（6 月） — 只读冻结规则
    6. 生成信号
    7. 模拟成交（T+1 open + square-root slippage）
    8. 记录 research / tradable / (future) actual-live 三轨

# 聚合所有 validation_window → 真 OOS 指标
```

**任务**
- [ ] **B-1** `walk_forward_runner.py`：3 年训练 / 6 月验证 / 1 月滚动
- [ ] **B-2** Ledoit-Wolf 协方差 shrinkage
- [ ] **B-3** 三轨道组合生成器
- [ ] **B-4** 成交成本模型：`α_bp > 2 × (fee_bp + half_spread_bp + impact_bp(qty/ADV))`
- [ ] **B-5** T+1 open 成交 + square-root slippage

**Exit**：
- 聚合 walk-forward 净 IR > 0.5（block-bootstrap CI 下界 > 0）
- 跑赢四基准（TOPIX TR / universe 等权 / sector-neutral / 现金）
- research vs tradable 轨迹偏差 < 50%

---

### Phase C — Intraday 生产线（已部分落地，2 周完成）

**目标**：把 14:45 JST 信号交付做成稳定日常。

**已完成（2026-04-14）**
- ✅ `intraday_decision.py` 基础版（refresh intraday + target_weights × SBI 实仓 → Discord）
- ✅ Discord webhook 接入 (`.env` WORKER_QUANT_ALERT_WEBHOOK_URL)

**剩余任务**
- [ ] **C-1** **修 paper/sprint strategy_id 污染**（P0 bug）—— paper_simulator 当前写到 `strategy_id='sprint'`，污染真实轨迹。必须改写到 `sprint_paper`
- [ ] **C-2** Briefing / action_plan 分 SBI / paper 两栏显示
- [ ] **C-3** Windows Task Scheduler 每交易日 14:45 JST 触发
- [ ] **C-4** Intraday_decision 出错重试 + 熔断（yfinance 失败时回退昨收 + 报警）
- [ ] **C-5** SBI fill 导入工具化（现状靠 `import_fills.py` 手动）—— 开发一个 CSV/截图解析器

**Exit**：每交易日 14:45 自动出单 + Discord 到达 + 用户可在 15:00 前下 SBI 单 + 次日 import_fills 对账。

---

### Phase D — Barra-lite 风险 + 统计（2 周）

**目标**：统计验证从 2/10 → 6/10。

**任务**
- [ ] **D-1(stats)** Newey-West 修正 t-stat（已有框架）
- [ ] **D-2(stats)** Block bootstrap CI（block 长度 = half-life × 3）
- [ ] **D-3(stats)** FDR (Benjamini-Hochberg)
- [ ] **D-4(stats)** Deflated Sharpe Ratio（Lopez de Prado 公式；读 experiment_log 取 N）
- [ ] **D-5(risk)** Barra-lite 5 因子：市场 / 33 业种 / size / vol / liquidity
- [ ] **D-6(risk)** 因子暴露计算 + residual risk 分解
- [ ] **D-7(risk)** 组合 beta / 行业中性约束（可选）

**Exit**：任何 walk-forward 结果必须带 (a) t-stat CI (b) DSR (c) 四基准超额 (d) Barra 风险分解。

---

## 任务清单（按优先级整合）

### 已完成（2026-04-13 至 2026-04-14）✅

- [x] v3 计划文档起草
- [x] experiment_log 预注册机制（P0）
- [x] reality_check 四基准报告（P0）
- [x] v1 晋升归档作废（P0）
- [x] PIT 审计 47 处 + 3 P0 修复（D-8 Step 1+2）
- [x] PIT 不变性测试 5 条
- [x] PIT 实证无污染验证（D-8 Step 3）
- [x] **intraday_decision.py 基础版（Phase C 起步）**

### 回滚 / 降级

- [x] ~~Paper 双步闸门默认开启~~ → 默认关闭（保留代码）
- [x] ~~冻结实盘建议~~ → 撤回

### P0（接下来 2 周）

- [ ] **C-1** 修 paper_simulator 污染真实 sprint strategy_id（**本周必做**）
- [ ] **C-2** Briefing / action_plan 分 SBI / paper 显示
- [ ] **C-3** Windows Task Scheduler 14:45 JST 触发配置
- [ ] **D-3** 1321.T total-return 序列（数据卫生，reality_check 更准）

### P1 — Alpha 扩展启动（3-4 周）

- [ ] **A-1** Alpha158 子集移植（60 个因子）
- [ ] **A-2** Cross-sectional rank
- [ ] **A-3** 行业中性化
- [ ] **A-4** Winsorize + z-score
- [ ] **A-5** 相关性矩阵 + 正交化
- [ ] **A-6** Marginal IC

### P2 — Walk-Forward Runner（3 周）

- [ ] **B-1** walk_forward_runner.py
- [ ] **B-2** Ledoit-Wolf shrinkage
- [ ] **B-3** 三轨道组合
- [ ] **B-4** 成交成本模型
- [ ] **B-5** T+1 open + square-root slippage

### P3 — Barra-lite + 统计（2 周）

- [ ] **D-1..7** 见 Phase D

### P4 — 数据卫生剩余（穿插做）

- [ ] **D-6** 基本面 available_at 审计
- [ ] **D-1** Survivorship
- [ ] **D-2/D-4/D-5/D-7** 剩余 4 项

---

## 时间表

| 周 | Phase -1 | Phase A | Phase B | Phase C | Phase D |
|---|---|---|---|---|---|
| W1 | D-3 | — | — | **C-1~C-3** | — |
| W2 | D-6 | A-1 | — | C-4 | — |
| W3 | D-1 | A-2~A-3 | — | C-5 | — |
| W4 | D-1 | A-4~A-6 | B-1 | — | — |
| W5 | D-2/D-4 | — | B-2~B-3 | — | D-1/D-2 |
| W6 | D-5/D-7 | — | B-4~B-5 | — | D-3/D-4 |
| W7 | — | — | 聚合验证 | — | D-5 |
| W8 | — | — | — | — | D-6/D-7 |
| W9 | — | — | 整合测试 | — | — |
| W10 | — | — | **Research Gate** | — | — |

---

## 成功指标（10 周后）

| 指标 | 当前 | 目标 |
|---|---|---|
| 项目总分 | 3.9/10 | **≥ 5.5/10** |
| 因子数 | 25 | **60+** |
| Walk-forward IR | n/a | **> 0.5, t-stat > 2** |
| 跑赢 TOPIX TR | n=7 勉强 +3.19% | **统计显著** |
| 跑赢持仓等权 | n=7 勉强 +11% | **持续正向** |
| DSR 显著性 | n/a | **> 0.95** |
| Paper/SBI reconciliation | 手动 | **日度自动** |

---

## 附录 A — 关于年化 600% 目标

用户明确目标：年化 ≥ 600%。本计划覆盖的信号优化能达到的合理上限估计 **30-80% 年化**（参照 Qlib 官方 benchmarks / AQR 因子投资论文 / Alphalens 顶尖样例）。

从信号到 600% 需要以下**同时发生**：
1. **杠杆 3-5×**（信用交易 / 保证金 / 期货）
2. **高胜率集中持仓**（3-5 只而非 15+ 只，单票权重 20%+）
3. **低回撤控制**（否则杠杆放大下单次爆仓）
4. **事件驱动机会**（IPO / 财报季 / 政策事件窗口）

这些**不在本改造计划范围**。本计划先把信号做到**真实超额 + 统计可验证**，然后 600% 目标通过另一条路径讨论（杠杆 + 集中 + 事件窗口设计）。

强行把 600% 目标塞进信号优化 = 过拟合地雷。分阶段做。

---

## 附录 B — 方法论红线（保留自 v2）

1. 没有 benchmark 不算策略（≥4 基准必须）
2. 没有置信区间的 IC 都是自欺
3. Out-of-sample 是物理隔离
4. 多重检验必须纳入
5. 先打平多基准再谈 alpha
6. 数据卫生优先于模型

---

## 附录 C — 明确不做的事

- ❌ 新数据源（先把 25 因子用透）
- ❌ LLM 新能力（已过拟合，降级为硬过滤）
- ❌ Intraday 策略（只做 intraday 交付）
- ❌ 衍生品 / 期权（600% 路径里再谈）
- ❌ Tick 级仿真（基建不足）
- ❌ Paper 双步闸门（已撤）
- ❌ 冻结实盘（已撤）
- ❌ 任何"系统看起来忙"的无效 step

---

**一句话总结 v3**：v2 把重心放错了位置（防御），v3 扳回进攻路线（Alpha + walk-forward）。治理/数据卫生继续走，但不再是主线。10 周目标是 3.9→5.5+ 分，600% 目标单独走附录 A 路径。
