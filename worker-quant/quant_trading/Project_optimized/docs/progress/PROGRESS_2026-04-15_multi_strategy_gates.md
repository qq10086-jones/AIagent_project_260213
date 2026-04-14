# PROGRESS 2026-04-15 — Multi-strategy registry + capital gates + P0 fail-closed

## 本阶段一句话

接续 04-14 的 walk-forward verdict，搭完**多策略 registry + 四道资本门槛 + fail-closed 安全层**，并把 sprint 日频下单模式降级为 **monthly-only + watchlist**。系统现在是一个**有证据门槛的 senior analyst**，用户是 PM（真金 100% 手动 SBI 执行，系统不自动下单）。

---

## 今日交付（1 次 commit: `3ee48a8`）

### 新增文件（11）

| 文件 | 职责 |
|---|---|
| `walk_forward_runner.py` | 月频 OOS 回测，支持 ablation / 合成 / 单因子 / long-short / T+1 open |
| `analyze_amihud_robustness.py` | Block-bootstrap CI + Deflated Sharpe Ratio |
| `strategy_registry.py` | 11 策略单一声明点（3 real + 8 paper）+ evidence |
| `capital_gate.py` | G1-G4 门槛评估 + audit JSON |
| `execution_gap_report.py` | real vs paper vs WF-expected 三轨 |
| `strategy_dashboard.py` | 统一每日视图 |
| `run_alpha_factor_decision.py` | 月频 decision emitter，支持 rotation SELL + fail-closed gate |
| `run_all_strategies.py` | 顶层 orchestrator（daily_run + 所有 real/paper + reports）|
| `run_all_papers.py` | 多 paper 批量调度（骨架） |
| `monthly_rebalance.py` | 月首真金建议生成 |
| `scheduled_all_strategies.cmd` + `scheduled_monthly_rebalance.cmd` | Windows Task Scheduler wrappers |
| `backfill_jquants_history.py` (stub) | 等 JQuants 凭证的 placeholder |
| `tests/test_capital_gate_failsafe.py` | 6 个 P0 回归测试 |

### 编辑（4）

- `sprint_signal.py`：`vol_z` 从 alpha 合成移除（保留为 entry filter）
- `kelly_sizer.py`：fallback 25% → 0（no-evidence no-position）+ 新增 `--kelly_bootstrap_pct`
- `make_decision.py`：capital_gate 挂钩 **fail-closed**（异常/未注册/paused 都拦 BUY，SELL 永远允许）+ `--bypass_capital_gate`
- `intraday_decision.py`：默认 `--watchlist_only`（不再自动推 BUY 建议到 Discord）

---

## Walk-forward OOS 证据（2024-01 → 2026-04, 23 个月）

### 三因子 sprint ablation

| 配置 | net cum | Sharpe | MaxDD | vs EW 月度 |
|---|---|---|---|---|
| 原 sprint 三因子（mom+hi52w+vol_z）| +14.3% | 0.40 | −18.5% | **−0.43%** |
| **mom_consist + high52w（移除 vol_z）** | **+26.1%** | **0.85** | −13.7% | **−0.13%** |
| **high52w 单因子** | **+30.1%** | **0.88** | −13.1% | **+0.02%** (≈EW) |
| 1321.T 基准 | +39.5% | 1.05 | −9.5% | — |
| 宇宙等权基准 | +31.7% | 1.48 | −7.9% | — |

### Amihud k=20 稳健性（stress matrix）

| ADV ≥ | slip bps | net cum | Sharpe | P(cum>0) | DSR p |
|---|---|---|---|---|---|
| 100M | 30 | +112% | 0.84 | 91% | **0.000** |
| 100M | 50 | +100% | 0.79 | 88% | 0.000 |
| 100M | 100 | +73% | 0.67 | 81% | 0.000 |
| 50M | 50 | +26% | 0.61 | 74% | 0.000 |
| 50M | 100 | **+7%** | 0.26 | 56% | 0.000 |

### 决定性 OOS split

| 窗口 | N | cum_net | Sharpe | 结论 |
|---|---|---|---|---|
| 2024-01 → 2026-04 IS | 23 | **+100%** | 0.79 | 看起来好 |
| **2023-04 → 2024-01 OOS** | 6 | **−3.1%** | **−0.52** | **edge 不存在** |

**核心发现：所有 30+ 个配置 DSR p=0.000** — 多重检验修正后**完全不显著**。N=23 月样本不够。

---

## 策略分配（post-verdict）

| strategy_id | tier | cap | state | G1 evidence.monthly_excess_vs_ew |
|---|---|---|---|---|
| `sprint` | real | ¥250k (已占用 3041.T) | **paused_sunk_only** | −0.13% (FAIL) |
| `high52w` | real | ¥100k | active | +0.02% |
| `amihud` | real | ¥100k | active | +1.87% (IS，已知 2023 OOS 负)|
| `sprint_paper` | paper | — | active | — |
| `high52w_paper` | paper | — | active | — |
| `amihud_paper` | paper | — | active | — |
| `amihud_k5_paper` / `k30_paper` / `adv50_paper` | paper | — | active | 参数敏感性探测 |
| `min_ret_paper` | paper | — | active | defensive 因子 |
| `mom_high52w_paper` | paper | — | active | 2f 组合对比 |

---

## 四道门槛（capital_gate.py）

- **G1 ENTRY**：walk-forward 月超额 vs EW > 0（real 准入）
- **G2 RETENTION**：rolling 3-month real PnL ≥ 0（留在 real）
- **G3 KILL**：MaxDD > 15% OR 3m PnL < −5% **OR since-inception < −5%**（< 3 月时）
- **G4 PROMOTION**：paper 6m Sharpe > 0.5 AND DSR p < 0.10（paper→real）

**今日 gate 状态**：sprint G1=FAIL（已 paused），high52w/amihud G1=OK active。

---

## P0 安全修复（Codex round-5 审视后）

| # | 修复 | 位置 |
|---|---|---|
| 1 | make_decision gate **fail-closed**，异常时拦 BUY | `make_decision.py:1310` |
| 2 | run_alpha_factor_decision **前置 gate + fail-closed** | `run_alpha_factor_decision.py:main` |
| 3 | **DestructiveRerunError**：已成交/filled 订单不允许覆盖 | `run_alpha_factor_decision._write_decision` |
| 4 | **Early-inception kill switch**（< 3 月 NAV 时 since-inception < −5% 触发） | `capital_gate.py:G3` |
| 5 | **Rotation SELL 逻辑**：monthly 重平衡自动 SELL 跌出 top-K 的持仓 | `run_alpha_factor_decision` |

### 6 个 P0 回归测试（`tests/test_capital_gate_failsafe.py`）

- G3 early-inception trip on inception <-5%（< 3m 数据）
- G3 OK when loss within tolerance
- Destructive rerun refused on filled orders
- Destructive rerun refused when fills exist
- Benign rerun allowed (only proposed orders)
- Rotation SELL exists in code path

**测试总数 193 → 199 全 pass**。

---

## 任务 schedule 调整

| 时间 (JST) | 任务 | 变化 |
|---|---|---|
| 09:00 | morning_briefing（旧） | 无变化 |
| **10:00** | **monthly_rebalance.py** (NEW) | 写月首真金建议到 DB（status=proposed），用户手动 SBI 执行 |
| 14:45 | intraday_decision.py | **降级 watchlist_only**，不再 Discord push BUY 订单 |
| **17:00** | **run_all_strategies.py** (NEW) | 跑 daily_run + 所有 real/paper alpha + gate + reports |

- 旧 `scheduled_daily_run.cmd` 建议禁用（其 work 现在由 `run_all_strategies.py` step 1 调用）
- 挂 scheduler 的 PowerShell 命令见 `docs/scheduler_setup.md`

---

## 重要职责边界澄清

**系统职责（自动可跑）**：
- 产生月频建议（写 `orders` 表 status='proposed'）
- paper PnL 仿真（写 `fills` 表）
- Dashboard + gate state + execution gap report
- Discord 推送 watchlist（默认不推订单）

**用户职责（手动）**：
- 审视 dashboard 决定是否执行建议
- 在 SBI 手动下单（限价/市价）
- `record_sbi_order` / `record_sbi_fill` CLI 回录
- 设 ATR 止损（系统不自动止损）
- 最终 override 决定（可 `--bypass_capital_gate`）

**系统从不**：直接触达 SBI、代替用户下单、在无人值守的情况下动真金。

---

## 实盘状态（2026-04-15 EOD）

- **SBI 真账户 NAV**: ¥400,145
- **现金**: ¥166,545
- **持仓**: 3041.T × 400 @ ¥585（2026-04-14 成交，sunk position）
- **挂单**: 无
- **建议动作**: 设 ¥550 硬止损（ATR −6%）在 SBI 端

---

## 遗留任务（优先级排序）

### P1（本周）

- [ ] `dsr_p=0.0` 语义修（0.0 < 0.10 反向激励 G4 promotion）—— 应改 None + 要求 n_periods 阈值
- [ ] `execution_gap_report` inception 对齐——real 与 paper 起止不同，gap 当前 misleading
- [ ] `intraday_decision` breaking change 说明（legacy .bat 可能期望旧行为）
- [ ] evidence 自动化：WF runner → registry 同步（现在手填）

### P2（下周+）

- [ ] JQuants 凭证 + 2018-2023 回填（样本 23 → 100 月）
- [ ] Survivorship 审计（对退市票）
- [ ] `walk_forward_runner` 加 purge + sqrt impact
- [ ] Briefing v2 集成 dashboard

### P3（月+）

- [ ] 6 个月 paper shadow 数据积累
- [ ] Paper→real G4 promotion 首次评估
- [ ] 如果 amihud G3 kill → 重新分配 ¥100k 给下一个候选

---

## 关键结论

1. **600% 年化目标**被降级为"挑战目标"，项目实际可达**年化 15-45%**（乐观 IS，诚实 OOS 约指数水平）
2. **¥400k 定义为 R&D budget**（可承受损失），但用户要求真钱必须带正期望 → sprint 被 gate 按住不出新建议
3. **Paper + real 并行**而不是"paper 先验证再真金"——paper 主要价值是 augmentation 而非 validation
4. 当前系统 **安全可自动跑**（因为只产建议），但**任何真金动作必须用户主动**

---

## 下次会话建议起点

- 让系统跑 1-3 天真实样本，观察 dashboard 是否按预期显示
- 如果有出现 sprint 误推 BUY / amihud 异常回撤，review gate 日志
- 补 P1 的 dsr_p + execution_gap inception 修
- JQuants 凭证到手后启动 backfill

---

**Commit**: `3ee48a8` `quant(v3.8): walk-forward verdict + multi-strategy gates + P0 fail-closed`
**Tests**: 199/199 PASS
**New files**: 13 | **Edited**: 4
