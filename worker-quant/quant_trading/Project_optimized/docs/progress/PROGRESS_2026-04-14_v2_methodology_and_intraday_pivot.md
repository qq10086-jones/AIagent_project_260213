# PROGRESS 2026-04-14 — v3 转向 + intraday 生产线 + A-1 因子库启动

## 本阶段主题

两个并行工作面：
1. **治理底线**：继续修完 codex 提出的数据正确性/幂等/PIT 风险点
2. **策略进攻**：启动 v3 Phase A 主战场 —— Alpha-extended 因子库

**重要修正**：v2 偏向治理/防御，方向错位。v3 计划（`docs/design/2026-04-13_quant_refactor_plan.md`）把主战场扳回 Alpha 因子扩展 + walk-forward runner。对标 Qlib / Alphalens 评分 **3.9/10**，10 周目标 5.5+/10。

---

## 已完成（截至 2026-04-14 下午）

### 方法论治理（2026-04-13 遗留完成）

| 任务 | 产物 |
|---|---|
| v3 重构计划（3 版迭代，codex review 定稿） | `docs/design/2026-04-13_quant_refactor_plan.md` |
| 预注册协议 | `experiment_log.py` + `docs/design/experiment_log_schema.md` |
| v1 晋升作废 | `archive_stale_promotion.py` → `voided_awaiting_v2_gate` |
| Reality check 四基准 | `reality_check.py` + `reports/reality_check_2026-04-13.md` |
| PIT 审计 47 点 + 3 P0 修复 | `docs/design/pit_audit_2026-04-13.md`, `compute_ic.py`, `compute_price_features.py`, `daily_run.py` |
| PIT 不变性测试 + 实证无污染 | `tests/test_pit_guards.py`, `reports/pit_parity_2026-04-08.md` |

### Paper/SBI 轨道分离 (C-1)

| 任务 | 产物 |
|---|---|
| paper_execute 写 `_paper` 后缀 | `paper_execute.py` |
| 历史数据迁移脚本 + 已跑 | `tools/migrate_paper_strategy_id.py` (+ 已 apply) |
| News 采集 subprocess 编码修复 | `quant_briefing.py` (encoding=utf-8) |

### SBI 手动订单生命周期 (C-5 / C-5b)

| 任务 | 产物 |
|---|---|
| 挂单录入 CLI (place/list/cancel) | `tools/record_sbi_order.py` |
| 成交确认 CLI（幂等/partial） | `tools/record_sbi_fill.py` |
| Briefing SBI/paper 双栏 (C-2) | `quant_briefing.py` |
| 8 个生命周期单测 | `tests/test_sbi_order_lifecycle.py` |
| execution_report 按 strategy_id 隔离 | `execution_report.py` |

### Intraday 生产线 (C 阶段)

| 任务 | 产物 |
|---|---|
| `intraday_decision.py` 基础版 | 14:45 JST refresh intraday + target_weights → Discord webhook |
| 首次人机闭环验证 | 用户挂单 BUY 3041.T 400@¥585 依 14:45 信号 |

### A-1 Alpha-Extended 因子库（v3 Phase A 主战场启动）

| 任务 | 产物 |
|---|---|
| 因子目录骨架 | `factors/` 模块（momentum / lottery / liquidity / range_vol / registry） |
| **11 个学术因子**（每个带出处） | 见下表 |
| 18 单元测试 | `tests/test_alpha_extended_factors.py` |
| 11 因子 + 3 supersede 全部 preregister | `reports/experiment_log.jsonl` |
| Codex 2 轮审视 + 全部修复 | 见"Codex 累计处理"段 |

**当前 Alpha-Extended 因子清单**：

| 家族 | 因子 | 出处 |
|---|---|---|
| Momentum | `alpha_roc_3` / `alpha_roc_10` / `alpha_reversal_1` / `alpha_jt_mom_6m_skip1m` | Jegadeesh-Titman 1993 / Jegadeesh 1990 |
| Lottery | `alpha_max_ret_20` / `alpha_min_ret_20` | Bali-Cakici-Whitelaw 2011 |
| Higher moment | `alpha_ret_skew_60` | Harvey-Siddique 2000 |
| Liquidity | `alpha_amihud_20`（支持 exchange turnover override） | Amihud 2002 |
| Range vol | `alpha_range_proxy_20` / `alpha_hl_ratio_20` / `alpha_parkinson_vol_20` | Parkinson 1980 / Alizadeh-Brandt-Diebold 2002 |

---

## Codex 累计处理（4 轮审视）

| 轮 | 提出 | 修 / 接受 / 延后 |
|---|---|---|
| 1 (C-1/C-5 初版) | 5 | 3 修 / 2 延后 |
| 2 (C-5 round-2) | 5 | 3 修 / 2 延后 |
| 3 (A-1 初版) | 6 | 4 修 / 2 接受 |
| 4 (A-1 复核) | 6 | 3 修 / 3 延后 |
| **合计** | **22 项** | **13 修 + 3 接受 + 6 延后** |

### 本轮处理的关键数学/归因错误（A-1）

- **JT momentum 归因错**：`alpha_roc_120` 不是 Jegadeesh-Titman 经典 → 改为 `alpha_jt_mom_6m_skip1m`（window=120 skip=21），跳过最近 21 天避免 ST reversal 污染
- **Parkinson 公式错**：原 `(H-L)/close` 不是 Parkinson variance → 新增 `alpha_parkinson_vol_20` 严格按 `sqrt(mean(ln(H/L)²)/(4 ln 2))`；原 proxy 改名 `alpha_range_proxy_20` 并修正 citation
- **Amihud 复权风险**：添加 `dollar_volume` 参数 + 复权警告 + registry 层 `optional_kwargs` 透传机制
- **Rolling min_periods**：收紧到 `=window`（防止早期窗口分布偏差）

### 本轮延后（Phase A-2/B 范畴）

- JT 21/120 交易日近似 vs 月度组合 → Phase B 统一
- Amihud 横截面可比性（停牌股估计方差不同）→ A-2 加 valid_count
- Range 家族高度同源 → A-2 cross-sectional rank + 正交化
- execution_report / post_trade / app CLI 传递 strategy_id
- evaluate_promotion paper_days 按 strategy_id 过滤（v1 已 void，非阻塞）

---

## 关键数据 + 账户状态

### 测试规模
- 初始 137 → **192**（+55，全 PASS）

### 真实 SBI 账户（2026-04-14）
- NAV: **¥400,545** 全现金（2026-04-10 后）
- **Pending 挂单**：BUY 3041.T 400@¥585 (order_id `2026-04-14__sbi__f1cba08b79`, status=open)
  - 按 14:45 模型信号，首次人机闭环
  - 盘中 ¥588 越过限价，成交概率 ~40%

### Paper 轨道（与 SBI 物理分离）
- strategy_id=`sprint_paper`
- 2 只历史 paper 持仓：3041.T + 7984.T（2026-04-10 进的 paper 单）

### experiment_log 条目
- 原 2 条（governance）+ 11 因子 + 3 supersede + 2 A-1 revision = **18 条**

### Git 推送
- 11 次 commit，main 分支，全部 push 到 GitHub

---

## 遗留问题（优先级排序）

### P0 — Phase A 继续推进（本周主线）

- [ ] **A-2 Cross-sectional rank** — factors.transforms 模块，日内横截面 rank 到 [0,1]
- [ ] **A-3 行业中性化** — 每因子减去 TSE 33 业种中位数（需要先有业种映射表）
- [ ] **A-4 Winsorize + z-score 标准化**
- [ ] **A-5 因子正交化 / 相关性剪枝**（range 家族、reversal vs roc_1 高度同源）
- [ ] **集成到 feature_daily 流水线** — `compute_alpha_extended_features.py`

### P1 — Phase B Walk-Forward Runner

- [ ] `walk_forward_runner.py` — 3 年训练 / 6 月验证 / 1 月滚动
- [ ] Newey-West t-stat + Block bootstrap CI
- [ ] Ledoit-Wolf 协方差 shrinkage
- [ ] T+1 open + square-root slippage

### P2 — Phase D 统计/风险

- [ ] Deflated Sharpe Ratio（读 experiment_log 取 N）
- [ ] FDR (Benjamini-Hochberg)
- [ ] Barra-lite 5 因子风险分解

### P3 — Intraday 生产线完善

- [ ] C-3 Windows Task Scheduler 14:45 JST 自动触发
- [ ] C-4 yfinance 熔断 + 回退昨收
- [ ] post_trade / sync_broker_fills / app CLI 传 strategy_id
- [ ] evaluate_promotion paper_days 按 strategy_id 过滤（v1 已 void，低优）

### P4 — Phase -1 数据卫生剩余

- [ ] D-3 1321.T total-return 序列
- [ ] D-6 基本面 available_at vs report_date 审计
- [ ] D-1 Survivorship / 退市票
- [ ] D-2 / D-4 / D-5 / D-7 剩余 4 项

---

## 当前项目评分（自评 + Codex 审视）

| 维度 | 当前 | 10 周目标 |
|---|---|---|
| 数据覆盖 | 5/10 | 6/10 |
| **因子库** | **3/10 → 4/10** (A-1 初阶完成) | **5/10** |
| 信号模型 | 4/10 | 5/10 |
| **回测严谨性** | **2/10** | **6/10** (Phase B) |
| 统计验证 | 2/10 | 6/10 |
| 组合构造 | 3/10 | 5/10 |
| 执行建模 | 3/10 | 5/10 |
| 实盘集成 | 6/10 | 7/10 |
| 治理/可复现 | 5/10 → **6/10** | 7/10 |
| 运维成熟 | 6/10 | 7/10 |

**总分 3.9/10 → 本阶段 ≈ 4.2/10**（治理+1，因子库 0→1）。核心提升仍靠 Phase B walk-forward。

---

## 方向性原则（不变）

- 模型经多轮认证，**治理修复不质疑模型**
- 14:45 JST 出单 + Discord + 用户跟 SBI 是核心节拍
- "不发明因子"：每个新因子必须有 published 出处 + experiment_log 预注册
- Paper / SBI 物理分离，absolute 诚实对账

---

**下一次开工从 A-2 cross-sectional rank + 集成到 feature_daily 流水线开始。**
