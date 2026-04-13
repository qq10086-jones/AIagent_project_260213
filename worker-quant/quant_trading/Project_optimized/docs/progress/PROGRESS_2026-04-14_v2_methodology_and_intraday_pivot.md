# PROGRESS 2026-04-14 — v2 方法论修复 + 决策时点转向盘中

## 本阶段主题

从 "盘后自动成交" → "盘中 14:30 决策 + 用户实时跟单 SBI"。
同时把底层方法论（PIT、预注册、晋升作废）一并修干净，为后续策略优化提供可信地基。

---

## 已完成（可上线）

### 1. 方法论治理

| 任务 | 产物 | 说明 |
|---|---|---|
| v2 重构计划 | `docs/design/2026-04-13_quant_refactor_plan.md` | 4 Phase，10 周路线图；核心改动：walk-forward 把选择类决策嵌进训练窗 |
| 预注册协议 | `experiment_log.py`, `docs/design/experiment_log_schema.md` | 任何新因子/阈值/规则先写 JSONL 再跑；供 Deflated Sharpe / FDR 校正使用 |
| v1 晋升作废 | `archive_stale_promotion.py` | 原 `eligible_for_promotion` 归档 → `voided_awaiting_v2_gate`；原文件保留在 `reports/archive/voided_promotions_v1/20260413T144243Z/` |
| Reality check 四基准 | `reality_check.py`, `reports/reality_check_2026-04-13.md` | 策略 −0.01% vs TOPIX −3.21% vs 持仓等权 −11.12%（n=7 警告） |

### 2. PIT 数据卫生

| 任务 | 修改 | 验证 |
|---|---|---|
| 审计 47 处查询 | `docs/design/pit_audit_2026-04-13.md` | 6 处 leaky（3 P0 已修） |
| compute_ic.py:477 | 加 `WHERE date <= :asof` + asof_override 锚点 | 测试覆盖 |
| compute_price_features.py:load_prices_from_db | 加 `asof: Optional[str]` 参数 | 测试覆盖 |
| daily_run.py:_latest_fundamental_status | 加 `asof` 参数 + 4 处调用点透传 | 测试覆盖 |
| 不变性测试 | `tests/test_pit_guards.py` (5 测试) | 注入未来数据后输出不变 |
| 存量 feature_daily 污染核查 | `tools/pit_parity_check.py` → `reports/pit_parity_2026-04-08.md` | **零污染**，无需重建 |

### 3. Paper 闸门（已回滚默认行为）

原来在 T0.1 加了 `require_approval=true` 默认，导致决策更晚 —— 和用户"盘中出单"需求相反，**已回滚**。

| 文件 | 最终状态 |
|---|---|
| `paper_execute.py` | 代码保留 `--require_approval` / `--approve` / fingerprint / CAS，但默认关 |
| `config.yaml` | `paper.require_approval: false`（默认自动成交） |
| `daily_run.py` | awaiting_approval 分支保留但默认不触发 |

审批闸门基建留作可选（多用户/审计场景可开），不影响日常流水线。

### 4. 测试状态

- 总计 **164 / 164 PASS**（原 137 + 本阶段 27 新增）
- 新增测试集中在 `test_experiment_log`、`test_paper_approval_gate`、`test_reality_check`、`test_archive_stale_promotion`、`test_pit_guards`

---

## 遗留问题（优先级排序）

### P0 — 决策时点重构（本阶段核心遗留）

**问题**：`daily_run.py` 16:30 JST 启动（收盘后），`paper_execute` 收盘后回填成交，用户看到信号时**已来不及在 SBI 跟单**。

**期望**：
- 14:30 JST 盘中出决策（收盘前 30 分钟）
- 实时推送（Discord webhook / 文件 / 控制台）
- 用户手动下单到 SBI
- Paper 轨迹维持自动化（收盘后补成交即可）

**下一步任务**：
1. `intraday_decision.py` —— 14:30 触发，拉 intraday 价格 + factor_registry 权重 + 当前持仓，调用 `make_decision` 出单
2. Discord webhook 新事件类型 `intraday_decision`（已有基建 `_post_runtime_alert_webhook`）
3. Windows 任务计划器每交易日 14:30 触发
4. **T+1 回测**：近 20 个交易日回放，决策按 14:30 价而非 next_open 成交，产出对比 NAV 曲线 —— 这是后续所有策略调参的基线数据

### P1 — 买卖策略优化（用户原始需求）

必须在 P0 完成后启动，否则回测假设（`close → next_open fill`）和真实跟单节奏不一致。

候选方向（按 Codex 建议）：
- 周频调仓 vs 当前日频
- lot_size + cash 约束下的 tradable portfolio 验证
- 三基准对齐（TOPIX TR / 持仓等权 / 现金）

**硬性要求**：启动前所有新因子/阈值必须先走 `experiment_log.preregister()`。

### P2 — PIT 剩余漏洞（非阻塞）

审计报告剩 3 处 P1 + 2 处 P2：
- `market_data_utils.py:10,46` — `MAX(date)` 系统时钟；建议加 `@live_only` 注释
- `macro_event_detector.py:192` — `ORDER BY asof DESC LIMIT 5` 无 WHERE
- `cross_asset_signals.py:342` — `LIMIT 1` 无 asof 参数
- `factor_registry` 元数据语义（`compute_ic.py:530` / `evaluate_promotion.py:193`）
- `db_update.py:197` ETL 路径标注

### P3 — Phase -1 数据卫生剩余 7 项

v2 计划的 D-1 ~ D-7：
- D-1 survivorship bias（退市票）
- D-2 公司行动复权
- D-3 1321.T total-return（当前 reality_check 只用 price-return，低估 TOPIX 表现）
- D-4 TSE 手数单位历史快照
- D-5 停牌/涨跌停标记
- D-6 基本面 `available_at` vs `report_date`
- D-7 跨市场时区 UTC 化

这些不阻塞 P0/P1，可穿插进行。D-3 工作量最小、收益立即可见（reality_check 基准更诚实）。

### P4 — Paper 闸门配套（如果未来要启用）

当前 `require_approval=false`。若将来启用，尚缺：
- approval 审计 jsonl
- reject / cancel / void 路径
- `stale_orders_expired` 同步清理 `awaiting_approval`
- PENDING_APPROVAL.md 展示 expected_value

---

## 方向性说明

**项目定位未变**：alpha 研究 + 决策辅助。不是"已证伪的玩具"，也不是"可投产赚钱机"。当前阶段 = **数据采集期**（模型有基础可靠性，需更多样本验证），本阶段的治理修复是为这个数据采集期**把底层洗干净**，让后续策略优化可信。

**决策节奏转向**：系统每日 14:30 JST 出单 → 用户实时跟 SBI → 收盘后 paper 自动结算 + NAV 入 reconciliation。这是后续所有工作的前提节拍。

---

## 累计指标

- 新建文件: 12（代码 4 + 测试 5 + 文档 3）
- 修改生产代码: 5 个（paper_execute / config.yaml / daily_run / compute_ic / compute_price_features）
- 测试: 137 → **164** (+27)
- experiment_log 条目: 3（governance 类）
- 已生成报告: reality_check / pit_parity / pit_audit / promotion_voided

---

**下一次开工从 `intraday_decision.py` 开始。**
