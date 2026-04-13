# Quant 系统方法论改造计划 v2

**日期**: 2026-04-13（v2 并入 Codex review 反馈）
**立场**: 从"工程能跑"转向"统计能站住"
**核心修正**: v1 把因子筛选放在 walk-forward 之外 → 二次过拟合。v2 把**所有自由度决策塞进 walk-forward 训练窗口**，验证窗口只读冻结产物。

---

## 一、指导原则（贴墙上）

1. **没有 benchmark 不算策略** —— 每份报告带多基准：TOPIX total-return / 可投资 universe 等权 / sector-neutral / beta-adjusted alpha
2. **没有置信区间的 IC 都是自欺** —— IC/ICIR 必须带 Newey-West t-stat + block-bootstrap CI
3. **Out-of-sample 是物理隔离** —— 因子筛选、权重、阈值、regime 规则、新闻过滤**全部**只能用训练窗数据
4. **多重检验必须纳入** —— 每次尝试写入 `experiment_log.jsonl`；最终评估用 **Deflated Sharpe / White's Reality Check / FDR** 校正
5. **先打平多基准再谈 alpha** —— 打不平就关机
6. **数据卫生优先于模型** —— 日本市场 idiosyncratic pitfalls（survivorship / 复权 / 手数 / 停牌）在建模前解决

---

## 二、Phase 结构（改为并行+嵌套，不再线性）

```
Phase -1 (数据卫生)  ─────► 必须先完成，否则一切白搭
        │
        ▼
Phase 0 (止血诊断) ──┐
                    ├─► Phase A (Walk-Forward Pipeline, 贯穿)
Phase -1 交付物 ────┘      ├─ 因子筛选（训练窗内）
                           ├─ 正交化 + 权重（训练窗内）
                           ├─ 组合构造（训练窗内）
                           └─ 验证窗评估（只读冻结产物）
                    │
Phase B (LLM 降级) ─┤   并行
Phase C (执行建模) ─┘
                    │
                    ▼
          10 周 gate: Research-Framework Gate
                    │
                    ▼  (通过后才进入)
          Live-Promotion Gate (独立时间线, ≥6 个月 paper-live reconciliation)
```

---

### Phase -1: 数据卫生（2-3 周，必须先做）
**目标**：消除日本市场特有的 bias 与数据陷阱。没有这一步，后面所有 IC 都是垃圾。

**任务**
- **D-1** survivorship bias 审计：确认数据库包含已退市票；补齐历史成分股快照
- **D-2** 公司行动复权：拆股/合股/股权分红的价格&数量复权链路校验
- **D-3** 分红处理：构造 1321.T **total-return** 序列（非 price-return）作为真 benchmark
- **D-4** TSE 手数单位：每只票的历史 `lot_size` 快照（部分票曾从 1000→100 变更）
- **D-5** 停牌 / 涨跌停 / 特别气配：在 fills 建模时标记**不可成交状态**，禁止假装成交
- **D-6** 财报发布日期滞后：基本面因子的 `available_at` 必须用实际公告日而非 `report_date`
- **D-7** 节假日对齐 + 跨市场时区：日经收盘/美股开盘/WTI 时间戳全部 UTC 化
- **D-8** PIT (point-in-time) 数据库快照：每个历史 asof 能重放当时可见的数据

**Exit**：所有 D-x 有单元测试，任给 `asof`，查询返回结果 ≡ 当时可见数据。

---

### Phase 0: 止血与诊断（1 周，可与 Phase -1 并行）

- **T0.1** 禁用 `paper_execute` 自动成交 → 改 `proposed` 状态待人工确认（也解决用户的盘后知情问题）
- **T0.2** 产出 `reports/reality_check_2026-04-13.md`：2026-02 起 NAV 扣费后 **vs 四条基准**
  - 1321.T total-return
  - 可投资 universe 等权
  - sector-neutral 等权
  - 现金持有（零风险参照）
- **T0.3** 当前 shadow_ic 晋升结果**全部作废并归档**（30 天 OOS 方法论不成立）
- **T0.4** 建 `experiment_log.jsonl`：从今天起任何参数/因子/阈值尝试必须落此日志（供后续 Deflated Sharpe 校正）

---

### Phase A: Walk-Forward Pipeline（4-6 周，核心重构）
**核心**：把 v1 中 Phase 1/2/3 顺序执行的错误改正 —— **所有自由度决策嵌入训练窗**。

**管道结构（每个 walk-forward 窗口内完整跑一遍）**

```
for each (train_window, validation_window) in rolling_splits:
    # ===== 训练窗 (只看 train_window 数据) =====
    1. 全因子 IC 诊断 (Newey-West + block bootstrap)
    2. 单因子筛选 (|t-stat|>2, half-life≥2日, FDR 校正)
    3. 相关性剪枝 (|corr|>0.7 剔除)
    4. Gram-Schmidt 正交化
    5. Marginal IC 选 ≤5 因子
    6. 权重确定 (equal-weight 或 max-IR with Ledoit-Wolf shrinkage)
    7. 组合构造参数 (目标持仓数 / 行业约束 / turnover 上限) 冻结
    8. 成本模型参数 (spread proxy, ADV participation) 校准

    # ===== 验证窗 (只读冻结产物，禁止回看调参) =====
    9. 用 1-8 的冻结规则生成信号
    10. 构造 research_portfolio (满配 N≥15)
    11. 构造 tradable_portfolio (lot_size + 停牌 + cash 约束下 top-K)
    12. 记录两条轨迹 + 多基准超额

# 聚合所有 validation_window 的 IR / 收益 → 真 OOS 指标
```

**任务**
- **A-1** `walk_forward_runner.py`：3 年训练 / 6 月验证 / 1 月滚动
- **A-2** Newey-West IC 工具 + 自相关修正 t-stat
- **A-3** Block bootstrap CI（block 长度 = half-life × 3）
- **A-4** FDR (Benjamini-Hochberg) + Deflated Sharpe 实现
- **A-5** 因子正交化 + marginal IC (全部在训练窗内)
- **A-6** `research_portfolio` / `tradable_portfolio` 双轨生成器
- **A-7** Ledoit-Wolf 协方差 shrinkage
- **A-8** 行业暴露约束 (行业数据源锁定 TSE 33 业种)
- **A-9** 调仓频率：默认**周频**，在 walk-forward 内对比周频/双周频/月频
- **A-10** 组合级回撤 kill-switch（保留），**个股 kill-switch**（停牌/退市/财务造假/跌停无法出）而非 ATR 止损

**Exit**：聚合 walk-forward 指标
- 净 IR > 0.5 且 t-stat > 2（用 block bootstrap 算）
- **四基准全部跑赢**（TOPIX TR / universe 等权 / sector-neutral / 现金）
- tradable_portfolio 与 research_portfolio 的 IR 差距 < 50%（否则说明 lot/cash 约束吃掉 alpha）

---

### Phase B: LLM / 宏观层降级（并行）

**定位**：LLM 与新闻**不是 alpha 源**，只能做风控硬过滤。

- **B-1** LLM regime 只影响**全局仓位系数**（降仓/维持），**不**进选股/因子权重
- **B-2** 新闻过滤：negative_news 剔除个股 → 二选一预注册：
  - (a) 过滤前 vs 过滤后两条轨迹都保留，晋升看过滤后
  - (b) 阈值规则冻结进训练窗，验证窗不得调整
- **B-3** LLM 输出的样本**从 IC 统计中剔除**（避免 universe selection 污染）
- **B-4** regime 阈值、新闻类别、负面定义全部写进 `docs/design/risk_filter_spec.md` **预注册**

---

### Phase C: 执行建模（并行，2-3 周）
**定位**：daily OHLCV 近似，明确声明**不是 tick 级仿真**。

- **C-1** Paper 成交改 **T+1 open** 填充，slippage = `α × (qty / ADV_20)` 简单 square-root 近似
- **C-2** 成本模型明确单位：
  - `alpha`: 训练窗内估计的**周频超额收益预期** (bp)
  - `cost`: `fee_bp + half_spread_bp + impact_bp(qty/ADV)`
  - 信号入池条件：`expected_alpha_bp > 2 × cost_bp`（双倍 margin）
- **C-3** `research_portfolio` / `tradable_portfolio` / `actual_live_portfolio` **三条轨迹 reconciliation 日报**
- **C-4** 限制：日成交上限 = 20% × 该票 20 日 ADV（硬约束）

---

## 三、10 周 Gate：**Research-Framework Gate**（非 Live Gate）

10 周后只判以下**三个二元问题**，全 YES 才进入 Live Gate 倒计时；不是"是否上 live"。

1. **Phase -1 数据卫生是否干净？** —— D-1 到 D-8 全部通过单元测试
2. **Walk-forward 聚合是否显著跑赢四基准？** —— 净 IR > 0.5，block-bootstrap 95% CI 下界 > 0
3. **Deflated Sharpe 是否仍然显著？** —— 校正 `experiment_log` 尝试次数后，DSR > 0.95

### Live-Promotion Gate（独立时间线）
- Research-Framework Gate 通过后，**冻结全部参数**，启动 paper-live reconciliation
- 至少 **6 个月** `tradable_portfolio` 与 `actual_live_portfolio` 日度对账
- Paper vs live IR 差 < 30% 且同向
- 期间**任何参数改动回滚到 Phase A**，paper 计时归零

---

## 四、任务清单（按依赖排序）

### P-1 数据卫生（阻塞一切）
- [ ] D-1 Survivorship 审计 + 退市票补齐
- [ ] D-2 公司行动复权链路验证
- [ ] D-3 1321.T total-return 序列构造
- [ ] D-4 每票历史 lot_size 快照
- [ ] D-5 停牌/涨跌停/特别气配标记
- [ ] D-6 基本面 `available_at` = 实际公告日
- [ ] D-7 跨市场时区 UTC 化
- [ ] D-8 PIT 数据库快照 + asof 查询测试

### P0 止血（本周）
- [ ] T0.1 paper_execute 改 proposed 状态
- [ ] T0.2 `reality_check_2026-04-13.md`（四基准对比）
- [ ] T0.3 shadow_ic 历史晋升结果作废
- [ ] T0.4 `experiment_log.jsonl` 启用（预注册机制）

### PA Walk-Forward Pipeline
- [ ] A-1 `walk_forward_runner.py`
- [ ] A-2 Newey-West IC 工具
- [ ] A-3 Block bootstrap CI
- [ ] A-4 FDR + Deflated Sharpe
- [ ] A-5 训练窗内因子筛选+正交化+marginal IC
- [ ] A-6 research / tradable 双轨组合
- [ ] A-7 Ledoit-Wolf 协方差 shrinkage
- [ ] A-8 TSE 33 业种暴露约束
- [ ] A-9 调仓频率 grid（周/双周/月）
- [ ] A-10 组合 kill-switch + 个股 idiosyncratic kill-switch

### PB LLM 降级（并行）
- [ ] B-1 regime 只动全局仓位
- [ ] B-2 新闻硬过滤（预注册+双轨对比）
- [ ] B-3 LLM 样本剔除出 IC 统计
- [ ] B-4 `risk_filter_spec.md` 预注册文档

### PC 执行建模（并行）
- [ ] C-1 T+1 open + square-root slippage
- [ ] C-2 成本-alpha gate（单位明确 + 双倍 margin）
- [ ] C-3 三轨道 reconciliation 日报
- [ ] C-4 日成交 ≤ 20% ADV 硬约束

### PR 报告层（收尾）
- [ ] R-1 Briefing 首段 = 四基准对比 + DSR
- [ ] R-2 每因子展示 t-stat + block-bootstrap CI
- [ ] R-3 三轨道轨迹偏离日报

---

## 五、时间表

| 周 | Phase -1 | Phase 0 | Phase A | Phase B | Phase C |
|----|----------|---------|---------|---------|---------|
| W1 | D-1~D-3  | T0.1~T0.4 |          |         |         |
| W2 | D-4~D-6  |         |  A-1~A-2 |  B-4    |         |
| W3 | D-7~D-8  |         |  A-3~A-4 |  B-1~B-3|  C-1    |
| W4 |          |         |  A-5     |         |  C-2    |
| W5 |          |         |  A-6~A-7 |         |  C-3~C-4|
| W6 |          |         |  A-8     |         |         |
| W7 |          |         |  A-9     |         |         |
| W8 |          |         |  A-10    |         |         |
| W9 |          |         |  聚合验证|         |         |
| W10|          |         | **Research-Framework Gate** |  |  |

通过后 → Live-Promotion Gate ≥ 6 个月（与 10 周时间线解耦）。

---

## 六、不做什么（明确划线）

- ❌ 不在 walk-forward 之外筛因子 / 调权重 / 选阈值
- ❌ 不用 TOPIX 单一 benchmark
- ❌ 不把 10 周当 live gate
- ❌ 不加新因子、不加新 LLM 能力、不做 intraday/tick、不做期权、不上杠杆
- ❌ 不把 Barra 简化成"市场+行业+size"还自称 Barra —— 我们用的就叫 **暴露回归 v1**
- ❌ 不删个股 kill-switch（保留 idiosyncratic tail 防线）
- ❌ 不保留任何"系统看起来忙"的无效 step

---

## 七、预注册协议（Preregistration）

从 2026-04-13 起：

1. 任何**新因子 / 新阈值 / 新规则**在**被测试前**写入 `experiment_log.jsonl`
2. 同一假设修改 > 3 次触发 "paradigm-shift flag"，该因子族**在当前 walk-forward 周期内作废**
3. 年度做一次 Deflated Sharpe 重算，所有正在运行策略**重过**晋升闸门

---

## 八、结论

v1 的致命错 = 把方法论改造本身搞成了过拟合工程。v2 核心就一件事：**把所有涉及选择的自由度塞进训练窗**，外面只剩验证。

做减法（5 因子、周频、单一 universe、LLM 降级）+ 数据卫生先行 + 多重检验校正 + 两层 gate。

两个可能结局：
- **结局 A**：walk-forward 聚合 IR 显著 → 进入 6 个月 live reconciliation → 可能跑出真 edge
- **结局 B**：跑不赢四基准 / DSR 不显著 → 项目**干脆地证伪**，转为数据/工程能力库

两个都比现在强。
