# Worker-Quant — Universe 扩容与选股 Alpha 管线重构

**作者**: Senior Quant Architect
**创建日期**: 2026-04-10
**状态**: DRAFT — 待审批
**对应任务清单**: `../tasks/TASKS_2026-04-10_Universe_Expansion_Alpha_Pipeline.md`
**上承文档**: `PATCH_2026-04-08_Model_Alpha_Optimization.md`, `DESIGN_v3.0_Dual_Strategy_Architecture.md`
**触发原因**: 选股管线审计发现系统性架构缺陷 — Universe 硬编码 97 只蓝筹，Screener 不含 alpha 信号，Sprint 动量策略在窄截面上统计失效，导致 entry_ok 长期全 False。

---

## 0. 问题审计

### 0.1 当前选股管线结构性评分

| 层级 | 组件 | 现状 | 评分 | 核心缺陷 |
|------|------|------|------|----------|
| L0 Universe | `db_update.py TARGET_UNIVERSE` | 硬编码 97 只，人工挑选 | **1/10** | 搜索空间砍掉 97%，alpha 上限被锁死 |
| L1 Screener | `screener.py screen()` | `log1p(ADV) - |log(vol)-log(0.02)|` | **2/10** | 流动性排序，零预测能力 |
| L2 Sprint 信号 | `sprint_signal.py sprint_score()` | 3 因子等权，截面 ~13 只 | **3/10** | 百分位在 13 只里无统计意义 |
| L3 Entry Filter | `sprint_entry_check()` | `mom_consist_pctile >= 0.80` | **2/10** | 97 只蓝筹天然低动量，长期 0/13 通过 |
| L4 Regime | V2 连续化 + 宏观事件 | 已完成 | **8/10** | 好，但上游没有喂给它合格的候选 |
| L5 风控 | ATR 止损 + trailing + 组合回撤 | 已完成 | **8/10** | 好 |
| L6 IC 学习 | Spearman IC + EWMA + 因子注册表 | 架子对 | **5/10** | 截面 25-50 只时 IC 信噪比极低 |

### 0.2 数学论证：为什么 97 只不够

**动量策略的统计前提**：

截面选股依赖尾部分布。假设收益率 r ~ N(0, σ)，从 N 只中选 top-k 的期望超额收益：

```
E[r_top_k] ≈ σ × Φ^(-1)(1 - k/N)
```

| Universe N | top_k=3 | 期望超额 (σ=2%) |
|------------|---------|-----------------|
| 97 | 3 | σ × 1.63 = 3.26% |
| 500 | 3 | σ × 2.43 = 4.86% |
| 1000 | 3 | σ × 2.78 = 5.56% |
| 1500 | 3 | σ × 2.97 = 5.94% |

**N=97 → N=1000 的 alpha 天花板提升约 70%**。这不是优化，这是修复。

**Spearman IC 精度**：

IC 的标准误 ≈ `1/sqrt(N)`。N=50 时 SE=0.14，N=500 时 SE=0.045。当前 IC 估计的置信区间是合理值的 3 倍宽，因子权重学习几乎是噪声拟合。

### 0.3 Screener 评分函数审计

当前 `screener.py:334-340`:
```python
score += np.log1p(max(m_adv, 0.0))                        # 流动性越大越好
score -= abs(np.log(max(m_vol, 1e-8)) - np.log(0.02))     # 波动率越接近2%越好
score -= m_missing * 50.0                                   # 数据完整性
```

**诊断**：这是一个数据质量排序函数，不是 alpha 排序函数。它回答的问题是"哪只股票的数据最好"，而不是"哪只股票最可能涨"。在正确的架构中，这类过滤应该是硬性门槛（pass/fail），不应该参与评分排序。

---

## 1. 设计目标

### 1.1 核心目标

| ID | 目标 | 量化指标 |
|----|------|----------|
| G-1 | 扩大 alpha 搜索空间 | Universe ≥ 500 只（TSE Prime + Standard 高流动性） |
| G-2 | 引入 alpha 预筛选 | Screener 评分包含至少 1 个 alpha 因子（动量/价值） |
| G-3 | 恢复截面统计有效性 | Sprint 截面 ≥ 50 只，entry_ok 通过率 ≥ 5% |
| G-4 | 向后兼容 | 不破坏现有 regime/风控/IC 学习/daily_run 管线 |
| G-5 | 数据源免费 | 仅用 yfinance / J-Quants Free / 公开数据 |

### 1.2 非目标（本轮不做）

- Harvest 策略激活（等 NAV 达标）
- sprint_score_v2 上线（等 IC 数据积累）
- 多市场扩展（美股/港股）
- 实时盘中信号（仍为收盘后批量）

---

## 2. 架构设计

### 2.1 新增组件

```
                        ┌─────────────────────┐
                        │  universe_builder.py │  ← 新增：动态 Universe 构建器
                        │  (周末/周一运行)      │
                        └──────────┬──────────┘
                                   │ universe.json (500-1000 只)
                                   ▼
┌──────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│ db_update.py │ ←──│ universe.json        │    │ config.yaml     │
│ (每日 16:30) │    │ + TARGET_UNIVERSE    │    │ universe_file:  │
└──────┬───────┘    │   (保留为 fallback)  │    │  universe.json  │
       │            └──────────────────────┘    └─────────────────┘
       ▼
┌──────────────┐    评分重构:
│ screener.py  │    hard_filter(ADV, vol, missing, lot_cost)
│ (每日)       │ →  alpha_score = w1×mom_20 + w2×reversal + w3×quality
└──────┬───────┘    output: top_k=50 (Sprint) / top_k=100 (Harvest)
       │
       ▼
┌──────────────────┐
│ sprint_signal.py │  截面 50+ 只，百分位排名恢复统计意义
│ (不改核心逻辑)    │  entry_ok 通过率从 ~0% → ~5-15%
└──────────────────┘
```

### 2.2 universe_builder.py — 动态 Universe 构建器

**职责**：每周构建一次 Universe，输出 `universe.json`

**数据源优先级**：
1. **yfinance screener** — `yf.screen()` 或遍历已知 TSE 代码范围
2. **J-Quants Free API** — 如可用，直接拉取上市公司列表
3. **静态种子列表** — 当前 97 只作为 fallback seed

**筛选逻辑**：

```
Phase 1: 获取候选池
  - TSE Prime 全量 (~1,650 只)
  - TSE Standard 中 ADV > 50M JPY 的 (~200-400 只)
  → 合计 ~1,800-2,000 只

Phase 2: 流动性硬过滤
  - 20日 ADV ≥ 10M JPY (Sprint 用 min_adv_floor)
  - 数据完整性 ≥ 95% (最近 60 交易日)
  - 非 ETF/REIT/优先股（排除特殊品种）
  → 预计存活 ~800-1,200 只

Phase 3: 单手成本过滤
  - Sprint: max_cost_per_lot ≤ 500K JPY（先放宽，让截面够大）
  - 最终由 screener 的 strategy-specific max_cost_per_lot 二次过滤
  → 预计存活 ~500-800 只

Output: universe.json
  格式: [{"symbol": "XXXX.T", "name": "...", "sector": "...", "market": "Prime/Standard"}]
```

**TSE 代码获取方案**：

方案 A — yfinance 批量验证：
```python
# TSE 代码范围: 1000.T ~ 9999.T (4位) + 部分5位
# 用 yfinance 批量 download 验证存活性
# 每周一次，缓存结果
import yfinance as yf
data = yf.download(tickers_batch, period="5d", group_by="ticker", threads=True)
alive = [t for t in tickers_batch if not data[t]["Close"].isna().all()]
```

方案 B — J-Quants Listed API (Free tier):
```
GET /v1/listed/info
→ 全量上市公司列表，含市场区分/行业/代码
```

方案 C — 静态 TSE 列表文件 + 增量验证：
```
维护 tse_all_tickers.csv（从 JPX 官网下载，季度更新）
每周验证数据可用性，剔除退市/停牌
```

**推荐**: 方案 B (J-Quants) 为主，方案 A (yfinance) 为 fallback。方案 C 作为冷启动种子。

### 2.3 screener.py 评分重构

**现状**: 单一评分 = 流动性 + 波动率偏好 + 数据完整性（零 alpha）

**重构为两阶段**:

```
Stage 1: Hard Filter (pass/fail，不参与评分)
  - ADV ≥ min_adv
  - 0.5% ≤ daily_vol ≤ 6%
  - missing ≤ 2%
  - cost_per_lot ≤ max_cost_per_lot
  - fundamental 硬否决（营业利润率 < -15% AND OCF < 0）

Stage 2: Alpha Score (用于排序选 top_k)
  score = Σ w_i × rank_percentile(factor_i)

  因子候选:
  ┌────────────────────┬────────┬───────────────────────────────────┐
  │ 因子               │ 权重   │ 理由                              │
  ├────────────────────┼────────┼───────────────────────────────────┤
  │ mom_20 (20日收益率) │ 0.30   │ 短期动量，Sprint 核心信号          │
  │ mom_60 (60日收益率) │ 0.20   │ 中期趋势确认                      │
  │ vol_adj_mom20      │ 0.15   │ 风险调整后动量，信噪比更高          │
  │ sharpe_20          │ 0.15   │ 近期风险收益比                     │
  │ fundamental_score  │ 0.10   │ 基本面质量（已有）                 │
  │ adv_rank           │ 0.10   │ 流动性偏好（降级为小权重）          │
  └────────────────────┴────────┴───────────────────────────────────┘

  注意: 权重为初始值，后续由 IC 学习系统自适应调整
```

**为什么选这些因子**:
- `mom_20` + `mom_60`: 日本市场动量效应在 1-3 月窗口最强（Asness et al. 2013 跨市场验证）
- `vol_adj_mom20`: 消除高波动带来的假动量信号
- `sharpe_20`: 过滤"涨了但全靠一天跳空"的低质量动量
- `fundamental_score`: 复用已有基本面 overlay，避免 value trap
- `adv_rank`: 流动性仍然重要，但从"唯一标准"降级为"加分项"

### 2.4 db_update.py 适配

**改动最小化原则**:

```python
# 现有逻辑保留为 fallback
TARGET_UNIVERSE: List[Tuple[str, str, str]] = [...]  # 不删除

def load_universe(path: str) -> List[Tuple[str, str, str]]:
    """优先从 universe.json 加载，失败则用 TARGET_UNIVERSE。"""
    if not path:
        return TARGET_UNIVERSE
    # 已有 JSON/YAML 解析逻辑，无需改动
```

**config.yaml 改动**:
```yaml
update:
  universe_file: universe.json   # null → universe.json
```

**数据量影响评估**:
- 当前 97 只 × 730 天 ≈ 70K 行 daily_prices
- 扩容后 800 只 × 730 天 ≈ 580K 行（增长 ~8x）
- SQLite 完全能承受，无需迁移
- yfinance 批量下载 800 只约 3-5 分钟（threads=True）
- 日增量更新 800 只约 2-3 分钟

### 2.5 Sprint 信号链影响分析

| 组件 | 是否需要改动 | 说明 |
|------|-------------|------|
| `sprint_signal.py` | **不改** | 截面自动变大，百分位自然恢复意义 |
| `sprint_entry_check()` | **不改** | `pctile >= 0.80` 在 50+ 截面里约 10 只通过 |
| `compute_price_features.py` | **不改** | 按 symbol 逐只计算，自动适应 |
| `compute_ic.py` | **不改** | 截面扩大后 IC 精度自动提高 |
| `model_ridge.py` | **不改** | 输入来自 feature_daily，自动适应 |
| `kelly_sizer.py` | **不改** | 输入为 target_weight，不依赖 universe 大小 |
| `make_decision.py` | **不改** | 输入为 sprint_candidates.csv |
| `paper_execute.py` | **不改** | 输入为 decision 输出 |

**结论: 下游全部零改动**。这是正确的架构 — 只改数据输入层，不碰策略逻辑层。

---

## 3. 分阶段实施计划

### Phase A: Universe 扩容（核心，最高优先级）

新建 `universe_builder.py`，实现动态 Universe 构建：
- 数据源接入（J-Quants / yfinance / 静态列表）
- 流动性 + 数据完整性 + 单手成本硬过滤
- 输出 `universe.json`，格式兼容 `db_update.py load_universe()`
- `config.yaml` 指向 `universe.json`
- 验证: `db_update.py` 能正确加载扩容后的 universe

**预期产出**: Universe 从 97 → 500+ 只

### Phase B: Screener Alpha 评分重构

重构 `screener.py` 评分函数：
- 流动性/波动率/数据完整性降级为 hard filter
- 新增 alpha 评分: 动量 + 风险调整 + 基本面 + 流动性加分
- 因子用截面百分位标准化（rank percentile），消除量纲
- 保持 `apply_fundamental_overlay()` 不变
- 验证: 新评分的 top_k 与旧评分重叠度 <80%（说明新因子有区分度）

**预期产出**: Screener 输出包含 alpha 信号

### Phase C: 管线集成与验证

- `daily_run.py` 集成: 周一自动调用 `universe_builder.py`
- `compute_price_features.py` 性能验证: 800 只 × 730 天计算时间 < 60s
- IC 截面验证: `compute_ic.py` 在新截面上的 IC 标准误下降 ≥ 50%
- Sprint entry_ok 通过率验证: ≥ 5%（历史回测 30 天）
- 端到端 smoke test: `daily_run.py --config config.yaml` 全链路跑通

### Phase D: 回测对比与上线

- 回测: 新 Universe + 新 Screener vs 旧系统，对比 Sharpe / MaxDD / 换手率
- 如回测 Sharpe 提升 ≥ 0.2 或 entry_ok 通过率显著提升，切换生产
- 旧 `TARGET_UNIVERSE` 保留为 fallback，`config.yaml` 一行切换

---

## 4. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| yfinance 批量下载被限流 | 中 | 数据更新延迟 | 分批下载 + 指数退避 + J-Quants 备选 |
| 800 只数据量导致 daily_run 超时 | 低 | 16:30 任务延迟 | yfinance threads=True 并行; 监控耗时 |
| 新 Universe 引入低质量小盘股 | 中 | 噪声增加 | 硬过滤 ADV ≥ 10M JPY; 基本面 overlay 兜底 |
| Screener alpha 因子过拟合 | 低 | 选出伪动量 | 用 rank percentile 而非原始值; IC 监控 |
| 新旧系统切换期信号不一致 | 低 | 误操作 | Phase D 回测确认后一次性切换 |
| J-Quants Free 配额不够 | 中 | 无法拉全量列表 | yfinance 代码范围扫描 fallback |

---

## 5. 验收标准

| ID | 指标 | 阈值 | 验证方法 |
|----|------|------|----------|
| V-1 | Universe 大小 | ≥ 500 只 | `wc -l universe.json` |
| V-2 | Screener 截面 | top_k ≥ 50 (Sprint) | `selected_tickers.json` count |
| V-3 | Sprint entry_ok 通过率 | ≥ 5% (≥ 2-3 只) | `sprint_candidates.csv` |
| V-4 | IC 标准误下降 | ≥ 40% vs 旧系统 | `factor_registry` n_observations 对比 |
| V-5 | daily_run 总耗时 | ≤ 15 分钟 | 计时日志 |
| V-6 | 回测 Sharpe | ≥ 旧系统 Sharpe | `run_risk_backtest_comparison.py` |
| V-7 | 全量测试通过 | 81/81 quant 绿 | `pytest tests/` |
| V-8 | 下游零 regression | regime/风控/决策输出不变 | diff 对比 |

---

## 6. 工期估算

| Phase | 工作量 | 依赖 |
|-------|--------|------|
| A: Universe 扩容 | 核心开发 | 无 |
| B: Screener 重构 | 中等开发 | Phase A 完成 |
| C: 管线集成 | 集成测试 | Phase A + B 完成 |
| D: 回测上线 | 验证 | Phase C 完成 |

---

*最后更新: 2026-04-10*
