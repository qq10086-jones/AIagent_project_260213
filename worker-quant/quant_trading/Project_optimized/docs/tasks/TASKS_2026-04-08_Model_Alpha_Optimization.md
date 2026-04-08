# TASKS: 2026-04-08 模型 Alpha 优化

参考设计文档：
`docs/design/PATCH_2026-04-08_Model_Alpha_Optimization.md`

优先级说明：[P0] 阻塞性 / [P1] 高优先 / [P2] 中等 / [P3] 低优先/长期
估时单位：天（1天 = 有效工作日，含测试）

---

## Phase 1: 跨资产领先指标体系（P0，预计 2 天）

**最高优先级**：零成本 alpha 源，不改现有逻辑，纯增量。

### [P0] T1-1. 跨资产数据采集模块

- [ ] **T1-1a** 新建 `cross_asset_signals.py`
  - `fetch_cross_asset_snapshot()` → 从 yfinance 获取:
    - `^GSPC` (S&P500 收盘价 + 隔夜收益率)
    - `USDJPY=X` (美元/日元 + 24h 变化率)
    - `^VIX` (VIX 收盘值 + 日变化率)
    - `NKD=F` (CME 日经期货 + vs 前日日经 gap%)
  - 所有 fetch 包在 try/except，单项失败不影响其他
  - 超时 30s，retry 2 次
  - 文件路径: `cross_asset_signals.py`（新建）

- [ ] **T1-1b** `compute_cross_asset_regime_signal(snapshot)` 函数
  - 输入: fetch 结果 dict
  - z-score 标准化（滚动 20 日均值/标准差，从 DB 历史计算）
  - 加权求和 → sigmoid → `cross_asset_score` [0, 1]
  - 初始权重: sp500=0.35, usdjpy=0.20, vix=0.20, futures=0.25
  - 输出 `regime_adjustment`: "upgrade" / "neutral" / "downgrade"
  - 文件路径: `cross_asset_signals.py`

- [ ] **T1-1c** 数据库表 `cross_asset_snapshots`
  - `trade_schema.py` 新增 `ensure_cross_asset_tables(conn)`
  - 主键: `asof`
  - 列: asof, ts, sp500_close, sp500_overnight_pct, usdjpy, usdjpy_change_pct, vix_close, vix_change_pct, nk_futures, nk_futures_gap_pct, cross_asset_score, regime_adjustment
  - 文件路径: `trade_schema.py`

### [P0] T1-2. 管线集成

- [ ] **T1-2a** `morning_briefing.bat` 新增 Step 0: 跨资产采集
  - 在 db_update 之前调用 `python cross_asset_signals.py --db japan_market.db`
  - 失败不阻塞后续步骤
  - 文件路径: `morning_briefing.bat`

- [ ] **T1-2b** 晨报 v2 新增"跨资产信号"小节
  - `quant_briefing.py` 读取 `cross_asset_snapshots` 最新行
  - 输出格式:
    ```
    ## 零、隔夜跨资产信号
    - S&P500: 5123 (+1.2%)  USD/JPY: 148.3 (+0.45%)
    - VIX: 22.1 (-3.2%)  日经期货: 37250 (gap +1.2%)
    - 跨资产 regime 信号: 0.72 (偏多) → 建议 UPGRADE regime
    ```
  - 文件路径: `quant_briefing.py`

- [ ] **T1-2c** `action_plan_builder.py` 读取跨资产信号
  - `action_plan_today.json` 新增 `cross_asset` 字段
  - action_summary 文本包含跨资产 regime 调整建议
  - 文件路径: `action_plan_builder.py`

### [P0] T1-3. 单元测试

- [ ] **T1-3a** `tests/test_cross_asset_signals.py`
  - Case 1: 正常数据 → score 在 [0, 1] 范围内
  - Case 2: 部分数据缺失 → fallback 到中性(0.5)，不报错
  - Case 3: 全部数据缺失 → score=0.5, adjustment=neutral
  - Case 4: 极端行情（美股 -5%, VIX 暴涨） → score < 0.2
  - Case 5: 强 risk-on（美股 +2%, VIX 跌, 期货涨） → score > 0.7
  - Case 6: DB 写入/读取往返正确
  - 文件路径: `tests/test_cross_asset_signals.py`（新建）

### [P0] T1-4. 历史数据回填

- [ ] **T1-4a** 回填脚本 `scripts/backfill_cross_asset.py`
  - 回填最近 60 个交易日的跨资产数据到 `cross_asset_snapshots`
  - 用于 z-score 标准化的历史基准
  - 文件路径: `scripts/backfill_cross_asset.py`（新建）

---

## Phase 2: Regime 连续化（P1，预计 3 天）

**依赖**: Phase 1 完成（需要 cross_asset_score 作为输入）

### [P1] T2-1. Regime Score V2 核心算法

- [ ] **T2-1a** `benchmark_regime.py` 新增 `compute_regime_score_v2()`
  - 输入: px_b, fast_ma, slow_ma, ma_slope_5d, volume_ratio, cross_asset_score
  - 5 分量加权: ma_signal(0.30) + slope_signal(0.20) + price_position(0.15) + volume_signal(0.10) + cross_asset(0.25)
  - 输出: regime_score float [0, 1]
  - 文件路径: `benchmark_regime.py`

- [ ] **T2-1b** `benchmark_regime.py` 新增 `regime_score_to_scale()`
  - 将 regime_score 映射为 position scale:
    - score > full_position_threshold(0.70) → 1.0
    - score < zero_position_threshold(0.15) → off_scale
    - 中间线性插值
  - 文件路径: `benchmark_regime.py`

- [ ] **T2-1c** MA 斜率计算辅助函数
  - `compute_ma_slope(conn, symbol, asof, ma_window=20, slope_window=5)`
  - 返回 MA20 过去 5 天的斜率（归一化为 %/day）
  - 文件路径: `benchmark_regime.py`

- [ ] **T2-1d** 量能倍数计算辅助函数
  - `compute_volume_ratio(conn, symbol, asof, lookback=20)`
  - 返回今日量 / 20 日均量
  - 文件路径: `benchmark_regime.py`

### [P1] T2-2. Config 集成

- [ ] **T2-2a** `config.yaml` 新增 regime v2 参数
  ```yaml
  benchmark_regime:
    version: "v2"
    v2_weights: {ma_signal: 0.30, slope_signal: 0.20, ...}
    v2_thresholds: {full_position: 0.70, zero_position: 0.15}
  ```
  - 初始设为 `version: "v1"` 保持向后兼容
  - 文件路径: `config.yaml`

- [ ] **T2-2b** `daily_run.py` / `make_decision.py` 根据 config version 分发
  - `version: "v1"` → 调原 `benchmark_regime_scale_v2()`
  - `version: "v2"` → 调新 `compute_regime_score_v2()` + `regime_score_to_scale()`
  - regime_diagnosis.json 新增 `regime_score` 字段
  - 文件路径: `daily_run.py`, `make_decision.py`

### [P1] T2-3. 回测验证

- [ ] **T2-3a** `run_regime_v2_backtest.py` 新建
  - 使用 `ss7_sqlite_news_overlay.py` 回测引擎
  - 对比 regime v1 vs v2 在相同回测窗口的:
    - Sharpe, MaxDD, 胜率, 年化收益
    - V 型反转场景的表现差异（手动标注 3-5 个历史 V 反转日期）
  - 输出对比表到 `reports/regime_v2_backtest_comparison.json`
  - 文件路径: `run_regime_v2_backtest.py`（新建）

### [P1] T2-4. 单元测试

- [ ] **T2-4a** `tests/test_regime_v2.py`
  - Case 1: 强 risk-on 输入 → score > 0.70
  - Case 2: 强 risk-off 输入 → score < 0.20
  - Case 3: 04-08 回推数据 → score ≈ 0.45-0.55（允许半仓）
  - Case 4: 全 NaN 输入 → fallback 到 v1 逻辑
  - Case 5: regime_score_to_scale 线性插值正确
  - Case 6: MA 斜率计算边界（不足 5 日数据）
  - Case 7: config version="v1" → 行为不变（兼容性）
  - 文件路径: `tests/test_regime_v2.py`（新建）

---

## Phase 3: Sprint Score 因子重整（P2，预计 2 天）

**依赖**: Phase 2 完成（需要 regime_score 做因子动态加权）

### [P2] T3-1. 基本面因子接入 IC 管线

- [ ] **T3-1a** `compute_ic.py` 扩展因子列表
  - 在因子扫描列表中加入 10 个基本面因子:
    `value_bp, roa_op, cfo_assets, accruals_inv, margin_op, growth_rev_yoy, growth_op_yoy, guidance_delta, leverage_safety, dividend_yield`
  - 文件路径: `compute_ic.py`

- [ ] **T3-1b** 首次运行 compute_ic 填充基本面 IC
  - 手动运行一次确认数据写入 factor_registry
  - 验证: 10 个基本面因子在 factor_registry 有 IC 值
  - 操作: 命令行运行

### [P2] T3-2. 低样本 ICIR Bayesian Shrinkage

- [ ] **T3-2a** `sprint_signal.py` 新增 `shrink_icir()`
  - `effective_icir = icir × (n / (n + 30)) + 0.05 × (30 / (n + 30))`
  - n_obs < 5 时几乎完全用先验 (0.05)
  - n_obs > 50 时几乎不收缩
  - 文件路径: `sprint_signal.py`

### [P2] T3-3. Sprint Score V2

- [ ] **T3-3a** `sprint_signal.py` 新增 `sprint_score_v2()`
  - 读取 factor_registry 全部 is_active=1 因子的 ICIR
  - 应用 shrink_icir + tier_mult + regime_factor_tilt
  - 因子分类标签: momentum / mean_reversion / risk / fundamental
  - 按 regime_score 动态调权:
    - regime 高 → 加权 momentum, 减权 value
    - regime 低 → 加权 value/defensive, 减权 momentum
  - 文件路径: `sprint_signal.py`

- [ ] **T3-3b** `generate_sprint_artifacts()` 支持 v2 评分
  - config 新增 `sprint_signal.score_version: "v2"`
  - v1 / v2 切换，v1 保持不动
  - 文件路径: `sprint_signal.py`

### [P2] T3-4. 因子分类标签配置

- [ ] **T3-4a** `config.yaml` 新增因子类别映射
  ```yaml
  factor_categories:
    momentum: [mom_consist, ret20, ret60, mom_12_1, high52w]
    mean_reversion: [rsi14, z_20, vol_z]
    risk: [sharpe_60, sharpe_20, sortino_60, vol_stability, vol_adj_mom20]
    fundamental: [value_bp, roa_op, cfo_assets, accruals_inv, margin_op,
                   growth_rev_yoy, growth_op_yoy, guidance_delta, leverage_safety, dividend_yield]
    neutral: [ret1, ret5, slope60, ma_gap, vol20, vol60]
  ```
  - 文件路径: `config.yaml`

### [P2] T3-5. 单元测试

- [ ] **T3-5a** `tests/test_sprint_score_v2.py`
  - Case 1: shrink_icir n=2 → 接近先验(0.05)
  - Case 2: shrink_icir n=100 → 接近原始 ICIR
  - Case 3: regime 高(0.8) → momentum 因子权重 > fundamental
  - Case 4: regime 低(0.2) → fundamental 因子权重 > momentum
  - Case 5: v2 score 与 v1 score 排名相关性 > 0.7（不应剧烈偏离）
  - Case 6: 全因子 NaN → fallback 到 v1 逻辑
  - 文件路径: `tests/test_sprint_score_v2.py`（新建）

---

## Phase 4: 执行时机优化（P2，预计 1 天）

**依赖**: 无（可与 Phase 2-3 并行）

### [P2] T4-1. 限价区间建议

- [ ] **T4-1a** `make_decision.py` 新增 `compute_entry_zone()`
  - 基于 last_close + ATR 计算:
    - `target_limit`: 回调 0.3 ATR（耐心入场）
    - `aggressive_limit`: 追涨 0.2 ATR（强信号时）
    - `walk_away_price`: 超过 1 ATR 放弃
  - 文件路径: `make_decision.py`

- [ ] **T4-1b** `orders_proposal.csv` 新增列
  - 新增: `target_limit`, `aggressive_limit`, `walk_away_price`
  - 保持 backward compatible: 现有列不变
  - 文件路径: `make_decision.py`

### [P2] T4-2. 盘中限价提醒

- [ ] **T4-2a** `intraday_monitor.py` 新增限价区间检查
  - 读取 `action_plan_today.json` 中的限价区间
  - 盘中价格进入 target_limit ±0.2% 时推送 Discord:
    "4005.T 当前 527.5，接近目标入场价 527.0，建议挂限价单"
  - 盘中价格突破 walk_away_price 时推送:
    "7267.T 已涨至 1295，超过放弃价 1285，建议放弃本次信号"
  - 文件路径: `intraday_monitor.py`

### [P2] T4-3. 单元测试

- [ ] **T4-3a** `tests/test_entry_zone.py`
  - Case 1: BUY 信号 → target < last_close < aggressive < walk_away
  - Case 2: SELL 信号 → 方向相反
  - Case 3: ATR 大(高波动) → 区间更宽
  - Case 4: ATR 小(低波动) → 区间更窄
  - 文件路径: `tests/test_entry_zone.py`（新建）

---

## Phase 5: Shadow 观察期 + 切换（P1，预计 5-10 天）

### [P1] T5-1. Shadow 模式运行

- [ ] **T5-1a** 所有新模块以 shadow 模式运行
  - 跨资产信号: 写入 DB + 显示在晨报，但不影响 regime 判定
  - Regime v2: 计算 score 并写入 regime_diagnosis.json 的新字段，但实际决策仍用 v1
  - Sprint score v2: 并行计算 v1 + v2 分数，对比排名差异
  - 限价建议: 在 orders_proposal.csv 中附加，但不改变主信号

- [ ] **T5-1b** Shadow 对比仪表盘
  - `reports/shadow_v2_comparison.json`: 每日记录 v1 vs v2 差异
  - 累计 10 个交易日后生成对比报告

### [P1] T5-2. 切换条件

- [ ] **T5-2a** 切换判据（需人工确认）:
  - Regime v2 在历史 V 反转日的 recall > 70%
  - Sprint score v2 排名与 v1 Spearman ρ > 0.6
  - 跨资产数据可用率 > 90%
  - 无异常极端值

- [ ] **T5-2b** 切换操作
  - `config.yaml`: `benchmark_regime.version: "v2"`, `sprint_signal.score_version: "v2"`
  - 保留 v1 代码不删除（fallback）

---

## 汇总

| Phase | 内容 | 优先级 | 预计工期 | 依赖 |
|-------|------|--------|----------|------|
| 1 | 跨资产领先指标 | P0 | 2 天 | 无 |
| 2 | Regime 连续化 | P1 | 3 天 | Phase 1 |
| 3 | Sprint Score 重整 | P2 | 2 天 | Phase 2 |
| 4 | 执行时机优化 | P2 | 1 天 | 无（可并行） |
| 5 | Shadow + 切换 | P1 | 5-10 天 | Phase 1-4 |
| **总计** | | | **8-10 天 开发 + 5-10 天观察** | |

### 新增/修改文件清单

| 文件 | 状态 |
|------|------|
| `cross_asset_signals.py` | 新建 |
| `scripts/backfill_cross_asset.py` | 新建 |
| `run_regime_v2_backtest.py` | 新建 |
| `tests/test_cross_asset_signals.py` | 新建 |
| `tests/test_regime_v2.py` | 新建 |
| `tests/test_sprint_score_v2.py` | 新建 |
| `tests/test_entry_zone.py` | 新建 |
| `benchmark_regime.py` | 修改 (+regime_score_v2) |
| `sprint_signal.py` | 修改 (+sprint_score_v2, +shrink_icir) |
| `make_decision.py` | 修改 (+entry_zone, +regime v2 分发) |
| `daily_run.py` | 修改 (+regime version 分发) |
| `trade_schema.py` | 修改 (+cross_asset_snapshots 表) |
| `compute_ic.py` | 修改 (+基本面因子列表) |
| `quant_briefing.py` | 修改 (+跨资产小节) |
| `action_plan_builder.py` | 修改 (+cross_asset 字段) |
| `intraday_monitor.py` | 修改 (+限价区间检查) |
| `morning_briefing.bat` | 修改 (+跨资产采集) |
| `config.yaml` | 修改 (+regime v2 参数, +因子分类) |

---

*本任务清单与设计文档 `PATCH_2026-04-08_Model_Alpha_Optimization.md` 对应，待用户审批后开始实施。*
