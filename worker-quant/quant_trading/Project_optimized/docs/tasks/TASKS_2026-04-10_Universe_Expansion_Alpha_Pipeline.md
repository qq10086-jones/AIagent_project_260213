# 任务清单 — Universe 扩容与选股 Alpha 管线重构

**设计文档**: `../design/PATCH_2026-04-10_Universe_Expansion_Alpha_Pipeline.md`
**创建日期**: 2026-04-10
**状态**: PENDING

---

## Phase A: Universe 扩容（最高优先级）

### A-1: TSE 上市公司代码获取

- [ ] **A-1a**: 调研 J-Quants Free API `/v1/listed/info` 可用性
  - 确认 Free tier 是否支持全量上市公司列表
  - 记录 API 限流策略（QPS / 日配额）
  - 如不可用，记录原因，切换到 A-1b

- [ ] **A-1b**: yfinance TSE 代码批量验证方案
  - TSE 4 位代码范围: 1000-9999 + `.T` 后缀
  - 用 `yf.download(batch, period="5d")` 验证存活性
  - 估算全量扫描耗时，设计分批策略（每批 100-200 只）
  - 处理退市/停牌/无数据代码

- [ ] **A-1c**: JPX 官网静态列表备选
  - 下载 TSE 上市公司 CSV（东证一覧）
  - 解析为 `(symbol, name, sector, market)` 格式
  - 作为冷启动种子 + 季度更新源

### A-2: universe_builder.py 实现

- [ ] **A-2a**: 核心逻辑
  - 新建 `universe_builder.py`
  - 输入: 数据源配置（J-Quants / yfinance / 静态文件）
  - Phase 1: 获取候选池（TSE Prime + Standard）
  - Phase 2: 流动性硬过滤（ADV ≥ 10M JPY, 数据完整性 ≥ 95%）
  - Phase 3: 品种过滤（排除 ETF/REIT/优先股/信托受益证券）
  - Phase 4: 单手成本过滤（≤ 500K JPY，可配置）
  - 输出: `universe.json`

- [ ] **A-2b**: 输出格式兼容
  - `universe.json` 格式需兼容 `db_update.py load_universe()`
  - 即 `List[Tuple[str, str, str]]` 或等价 JSON
  - 包含 `TARGET_UNIVERSE` 中的全部 97 只（superset 保证）
  - 添加元数据: 构建日期、数据源、候选池总数、存活数

- [ ] **A-2c**: 增量更新机制
  - 首次运行: 全量扫描（预计 10-20 分钟）
  - 后续运行: 读取上次 universe.json，只验证增量（新上市/退市）
  - 周一 06:00 JST 自动运行（在 daily_run 之前）

- [ ] **A-2d**: 行业/板块分类
  - J-Quants 有 33 业种分类，直接复用
  - yfinance 回退: `info["sector"]` + `info["industry"]`
  - 映射到统一行业分类（保持与现有 `tickers.sector` 兼容）

### A-3: config.yaml 与 db_update.py 适配

- [ ] **A-3a**: `config.yaml` 添加 universe 配置块
  ```yaml
  universe:
    file: universe.json           # 动态 Universe 文件路径
    fallback_to_hardcoded: true   # universe.json 不存在时用 TARGET_UNIVERSE
    rebuild_day: monday           # 自动重建日
    min_adv_universe: 10000000    # Universe 级流动性下限
    max_cost_per_lot_universe: 500000
    exclude_etf: true
    exclude_reit: true
  ```

- [ ] **A-3b**: `db_update.py` 适配
  - `load_universe()` 优先读 `config.yaml -> universe.file`
  - 新 universe 可能有 800+ 只，验证 yfinance 批量下载性能
  - 添加分批下载 + 进度条（tqdm）
  - 失败重试: 单只下载失败不阻塞整批

### A-4: 测试与验证

- [ ] **A-4a**: 单元测试
  - `test_universe_builder.py`: 格式验证、去重、排除逻辑
  - mock 数据源，验证 fallback 链路

- [ ] **A-4b**: 集成验证
  - 运行 `universe_builder.py` → `universe.json`
  - 运行 `db_update.py --universe universe.json`
  - 确认 `tickers` 表和 `daily_prices` 表正确扩容
  - 确认原有 97 只全部包含（superset）

---

## Phase B: Screener Alpha 评分重构

### B-1: 评分函数拆分

- [ ] **B-1a**: 抽离 hard filter
  - 将 ADV / vol / missing / lot_cost 判断从评分中移出
  - 改为独立的 `_hard_filter(row, cfg) -> (pass: bool, reasons: list)`
  - 评分函数不再包含这些维度

- [ ] **B-1b**: 新增 alpha 评分函数
  - 新函数: `_alpha_score(sym, px_panel, vol_panel, fundamental_score, cfg) -> float`
  - 因子:
    - `mom_20`: 20日收益率，截面 rank percentile
    - `mom_60`: 60日收益率，截面 rank percentile
    - `vol_adj_mom20`: 波动率调整动量，截面 rank percentile
    - `sharpe_20`: 20日滚动夏普，截面 rank percentile
    - `fundamental_score`: 来自现有 overlay（已经是 0-1 分数）
    - `adv_rank`: ADV 截面 rank percentile（流动性加分）
  - 所有因子用 rank percentile 标准化，消除量纲和异常值影响
  - 加权求和，初始权重硬编码，后续可从 `factor_registry` 读取

- [ ] **B-1c**: ScreenConfig 扩展
  ```python
  @dataclass
  class ScreenConfig:
      # ... 现有字段保留 ...
      # 新增: alpha 评分权重
      alpha_weights: dict = field(default_factory=lambda: {
          "mom_20": 0.30,
          "mom_60": 0.20,
          "vol_adj_mom20": 0.15,
          "sharpe_20": 0.15,
          "fundamental_score": 0.10,
          "adv_rank": 0.10,
      })
      use_alpha_scoring: bool = True  # feature flag，可回退旧逻辑
  ```

### B-2: 重叠度验证

- [ ] **B-2a**: 对比新旧评分 top_k 重叠度
  - 同一 asof，分别用旧评分和新评分跑 screener
  - 计算 Jaccard 相似度: `|A ∩ B| / |A ∪ B|`
  - 目标: Jaccard < 0.80（说明新因子有区分度）
  - 如 Jaccard > 0.90，说明权重需要调整

- [ ] **B-2b**: 新评分的 alpha 预测性检验
  - 用过去 60 天数据，计算新评分 vs 5日后收益率的 IC
  - IC > 0.03 且 t-stat > 1.5 视为有效

### B-3: 测试

- [ ] **B-3a**: 单元测试
  - `test_screener_alpha.py`: hard filter 逻辑、alpha 评分计算
  - 验证 feature flag `use_alpha_scoring=False` 回退旧逻辑

- [ ] **B-3b**: 回归测试
  - `use_alpha_scoring=False` 时，输出与旧版完全一致

---

## Phase C: 管线集成

### C-1: daily_run.py 集成

- [ ] **C-1a**: 添加 universe 刷新步骤
  - `daily_run.py` 新增 Step 0: 检查是否周一 → 调用 `universe_builder.py`
  - 非周一跳过，直接用现有 `universe.json`
  - `universe.json` 不存在时: 首次强制运行 builder，不依赖星期

- [ ] **C-1b**: compute_price_features 性能验证
  - 800 只 × 730 天计算时间 < 60s（当前 97 只约 5-8s）
  - 如超时: 考虑只计算最近 252 天（1 年）的特征

### C-2: IC 学习系统验证

- [ ] **C-2a**: 截面扩大后 IC 精度验证
  - 运行 `compute_ic.py` 在新 Universe 上
  - 对比 `factor_registry` 中 `n_observations` 和 IC 标准误
  - 预期: SE 下降 ≥ 40%（因截面从 ~50 扩大到 ~200+）

- [ ] **C-2b**: 因子权重稳定性
  - 连续 5 个 asof 日期运行 IC 计算
  - 检查 `weight` 列的日间变化幅度
  - 预期: 比旧系统更稳定（因为 IC 估计更精确）

### C-3: Sprint entry_ok 通过率验证

- [ ] **C-3a**: 历史回测 entry_ok
  - 用过去 30 个交易日，逐日跑 sprint_signal
  - 统计每日 entry_ok=True 的数量
  - 预期: 日均 ≥ 2 只（通过率 ≥ 4%）
  - 旧系统对照: 日均 ~0 只

### C-4: 端到端 Smoke Test

- [ ] **C-4a**: 全链路跑通
  ```bash
  python universe_builder.py --config config.yaml
  python daily_run.py --config config.yaml
  python quant_briefing.py --mode full --output-version v2
  ```
  - 无报错
  - `reports/briefing_v2_latest.md` 正常生成
  - `sprint_candidates.csv` 行数 ≥ 50

- [ ] **C-4b**: 全量测试
  - `pytest tests/ -v` 全绿（81/81 + 新增测试）

---

## Phase D: 回测对比与上线

### D-1: 回测

- [ ] **D-1a**: 对比回测
  - 新系统 vs 旧系统，回测期 2025-04-01 ~ 2026-04-09（1 年）
  - 指标: Sharpe / Sortino / MaxDD / 年化收益 / 换手率
  - 用 `run_risk_backtest_comparison.py` 框架

- [ ] **D-1b**: 风险指标检查
  - 新系统 MaxDD 不超过旧系统 × 1.5
  - 换手率 ≤ 200% 年化（避免过度交易）

### D-2: 上线切换

- [ ] **D-2a**: 生产切换
  - `config.yaml`: `universe_file: universe.json` + `use_alpha_scoring: true`
  - 旧 `TARGET_UNIVERSE` 保留不删（fallback）
  - Windows Task Scheduler 确认 `universe_builder.py` 周一 06:00 JST

- [ ] **D-2b**: 监控
  - 首周每日检查: universe.json 大小、screener 输出、entry_ok 通过率
  - IC 学习收敛速度: 新因子权重是否在 2 周内稳定

### D-3: 文档更新

- [ ] **D-3a**: 更新 `CLAUDE.md`
  - 新增 `universe_builder.py` 说明
  - 更新场景表（新增 "更新 Universe" 场景）
  - 更新关键参数表

- [ ] **D-3b**: 更新 Memory
  - 更新 pipeline chain（新增 Step 0: universe_builder）
  - 更新候选池大小（97 → 500+）

---

## 任务依赖关系

```
A-1 (数据源调研)
 └→ A-2 (builder 实现) → A-3 (config 适配) → A-4 (测试)
                                                  │
B-1 (评分重构) → B-2 (验证) → B-3 (测试) ────────┤
                                                  │
                                                  ▼
                                          C-1 ~ C-4 (集成)
                                                  │
                                                  ▼
                                          D-1 ~ D-3 (回测上线)
```

Phase A 和 Phase B 可并行开发，Phase C 需要两者都完成。

---

## 优先级排序

| 优先级 | 任务 | 理由 |
|--------|------|------|
| P0 | A-1, A-2 | 没有 Universe 扩容，其他一切都是空谈 |
| P0 | A-3, A-4 | Universe 要能接入现有管线 |
| P1 | B-1 | Screener alpha 是第二重要的改进 |
| P1 | C-1, C-3, C-4 | 集成和 entry_ok 验证是上线前提 |
| P2 | B-2, C-2 | 统计验证，重要但不阻塞上线 |
| P2 | D-1 | 回测对比，建议做但不强制 |
| P3 | D-2, D-3 | 上线和文档，Phase C 通过后执行 |

---

*最后更新: 2026-04-10*
