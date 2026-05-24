# Data And Open Source Selection

## 1. Recommendation

第一阶段不要 fork 大型开源交易平台。主工程使用本项目，复用旧项目的数据和执行建议经验，再按需吸收开源项目能力。

## 2. Open Source Candidates

### OpenBB

Use case: 数据研究接口和外部市场数据探索。

Decision: watch and spike only.

Reason:

- 优点：覆盖范围广，适合研究和数据探索。
- 风险：直接作为核心生产数据源会增加依赖和稳定性风险。
- 规则：未完成 P6-01 前不能替代本地数据链路。

### vectorbt

Use case: 快速批量回测止盈、止损、换仓阈值。

Decision: recommended for backtest spike.

Reason:

- 优点：适合参数矩阵和信号向量化回测。
- 风险：新闻事件和盘中执行细节需要额外适配。
- 规则：只作为回测工具，不作为生产信号引擎。

### Backtrader

Use case: 事件驱动回测参考。

Decision: not primary.

Reason:

- 优点：成熟，文档多。
- 风险：项目较老，对本策略的新闻/主题/盘中轮动不是最贴合。

### QuantConnect LEAN

Use case: 完整机构级回测/实盘框架。

Decision: not selected for phase 1.

Reason:

- 优点：完整。
- 风险：太重，迁移成本高，不适合作为当前本地工具的第一阶段底座。

### NautilusTrader

Use case: 专业事件驱动交易系统。

Decision: future candidate only.

Reason:

- 优点：架构强，适合更专业的实盘执行。
- 风险：当前策略更需要热点发现、候选排序和人工执行建议，不需要先上重型执行引擎。

## 3. Existing Local Sources

### Project_optimized

Candidate migration sources:

- `candidate_ranker.py`
- `intraday_decision.py`
- `ss7_sqlite_news_overlay.py`
- `news_to_db.py`
- `risk_kill_switch.py`
- `execution_report.py`
- `japan_market.db`

Migration rule: wrap through adapters first; do not copy large scripts blindly.

### Project_v5

Candidate migration sources:

- `src/strategy_v7/theme_detector.py`
- `src/strategy_v7/news_aggregator.py`
- `src/strategy_v7/feature_builder_v7.py`
- `src/strategy_v7/risk_governor_v7.py`
- `docs/PROJECT_V7_DESIGN.md`

Migration rule: use V7 concepts, but rewrite module boundaries for this project.

## 4. Data Priority

1. Existing local SQLite and cached prices.
2. Existing TDnet/EDINET/news cache.
3. Free public data for external temperature.
4. Optional OpenBB research adapter.
5. Paid or broker data only after paper value is proven.

