# OpenClaw Nexus v1.2.7 Execution Strategy Patch

## 1. 简介 (Introduction)
针对用户在 Discord 中提出的复杂交易执行与本金受限下的选股策略问题，本 Patch 旨在增强 `worker-quant` 的执行定价能力与组合建议逻辑。

## 2. 核心改进 (Core Enhancements)

### 2.1 挂单定价逻辑 (Execution Pricing Module)
目前的工具仅支持市价单 (MKT) 或简单的昨日收盘价参考。
- **改进方案**: 
  - 在 `deep_analysis` 中引入 ATR (Average True Range) 计算支撑/阻力位。
  - 增加 `quant.calc_limit_price` 工具，根据盘口波动率给出“激进/稳健/保守”三种挂单价格建议。
  - 对于卖出指令，建议参考：`昨日收盘价 + 0.5 * ATR` (止盈挂单) 或 `5日均线`。

### 2.2 小额本金组合策略 (Small-Cap Portfolio Recommender)
针对用户 40w 日元的本金（在日股 100 股一手规则下受限严重）提供定制化筛选。
- **改进方案**:
  - 在 `discovery_workflow` 中加入 `capital_constraint` 参数。
  - **逻辑**: 筛选 `股价 * 100 < (本金 * 0.25)` 的股票，确保单仓位不超过 25%，实现基本分散。
  - **示例**: 针对 40w 日元，重点推荐单价在 1000 JPY 以下的高质量个股（如 NTT 9432.T 等）。

### 2.3 调仓与建仓建议 (Onboarding Strategy)
- **分批建仓 (Pyramid Entry)**: 建议将 40w 日元分为三份，首笔 15w，观察趋势后再追加。
- **空仓期策略**: 在无合适标的时，建议资金暂留 Redis/DB 记录的 Cash 账户，并触发 `news.daily_report` 监控潜在机会。

## 3. 待实现代码清单 (To-Do List)
1. [ ] 修改 `worker.py`: 增加 `_calculate_atr(symbol, period=14)` 内部函数。
2. [ ] 修改 `worker.py`: 在 `deep_analysis` 输出中增加 `execution_suggestions` 字段。
3. [ ] 修改 `discovery_workflow`: 增加 `max_price` 过滤逻辑，支持 `starting_capital` 敏感型推荐。

## 4. 指令示例 (Command Examples)
- `/analyze 9432.T mode:sell` -> 返回包含建议挂单价的卖出分析。
- `/discovery capital:400000` -> 返回适配 40w 日元的小额绩优股组合。
