# Japan Quant Strategy — Project_optimized (Production)

## 项目概述

针对东京证券交易所（TSE）的系统化量化交易框架。目标：100万日元本金实现20%+ 年化收益。

## 当前生产版本

**核心模型文件**：`ss7_sqlite_news_overlay.py`

这是最新的生产级模型，包含：
- Ridge 回归 + 均值方差优化
- 新闻情绪门控层（F/A/U 三因子）
- 单股止损（8%）+ 组合最大回撤控制（12%/18%）
- 资本依赖的执行模型（100手、手续费、滑点、市场冲击）

## 版本演化路径

```
ss4 → ss4_optimized → ss5 → ss6 (执行模型) → ss6_sqlite (DB版) → ss7_sqlite_news_overlay (当前)
```
旧版本存放于 `../archive/models/`（仅供参考，不运行）。

## 快速启动

```bash
cd quant_trading/Project_optimized

# 1. 更新数据库（约100只TSE股票）
python db_update.py --db japan_market.db

# 2. 运行完整流水线（数据→筛选→回测→报告）
python run_pipeline.py --config config.yaml

# 3. 生成交易决策
python make_decision.py --db japan_market.db --cash 1000000 --lot 100

# 4. 启用新闻门控（需提供news.csv）
SS6_NEWS_ON=1 SS6_NEWS_CSV=news.csv python ss7_sqlite_news_overlay.py
```

## 关键配置（config.yaml）

| 参数 | 当前值 | 说明 |
|------|--------|------|
| initial_capital | 1,000,000 JPY | 100万日元本金 |
| lot_size_default | 100 | 日本股票标准手数 |
| fee_bps | 5.0 | 手续费（bps） |
| slippage_bps | 5.0 | 滑点（bps） |
| impact_k | 0.5 | 市场冲击系数 |
| stop_loss_pct | 0.08 | 单股止损阈值（8%） |
| max_dd_half | 0.12 | 回撤12%降半仓 |
| max_dd_full | 0.18 | 回撤18%全平仓 |

## 文件结构

```
Project_optimized/
├── ss7_sqlite_news_overlay.py  ★ 核心模型（最新）
├── db_update.py                数据库更新（~100只TSE股票）
├── screener.py                 选股过滤（含1手成本过滤）
├── run_pipeline.py             完整流水线入口
├── make_decision.py            交易决策生成（含回撤检查）
├── daily_run.py                每日定时任务
├── app.py                      Streamlit Dashboard
├── market_db_v2.py             SQLite数据库接口
├── trade_schema.py             交易记录数据库schema
├── config.yaml                 ★ 核心配置文件
├── execution_report.py         执行报告
├── report_obsidian.py          Obsidian报告生成
├── import_fills.py             导入成交记录
├── build_positions.py          构建持仓
├── build_account_snapshot.py   账户快照
├── post_trade.py               盘后处理
├── manual_fills_entry.py       手动录入成交
├── report_utils.py             报告工具
└── artifacts/                  决策记录（自动生成）
    └── decision/YYYY-MM-DD/    每次决策的完整审计轨迹
```

## 新闻因子系统

新闻数据 CSV 格式：
```csv
date,ticker,sent,weight,conf
2026-01-05,7203.T,0.8,1.5,0.9
```
- `sent`: 情绪分 [-1, 1]，正向看涨
- `weight`: 新闻权重（重要性）
- `conf`: 置信度 [0, 1]

门控逻辑（不直接进入预测器，仅作风险过滤）：
- **F**（方向性情绪）：加权平均情绪
- **A**（关注度）：注意力强度（过热时降仓）
- **U**（分歧度）：矛盾信号强度（不确定时降仓）

## 股票宇宙

涵盖约100只TSE Prime流动性股票，跨12个行业：半导体/AI、电子机械、重工国防、商社贸易、金融保险、汽车运输、能源化工、钢铁材料、电力公用、电信、消费零售、医疗互联网。

通过 screener 自动过滤：
- ADV > 2000万JPY
- 1手成本 < 15万JPY（确保100万日元可分散持有≥6只）
- 日波动率 0.5%-6%
