# TASKS: 2026-04-07 风控机制加固

参考设计文档：
`docs/design/PATCH_2026-04-07_Risk_Management_Hardening.md`

优先级说明：[P0] 阻塞性 / [P1] 高优先 / [P2] 中等 / [P3] 低优先/长期
估时单位：天（1天 = 有效工作日，含测试）

---

## 阶段一：止损执行链路（P0，预计 2 天）

**致命缺陷修复**：实盘链路无价格止损。所有后续任务可独立但此阶段最优先。

### [P0] T1. ATR 计算集成

- [x] **T1-1** `make_decision.py` 新增 `_compute_atr_pct(conn, symbol, asof, window=20) -> float`
  - 从 `daily_prices` 读取最近 `window+1` 条 OHLC
  - 计算 True Range 序列: `max(H-L, |H-prev_C|, |L-prev_C|)`
  - 返回 `mean(TR) / latest_close` 百分比
  - 边界处理：数据不足时返回 `stop_loss_min_pct` 作为 fallback
  - 文件路径：`make_decision.py`
  - 验证：`python -c "from make_decision import _compute_atr_pct; ..."`

- [x] **T1-2** 单元测试：`tests/test_atr_computation.py`
  - Case 1: 正常 20 日数据，验证 ATR 计算正确
  - Case 2: 数据不足（< 5 条），验证 fallback 行为
  - Case 3: 高波动股 vs 低波动股，验证 clip 到 [min_pct, max_pct]

---

### [P0] T2. sprint_exit_check_v2 实现

- [x] **T2-1** `sprint_signal.py` 新增 `sprint_exit_check_v2()` 函数
  - 保留原有 3 个退出条件（benchmark_off / holding_period / volume_reversal）
  - 新增条件 4: 价格止损
    ```python
    if avg_cost and current_price:
        stop_pct = clip(atr_pct * vol_mult, min_pct, max_pct)
        if current_price <= avg_cost * (1 - stop_pct):
            return True, "price_stop_loss"
    ```
  - 新增条件 5: trailing protect
    ```python
    if high_since_entry and avg_cost and current_price:
        if high_since_entry >= avg_cost * (1 + activate_pct):
            if current_price < high_since_entry * (1 - trailing_stop_pct):
                return True, "trailing_protect"
    ```
  - 缺少参数时静默退化为原有行为（不报错）
  - 文件路径：`sprint_signal.py`

- [x] **T2-2** 单元测试：`tests/test_sprint_exit_v2.py`
  - Case 1: 价格跌破 ATR×3 → 返回 `(True, "price_stop_loss")`
  - Case 2: 价格在止损线上方 → 不触发，走原有逻辑
  - Case 3: 浮盈 5% 后回落 2% → 返回 `(True, "trailing_protect")`
  - Case 4: 浮盈 2%（未达 3% 激活线） → trailing 不触发
  - Case 5: 无 avg_cost/current_price → 退化为原有 3 条件
  - Case 6: benchmark_off 优先级高于价格止损（确保不冲突）

---

### [P0] T3. make_decision.py 集成止损检查

- [x] **T3-1** `make_decision.py` 修改决策生成流程，在信号计算前增加止损检查
  - 读取当前 positions: `symbol, avg_cost, high_since_entry`
  - 对每只持仓计算 ATR_pct（调用 T1-1）
  - 调用 `sprint_exit_check_v2()` 检查每只持仓
  - 触发止损的 symbol 加入 `forced_exit_tickers` 列表
  - `forced_exit_tickers` 传入 `execute_rebalance()` 的同名参数（已存在）
  - 在 decision artifact 中记录止损原因
  - 文件路径：`make_decision.py`

- [x] **T3-2** 集成测试：`tests/test_decision_stop_loss_integration.py`
  - 构造 mock DB：positions 有一只跌破止损线
  - 验证 `make_decision` 生成的 orders 包含该 symbol 的 SELL 指令
  - 验证 reason 字段包含 `"price_stop_loss"`

---

## 阶段二：持仓追踪增强（P1，预计 1 天）

### [P1] T4. positions 表 schema 扩展

- [x] **T4-1** `trade_schema.py` migration: positions 表新增 `high_since_entry REAL` 和 `entry_date TEXT`
  - 使用 `ALTER TABLE ... ADD COLUMN` + try/except 幂等模式
  - 现有数据 `high_since_entry` 初始化为当前 `market_price`
  - 现有数据 `entry_date` 初始化为当前 `asof`
  - 文件路径：`trade_schema.py`

- [x] **T4-2** `build_positions.py` 更新逻辑：每日收盘后更新 `high_since_entry`
  - `high_since_entry = max(prev_high_since_entry, today_high)`
  - `today_high` 从 `daily_prices` 取当日 high
  - 新建仓时 `high_since_entry = entry_price`, `entry_date = asof`
  - 文件路径：`build_positions.py`

- [x] **T4-3** 单元测试
  - Case 1: 新建仓 → high_since_entry = entry_price
  - Case 2: 连续3天上涨 → high_since_entry 递增
  - Case 3: 上涨后下跌 → high_since_entry 不变

---

## 阶段三：参数调优（P1，预计 0.5 天）

### [P1] T5. config.yaml Sprint 参数收紧

- [x] **T5-1** `config.yaml` `strategy_profiles.sprint` 更新以下参数：
  ```yaml
  stop_loss_vol_mult: 3.0        # 原值无（全局 6.0）
  stop_loss_min_pct: 0.04        # 原值 0.06
  stop_loss_max_pct: 0.12        # 原值 0.20
  max_position_pct: 0.35         # 原值 0.50
  max_dd_half: 0.08              # 原值 0.12（全局）
  max_dd_full: 0.12              # 原值 0.18（全局）
  trailing_activate_pct: 0.03    # 新增
  trailing_stop_pct: 0.02        # 新增
  ```
  - 文件路径：`config.yaml`
  - 验证：`python -c "import yaml; s=yaml.safe_load(open('config.yaml'))['strategy_profiles']['sprint']; assert s['stop_loss_vol_mult']==3.0; assert s['max_position_pct']==0.35"`

- [x] **T5-2** `make_decision.py` 读取 per-strategy 止损参数
  - 优先读取 `strategy_profiles.sprint.stop_loss_*`
  - Fallback 到全局 `model.exec.stop_loss_*`
  - 文件路径：`make_decision.py`

---

## 阶段四：回测验证（P2，预计 1 天）

### [P2] T6. 回测对比

- [x] **T6-1** ss7 回测引擎已原生支持 trailing stop / dynamic ATR stop loss，参数兼容
  - ss7 `BTConfig` 已有 `trailing_activate_pct`, `trailing_stop_pct`, `stop_loss_vol_mult` 等参数
  - 回测与实盘使用相同的止损逻辑（ATR 动态止损 + trailing protect）
  - 无需替换代码，只需传入 Sprint 加固参数即可
  - 验证脚本：`run_risk_backtest_comparison.py`

- [x] **T6-2** 回测对比完成
  - Baseline: vol_mult=6.0, 无 trailing, dd_half=0.12, dd_full=0.18
  - Variant: vol_mult=3.0, trailing 3%/2%, dd_half=0.08, dd_full=0.12
  - 结果: Sharpe 0.92→1.18, MaxDD -13.5%→-7.5%, Sortino 1.47→2.12
  - 收益略降 (-8pp) 但风险调整收益大幅提升
  - 输出报告：`reports/risk_hardening_backtest_comparison.md`

- [x] **T6-3** 全量测试回归
  - `python -m pytest tests/ -v` 全部通过
  - 确认现有 25/25 quant 测试不退化
  - 新增测试（T1-2, T2-2, T3-2, T4-3）全部通过

---

## 阶段五：长期优化（P3，无固定排期）

### [P3] T7. 组合回撤执行链路

- [ ] **T7-1** `make_decision.py` 增加组合级回撤检查
  - 读取 `account_snapshots` 计算当前回撤
  - 触发 `max_dd_half` → 所有持仓权重减半
  - 触发 `max_dd_full` → 全平仓，进入 cooldown
  - 当前此逻辑仅在 ss7 回测中存在，实盘缺失
  - 文件路径：`make_decision.py`

### [P3] T8. Harvest 策略止损参数独立配置

- [ ] **T8-1** 当 Harvest 激活（NAV ≥ 200万）时，使用独立止损参数
  - Harvest 适合更宽的止损：`vol_mult=4.0`, `min_pct=0.05`, `max_pct=0.15`
  - 当前优先级低，等 Sprint 阶段完成后再配置

### [P3] T9. 盘中实时止损监控

- [ ] **T9-1** `monitor_live_orders.py` 增加盘中止损价格监控
  - 基于 intraday_quotes 表的实时数据
  - 触发止损时通过 Discord webhook 发送警报
  - 需要配合 `intraday_update.py` 的定时运行

---

## 验收标准

| 阶段 | 验收条件 | 阻塞关系 |
|------|----------|----------|
| 阶段一 | T1-T3 全部完成，6 个新测试全绿，实盘链路可触发价格止损 | 无前置 |
| 阶段二 | T4 完成，positions 表新增列可读写 | 依赖阶段一 |
| 阶段三 | T5 完成，config 参数更新，make_decision 读取正确 | 依赖阶段一 |
| 阶段四 | T6 完成，回测报告产出，全量测试通过 | 依赖阶段一二三 |
| 阶段五 | T7-T9，长期项，不阻塞发布 | — |

**总估时**: 阶段一~四合计约 **4.5 天**

---

## 风险与注意事项

1. **向后兼容**：`sprint_exit_check()` 原函数保留不动，新增 `_v2` 版本。调用方逐步迁移。
2. **测试不退化**：每个阶段完成后跑 `pytest tests/ -v`，确保现有 25 个测试不破。
3. **回测 bias**：trailing stop 参数（3%/2%）基于当前市场波动率设定，需定期随 ATR 变化 review。
4. **Harvest 隔离**：所有改动仅影响 Sprint 策略。Harvest 使用全局参数，当前 disabled 不受影响。
5. **不修改 ss7 核心**：`ss7_sqlite_news_overlay.py` 的止损逻辑保持独立，T6-1 仅为回测对比，不改变生产信号链路。

---

*文档结束*
