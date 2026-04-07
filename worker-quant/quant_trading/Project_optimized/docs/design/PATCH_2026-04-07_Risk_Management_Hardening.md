# Worker-Quant — 风控机制加固补丁设计文档

**作者**: PM + Quant Architect (Senior Review)
**创建日期**: 2026-04-07
**状态**: APPROVED — 待实施
**对应任务清单**: `../tasks/TASKS_2026-04-07_Risk_Management_Hardening.md`
**上承文档**: `DESIGN_v3.0_Dual_Strategy_Architecture.md`
**触发原因**: 实盘持仓 7267.T 信号退化后无自动止损触发，暴露风控执行链路缺失

---

## 0. 审计摘要

### 0.1 审计范围

| 模块 | 文件 | 审查结论 |
|------|------|----------|
| 止损参数 | `config.yaml` | 参数存在但未被实盘链路消费 |
| Sprint 退出逻辑 | `sprint_signal.py:sprint_exit_check()` | **无价格止损条件** |
| 决策生成 | `make_decision.py` | 不读取持仓成本，不检查止损 |
| 执行模型 | `execution_model.py` | 纯仓位调整，无风控门控 |
| Kelly 仓位 | `kelly_sizer.py` | 框架正确，上限参数偏激进 |
| 组合回撤 | `config.yaml` | 门槛偏松，适合 Harvest 不适合 Sprint |
| Benchmark Regime | `benchmark_regime.py` | 设计精良，无需修改 |
| 止盈 | 无 | **完全缺失** |

### 0.2 风险评级

| 缺陷 | 严重度 | 影响 |
|------|--------|------|
| 实盘无止损执行 | **CRITICAL** | 单笔亏损可超 ATR×6 = ~9% 才被动退出 |
| 止盈缺失 | HIGH | 盈利回吐，持有期到期才退出 |
| ATR 乘数偏大 | MEDIUM | Sprint 短周期下止损线等于摆设 |
| Kelly 上限偏高 | MEDIUM | 极端 edge 时单票 50% 暴露 |
| 组合回撤门槛偏松 | LOW | 12%/18% 对 40 万本金稍宽 |

---

## 1. 补丁设计：价格止损执行链路

### 1.1 问题描述

当前 `sprint_exit_check()` 只有三个退出条件：

```python
def sprint_exit_check(row, holding_days, benchmark_state):
    if benchmark_state == "off": return True, "benchmark_off"
    if holding_days >= target: return True, "holding_period"
    if vol_z < -0.5: return True, "volume_reversal"
```

**缺失**：无论股价跌多少，只要 regime 不是 off、持有天数没到、vol_z 没反转，模型不会触发卖出。

### 1.2 设计方案

在 `sprint_exit_check()` 新增两个退出条件：

```
条件 4: 价格止损 (stop_loss)
  触发: current_price <= avg_cost × (1 - stop_loss_pct)
  stop_loss_pct = min(max(ATR_20 × vol_mult, min_pct), max_pct)
  退出原因: "price_stop_loss"

条件 5: 保本止盈 (trailing_protect)
  触发: 曾浮盈 > trailing_activate_pct 且当前价格 < 最高价 × (1 - trailing_stop_pct)
  退出原因: "trailing_protect"
```

### 1.3 数据流

```
make_decision.py
  │
  ├── 读取 positions 表: symbol, avg_cost
  ├── 读取 daily_prices 表: 最新 close, ATR_20
  ├── 读取 config.yaml: stop_loss_vol_mult, min_pct, max_pct, trailing 参数
  │
  ├── 对每只持仓调用 sprint_exit_check_v2()
  │     ├── 原有 3 条件不变
  │     ├── + 价格止损检查
  │     └── + trailing protect 检查
  │
  └── 触发退出的持仓 → forced_exit_tickers → 传入 execute_rebalance()
```

### 1.4 接口变更

#### `sprint_signal.py`

```python
# 新函数签名
def sprint_exit_check_v2(
    row: pd.Series,
    holding_days: int,
    benchmark_state: str,
    *,
    avg_cost: float | None = None,
    current_price: float | None = None,
    high_since_entry: float | None = None,
    atr_pct: float | None = None,
    stop_loss_config: dict | None = None,
) -> tuple[bool, str]:
```

#### `config.yaml` 新增参数

```yaml
strategy_profiles:
  sprint:
    # 止损参数（Sprint 适配）
    stop_loss_vol_mult: 3.0        # ATR 乘数（从 6.0 降为 3.0）
    stop_loss_min_pct: 0.04        # 最低止损 4%
    stop_loss_max_pct: 0.12        # 最高止损 12%
    # 移动止盈参数
    trailing_activate_pct: 0.03    # 浮盈超 3% 激活保本
    trailing_stop_pct: 0.02        # 从最高点回撤 2% 触发
    # 仓位上限（收紧）
    max_position_pct: 0.35         # 从 0.50 降为 0.35
```

---

## 2. 补丁设计：ATR 计算集成

### 2.1 问题

ATR 参数 (`atr_window`, `stop_loss_vol_mult`) 仅在 `ss7_sqlite_news_overlay.py` 回测引擎中使用。实盘链路 (`make_decision.py`) 不计算 ATR。

### 2.2 方案

在 `make_decision.py` 中新增 `_compute_atr_pct()` 函数：

```python
def _compute_atr_pct(conn, symbol, asof, window=20):
    """计算 ATR 占收盘价百分比"""
    rows = conn.execute("""
        SELECT high, low, close FROM daily_prices
        WHERE symbol=? AND date<=?
        ORDER BY date DESC LIMIT ?
    """, (symbol, asof, window + 1)).fetchall()
    # 计算 True Range, 返回 ATR / close 的百分比
```

### 2.3 ATR 止损线计算公式

```
stop_loss_pct = clip(ATR_20_pct × vol_mult, min_pct, max_pct)
stop_price = avg_cost × (1 - stop_loss_pct)
```

**示例** (7267.T):
- 日均波幅约 1.5%, ATR_20_pct ≈ 0.015
- vol_mult = 3.0 → stop_loss_pct = 4.5%
- avg_cost = 1259.5 → stop_price = 1259.5 × 0.955 = **1202.8**
- clip 后（min 4%, max 12%）：stop_loss_pct = 4.5% ✓

对比原参数 (vol_mult=6.0):
- stop_loss_pct = 9.0% → stop_price = 1146.1（跌到 1146 才止损，不可接受）

---

## 3. 补丁设计：持仓最高价追踪 (trailing stop 前置条件)

### 3.1 问题

当前数据库无持仓期间最高价记录。trailing stop 需要 `high_since_entry`。

### 3.2 方案

扩展 `positions` 表，新增 `high_since_entry` 列：

```sql
ALTER TABLE positions ADD COLUMN high_since_entry REAL;
ALTER TABLE positions ADD COLUMN entry_date TEXT;
```

在 `build_positions.py` 每日更新时，取 `max(high_since_entry, today_high)` 写入。

---

## 4. 补丁设计：组合回撤门槛适配

### 4.1 当前值

```yaml
max_dd_half: 0.12    # ¥48,000 降半仓
max_dd_full: 0.18    # ¥72,000 全平仓
```

### 4.2 Sprint 适配值

```yaml
max_dd_half: 0.08    # ¥32,000 降半仓
max_dd_full: 0.12    # ¥48,000 全平仓
```

**理由**: Sprint 集中持仓 3 只，单票波动传导更剧烈。12%/18% 在 Harvest（12只分散持仓）下合适，Sprint 需要更敏感。

### 4.3 实现

这些参数已经可以按 `strategy_profiles` 分策略配置，只需在 sprint 配置块下覆盖即可。`make_decision.py` 已经读取 per-strategy config。

---

## 5. 参数变更汇总

| 参数 | 当前值 | 新值 | 位置 | 理由 |
|------|--------|------|------|------|
| `stop_loss_vol_mult` (Sprint) | 6.0 (全局) | **3.0** (Sprint) | `strategy_profiles.sprint` | Sprint 持有期 5 天，6×ATR 止损永远触发不了 |
| `stop_loss_min_pct` | 0.06 | **0.04** | `strategy_profiles.sprint` | 适配短周期 |
| `stop_loss_max_pct` | 0.20 | **0.12** | `strategy_profiles.sprint` | 20% 对 40 万本金太宽 |
| `max_position_pct` | 0.50 | **0.35** | `strategy_profiles.sprint` | 防止单票过度集中 |
| `max_dd_half` | 0.12 | **0.08** | `strategy_profiles.sprint` | 集中组合需更敏感 |
| `max_dd_full` | 0.18 | **0.12** | `strategy_profiles.sprint` | 同上 |
| `trailing_activate_pct` | 无 | **0.03** | `strategy_profiles.sprint` | 新增：浮盈 3% 激活保本 |
| `trailing_stop_pct` | 无 | **0.02** | `strategy_profiles.sprint` | 新增：最高价回撤 2% 触发 |

---

## 6. 向后兼容

- `sprint_exit_check()` 保留不动，新增 `sprint_exit_check_v2()`
- `sprint_exit_check_v2()` 在缺少 `avg_cost` / `current_price` 参数时，退化为原有行为（仅 3 条件检查）
- 全局 `exec.stop_loss_*` 参数保留作为 fallback，`strategy_profiles.sprint.*` 优先级更高
- `positions` 表新增列使用 `ALTER TABLE ... ADD COLUMN` + try/except，向后兼容
- 所有变更对 Harvest 策略无影响（Harvest 仍使用全局参数）

---

## 7. 回测验证要求

实施完成后，需运行以下验证：

1. `python -m pytest tests/ -v` — 全量测试不退化
2. 手动构造场景：持仓跌破 ATR×3，验证 `sprint_exit_check_v2()` 返回 `(True, "price_stop_loss")`
3. 手动构造场景：浮盈 5% 后回落 2%，验证返回 `(True, "trailing_protect")`
4. 回测对比：使用 `ss7_sqlite_news_overlay.py` 对比加入止损/止盈前后的 Sharpe / MaxDD

---

*文档结束*
