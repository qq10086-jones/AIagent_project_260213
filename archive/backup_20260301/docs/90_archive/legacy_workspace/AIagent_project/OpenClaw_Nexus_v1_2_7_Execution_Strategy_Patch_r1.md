# OpenClaw Nexus v1.2.7 Execution Strategy Patch (r1)

## 1. 简介 (Introduction)
针对用户在 Discord 中提出的交易执行（挂单定价）与**小额本金/离散手数（日本股票 100 股一手）**带来的组合构建问题，本 Patch 聚焦在 **Execution Layer（执行层）** 与 **Capital Constraint Layer（资金约束层）** 的工程化补齐：  
- 不修改你的核心选股/信号算法（Signal Layer）。  
- 仅补充“如何下单、下多少、挂在哪、如何自测”的可落地规则与接口。

---

## 2. 设计原则 (Design Principles)
1. **解耦**：Signal 与 Execution 分离，Execution 可被 `/analyze`、`/discovery`、`/rebalance` 复用。  
2. **可验证**：每条规则都能写单元测试/回放测试验收。  
3. **离散可执行**：所有 sizing 输出必须能落到 100 股单位，并考虑手续费/滑点 buffer。  
4. **弹性资金**：本金不锁死；允许 **+20%~+30% 向上增量**（预留余量）。

---

## 3. 核心改进 (Core Enhancements)

### 3.1 执行层接口拆分 (Signal ↔ Execution Contract)

#### 3.1.1 Signal Layer（不改你的核心算法）
建议标准化输出（示例字段）：
```json
{
  "symbol": "9432.T",
  "signal": {
    "side": "buy|sell|hold",
    "strength": 0.0,
    "time_horizon": "swing|position",
    "risk_hint": {
      "stop_ref": null,
      "take_profit_ref": null
    }
  }
}
```

#### 3.1.2 Execution Layer（本 patch 重点新增/固化）
新增执行建议输出（供 UI/Discord 直接展示）：
```json
{
  "execution_suggestions": {
    "urgency": "aggressive|balanced|patient",
    "validity": "DAY|GTC",
    "limit_prices": {
      "aggressive": 0,
      "balanced": 0,
      "patient": 0
    },
    "sizing": {
      "lot_size": 100,
      "shares": 0,
      "notional_jpy": 0,
      "cash_buffer_jpy": 0,
      "reason_if_zero": ""
    },
    "constraints_snapshot": {
      "tick_size": 0,
      "max_position_pct": 0.25,
      "capital_base_jpy": 0,
      "capital_headroom_pct": 0.2,
      "capital_budget_jpy": 0
    }
  }
}
```

---

### 3.2 挂单定价逻辑 (Execution Pricing Module)

#### 3.2.1 新增工具：`quant.calc_limit_price`
目标：根据波动与（可得则）盘口信息，输出 “激进 / 稳健 / 保守” 三档挂单价，并处理边界条件。

**输入建议：**
- `symbol`
- `side`（buy / sell）
- `asof`（YYYY-MM-DD）
- `close`（昨日收盘）
- `atr14`（14日 ATR）
- `ma5`（5日均线，可选）
- `best_bid/best_ask/spread`（可选，有就用）
- `tick_size`（必须：价格最小跳动）
- `price_guard_pct`（默认 0.10：昨日收盘 ±10% 的安全栏，可配置）

**核心逻辑（建议实现为可测规则，而非“描述性建议”）：**
1. **ATR clamp（防极端）**  
   - `atr_eff = clamp(atr14, atr_min, atr_max)`  
   - 默认：`atr_min = close * 0.003`（0.3%）  
   - 默认：`atr_max = close * 0.08`（8%）

2. **基础锚点（anchor）**  
   - `anchor = close`  
   - 若 `ma5` 可得：  
     - 买入：`anchor = min(close, ma5)`（更贴近回调）  
     - 卖出：`anchor = max(close, ma5)`（更贴近反弹/趋势）

3. **三档价格（以 ATR 为步进）**  
   - 买入（BUY）：
     - aggressive: `anchor + 0.15 * atr_eff`
     - balanced:   `anchor - 0.10 * atr_eff`
     - patient:    `anchor - 0.35 * atr_eff`
   - 卖出（SELL）：
     - aggressive: `anchor - 0.15 * atr_eff`
     - balanced:   `anchor + 0.10 * atr_eff`
     - patient:    `anchor + 0.35 * atr_eff`

4. **盘口约束（可得则 clamp 到合理区间）**  
   - BUY：`limit <= best_ask + k*spread` 且 `limit >= best_bid`（默认 k=1）  
   - SELL：`limit >= best_bid - k*spread` 且 `limit <= best_ask`  

5. **价格安全栏（必做）**  
   - `lower_guard = close * (1 - price_guard_pct)`  
   - `upper_guard = close * (1 + price_guard_pct)`  
   - 三档价格必须 clamp 到 `[lower_guard, upper_guard]`

6. **tick size 取整（必做）**  
   - BUY：向下取整到 tick（避免报更高价）  
   - SELL：向上取整到 tick（避免报更低价）

> 说明：以上参数（0.15/0.10/0.35、0.3%/8%、±10%）建议作为默认值写入配置，后续可在 UI 侧提供“保守/激进”滑杆，而不改核心逻辑。

---

### 3.3 小额本金组合策略（含 20%~30% 余量）(Capital-Constrained Portfolio)

#### 3.3.1 资金预算定义（不锁死本金）
输入：`capital_base_jpy`（用户声明本金）  
配置：`capital_headroom_pct`（默认 0.25，允许范围 0.20~0.30）

- `capital_budget_jpy = capital_base_jpy * (1 + capital_headroom_pct)`
- 同时输出两套口径，避免“预算”被误解成必须用满：  
  - **base**：用于风险控制与仓位上限  
  - **budget**：用于可执行性（允许多买一点以满足 100 股手数离散）

#### 3.3.2 单仓位上限与可买性筛选
默认 `max_position_pct = 0.25`（可配置）

- 单仓上限（按 base 计算更稳健）：  
  - `max_pos_jpy = capital_base_jpy * max_position_pct`
- 可买性硬过滤（100 股一手）：  
  - `price * 100 <= max_pos_jpy`
- 若用户允许更集中，也可开关：用 budget 计算上限（不推荐默认启用）。

#### 3.3.3 增加两条“执行可落地”过滤（建议默认开启）
1. **流动性过滤（避免滑点/挂不到）**  
   - 例如：`avg_daily_value_20d >= X`（X 可先设 50,000,000 JPY 作为保守默认，后续按你数据源调整）
2. **波动→仓位上限联动（风险层，不动选股因子）**  
   - 例如：若 `atr_eff/close > 0.04`（4%），则将该标的 `max_position_pct` 从 25% 下调到 15%（可配置）

#### 3.3.4 输出必须包含“可执行手数”
`discovery_workflow` 输出建议包含：
- 每个标的可买手数 `lots = floor(max_pos_jpy / (price*100))`（至少 1 才入选）
- 推荐买入的 `shares = lots * 100`
- 预计占用资金与剩余现金（含 buffer）

---

### 3.4 调仓与建仓建议 (Onboarding / Rebalance Strategy)

#### 3.4.1 分批建仓（Pyramid Entry）输出“可计算计划”
建议默认 3 档：
- tranche1：40%（先建底仓）
- tranche2：30%（趋势确认/回撤到位）
- tranche3：30%（突破/继续走强）

触发条件示例（写进文档，便于实现为规则）：
- tranche2（回撤加仓）：价格回撤至 `ma5` 附近或 `-0.3*ATR`  
- tranche3（突破加仓）：价格创新高或站上关键均线且 `strength` 上升

输出字段建议：
```json
{
  "entry_plan": {
    "tranche1": {"pct": 0.40, "trigger": "now"},
    "tranche2": {"pct": 0.30, "trigger": "pullback_to_ma5_or_-0.3ATR"},
    "tranche3": {"pct": 0.30, "trigger": "breakout_or_trend_confirm"}
  }
}
```

#### 3.4.2 空仓期策略（保持不变，仅明确触发条件）
- 当 `discovery_workflow` 找不到满足 **可买性 + 流动性 + 风险上限** 的标的：  
  - 记录为 `CASH`  
  - 触发 `news.daily_report` +（可选）`news.preclose_brief_jp` 做机会监控

---

## 4. 验收与测试 (Acceptance & Tests)

### 4.1 回放测试（Replay Test）
给定过去 60 天 OHLC（以及可得则 bid/ask）：
- 三档 limit 价格必须满足：  
  - BUY：patient ≤ balanced ≤ aggressive（单调）  
  - SELL：aggressive ≤ balanced ≤ patient（单调）
- 三档价格必须落在 `[close*(1-guard), close*(1+guard)]`
- tick 取整方向正确（BUY 向下，SELL 向上）

### 4.2 资金约束测试（Capital Constraint Test）
输入 `capital_base_jpy=400000`，`headroom=0.25`：
- 每个标的：`shares` 必须是 100 的倍数  
- 任何单标的占用资金（按 base 上限）不得超过 `25% * base`  
- 总占用资金不得超过 `budget`（含手续费/滑点 buffer）

---

## 5. 待实现代码清单 (To-Do List)
1. [ ] `worker.py`：增加 `_calculate_atr(symbol, period=14)`（或复用现有行情模块）  
2. [ ] 新增 `quant.calc_limit_price`：实现三档定价 + tick + guard +（可选）盘口 clamp  
3. [ ] `worker.py deep_analysis`：输出 `execution_suggestions`（含 limit_prices、sizing、constraints_snapshot）  
4. [ ] `discovery_workflow`：加入 `capital_base_jpy`、`capital_headroom_pct`，并输出可执行 `shares`（100 股单位）  
5. [ ] `discovery_workflow`：加入流动性过滤（数据源可先占位实现）  
6. [ ] 增加回放测试与资金约束测试（pytest 或简单脚本均可）

---

## 6. 指令示例 (Command Examples)
- `/analyze 9432.T mode:sell`  
  - 返回包含 `execution_suggestions.limit_prices`（三档）与 `tick/guard` 信息的卖出分析。

- `/discovery capital_base:400000 headroom:0.25`  
  - 返回适配小额本金的组合建议（每个标的给出可买 `shares`，并说明若为 0 的原因）。

- `/rebalance asof:2026-02-27 capital_base:400000 headroom:0.25`  
  - 输出 trade list（按 100 股取整）、每笔建议 limit（三档）、预计现金余额、以及无法成交/无法买入的解释字段。
