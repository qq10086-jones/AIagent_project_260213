# Worker-Quant — 模型 Alpha 优化补丁设计文档

**作者**: Senior Quant Architect
**创建日期**: 2026-04-08
**状态**: DRAFT — 待审批
**对应任务清单**: `../tasks/TASKS_2026-04-08_Model_Alpha_Optimization.md`
**上承文档**: `PATCH_2026-04-07_Risk_Management_Hardening.md`, `PATCH_2026-04-07_Execution_Discipline_and_Regime_Optimization.md`
**触发原因**: 04-08 大盘暴涨 5%+，regime=OFF 导致完全踏空。模型审计发现择时层过于粗糙，跨资产信号完全缺失，Sprint 评分仅用 3/29 因子。

---

## 0. 问题审计

### 0.1 结构性矛盾

| 模型层 | 现状 | 评分 | 问题 |
|--------|------|------|------|
| 选股 | 29 因子 Ridge + screener | 8/10 | 因子利用率低（仅 3 个参与 sprint_score） |
| 择时 | MA20/MA60 交叉，二值 on/off | 4/10 | V 型反转踏空、窄幅震荡假信号 |
| 仓位 | Half-Kelly | 7/10 | 依赖 regime 二值输入，无法连续调节 |
| 退出 | ATR 止损 + trailing | 7/10 | 功能完备 |
| 跨资产 | 完全缺失 | 1/10 | 美股隔夜、USD/JPY 等免费 alpha 未利用 |

### 0.2 04-08 事件复盘

| 指标 | 04-07 (regime OFF) | 04-08 (盘中) |
|------|---------------------|--------------|
| 日经 ETF | 55,950 | 58,830 (+5.09%) |
| fast_ma | 55,559 | 需重算 |
| slow_ma | 56,638 | 需重算 |
| regime | OFF | OFF (MA 仍死叉) |
| 7267.T | 1252 (我们 1247.5 卖出) | 1286 (+2.7%) |

**问题本质**：MA 是滞后指标，在 V 型反转中有 1-3 天延迟。当日经从 55,950 暴涨到 58,830 (+5.15%) 时，MA20 仍未上穿 MA60，regime 仍输出 OFF。但实际上市场情绪已彻底反转。

### 0.3 因子利用率审计

```
总注册因子: 29
sprint_score 实际使用: 3 (mom_consist, high52w, vol_z)
factor_registry 有 IC: 19 (技术因子)
factor_registry 无 IC: 10 (全部基本面因子)
有效 ICIR (n_obs >= 20): 12
低样本 ICIR (n_obs < 5): 3 (vol_stability, sharpe_20, sortino_60)
```

10 个基本面因子数据已在 `feature_daily`（~385 行/因子），但从未接入 IC 计算管线，权重恒为 0。这是白白浪费的 alpha。

### 0.4 跨资产信息差分析

日股交易日时间轴（JST）：

```
           日股收盘     daily_run    美股开盘    美股收盘    晨报推送    日股开盘
  15:00 ──── 16:30 ──── 22:30 ──── 05:00 ──── 07:30 ──── 09:00
  ▲                                  ▲                       ▲
  当日数据                        14h 信息差              可用于决策
```

从 daily_run(16:30) 到次日晨报(07:30)，有 **15 小时窗口** 可以吸收：
- S&P500 涨跌（与日经次日相关性 ~0.6）
- USD/JPY 变化（与出口股相关性 ~0.4-0.7）
- VIX 变化（恐慌指标，比缺失的 1552.T 可靠）
- CME 日经期货（最直接的次日定价锚）

这些数据全部可以通过 yfinance 免费获取，零成本。

---

## 1. 优化一：跨资产领先指标体系 [P0]

### 1.1 设计目标

在晨报（07:30 JST）中接入美股隔夜数据，为 regime 判断和 action_plan 提供领先信号。

### 1.2 新因子定义

| 因子名 | 数据源 | 计算方式 | 更新时机 |
|--------|--------|----------|----------|
| `sp500_overnight` | `^GSPC` | 美股当日收盘收益率 | 晨报 07:30 |
| `usdjpy_change` | `USDJPY=X` | 24h 变化率 | 晨报 07:30 |
| `vix_level` | `^VIX` | VIX 收盘值 | 晨报 07:30 |
| `vix_change` | `^VIX` | VIX 日变化率 | 晨报 07:30 |
| `nk_futures_gap` | `NKD=F` | CME 日经期货 vs 前日收盘的 gap% | 晨报 07:30 |

### 1.3 数据采集模块

新建 `cross_asset_signals.py`：

```python
def fetch_cross_asset_snapshot() -> dict:
    """07:30 JST 调用，获取隔夜跨资产信号。
    Returns:
        {
            "asof": "2026-04-08",
            "sp500_close": 5123.4,
            "sp500_overnight_pct": -0.82,
            "usdjpy": 148.3,
            "usdjpy_change_pct": +0.45,
            "vix_close": 22.1,
            "vix_change_pct": +5.3,
            "nk_futures": 37250,
            "nk_futures_gap_pct": +1.2,
            "fetch_ts": "2026-04-08T07:30:00+09:00",
        }
    """

def compute_cross_asset_regime_signal(snapshot: dict) -> dict:
    """将跨资产数据转化为 regime 调整信号。
    Returns:
        {
            "cross_asset_score": 0.65,   # 0-1, 越高越 risk-on
            "components": {...},
            "regime_adjustment": "upgrade",  # upgrade / neutral / downgrade
        }
    """
```

### 1.4 评分公式

```
cross_asset_score = sigmoid(
    w1 × z(sp500_overnight)      # 美股隔夜收益 z-score
  + w2 × z(usdjpy_change)        # 日元走弱 = 利好出口股
  + w3 × (-z(vix_change))        # VIX 下降 = risk-on
  + w4 × z(nk_futures_gap)       # 期货溢价 = 市场看多
)
```

初始权重（基于日股文献和经验）：
- w1 = 0.35 (S&P500 最强领先指标)
- w2 = 0.20 (汇率对出口股)
- w3 = 0.20 (波动率)
- w4 = 0.25 (期货最直接)

sigmoid 将输出映射到 [0, 1] 区间。

### 1.5 数据库存储

`cross_asset_snapshots` 表：

```sql
CREATE TABLE cross_asset_snapshots (
    asof TEXT NOT NULL,
    ts TEXT NOT NULL,
    sp500_close REAL,
    sp500_overnight_pct REAL,
    usdjpy REAL,
    usdjpy_change_pct REAL,
    vix_close REAL,
    vix_change_pct REAL,
    nk_futures REAL,
    nk_futures_gap_pct REAL,
    cross_asset_score REAL,
    regime_adjustment TEXT,
    PRIMARY KEY (asof)
);
```

### 1.6 集成点

- `morning_briefing.bat` Step 1 新增: 调用 `cross_asset_signals.py`
- `action_plan_builder.py` 读取 `cross_asset_snapshots` 最新行
- 晨报 v2 新增: "跨资产信号" 小节
- `benchmark_regime.py` 新增 `cross_asset_score` 作为可选输入

---

## 2. 优化二：Regime 连续化 [P1]

### 2.1 设计目标

将 regime 从二值 {on, off} 升级为连续分数 [0.0, 1.0]，直接作为仓位缩放因子。消除 V 型反转中的盲区。

### 2.2 当前问题（数学分析）

当前 regime 判定：

```
IF fast_ma(20) > slow_ma(60) AND price > exit_line:
    state = "on"   →  scale = 1.0
ELSE IF fast_ma(20) < slow_ma(60) AND price < enter_line:
    state = "off"  →  scale = 0.25
```

MA 交叉本质是**差分滤波器**，对价格序列有 ~(window/2) 天的延迟。MA20/MA60 的延迟约 10-30 天。当市场一天暴涨 5%，MA20 仅上移约 0.25%（= 5%/20），需要持续 3-5 天才能翻转。

### 2.3 新 Regime 模型: `regime_score_v2`

```python
def compute_regime_score_v2(
    px_b: float,              # benchmark 当前价
    fast_ma: float,           # MA20
    slow_ma: float,           # MA60
    ma_slope_5d: float,       # MA20 5日斜率 (归一化)
    volume_ratio: float,      # 今日量 / 20日均量
    cross_asset_score: float, # 跨资产信号 (0-1)
    breadth: float | None,    # 涨跌比 (可选, 后期接入)
) -> float:
    """
    Returns regime_score in [0.0, 1.0].

    公式:
        raw = w_ma × ma_signal
            + w_slope × slope_signal
            + w_px × price_position_signal
            + w_vol × volume_signal
            + w_cross × cross_asset_score

    其中:
        ma_signal     = tanh((fast_ma - slow_ma) / slow_ma × k_ma)
                        # MA 间距归一化, 0 = 交叉点, >0 = 金叉
        slope_signal  = clip(ma_slope_5d × k_slope, -1, 1)
                        # MA 加速度, 捕捉趋势刚启动的瞬间
        price_position = tanh((px_b - slow_ma) / slow_ma × k_px)
                        # 价格相对慢线位置
        volume_signal = clip(log(volume_ratio) / log(5), 0, 1)
                        # 量能确认, 5x 以上视为强确认
    """
```

### 2.4 权重设定（初始值，待回测校准）

| 分量 | 权重 | 理由 |
|------|------|------|
| ma_signal | 0.30 | 传统趋势核心，保持但降权 |
| slope_signal | 0.20 | 捕捉 MA 转向的**加速度**——V 反转时斜率先变 |
| price_position | 0.15 | 价格突破位置 |
| volume_signal | 0.10 | 量能确认（防止假突破） |
| cross_asset_score | 0.25 | 跨资产领先信号 |

### 2.5 与现有系统的兼容性

`benchmark_regime.py` 新增函数 `benchmark_regime_score_v2()`，保留原 `benchmark_regime_scale_v2()` 不动。
通过 `config.yaml` 切换：

```yaml
benchmark_regime:
  version: "v2"          # "v1" = 原二值逻辑, "v2" = 连续化
  v2_weights:
    ma_signal: 0.30
    slope_signal: 0.20
    price_position: 0.15
    volume_signal: 0.10
    cross_asset: 0.25
  v2_thresholds:
    full_position: 0.70    # score > 0.7 → scale=1.0
    zero_position: 0.15    # score < 0.15 → scale=off_scale
    # 0.15-0.70 → 线性插值
```

### 2.6 04-08 回推验证

如果 regime_v2 在 04-07 收盘时运行：

```
ma_signal      = tanh((55559-56638)/56638 × 50) = tanh(-0.95) ≈ -0.74  → 负面
slope_signal   = MA20 过去5日从 55800→55559, slope ≈ -0.043%/day       → 负面
price_position = tanh((55950-56638)/56638 × 50) = tanh(-0.61) ≈ -0.54  → 负面
volume_signal  = 正常量能 → 0.3
cross_asset    = 假设美股当晚涨 2%, VIX 跌, 期货涨 → 0.75

raw = 0.30×(-0.74) + 0.20×(-0.3) + 0.15×(-0.54) + 0.10×0.3 + 0.25×0.75
    = -0.222 + (-0.06) + (-0.081) + 0.03 + 0.1875
    = -0.146

regime_score ≈ sigmoid(-0.146) ≈ 0.46
```

在 v1 下这是 OFF (scale=0.25)，在 v2 下这是 **scale ≈ 0.55**——允许半仓试探性建仓。

如果 04-08 盘中美股大涨后更新晨报：
- cross_asset_score 会跳到 ~0.85
- volume_signal 会跳到 ~0.9 (14.7x 量能)
- regime_score 会跳到 ~0.60-0.65

**结论**：v2 不会在 04-08 踏空，会以 50-65% 仓位入场，同时在真正的下跌趋势中仍然 <0.20 保持防御。

---

## 3. 优化三：Sprint Score 因子重整 [P2]

### 3.1 设计目标

将 sprint_score 从 3 因子硬编码升级为全因子自适应加权，充分利用已有的 29 个因子。

### 3.2 当前 sprint_score 逻辑

```python
# sprint_signal.py:41
def sprint_score(features, ic_weights=None):
    weights = ic_weights or {"mom_consist": 1.0, "high52w": 1.0, "vol_z": 1.0}
    # 仅遍历 3 个硬编码因子
    for factor in ["mom_consist", "high52w", "vol_z"]:
        ...
```

### 3.3 新评分模型: `sprint_score_v2`

```python
def sprint_score_v2(
    features: pd.DataFrame,
    factor_registry: dict,      # {factor_name: {icir, n_obs, weight, is_active}}
    tier_config: dict,          # core / candidate / excluded
    regime_score: float = 0.5,  # 用于因子动态加权
) -> pd.Series:
    """
    全因子自适应评分。

    权重计算:
        raw_weight = ICIR × confidence_shrink(n_obs) × tier_mult
        adaptive_weight = raw_weight × regime_factor_tilt(regime_score, factor_type)

    confidence_shrink(n_obs):
        # 低样本收缩: n<20 → 强收缩, n>50 → 几乎不收缩
        shrink = min(n_obs / 50, 1.0)
        return icir × shrink + prior_icir × (1 - shrink)

    tier_mult:
        core = 1.0, candidate = 0.7, fundamental_pending = 0.5, excluded = 0.0

    regime_factor_tilt:
        # regime 高时偏重动量因子, regime 低时偏重防御/价值因子
        if factor_type == "momentum":
            return 0.5 + 0.5 × regime_score
        elif factor_type == "value/defensive":
            return 1.0 - 0.3 × regime_score
        else:
            return 1.0
    """
```

### 3.4 因子分类标签

| 类别 | 因子 | regime 高加权 | regime 低加权 |
|------|------|:----------:|:----------:|
| momentum | mom_consist, ret20, ret60, mom_12_1, high52w | 加强 | 减弱 |
| mean_reversion | rsi14, z_20, vol_z | 减弱 | 加强 |
| risk | sharpe_60, sharpe_20, sortino_60, vol_stability | 中性 | 中性 |
| fundamental | value_bp, roa_op, cfo_assets, margin_op, etc. | 减弱 | 加强 |

### 3.5 基本面因子 IC 接入

当前问题：10 个基本面因子从未接入 `compute_ic.py`，因此 `factor_registry` 中无 IC 数据。

修复：
1. `compute_ic.py` 扩展 feature_list，加入 10 个基本面因子名
2. 首次运行后 `factor_registry` 会填入 IC/ICIR
3. `sprint_score_v2` 自动读取并加权

### 3.6 低样本 ICIR 修复

当前 3 个因子（vol_stability, sharpe_20, sortino_60）仅 2 个观测，ICIR 统计不可靠。

修复：Bayesian shrinkage

```
effective_icir = icir × shrink_factor + prior × (1 - shrink_factor)
shrink_factor = n_obs / (n_obs + shrink_constant)  # shrink_constant = 30
prior = 0.05  # 弱先验（微弱正向）
```

n=2 时 shrink_factor = 2/32 = 0.0625，几乎完全用先验，避免异常值主导。

---

## 4. 优化四：执行时机优化 [P2]

### 4.1 设计目标

`make_decision.py` 输出附带限价区间建议，配合 `intraday_monitor.py` 在价格进入区间时触发提醒。

### 4.2 限价建议模型

```python
def compute_entry_zone(
    symbol: str,
    signal_side: str,         # BUY / SELL
    last_close: float,
    atr_pct: float,
    signal_strength: float,   # sprint_score 归一化
) -> dict:
    """
    Returns:
        {
            "aggressive_limit": 价格,   # 开盘追进（信号强时）
            "target_limit": 价格,       # 目标入场价（回调等）
            "walk_away_price": 价格,    # 超过此价放弃
        }
    """
    if signal_side == "BUY":
        # 基础策略: 开盘后等回调 0.3-0.5 ATR
        target = last_close * (1 - atr_pct * 0.3)
        aggressive = last_close * (1 + atr_pct * 0.2)  # 最多追 0.2 ATR
        walk_away = last_close * (1 + atr_pct * 1.0)   # 涨超 1 ATR 放弃
    else:  # SELL
        target = last_close * (1 + atr_pct * 0.3)
        aggressive = last_close * (1 - atr_pct * 0.2)
        walk_away = last_close * (1 - atr_pct * 1.0)

    return {
        "aggressive_limit": round(aggressive, 1),
        "target_limit": round(target, 1),
        "walk_away_price": round(walk_away, 1),
    }
```

### 4.3 集成

- `make_decision.py` → `orders_proposal.csv` 新增列: `target_limit`, `aggressive_limit`, `walk_away_price`
- `action_plan_builder.py` → `action_plan_today.json` 包含限价区间
- `intraday_monitor.py` → 盘中价格进入 target_limit ±0.2% 时触发 Discord 推送:
  "4005.T 当前 528.5，接近目标入场价 527.0，建议挂限价单"

---

## 5. 不做清单（Anti-Scope）

| 排除项 | 理由 |
|--------|------|
| 换用 ML / 深度学习 | 20 只股票 × 50 天，样本远不足，过拟合风险 >80% |
| 新增更多技术因子 | 现有 15 个已严重共线性（ret20/ret60/mom_12_1 ρ>0.7） |
| 追求 90% 胜率 | Sprint 策略本质低胜率高赔率，期望值 > 胜率 |
| 分钟级高频化 | 40 万 JPY 本金，手续费吃掉 alpha |
| 自动下单 broker API | 当前阶段手动执行已够，自动化风险高 |
| 因子共线性去除 | 有效但优先级低于上述 4 项，排入后续迭代 |

---

## 6. 预期收益

### 6.1 回测预估（保守）

| 指标 | 当前 (v1) | 优化后 (v2, 预期) | 提升来源 |
|------|-----------|-------------------|----------|
| Sharpe | 1.18 | 1.4-1.6 | Regime 连续化减少踏空 |
| MaxDD | -7.5% | -6.0% ~ -7.0% | 跨资产提前预警 |
| 胜率 | ~45% | ~50% | 执行优化 + 限价 |
| Regime 误判率 | ~30% | ~15% | 多维度信号融合 |
| 执行滑点 (每手) | ¥1,000-2,000 | ¥300-500 | 限价区间 |

### 6.2 上线路径

```
Phase 1 (1-2天): 跨资产信号采集 → shadow 模式（仅显示在晨报，不影响决策）
Phase 2 (3-5天): Regime v2 实现 → 回测对比 → shadow 切换
Phase 3 (2-3天): Sprint score v2 → 基本面因子接入 IC → shadow
Phase 4 (1-2天): 执行限价建议 → 接入 intraday_monitor
Phase 5 (5-10天): Shadow 期观察 → 逐步切换为 live
```

---

## 7. 风险与回退

| 风险 | 缓解 |
|------|------|
| Regime v2 回测过拟合 | Phase 2 先 shadow 运行 2 周再切换 |
| 跨资产数据延迟/缺失 | 所有因子 fallback to 0.5（中性），不影响原 MA regime |
| 基本面 IC 样本不足 | Bayesian shrinkage 保护，低样本权重趋近先验 |
| yfinance 限流 | 已有 retry 机制；CME 期货可能需要替代源 |

---

*本文档待用户审批后进入实施阶段。*
