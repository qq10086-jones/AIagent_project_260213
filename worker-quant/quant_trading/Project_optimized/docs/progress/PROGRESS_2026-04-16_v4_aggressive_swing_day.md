# PROGRESS 2026-04-16 — v4.0 Aggressive Swing-Day + Alpha158 因子库

## 本阶段一句话

在 v3.8 walk-forward verdict 基础上，修复 3 个根因 bug（vol_z 符号反转 / min_adv 太低 / briefing 硬编码），将回测从 -9.5% 修复到 +44.4%；扩展因子库 3→68；实现 contrarian 研究并证伪 long-only 反转策略；最终落地 Core-Satellite 方案 B 的 Core 部分 + P&L 门控 + kill switch。

---

## 核心修复（3 个回测 -9.5% 的根因）

| Bug | 原值 | 修复值 | 影响 |
|---|---|---|---|
| **vol_z sign 搞反** | +1（买量能高潮） | **-1**（买缩量蓄势） | -9.5% → +35% |
| **min_adv 太低** | 5M（含微盘垃圾） | **50M**（流动大盘） | +35% → +48% |
| **briefing screener 硬编码** | 5M（不读 config） | 读 config.yaml | 候选池和回测一致 |

---

## 回测最终数据

### Sprint Trend（生产策略）

| 期间 | 毛收益 | 净收益 | Sharpe | MaxDD | vs 日经 | 成本模型 |
|---|---|---|---|---|---|---|
| 2025-01~04（15月） | +52.3% | **+44.4%** | 2.96 | -6.2% | -2% | sqrt 22bps |
| 2024-01~04（23月） | +51.0% | **+35.0%** | 1.11 | -18.8% | -4.5% | sqrt 22bps |

**Bootstrap 95% CI**：15月 Sharpe [+2.08, +13.48]（显著 > 0），23月 [-0.87, +4.53]（和日经 CI 重叠）

### 配置
```
因子: mom_consist(+1) + high52w(+1) + vol_z(-1)
宇宙: ADV ≥ 50M，top_k=10
仓位: aggressive_kelly fraction=0.75, max_single=60%
止损: ATR 2.0x (min 3%, max 8%)
Trailing: activate 1.5%, stop 1.2%
回撤: half at -10%, full exit at -20%
Leverage: 2x 信用, trigger score ≥ 0.85
```

### 证伪策略

| 策略 | 结果 | 原因 |
|---|---|---|
| Monthly Contrarian (5 factors, long-only) | **-29% vs 日经** | 月度 rebalance 杀掉 T+1 alpha |
| Satellite T+1 (daily, 2d hold, z>2σ) | **-100% (归零)** | Long-only 吃不到 rank alpha，牛市中反选 |

---

## 新增文件（12 个）

| 文件 | 职责 |
|---|---|
| `factor_library.py` | Alpha158 风格 41 个短期量价因子 |
| `factor_diagnostics.py` | alphalens-reloaded 因子体检 |
| `factor_diagnostics_batch.py` | 批量版 IC 诊断（全宇宙 40x 提速） |
| `factor_autofix.py` | IC 反向检测（防 vol_z 再搞反 6 个月） |
| `risk_kill_switch.py` | 4 规则熔断：单日-5%/3日-10%/连3止损/组合-20% |
| `feature_gate.py` | P&L 门控：实盘盈利覆盖年费才解锁付费 API |
| `satellite_signal.py` | T+1 反转候选扫描（默认关闭） |
| `satellite_backtest.py` | Satellite 策略 daily 回测（证伪用） |
| `contrarian_backtest.py` | Monthly contrarian 回测（证伪用） |
| `aggressive_kelly_sizer.py`（集成到 kelly_sizer.py） | Kelly 0.75 + signal/vol/regime 三重缩放 |

## 编辑文件（9 个）

| 文件 | 变更 |
|---|---|
| `config.yaml` | v4.0 全配置：aggressive params + kill switch + feature_gates + satellite_enabled=false |
| `sprint_signal.py` | vol_z weight=-1 恢复到合成 + entry_check 放宽 0.70/0.75 + caution 梯度 |
| `quant_briefing.py` | 持仓健康表（HoldScore/Exit信号/保护期） + 交易日志 + satellite 章节（带警告） + screener 读 config |
| `walk_forward_runner.py` | vol_z_sign=-1 默认 + sqrt 滑点 + purge/embargo + block bootstrap CI |
| `kelly_sizer.py` | AggressiveKellySizer 类 |
| `make_decision.py` | aggressive_kelly dispatch |
| `compute_price_features.py` | OHLCV 加载 + factor_library 41 因子接入 |
| `tests/test_sprint_signal.py` | 适配 v4.0 语义（entry 0.70, vol_z climax filter, vol spike 不再要求） |

## 测试状态

**199/199 PASS**（0 fail）

---

## 因子 IC 全面诊断（44 因子 × 10 个月 × 2000 股流动池）

### 发现

1. **Top 16 因子全部是短期反转 (negative IC)**：ksft2 t=-12.66, ma_gap_3 t=-6.74, kmid t=-7.37
2. **唯一正向趋势因子**：high52w t=+2.91, IC(T+21)=+0.124（长期越强）
3. **mom_consist 勉强显著**：t=+1.97, IC(T+21)=+0.056
4. **vol_z 不显著**：t=-1.04（但在组合中贡献 +3% alpha）
5. **反转 alpha 真实存在但 long-only 无法捕获**：需要 market-neutral long-short

### 战略结论

日股大盘流动池 (2025-2026) 的 alpha 分两层：
- **T+1~T+3 短期反转**（IC 极强但衰减快，需 long-short + 日频 rebalance）
- **T+20+ 中期趋势**（high52w 独大，月度 rebalance 可用）

当前 Sprint Trend 走第二条路（可行），Satellite 走第一条路（失败，因 long-only）。

---

## Feature Gate 状态

| API | 阈值 | 当前 P&L | 进度 |
|---|---|---|---|
| J-Quants Light | ¥19,800 | ¥420 | 🔒 2.1% |
| J-Quants Premium | ¥198,000 | ¥420 | 🔒 0.2% |
| Polygon Starter | ¥54,000 | ¥420 | 🔒 0.8% |

---

## 当前阶段

**4 周实盘观察期（2026-04-17 ~ 05-15）**

- Sprint Trend 策略生产运行
- Kill switch + factor IC 自动检测
- 不做新开发，等实盘数据
- 4 周后 review：P&L ≥ ¥20k → 推 PATCH 8 风险模型 + J-Quants

---

## 下一步 (Week 3+ 待解锁)

| 优先级 | 内容 | 前置条件 |
|---|---|---|
| 1 | PATCH 8 风险模型（行业中性化 + 协方差收缩） | 无 |
| 2 | J-Quants Light 数据源 | P&L gate ≥ ¥19,800 |
| 3 | Market-neutral long-short | SBI 信用账户 + J-Quants |
| 4 | LightGBM ML 模型 | J-Quants + PATCH 8 |

---

*Commit pending — 等用户确认后 commit*
