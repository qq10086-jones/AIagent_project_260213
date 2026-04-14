# PROGRESS 2026-04-14 (下午) — Walk-forward runner 落地 + 项目战略结论重置

## 本次会话一句话

**项目过去 6 个月的 sprint momentum 方向被 OOS 数据证伪；找到的替代候选（Amihud illiquidity）在更早的 2023 年段也不成立；N=23 + 30 次试错样本根本不足以识别任何真 alpha。** 诚实结论：目前没有可部署的信号。

---

## 实盘动作（已发生）

- 2026-04-14 SBI 成交：BUY 3041.T 400@¥585（14:45 信号挂单 → 日内触发）
- 成交后 NAV ¥400,145 / 现金 ¥166,545 / 持仓 3041.T × 400

---

## 代码改动

### (1) `kelly_sizer.py` — P0 实盘风险修复
- `fallback_position_pct` 默认 0.25 → **0.0**
  - 原逻辑："样本<30 时给 25% 单仓" 等于"无证据重仓"
- 新增 `--kelly_bootstrap_pct` CLI，可显式 opt-in bootstrap 仓位
- 新增单测 `test_low_sample_defaults_to_zero_no_evidence_no_position`

### (2) `sprint_signal.py` — 移除 vol_z from alpha composite
- `sprint_score()` 从 3 因子降为 2 因子：`mom_consist + high52w`
- 原因：ablation 证明 `vol_z` 让净收益从 +26.1% (2f) 掉到 +14.3% (3f)
- `vol_z` 保留为 entry filter（not linear alpha）
- 12 tests pass

### (3) `walk_forward_runner.py` — 新文件（~280 行）
- 月度 rebalance, top-K 等权, T+1 open 成交, 10bps fee + 配置 slippage
- 支持 `--factors` 子集、`--vol_z_sign`、`--alpha_factor` 单因子、`--alpha_composite` 多因子合成、`--long_short`
- 基准与策略窗口**完全对齐**（同 fill_day → exit_day）
- 宇宙排除 ETF/REIT（1XXX/2500-2599）
- v1 vs v0 改进：窗口对齐（mom_consist 20d→63d, vol_z 20d raw→60d log）、T+1 open、aligned bench

### (4) `analyze_amihud_robustness.py` — 新文件
- Block-bootstrap CI（N_sim=5000, block=3）
- Deflated Sharpe Ratio（Lopez de Prado, 30 trials prior）
- top-1 name contribution share, sector concentration

---

## Walk-forward v1 结果总表（2024-01 → 2026-04, 23 个月, top-K=5 unless noted）

### 原三因子 sprint 及其 ablation（用生产窗口 63/252/60, log-volume）

| 配置 | gross | **net** | Sharpe | MaxDD | vs EW 月 |
|---|---|---|---|---|---|
| 原三因子 baseline | +28.0% | **+14.3%** | 0.40 | −18.5% | −0.43% |
| mom_consist 单 | +30.4% | +18.9% | 0.58 | −24.2% | −0.36% |
| **high52w 单** | +42.7% | **+30.1%** | **0.88** | −13.1% | +0.02% |
| vol_z 单 (+1) | +25.7% | +12.1% | 0.33 | −14.6% | −0.43% |
| vol_z 单 (−1) | −4.7% | −14.7% | −0.28 | −29.4% | −1.75% |
| mom + high52w | +38.0% | +26.1% | 0.85 | −13.7% | −0.13% |
| mom + vol_z | +16.6% | **+4.2%** | 0.21 | −19.7% | −0.80% ← vol_z 毒药 |
| 3f with vol_z(−1) | +38.8% | +25.0% | 0.83 | −18.2% | −0.17% |
| high52w LONG-SHORT | +54.0% | +44.1% | 0.91 | −24.0% | +0.58% |
| 1321.T (基准) | | +39.5% | 1.05 | −9.5% | — |
| 宇宙等权（基准） | | +31.7% | 1.48 | −7.9% | — |

### A-1 新因子库单因子 ablation（top-K=5, ADV>5M, slip=15bps）

| 因子 | 文献方向 | net cum | Sharpe | MaxDD | 判决 |
|---|---|---|---|---|---|
| alpha_roc_3 (+) | Jegadeesh-Titman | **−88.5%** | −2.16 | −89% | 灾难 |
| alpha_roc_10 (+) | JT | −78.4% | −0.59 | −81% | 灾难 |
| alpha_jt_mom_6m_skip1m (+) | JT | −13.1% | 0.14 | −59% | 废 |
| alpha_reversal_1 (+) | Jegadeesh 90 | −34.9% | −0.48 | −39% | 废 |
| alpha_max_ret_20 (+) | BCW lottery | −71.9% | −1.34 | −77% | 灾难 |
| alpha_min_ret_20 (+) | — | +8.4% | 0.42 | **−3.3%** | defensive, 低 DD |
| alpha_ret_skew_60 (+) | Harvey-Siddique | +11.1% | 0.34 | −17% | 弱 |
| **alpha_amihud_20 (+)** | Amihud 2002 | **+21.6%** | 0.69 | −9.3% | 候选 |
| alpha_range_proxy_20 (+) | Parkinson-like | −58.8% | −0.88 | −66% | 灾难 |
| alpha_hl_ratio_20 (+) | ABD 2002 | −59.0% | −0.97 | −66% | 灾难 |
| alpha_parkinson_vol_20 (+) | Parkinson 80 | −55.0% | −0.76 | −70% | 灾难 |

**动量/范围类全部灾难（11 个因子 8 个是 Sharpe<0）**。日股 2024-2026 样本里 long-top-momentum 等于高位接盘。

### Amihud 稳健性矩阵（top-K=20, 2024-01 → 2026-04）

| ADV 门槛 | slip | cum_net | Sharpe | P(cum>0) | 备注 |
|---|---|---|---|---|---|
| 5M | 15 | +123% | 2.57 | — | 不可信（ADV 太松） |
| 20M | 30 | +34% | 0.93 | 85% | — |
| 20M | 100 | +7% | 0.28 | 56% | 无 edge |
| 50M | 30 | +34% | 0.75 | 80% | — |
| 50M | 100 | **+7%** | 0.26 | 56% | 无 edge |
| 100M | 30 | +112% | 0.84 | 91% | 看起来很好 |
| **100M** | **100** | **+73%** | **0.67** | **81%** | 看起来可执行 |

### **决定性 out-of-sample 验证**（ADV≥100M, slip=50bps, k=20）

| 窗口 | N | cum_net | Sharpe | P(>0) | 结论 |
|---|---|---|---|---|---|
| 2024-01 → 2026-04 (IS) | 23 | **+100%** | 0.79 | 88% | 看起来很好 |
| **2023-04 → 2024-01 (OOS 更早)** | 6 | **−3.1%** | **−0.52** | **25%** | **edge 不存在** |

**Amihud 是 regime-dependent，不是稳定 premium。** 2023 日股大盘股牵头，2024 起小盘轮动回来。

### 多重检验修正

所有配置 **DSR p = 0.000**（与 30 试错 prior 对比）。N=23 + 30 trials 根本不够排除 lucky outcome。

---

## 我作为 senior quant 的硬判断

1. **项目过去 6 个月的 sprint momentum 方向在 2024-2026 日股 OOS 是负 alpha**。原 3 因子 net +14.3% vs EW +31.7%。
2. **vol_z 是因子库里的毒药**（方向模糊的 event 信号，当线性 alpha 使用反向）。已移除。
3. **A-1 新增 11 个学术因子中 8 个灾难性亏损**，说明"不发明因子 + 学术引用"原则不能保证在日股样本上有效。
4. **Amihud illiquidity 在 IS 段看似有 edge，OOS 更早段证伪**。不是真 alpha，是样本内 regime。
5. **多重检验让 Sharpe 0.5-1.0 都不具备统计显著性**，N=23 个月的样本量与 30 次试错不兼容。
6. **项目当前没有任何一个可部署的信号**。

---

## 下一步路线（取决于用户战略决定）

### Path A：接受诚实结论，战略收缩

- 实盘只留 3041.T + 7984.T 现有两仓，设 ATR −6% 硬止损
- 自动建仓**无限期暂停**，直到找到通过 OOS + DSR 的信号
- 改 Phase A 扩 60 因子计划 → **先用 JQuants 批量回填 2018-2023 数据**（3 年以上样本才可能通过 DSR）
- 10 周里程碑调整：不是"5.5/10 对标 Qlib"，而是"证明任何一个策略 OOS + DSR 显著"

### Path B：承认 edge-seeking 不可行，转方向

- 放弃日频选股 alpha，转事件驱动（earnings surprise / guidance revision / macro shock）
- 或转为受控实验：每月小仓多策略并行（每策略 ¥20k），积累 PnL 样本
- 600% 年化目标必须独立讨论（不能单靠信号优化达成）

### Path C：保留现状 + 纯降级

- 既不扩因子也不改方向，当前所有自动建仓停用
- 每日脚本继续跑，但只产出 watchlist 不下单
- 用户手动在 SBI 做判断，每月 review 与机器信号分歧

---

## 未回答的关键问题

- DB 是否含退市票（survivorship）？Amihud 对这个敏感度高
- JQuants 能否回填 2018-2023 数据？样本翻 3 倍 DSR 才可能翻身
- 用户对 600% 年化的心理底线是"必须达成"还是"理想目标"？直接影响杠杆取舍
- 是否有意愿把 ¥400k 拆成 multi-strategy paper 实验田（每个策略 ¥20-50k）？

---

## 本次会话未修改的文件

所有修改已列出；未提交 git，等待战略决定后一并 commit。

reports/ 下新增：
- `walk_forward_v0.json`, `walk_forward_v1_baseline.json`
- 14 个 ablation 产物（abl_*, a1_*, cmp_*, amihud_*）
- `amihud_robustness.json`, `amihud_grid_*.json`, `amihud_oos_2023.json`, `amihud_is_2024.json`
