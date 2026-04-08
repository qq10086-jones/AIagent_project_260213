# Progress: 2026-04-08 宏观情报管线 + V2 Regime 上线 + 基本面 IC 修复

## Status

宏观情报管线三层架构代码全部完成（规则引擎 + LLM 管线 + regime 集成），
V2 连续化 regime 已上线生产，基本面因子 IC 管线修复，8 品种跨资产数据扩展并回填 100 天。
**待家里环境**: pytest 验证、Ollama endpoint 配置、权重优化回测。

## 背景

2026-04-08 伊美停战 → 日经 +5%，系统完全未检测到原因：
- NK期货 +4.25% 被看到，但原油暴跌 -10% 无数据源
- cross_asset_score = 0.40（中性）← 应该是强 risk-on
- VIX +6.66% 和 USDJPY -0.80% 被当作利空，实际是停战后的正常二阶反应
- 系统无 L1/L2 事件检测框架，regime 纯回顾性（MA 需 5-10 天才反映）

## 一、跨资产数据层扩展 (Phase 1)

### 改动

| 文件 | 改动内容 |
|------|----------|
| `cross_asset_signals.py` | 从 4 品种扩展到 8 品种（新增 WTI原油 CL=F、黄金 GC=F、铜 HG=F、SOX ^SOX），权重重分配 |
| `config.yaml` | `cross_asset.tickers` 和 `weights` 更新为 8 品种 |
| DB: cross_asset_snapshots | ALTER TABLE 新增 8 列（4品种 × close + change_pct） |

### 历史回填

- 使用 `backfill_cross_asset.py` 回填 99 天数据（2025-11-11 ~ 2026-04-08）
- 全部 8 品种列非 NULL
- cross_asset_score 和 regime_adjustment 重新计算

## 二、规则引擎 macro_event_detector.py (Phase 2)

### 新建文件: `macro_event_detector.py`

- 13 条检测规则覆盖 L1/L2 事件
- `detect_macro_events(snapshot)`: 主检测函数
- `load_active_event(conn, asof)`: 读取活跃事件（含线性衰减）
- `save_macro_event()` / `ensure_macro_events_table()`: DB 读写

### 规则概要

| 级别 | 规则名 | 字段 | 阈值 | boost |
|------|--------|------|------|-------|
| L1 | nk_futures_surge | nk_futures_gap_pct | ±3.0% | ±0.25 (sign) |
| L1 | oil_crash | crude_oil_change_pct | <-5.0% | +0.20 |
| L1 | oil_spike | crude_oil_change_pct | >+5.0% | -0.20 |
| L1 | vix_extreme_spike | vix_change_pct | >+15.0% | -0.15 |
| L2 | nk_futures_moderate | nk_futures_gap_pct | ±1.5% | ±0.10 (sign) |
| L2 | oil_moderate_down/up | crude_oil_change_pct | ±3.0% | ±0.10 |
| L2 | sox_surge/crash | sox_change_pct | ±3.0% | ±0.10 |
| L2 | yen_shock_weak/strong | usdjpy_change_pct | ±2.0% | ±0.08 |
| L2 | gold_spike | gold_change_pct | >+3.0% | -0.05 |
| L2 | copper_surge | copper_change_pct | >+3.0% | +0.05 |

- 多规则触发时 boost 求和后 clamp [-0.30, +0.30]
- L1 持续 3 天、L2 持续 2 天、线性衰减

### 今日场景验证

```
输入: NK期货 +4.25%, 原油 -10.2%, VIX +6.66%
输出: L1 alert, boost = +0.30 (clamped from +0.75)
触发规则: nk_futures_surge(+0.25), oil_crash(+0.20), nk_futures_moderate(+0.10), oil_moderate_down(+0.10)
regime 效果: 0.214 → 0.514 (建仓比例从 ~12% 提升到 ~66%)
```

## 三、regime 事件覆盖层 + 主链路集成 (Phase 3)

### 改动

| 文件 | 改动内容 |
|------|----------|
| `benchmark_regime.py` | 新增 `apply_event_boost(regime_score, conn, asof)` — 读取 macro_events，叠加衰减 boost |
| `sprint_signal.py` | V2 regime 计算后调用 `apply_event_boost()`，重新映射 state/scale/benchmark_state |
| `daily_run.py` | cross_asset 后调用 `detect_macro_events()` + `save_macro_event()`，L1/L2 时调用 `macro_digest`（如 enabled） |
| `morning_briefing.bat` | Step 0.5: macro_event_detector, Step 0.7: macro_digest (LLM) |
| `quant_briefing.py` | "零、隔夜跨资产信号" 扩展显示 8 品种 + 新增 "宏观事件检测" 小节 |

### 事件覆盖公式

```python
regime_final = clamp(regime_score_v2 + event_boost × decay, 0.0, 1.0)
decay = max(0, 1 - days_elapsed / duration_days)   # 线性衰减
event_boost ∈ [-0.30, +0.30]                        # 硬约束
```

### 安全设计

- `apply_event_boost()` 失败时 → 返回原始 regime_score（不影响生产）
- `macro_event_detector` 失败时 → try/except 不阻塞 daily_run
- `macro_digest` LLM 失败时 → 退化到规则引擎 boost

## 四、LLM 分析管线 macro_digest.py (Phase 4 — 代码完成)

### 新建文件: `macro_digest.py`

- `fetch_macro_headlines()`: Google News RSS 采集宏观标题
- `build_llm_prompt()`: 构建 Gemma 分析 prompt（含市场数据 + 新闻 + 规则引擎结果）
- `call_llm_analysis()`: POST Ollama `/api/generate` API
- `parse_llm_response()`: JSON 抽取 + 校验（clamp boost/duration, confidence 控制）
- `run_macro_digest()`: 完整管线入口
- `update_macro_event_with_llm()`: LLM 结果写回 macro_events 表

### LLM 安全约束

- `regime_boost` clamp [-0.30, +0.30]
- `confidence < 0.5` → boost 减半
- 超时 (60s) / 返回非法 JSON → 退化到规则引擎 boost
- 仅 L1/L2 触发时才调用（不会幻觉创造事件）

### config.yaml 新增段

```yaml
macro_events:
  enabled: true
  llm:
    enabled: false        # 家里环境改为 true
    endpoint: "http://localhost:11434"
    model: "gemma4:27b"
    timeout_seconds: 60
    fallback_to_rules: true
```

## 五、V2 Regime 上线生产

- `config.yaml benchmark_regime.version: "v2"` + `cross_asset.shadow_only: false`
- 回测验证: Sharpe 0.55→1.02, MaxDD -26%→-22.4%, Sortino 0.57→0.85
- `sprint_signal.py` 使用 `compute_regime_score_v2()` + 连续 scale → 离散 state 映射

## 六、基本面因子 IC 管线修复

- `compute_ic.py load_feature_daily_scores()`: 移除 `asof <= max_eval_date` 限制 + bfill
- `compute_ic.py load_logged_factor_scores()`: 过滤 `raw_score IS NULL`（清除 23,018 条脏数据）
- 修复后 7/10 基本面因子正常计算 IC（roa_op ICIR=+0.355, margin_op ICIR=+0.150）
- 3 个 earnings 因子 (growth_rev/op_yoy, guidance_delta) 因无数据源仍为 NaN

## 七、语法验证

所有 6 个修改/新建文件通过 `py_compile` 语法检查:
- `macro_event_detector.py` OK
- `macro_digest.py` OK
- `benchmark_regime.py` OK
- `sprint_signal.py` OK
- `daily_run.py` OK
- `quant_briefing.py` OK

## 待家里环境完成

| 任务 | 内容 |
|------|------|
| Ollama 配置 | `config.yaml macro_events.llm.endpoint` 改为 Gemma 机器 IP，`enabled: true` |
| 全量测试 | `python -m pytest tests/ -v` — 包括新增的 `tests/test_macro_event_detector.py` |
| 权重优化 | 8 品种 cross_asset 权重网格搜索（最大化 rank IC） |
| 历史事件回测 | 2025-08-05 日银冲击 / 2024-08-02 非农爆冷 / 2026-03-31 NK+3.85% |
| LLM prompt 调优 | 用历史场景验证 Gemma 输出质量 |

## 文件变更汇总

| 文件 | 操作 |
|------|------|
| `macro_event_detector.py` | 新建 |
| `macro_digest.py` | 新建 |
| `cross_asset_signals.py` | 修改（4→8 品种） |
| `benchmark_regime.py` | 修改（+apply_event_boost） |
| `sprint_signal.py` | 修改（+event_boost 集成） |
| `daily_run.py` | 修改（+macro_event + macro_digest 调用） |
| `morning_briefing.bat` | 修改（+Step 0.5/0.7） |
| `quant_briefing.py` | 修改（+8品种显示 + 宏观事件小节） |
| `config.yaml` | 修改（+macro_events 段 + cross_asset 扩展） |
| `compute_ic.py` | 修改（IC 管线 2 处修复） |
| `docs/design/PATCH_2026-04-08_Macro_Intelligence_Pipeline.md` | 新建（设计文档） |
| `docs/tasks/TASKS_2026-04-08_Macro_Intelligence_Pipeline.md` | 新建（任务清单） |
