# 任务清单: 宏观情报管线

**设计文档**: `docs/design/PATCH_2026-04-08_Macro_Intelligence_Pipeline.md`
**进展记录**: `docs/progress/PROGRESS_2026-04-08_macro_intelligence_pipeline.md`
**日期**: 2026-04-08

---

## Phase 1: 扩展跨资产数据层 (本机，无外部依赖) ✅ 完成

### T1-1: cross_asset_signals.py 扩展 4 个品种 ✅
- **文件**: `cross_asset_signals.py`
- **改动**:
  - `fetch_cross_asset_snapshot()` 新增: CL=F (WTI), GC=F (黄金), HG=F (铜), ^SOX
  - `ensure_cross_asset_table()` ALTER TABLE 加 8 列 (4品种 × close+change_pct)
  - `compute_cross_asset_regime_signal()` 更新权重分配 (8品种)
  - `save_cross_asset_snapshot()` 写入新列
- **验证**: 运行一次采集，确认 8 品种全部有数据

### T1-2: 回填扩展品种历史数据 ✅
- **依赖**: T1-1
- **改动**: 回填脚本，补充原油/黄金/铜/SOX 的 100 天历史
- **验证**: cross_asset_snapshots 表 100 行 × 全列非 NULL

### T1-3: config.yaml 更新跨资产配置 ✅
- **文件**: `config.yaml`
- **改动**: cross_asset.tickers 和 weights 扩展到 8 品种

### T1-4: 跨资产权重历史回归验证
- **改动**: 用 100 天数据跑各品种 vs 次日日经收益的单因子回归
- **输出**: 各品种 beta / R² / p-value → 确认权重合理性
- **标记**: 可在家里环境执行

---

## Phase 2: 规则引擎 macro_event_detector.py (本机，无外部依赖) ✅ 完成

### T2-1: 新建 macro_event_detector.py ✅
- **文件**: `macro_event_detector.py` (新建)
- **功能**:
  - `detect_macro_events(snapshot: dict) -> dict` — 规则引擎主函数
  - `MACRO_EVENT_RULES` — 阈值配置字典
  - 输入: cross_asset_snapshot
  - 输出: `{alert_level, triggered_rules, rule_boost, suggested_duration}`
- **规则**:
  - NK期货 ±3% → L1
  - 原油 ±5% → L1
  - VIX +15% → L1
  - SOX ±3% → L2
  - 原油 ±3% → L2
  - NK期货 ±1.5% → L2
  - 多规则 boost 求和后 clamp [-0.30, +0.30]

### T2-2: 新建 macro_events DB 表 ✅
- **文件**: `macro_event_detector.py` (ensure_macro_events_table)
- **Schema**: asof, ts, alert_level, triggered_rules, rule_boost, llm_boost, final_boost, duration_days, event_summary, event_type, llm_raw_json, source

### T2-3: macro_event_detector.py CLI 入口 ✅
- **改动**: `if __name__ == "__main__"` 支持 `--db japan_market.db`
- **功能**: 读取最新 cross_asset_snapshot → 检测 → 写入 macro_events → 打印

### T2-4: 测试 test_macro_event_detector.py
- **文件**: `tests/test_macro_event_detector.py` (新建)
- **测试用例**:
  - NK期货 +4.25% → L1, boost > 0
  - 原油 -10% → L1, boost > 0
  - 正常日 (所有指标 < 阈值) → none, boost = 0
  - 多规则同时触发 → boost 求和后 clamp
  - VIX +20% → L1, boost < 0 (利空)
  - SOX +4% + NK +2% → L2, boost > 0
- **标记**: 可在家里环境执行

---

## Phase 3: regime 事件覆盖层 (联动 quant 主链路) ✅ 完成

### T3-1: benchmark_regime.py 新增 apply_event_boost() ✅
- **文件**: `benchmark_regime.py`
- **新增函数**:
  ```python
  def apply_event_boost(
      regime_score: float,
      db_conn,
      asof: str,
  ) -> tuple[float, dict]:
      """读取 macro_events 表，叠加 event_boost (含衰减)"""
  ```
- **衰减逻辑**: `decay = max(0, 1 - days_elapsed / duration_days)`
- **输出**: `(regime_final, {"event_boost": ..., "decay": ..., "source_event": ...})`

### T3-2: sprint_signal.py 集成 event_boost ✅
- **文件**: `sprint_signal.py`
- **改动**: `generate_sprint_artifacts()` 在计算 regime_score_v2 后，调用 `apply_event_boost()` 得到 regime_final
- **约束**: event_boost 不改变 benchmark_state 的离散映射逻辑（保持兼容）

### T3-3: daily_run.py 接入 macro_event_detector ✅
- **文件**: `daily_run.py`
- **改动**:
  - 在 cross_asset_signals 之后调用 `macro_event_detector.py`
  - 将 alert_level 和 boost 写入 regime_diagnosis.json
- **约束**: macro_event_detector 失败不阻塞主链路 (try/except)

### T3-4: morning_briefing.bat 加入事件检测步骤 ✅
- **文件**: `morning_briefing.bat`
- **改动**: Step 0 之后加 Step 0.5: `python macro_event_detector.py --db japan_market.db`

### T3-5: quant_briefing.py 显示宏观事件 ✅
- **文件**: `quant_briefing.py`
- **改动**: 在"零、隔夜跨资产信号"小节后，新增"宏观事件检测"小节
- **显示**: alert_level, 触发规则, boost 值, 衰减状态

### T3-6: 测试 event_boost 集成
- **文件**: `tests/test_macro_event_detector.py` (追加)
- **测试**:
  - apply_event_boost 正常叠加
  - 衰减到 0 后不影响 regime
  - 无 macro_events 记录时返回原始 regime
  - boost clamp 测试
- **标记**: 可在家里环境执行

---

## Phase 4: LLM 宏观分析层 (代码完成，需家里环境配置/测试)

### T4-1: 新建 macro_digest.py ✅ (代码完成)
- **文件**: `macro_digest.py` (新建)
- **功能**:
  - `fetch_macro_headlines(queries) -> list[dict]` — Google News RSS 采集宏观标题
  - `call_llm_analysis(snapshot, headlines, alert_level) -> dict` — 调 Ollama API
  - `build_llm_prompt(snapshot, headlines, alert_level) -> str` — prompt 构建
  - `parse_llm_response(raw: str) -> dict` — JSON 解析 + 校验
  - `ensure_macro_headlines_table(conn)` — DB 表创建
- **Ollama 调用**: POST `{endpoint}/api/generate` with model=gemma4:27b
- **超时**: 60 秒，超时退化到规则引擎
- **触发条件**: alert_level in ("L1", "L2")

### T4-2: macro_digest.py CLI 入口 ✅ (代码完成)
- **改动**: `if __name__ == "__main__"` 支持 `--db` `--endpoint` `--model`
- **功能**: 读取最新 snapshot + macro_event → 如果 L1/L2 则调 LLM → 更新 macro_events

### T4-3: daily_run.py 接入 macro_digest ✅ (代码完成)
- **文件**: `daily_run.py`
- **改动**: macro_event_detector 之后，如果 alert_level in (L1, L2)，调用 macro_digest.py
- **约束**: LLM 失败不阻塞 (try/except + fallback_to_rules)

### T4-4: morning_briefing.bat 加入 LLM 分析步骤 ✅ (代码完成)
- **文件**: `morning_briefing.bat`
- **改动**: Step 0.5 之后加 Step 0.7: `python macro_digest.py --db japan_market.db`

### T4-5: config.yaml 增加 macro_events 配置段 ✅
- **文件**: `config.yaml`
- **改动**: 新增 `macro_events:` 段 (enabled, llm, rules, boost, alerts)

### T4-6: LLM prompt 调优和测试
- **依赖**: Gemma 4 27B 可用
- **测试**: 用 3-5 个历史场景验证 Gemma 输出质量
  - 2026-04-08 伊美停战 (L1, boost +0.30)
  - 2025-08-05 日银加息冲击 (L1, boost -0.25)
  - 普通日无事件 (不应调用 LLM)
- **标记**: 必须在家里环境执行

### T4-7: Discord 告警接入
- **文件**: `macro_event_detector.py` 或 `macro_digest.py`
- **改动**: L1 事件时推送 Discord webhook
- **格式**: 包含 alert_level、数据摘要、regime_boost、建议

---

## Phase 5: 验证和回测

### T5-1: 历史事件回测
- **内容**: 找 2024-2026 的 L1 级别事件日，验证规则引擎能否检测到
- **事件候选**:
  - 2025-08-05 日银加息冲击 (日经 -12%)
  - 2024-08-02 美国非农爆冷 (日经 -5%)
  - 2026-03-31 (NK期货 +3.85%)
  - 2026-04-08 伊美停战 (日经 +5%)
- **验证**: 回测 boost 后的 regime vs 无 boost 的 regime → 哪个仓位更合理

### T5-2: cross_asset 权重优化
- **内容**: 用 100+ 天数据跑 8 品种权重的网格搜索
- **目标**: 最大化 cross_asset_score vs 次日日经收益的 rank IC
- **输出**: 优化后的权重写入 config.yaml

### T5-3: 全量测试
- **内容**: `python -m pytest tests/ -v`
- **目标**: 所有测试通过，包括新增测试
- **标记**: 在家里环境执行

---

## 执行计划

| Phase | 在哪做 | 依赖 | 状态 |
|-------|--------|------|------|
| Phase 1 (T1-1~T1-3) | 办公室机器 | 无 | ✅ 完成 |
| Phase 2 (T2-1~T2-3) | 办公室机器 | Phase 1 | ✅ 完成 |
| Phase 3 (T3-1~T3-5) | 办公室机器 | Phase 2 | ✅ 完成 |
| Phase 4 (T4-1~T4-5) 代码 | 办公室机器 | Phase 2 | ✅ 代码完成 |
| T1-4, T2-4, T3-6 | 家里环境 | 需要 pytest | ⏳ 待执行 |
| T4-6, T4-7 | 家里环境 | 需要 Gemma + Tailscale | ⏳ 待执行 |
| Phase 5 (T5-1~T5-3) | 家里环境 | Phase 1-4 完成 |
