# 2026-03-09 QA Accelerated Validation Tasklist

## 1. 当前项目进展

截至 2026-03-09，项目主线状态如下：

- M2 `Core Orchestration`：已关闭
- M3 `vNext Service Layer`：已关闭
- M4 `LLM Dispatcher + Role Policy`：已关闭
- M5 `Workflow DAG Engine`：已关闭
- M6 `Parallel Rollout Readiness`：`GO_LIMITED_EXPOSURE`
- M7 `Limited Dynamic Routing v1`：已关闭，但为 `ACCEPTED WITH DEVIATION`
- M8 `Staging Evidence and Live Routing Validation`：已关闭

当前没有新的 active milestone，M9 尚未定义。

当前生产治理结论：

- `master_enabled=true`
- `dynamic_routing_enabled=false`
- `router_mode=static_policy_only`

这意味着：

- M6 的有限并行曝光已经批准进入生产策略
- M7 的动态路由基础设施已经完成，但生产上仍未开启
- 当前最正式的主线任务，仍然是积累 M6 的真实生产基线数据，再决定是否激活 M7

## 2. 本文档的定位

本文档对应的是今日必须完成的 QA 交付任务。

这组工作不等同于 “M7 生产放开批准”，也不能替代 1 到 2 周的真实生产观测；它的定位是：

`主线前的加速验证包`

也就是在不改变当前治理结论的前提下，尽快补齐今日可交付的测试证据、验证记录和报告框架。

## 3. 今日目标

今天的目标不是开启动态路由，而是完成以下事项：

1. 固化当前版本的自动化测试结果
2. 复核动态路由试验脚本的预演与降级行为
3. 确认生产配置与治理结论一致
4. 形成一份可以提交的 QA 测试摘要
5. 形成后续生产基线统计所需的最小工具/结构

## 4. 任务清单

### T0. 交付边界说明

目标：
明确今天交付的是 “加速验证包”，不是 “动态路由生产放开审批”。

输出：

- 一段说明文字，写入测试摘要或日报

验收标准：

- 明确说明今日结果仅代表阶段性验证
- 明确说明 M7 仍需独立 Architect sign-off

---

### T1. 自动化测试全量执行

目标：
确认最新代码在 orchestrator 与 brain 两个层面都可稳定通过回归验证。

执行项：

- 运行 `node --test orchestrator/test/*.test.js`
- 运行 `pytest brain/tests/`

输出：

- 通过数
- 失败数
- 执行时间
- 如有失败，记录失败用例名与根因

验收标准：

- orchestrator 测试全绿
- brain 测试全绿
- 结果被整理进测试摘要

---

### T2. 动态路由试验脚本验证

目标：
验证试验脚本在正常预演和不可用 drill 两种模式下都能产出结构化结果。

执行项：

- 运行 `node orchestrator/scripts/run_m7_dynamic_routing_trial.js`
- 运行 `node orchestrator/scripts/run_m7_dynamic_routing_trial.js --drill-unavailable`

关注点：

- 是否生成结果 JSON
- 是否正确区分 `dry_run_preflight` / `live_trial`
- drill 场景下是否触发 fallback
- 是否出现 threshold breached 提示
- 退出码是否符合预期

输出：

- 结果文件路径
- 核心指标摘要
- 异常/告警摘要

验收标准：

- 两种场景均可运行完成
- 结果结构可用于后续报告引用
- 不发生未预期崩溃

---

### T3. 配置与治理一致性复核

目标：
确认当前运行配置没有偏离治理结论。

检查项：

- `orchestrator/configs/production_parallel_rollout.json`
- `orchestrator/configs/parallel_exposure_policy.json`
- M8 Go/No-Go 文档中的当前决策

重点核对：

- `master_enabled=true`
- `dynamic_routing_enabled=false`
- `router_mode=static_policy_only`
- 曝光策略仍受 whitelist 约束

输出：

- 一段配置复核结论

验收标准：

- 配置与治理文档一致
- 不存在“代码允许但治理未批准”的隐性开关

---

### T4. 风险与证据缺口声明

目标：
把今天无法覆盖的内容明确写出来，避免测试结论被误解为生产批准。

必须列出的缺口：

- 尚未完成 1 到 2 周真实生产基线观测
- 尚未形成稳定的 `waterfall_stage_log` 生产统计样本
- 尚未完成 M7 动态路由生产激活审批

输出：

- 一段“已验证 / 未验证”声明

验收标准：

- 结论边界清楚
- 风险不被掩盖

---

### T5. 形成 QA 测试摘要文档

目标：
把今天的测试结论整理成可提交、可归档、可评审的文档。

建议结构：

1. 测试背景
2. 当前项目阶段
3. 测试范围
4. 执行命令
5. 结果摘要
6. 风险与缺口
7. 建议结论

输出：

- 1 份 Markdown 文档

验收标准：

- 任何阅读者都能快速知道：现在做到了什么、没做到什么、下一步是什么

---

### T6. 生产基线统计工具雏形（可选但建议今天完成）

目标：
提前搭好后续主线任务需要的统计工具，而不是等 1 到 2 周后再临时补。

建议内容：

- 从 `routing_decision_log` 读取样本
- 从 `waterfall_stage_log` 读取阶段耗时
- 输出 P50/P95
- 输出 deny reason 分布
- 输出 forced_sequential 比例
- 输出 cohort 维度统计

输出：

- 1 个统计脚本
- 1 份最小日报模板

验收标准：

- 脚本即使当前数据不多，也能跑通
- 后续只需要持续喂真实样本即可

## 5. 优先级排序

今天建议按这个顺序执行：

1. `T1` 自动化测试全量执行
2. `T2` 动态路由试验脚本验证
3. `T3` 配置与治理一致性复核
4. `T4` 风险与证据缺口声明
5. `T5` 形成 QA 测试摘要文档
6. `T6` 生产基线统计工具雏形

## 6. 任务定性

从项目 roadmap 的角度，这是一项 `支线任务`。

原因：

- 当前主线并不是“继续扩展测试脚本”，而是“运行 M6 并积累真实生产基线”
- 今天的工作主要是加速验证、固化证据、准备后续观测工具
- 它为主线服务，但本身不等于主线闭环

但从今日交付角度，这项支线任务的优先级应视为 `最高`。

## 7. 今日结论模板

建议今天最终对外使用如下结论口径：

> 今日已完成动态路由相关的加速验证包，包括自动化回归、试验脚本验证、配置一致性复核与风险边界声明。当前系统已具备继续观测 M6 生产并行行为的测试与证据基础，但本次工作不构成 M7 动态路由生产放开的最终审批依据。
