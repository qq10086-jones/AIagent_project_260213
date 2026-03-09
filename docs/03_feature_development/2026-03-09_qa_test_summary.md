# 2026-03-09 QA Test Summary

## 1. 背景

本次工作对应 `2026-03-09 QA Accelerated Validation Tasklist`。

目标不是推动 `M7 dynamic routing` 当天直接放开，而是在当前治理约束不变的前提下，完成一轮可交付的加速验证，补齐以下证据：

- 当前代码版本的自动化测试结果
- 动态路由试验脚本的正常预演与降级验证
- 当前生产配置与治理结论的一致性复核
- 今日测试结论的边界声明

## 2. 当前项目阶段

截至 2026-03-09，项目状态如下：

- M6：`GO_LIMITED_EXPOSURE`
- M7：`CLOSED (ACCEPTED WITH DEVIATION)`
- M8：`CLOSED`
- M9：尚未定义

当前生产配置结论：

- `master_enabled=true`
- `dynamic_routing_enabled=false`
- `router_mode=static_policy_only`

这意味着当前主线仍然是：

- 继续运行 M6 的受限并行曝光
- 累积真实生产基线
- 在独立评审后，再决定是否开启 M7 动态路由

## 3. 测试范围

本次测试覆盖以下内容：

1. `orchestrator` 自动化测试
2. `brain` 自动化测试
3. `run_m7_dynamic_routing_trial.js` 正常预演
4. `run_m7_dynamic_routing_trial.js --drill-unavailable` 降级 drill
5. 生产 rollout 配置检查
6. 本地真实 LLM 调用验证（`deepseek-r1:32b`）
7. 在线运行时验证（local orchestrator + docker infra）

本次测试未覆盖：

- 1 到 2 周真实生产观测
- 真实生产 `waterfall_stage_log` 长周期基线
- M7 动态路由生产激活审批
- 完整 `orchestrator HTTP -> workflow runtime -> worker -> artifact` 在线链路

## 4. 执行记录

### 4.1 Orchestrator Tests

执行命令：

`node --test orchestrator/test/*.test.js`

执行结果：

- 总计：127
- 通过：127
- 失败：0
- 结论：全绿

说明：

- 该命令在沙箱内触发 `spawn EPERM`
- 改为宿主机提权执行后，测试通过
- 这属于执行环境限制，不属于测试断言失败

### 4.2 Brain Tests

执行命令：

`python -m pytest brain/tests/`

执行结果：

- 总计：11
- 通过：11
- 失败：0
- 结论：全绿

附注：

- 本机初始环境缺少 `pytest`
- 已安装 `pytest` 与 `brain/requirements.txt` 所需依赖后完成执行
- 运行中出现 1 条 `PytestCacheWarning`，为缓存目录写入权限问题，不影响用例通过

### 4.3 Dynamic Routing Trial - Preflight

执行命令：

`node orchestrator/scripts/run_m7_dynamic_routing_trial.js --out orchestrator/artifacts/m7_trial/preflight_result_20260309_local.json`

结果文件：

- `orchestrator/artifacts/m7_trial/preflight_result_20260309_local.json`

关键结果：

- mode: `dry_run_preflight`
- total_cases: `50`
- static_gated_parallel: `0`
- dynamic_gated_parallel: `11`
- agreement_rate: `0.78`
- forced_sequential_count: `39`

前置检查结果：

- `dynamic_routing_enabled=false`
- 因此前置条件未满足，脚本不会进入 live trial，而是执行 dry run preflight

阈值结果：

- `forced_sequential_spike_pct=78%`
- threshold limit=`40%`
- 结果为 `breached`

说明：

- 本次预演脚本成功生成结构化结果文件
- 阈值 breach 属于脚本设计内的信号输出，不代表脚本自身崩溃
- 当前结果符合“动态路由尚未开启”的治理状态

### 4.4 Dynamic Routing Trial - Unavailability Drill

执行命令：

`node orchestrator/scripts/run_m7_dynamic_routing_trial.js --drill-unavailable --out orchestrator/artifacts/m7_trial/drill_unavailable_result_20260309_local.json`

结果文件：

- `orchestrator/artifacts/m7_trial/drill_unavailable_result_20260309_local.json`

关键结果：

- mode: `dry_run_preflight`
- total_cases: `50`
- dynamic_gated_parallel: `0`
- agreement_rate: `1.0`
- forced_sequential_count: `50`
- classifier availability: `0%`
- classifier failures: `50/50`

降级行为：

- 所有样本均退化到 `forced_sequential`
- `classifier_unavailable_fallback` 生效
- health alert 成功触发

阈值结果：

- `forced_sequential_spike_pct=100%`
- threshold limit=`40%`
- 结果为 `breached`

说明：

- drill 场景下的退化路径符合预期
- 没有出现未预期中断
- fallback 逻辑与告警逻辑均被实际触发

### 4.5 Live Local LLM Dispatch

执行方式：

- 使用项目内 `orchestrator/src/vnext/llm_dispatcher.js`
- 真实调用本机 `Ollama`
- 真实模型：`deepseek-r1:32b`

结果文件：

- `orchestrator/artifacts/canary/live_local_llm_dispatcher/live_local_llm_dispatcher_20260309.json`

关键结果：

- role: `backend`
- provider: `local_ollama`
- model: `deepseek-r1:32b`
- used_fallback: `false`
- contains_expected_token: `true`
- content_preview: `LIVE_DISPATCH_OK`
- latency_ms: `223825`

结论：

- 项目运行时代码已确认可以通过真实本地模型完成一次成功调度
- 本次不是 mock，也不是仅检查模型列表存在，而是实际完成了一次模型推理请求
- 但本地 `deepseek-r1:32b` 延迟较高，当前样本约为 `223.8s`

### 4.6 Live Runtime Validation

测试环境：

- 本机启动 `orchestrator`
- Docker 启动 `redis` / `db` / `brain` / `worker-coder`
- 入口地址：`http://localhost:3000`

已验证通过：

- `/health` 返回 `ok`
- `/chat` 的 direct reply 路径可用
- `/coder:` 前缀可强制进入 orchestrated workflow
- 高风险请求可进入 `waiting_approval=true`
- `POST /tasks/:id/reject` 可成功执行
- reject 后 run 状态变为 `failed`
- reject 后 task `error_code=APPROVAL_REJECTED`

本轮定位与修复：

- 修复 `infra/docker-compose.yml` 中 `orchestrator` 缺少 `contracts` 挂载的问题
- 修复 `live_validate_vnext_runtime.js` 的审批触发语句，使其与当前 `/coder:` 路由规则一致
- 修复 `live_validate_vnext_runtime.js` 的响应断言，使其适配当前 `workflow` 响应协议，而不是旧的 `task` 协议

修复后结果：

- `live_validate_vnext_runtime.js` 已可通过
- 在线链路验证覆盖了 health / direct chat / approval reject / approval approve
- 当前在线验证结论可视为 `pass`

附注：

- 容器版 `orchestrator` 在本次复测时因本机已有本地 `orchestrator` 占用 `3000` 端口而未直接复跑
- 但修复后的 live validation 已在本地在线服务上通过

## 5. 配置一致性复核

复核对象：

- `orchestrator/configs/production_parallel_rollout.json`

复核结果：

- `master_enabled=true`
- `force_sequential=false`
- `dynamic_routing_enabled=false`
- `router_mode=static_policy_only`

结论：

- 当前配置与 M8 治理结论一致
- 当前环境没有越过治理边界提前开启动态路由

## 6. 风险与证据缺口

### 已验证

- `orchestrator` 自动化测试可通过
- `brain` 自动化测试可通过
- 动态路由预演脚本可产出结构化结果
- classifier 不可用时可退化到 `forced_sequential`
- 当前 rollout 配置与治理文档一致
- 项目内 `llm_dispatcher` 可真实调用本机 `deepseek-r1:32b`
- 在线 API、审批 reject、审批 approve、worker 消费链路可真实运行

### 未验证

- 真实生产下的长期稳定性
- 真实生产 `waterfall_stage_log` 的 P50/P95 基线
- M7 动态路由在真实生产流量中的 uplift vs risk
- 开启 `dynamic_routing_enabled=true` 后的生产审批条件
- 完整在线运行时链路是否可在当前主机直接完成端到端执行
- 容器版 `orchestrator` 需要在释放本地 `3000` 端口后再做一次 compose 路径复验

### 风险说明

- 今日结果不能替代真实生产观察窗口
- 今日结果不能作为 M7 生产放开的最终批准依据
- 如需进入下一阶段，仍需独立 Architect sign-off

## 7. 结论

今日的加速验证已经完成，且结果可用于提交阶段性 QA 证据：

- `orchestrator`：127/127 PASS
- `brain`：11/11 PASS
- dynamic routing preflight：结果文件已生成
- unavailability drill：结果文件已生成，fallback 行为符合预期
- rollout 配置：与当前治理决策一致
- live local llm dispatch：已通过真实 `deepseek-r1:32b` 调用验证
- live runtime：修复后已通过

本次测试结论应表述为：

> 当前版本已经完成一轮可交付的加速验证，证明自动化测试链、预演脚本链与降级路径链均可工作，且当前配置未偏离治理决策。该结果可作为后续主线观测与激活评审的前置证据，但不构成 M7 动态路由生产放开的最终审批依据。

## 8. 产物清单

- `docs/03_feature_development/2026-03-09_qa_accelerated_validation_tasklist.md`
- `docs/03_feature_development/2026-03-09_qa_test_summary.md`
- `orchestrator/artifacts/m7_trial/preflight_result_20260309_local.json`
- `orchestrator/artifacts/m7_trial/drill_unavailable_result_20260309_local.json`
- `orchestrator/artifacts/canary/live_local_llm_dispatcher/live_local_llm_dispatcher_20260309.json`
- `orchestrator/artifacts/canary/live_vnext_runtime/live_vnext_runtime_report.json`
- `orchestrator/artifacts/canary/orchestrator_local_stdout.log`
- `orchestrator/artifacts/canary/orchestrator_local_stderr.log`
