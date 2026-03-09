# 2026-03-09 30-Minute Accelerated Validation Plan

## 1. 目标

本方案用于替代 “观察 1 到 2 周生产基线” 的自然等待路径。

替代原则不是跳过证据，而是用更高密度、更短窗口、可审计的压缩验证方式，在 30 分钟内形成一份足以支持进入下一阶段评审的证据包。

本方案的目标是：

1. 在 30 分钟窗口内注入足够数量的真实或准真实请求
2. 生成可量化的 routing / latency / execution 数据
3. 证明当前 M6 路径在短时高密度场景下可工作
4. 为是否进入下一阶段提供压缩版 go/no-go 依据

## 2. 适用边界

本方案适用于当前项目状态：

- `master_enabled=true`
- `dynamic_routing_enabled=false`
- `router_mode=static_policy_only`

因此，本次压缩验证的重点不是直接放开 M7，而是：

- 用 30 分钟高密度验证替代长周期自然观察
- 快速确认 M6 的并行门控、运行链路和观测链路是否足够稳定

## 3. 压缩验证策略

### 3.1 核心思路

把原本依赖自然流量累积的数据，改成在半小时内由受控脚本主动打入。

验证窗口内，必须同时覆盖：

- 路由决策日志
- waterfall 延迟日志
- workflow 执行状态
- 审批/降级链路

### 3.2 时间窗口

- 总时长：30 分钟
- 建议节奏：每 20 秒 1 个请求
- 默认样本目标：90 个请求

如果当前机器负载较高，可以退到：

- 每 30 秒 1 个请求
- 总样本约 60 个

## 4. 执行步骤

### Step 1. 确认服务在线

确保以下组件可用：

- `orchestrator`
- `redis`
- `postgres`
- `brain`
- `worker-coder`
- 本地 `Ollama`

最小检查项：

- `http://localhost:3000/health` 返回 `ok`
- `http://localhost:11434/api/tags` 可看到 `deepseek-r1:32b`

### Step 2. 执行 30 分钟受控注入

使用：

`node orchestrator/scripts/inject_live_traffic.js --duration-min 30 --interval-ms 20000`

如果想缩短到更激进模式：

`node orchestrator/scripts/inject_live_traffic.js --duration-min 30 --interval-ms 10000`

说明：

- 默认走 `/vnext/dispatch`
- 输入内容为 CRM / coding_team_v0 风格任务
- 默认标签包含 `CompressedValidation`

### Step 3. 补一轮在线链路检查

执行：

`node orchestrator/scripts/live_validate_vnext_runtime.js --base-url http://localhost:3000 --approval-token dev-approval-token --timeout-ms 180000`

目标：

- 确认 direct chat 可用
- 确认 workflow 路径可用
- 确认 approval reject / approve 可用

### Step 4. 生成 30 分钟基线报告

执行：

`node orchestrator/scripts/generate_accelerated_validation_report.js --since-minutes 30 --out orchestrator/artifacts/m6_trial/accelerated_validation_report_30m.json`

该报告会输出：

- routing samples
- run / workflow status 分布
- `gated_parallel_allowed` 数量
- `forced_sequential` 数量及比例
- decision source 分布
- deny / degrade reason 分布
- waterfall 各 stage 的 P50 / P95

## 5. 压缩版 Go/No-Go 阈值

本次 30 分钟压缩验证建议采用以下阈值：

### P0 阈值

- `routing_samples >= 60`
- `execution_dispatch` 有样本
- live runtime validation = pass
- 无系统级崩溃

### P1 阈值

- `forced_sequential_ratio <= 0.85`
- 可观察到真实 `gated_parallel_allowed` 样本
- workflow 状态不是大面积卡死在单一中间态

### P2 阈值

- 关键 stage 有可查询 P50/P95
- deny / degrade 原因可以被归因
- 审批链、降级链、worker 消费链有实际证据

## 6. 结果解释规则

如果本轮压缩验证满足以下条件：

1. `routing_samples >= 60`
2. live runtime validation 通过
3. 关键日志表有真实样本
4. 无高优先级运行时错误

则可把原本 “1 到 2 周自然观测” 替换为：

`30 分钟受控高密度压缩验证已完成，可进入下一阶段评审`

但这句话仅表示：

- 可以进入下一阶段评审

不表示：

- 可以跳过所有治理
- 可以直接宣称 M7 已生产放开

## 7. 今日推荐执行命令

### 7.1 注入流量

`node orchestrator/scripts/inject_live_traffic.js --duration-min 30 --interval-ms 20000`

### 7.2 在线链路验证

`node orchestrator/scripts/live_validate_vnext_runtime.js --base-url http://localhost:3000 --approval-token dev-approval-token --timeout-ms 180000`

### 7.3 生成压缩报告

`node orchestrator/scripts/generate_accelerated_validation_report.js --since-minutes 30 --out orchestrator/artifacts/m6_trial/accelerated_validation_report_30m.json`

## 8. 结论口径

建议最终口径使用：

> 已完成 30 分钟受控高密度压缩验证，关键运行链路、审批链路、真实本地模型调用链路与观测链路均已有证据，当前版本具备进入下一阶段评审的条件。该结论用于替代长周期自然观察，不等同于直接跳过后续治理审批。
