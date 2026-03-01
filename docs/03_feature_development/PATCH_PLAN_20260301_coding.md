# Coding 功能修复补丁文档（2026-03-01）

## 1. 目标
修复并完善 `coding` 新功能链路，使其达到“可稳定执行”的状态：
- Brain -> Orchestrator -> Redis -> Worker-Coder -> File/Command -> DB Fact -> Brain 回收

## 2. 已确认问题

### P0-1: Patch Manager 运行时错误
- 现象：`contentLines is not defined`
- 影响：`coding.patch` 必定失败
- 根因：变量命名不一致（`originalContentLines` 与 `contentLines` 混用）
- 受影响文件：
  - `worker-coder/patch_manager.js`
  - `orchestrator/src/patch_manager.js`

### P0-2: 命令执行被误拦截
- 现象：`coding.execute` 对任意命令都返回 forbidden
- 根因：`forbiddenChars` 包含空字符串 `""`，`command.includes("")` 恒为真
- 受影响文件：
  - `worker-coder/worker.js`

### P0-3: 队列路由竞争（跨 Worker 抢任务）
- 现象：`worker-coder` 与 `worker-quant` 共用 `stream:task` + `cg:workers`，可能误消费彼此任务
- 影响：任务丢失、错误重试、错误入 DLQ
- 修复策略：
  - 新增 coding 专用 stream：`stream:task:coding`
  - Orchestrator 按 `tool_name` 路由入队（`coding.*` -> coding stream）
  - `worker-coder` 改为仅消费 coding stream
- 受影响文件：
  - `orchestrator/src/index.js`
  - `worker-coder/worker.js`
  - `infra/docker-compose.yml`

### P1-1: 测试脚本仍使用旧 API
- 现象：部分测试仍请求 `/coding/patch` 与 `/coding/execute`
- 影响：与“去业务化”架构不一致，产生假失败
- 修复策略：改为调用 `/execute-tool`
- 受影响文件：
  - `test_coding.js`
  - `test_full_chain_integration.py`

## 3. 执行步骤
1. 修复 patch manager 变量错误，并增强替换逻辑稳定性。
2. 修复命令安全过滤器空字符串 bug。
3. 落地 coding 专用 stream 路由。
4. 更新测试脚本到新 API。
5. 运行最小回归：
   - `node test_coding_multi_blocks.js`
   - `node test_patch_edge_cases.js`

## 4. 验收标准
- `coding.patch` 可在多 block 场景成功修改文件。
- `coding.execute` 不再出现“全部命令被拦截”的误判。
- `coding.*` 任务不再被 quant worker 误消费。
- 测试脚本与当前架构（`/execute-tool`）一致。

## 5. 风险与后续
- 目前 `coding.patch` 在匹配策略上使用 `trim` 级别宽松匹配，可能误命中结构高度重复代码段。
- 后续建议：增加“最少上下文行”策略，或引入 AST/line-range patch 模式降低误替换概率。
