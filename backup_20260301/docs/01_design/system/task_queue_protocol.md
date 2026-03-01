# Task Queue Protocol (Redis Streams) — v1.2.1

本文件定义 Orchestrator（投递/编排）与 Worker（执行）之间的任务协议。
所有实现必须遵循本协议；如需调整，先更新本文件再改代码。

---

## 1. Streams & Consumer Group

### Streams
- stream:task    — 任务队列（Orchestrator -> Worker）
- stream:result  — 结果回写（Worker -> Orchestrator）
- stream:dlq     — 死信队列（Worker -> DLQ）

### Consumer Group
- group: cg:workers

---

## 2. Task Lifecycle & Status

### Task Status (逻辑状态)
- queued
- running
- succeeded
- failed
- waiting_approval (被 Guard 拦截，需要人审)

### 必须保证
- 每个 task_id 在最终状态（succeeded/failed）只能落一次“最终结果事件”
- 任务执行必须支持幂等（见 6. Idempotency）

---

## 3. ACK Semantics (XACK)

### 规则
- Worker 在成功或失败处理完成后，必须对 stream:task 执行 XACK。
- 在 XACK 之前，任务处于 pending 状态，系统认为仍在执行中。

### 注意：心跳（heartbeat）
- 心跳/进度更新不要高频写入 Redis stream（避免噪音与膨胀）。
- 心跳建议写入 DB event_log（event_type=task.heartbeat），或写入 metrics。

---

## 4. Visibility Timeout

### 定义
- VISIBILITY_TIMEOUT_S: 任务被某个 worker claim 后，超过该时间仍未 XACK，则认为可能卡死/掉线。

### 默认建议
- VISIBILITY_TIMEOUT_S = 300 (5 minutes)
- 心跳间隔建议 HEARTBEAT_INTERVAL_S = 30

---

## 5. Reclaim (Pending Task Recovery)

### 目标
当 worker 异常退出导致任务停留在 pending 时，系统必须能回收并重新分配该任务。

### Redis >= 6.2
- 使用 XAUTOCLAIM 对超时 pending 任务进行回收

### Redis < 6.2
- 使用 XPENDING + XCLAIM 手动回收

### 规则
- 仅回收 idle_time >= VISIBILITY_TIMEOUT_S 的 pending 任务
- 回收后必须记录 event_log：task.reclaimed（包含原 consumer、new consumer、message_id）

---

## 6. Retry & Idempotency

### Retry
- MAX_RETRIES: 失败后最多重试次数（建议 3）
- retry_count 每失败一次 +1
- 可选：指数退避（例如 1m, 5m, 15m）或延迟队列策略（后续增强）

### Idempotency
- 每个任务必须携带 idempotency_key
- Worker 执行外部副作用动作（写外部系统/下单/发布）时必须检查幂等：
  - 相同 idempotency_key 不可重复产生副作用
  - 建议：在 DB 建 unique 索引 (idempotency_key)

---

## 7. DLQ (Dead Letter Queue)

### 触发条件
- retry_count > MAX_RETRIES
或
- 输入不可修复错误（schema invalid / missing required fields）

### 行为
- 将任务写入 stream:dlq（包含 task_id、tool_name、payload、error、retry_count、failed_at）
- 记录 event_log：task.dlq

### 运维
- 提供 dlq_replay 工具：支持 “修复后重放” 或 “标记为放弃”

---

## 8. Minimal Events to DB (event_log)

必须落库的关键事件（最低限度）：
- task.created
- task.claimed
- task.succeeded / task.failed
- task.reclaimed（如果发生）
- task.dlq（如果发生）
- approval.requested / approval.approved / approval.rejected（如果有 Guard 人审）

---

## 9. Acceptance Criteria (Done)

以下全部满足，视为协议落地完成：
1) 成功任务：created -> claimed -> succeeded，且最终 XACK，pending=0
2) 失败任务：重试至 MAX_RETRIES 后进入 DLQ，且 event_log 完整
3) 停掉 worker 后重启：超时 pending 任务可 reclaim 并完成
4) 相同 idempotency_key 的外部副作用不重复执行
