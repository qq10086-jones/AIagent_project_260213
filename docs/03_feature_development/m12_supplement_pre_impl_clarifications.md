# M12 补充说明：开工前必须对齐的 3 个实现细节

**关联文档：** M12 Revised Design v2.0  
**日期：** 2026-03-16  
**性质：** 设计补丁（Patch），合入主文档对应章节  

---

## 问题 1：如何精准识别"包含数据库依赖"？

**关联章节：** Section 3 (Preview Eligibility Matrix) → Section 5.2 Phase 1 (artifact_validate)

### 问题本质

文档定义了"有数据库写入依赖的应用不予支持"，但只在 Eligibility Matrix 中写了结论，没有定义**判别机制**。如果依赖 LLM 在 `deployment_metadata` 中自行声明"我不需要数据库"，这个判断是不可信的——LLM 可能遗漏依赖，也可能在 prompt 引导下刻意隐瞒。

### 修正方案：artifact_validate 阶段加入轻量级静态扫描

在 `deploy_preview` 子状态机的 Phase 1（artifact_validate）中，增加一层**依赖清单静态扫描**，作为 LLM 自声明之外的独立校验层。

#### 扫描规则

**Node.js 项目（检测 `package.json` 的 `dependencies` + `devDependencies`）：**

| 触发 Fallback 的包名模式 | 说明 |
|---|---|
| `pg`, `pg-native`, `pg-pool` | PostgreSQL 客户端 |
| `mysql`, `mysql2` | MySQL 客户端 |
| `mongoose`, `mongodb` | MongoDB 客户端 |
| `sequelize`, `typeorm`, `prisma`, `@prisma/client`, `knex` | ORM / 查询构建器 |
| `redis`, `ioredis` | Redis 客户端（作为主存储时） |
| `sqlite3`, `better-sqlite3` | SQLite（虽然可嵌入，但 preview 无持久磁盘，运行时会报错） |
| `mssql`, `tedious` | SQL Server 客户端 |

**Python 项目（检测 `requirements.txt` / `pyproject.toml`）：**

| 触发 Fallback 的包名模式 | 说明 |
|---|---|
| `psycopg2`, `psycopg`, `asyncpg` | PostgreSQL |
| `mysqlclient`, `pymysql`, `aiomysql` | MySQL |
| `pymongo`, `motor` | MongoDB |
| `sqlalchemy`, `django` (含 `DATABASES` 配置时), `tortoise-orm`, `peewee` | ORM |
| `redis`, `aioredis` | Redis |

**静态前端项目：**

静态前端（纯 HTML/CSS/JS）天然无服务端数据库依赖，扫描直接通过。

#### 扫描逻辑伪代码

```
function checkDatabaseDependency(releasePackPath):
    manifest = detectProjectType(releasePackPath)  // node | python | static

    if manifest == "static":
        return ELIGIBLE

    deps = extractDependencies(manifest)  // 从 package.json 或 requirements.txt 提取
    blockedDeps = deps ∩ DB_BLOCKLIST     // 与上述黑名单取交集

    if blockedDeps is not empty:
        return INELIGIBLE(reason: "检测到数据库依赖: " + blockedDeps.join(", "))

    return ELIGIBLE
```

#### 设计要点

1. **扫描结果优先级高于 LLM 自声明。** 即使 `deployment_metadata` 中没有标注数据库依赖，只要静态扫描命中黑名单，一律 fallback。
2. **黑名单可配置。** 存储为 JSON 配置文件（`config/db_dependency_blocklist.json`），不硬编码在扫描逻辑中，方便后续维护。
3. **扫描失败不阻塞。** 如果项目结构异常导致无法解析依赖文件（例如 monorepo、非标准目录结构），扫描返回 `UNKNOWN`，此时降级为信任 LLM 自声明 + 日志告警，而不是直接拒绝。
4. **性能开销极低。** 只做文本匹配，不安装依赖、不执行代码。预期耗时 < 100ms，不影响 Section 6 的延迟预算。

#### 对 Acceptance Criteria 的影响

**WS-40-03 追加 AC：**

> artifact_validate 阶段对包含 `pg`（Node.js）或 `psycopg2`（Python）依赖的 release pack 返回 INELIGIBLE，并在 fallback 消息中列出触发拦截的具体依赖名称。

**Q4 测试用例追加：**

> 测试输入：一个 `package.json` 中包含 `"mongoose": "^7.0.0"` 的 Node.js 项目。预期结果：deploy_preview 被跳过，用户收到 ZIP + 消息："检测到数据库依赖 (mongoose)，Live Preview 不适用于需要数据库的项目。"

---

## 问题 2：利用 Render Auto-Sleep 作为成本兜底

**关联章节：** Section 7.2 (TTL & Cleanup Mechanism)

### 问题本质

当前设计完全依赖我们自研的 Cleanup Daemon 来控制预览生命周期。如果 Daemon 本身宕机、部署故障、或 dead-letter 处理延迟，Render 上的容器可能持续运行数小时甚至数天，造成不可控的计费。

这是一个**单点故障**问题：我们用一个自己写的守护进程来保护成本安全，但这个进程本身没有"兜底者"。

### 修正方案：Render Auto-Sleep 作为第二层防线

#### 架构分层

| 层级 | 机制 | 触发条件 | 目的 |
|---|---|---|---|
| **L1：主动清理** | Cleanup Daemon（我们自研） | `expires_at < now` | 正常 TTL 到期清理 |
| **L2：被动休眠** | Render Auto-Sleep（平台原生） | 15 分钟无入站 HTTP 请求 | Daemon 宕机时的成本兜底 |
| **L3：硬性上限** | Render 月度 Spend Limit（平台配置） | 月消费触达阈值 | 极端情况下的最终止损 |

#### 实现细节

**L2 - Auto-Sleep 配置：**

在 `render_client.js` 的 `createService()` 方法中，创建服务时显式启用 Auto-Sleep：

```javascript
// adapters/render_client.js - createService 配置片段
const serviceConfig = {
  type: "web_service",
  plan: "starter",          // 最低规格，控制单实例成本
  autoDeploy: "no",         // 不自动重新部署
  // 关键：确保使用支持 Auto-Sleep 的计划
  // Render Starter plan 默认 15 分钟无请求后休眠
  // 休眠期间不计费
};
```

**行为预期：**

- 正常情况：Cleanup Daemon 在 TTL 到期后 5 分钟内删除服务（L1 生效，L2 不触发）。
- Daemon 延迟：如果 Daemon 在 TTL 到期后 15 分钟仍未清理，Render Auto-Sleep 让容器休眠，停止计费（L2 生效）。
- Daemon 宕机：容器在最后一次用户访问后 15 分钟休眠。即使永远没有被删除，也不会持续产生费用。
- 极端情况：如果大量休眠容器累积占用资源配额，月度 Spend Limit（L3）作为最终止损。

**L3 - Spend Limit 配置：**

在 Render Dashboard 或通过 API 设置：

| 配置项 | 值 | 说明 |
|---|---|---|
| 月度 Spend Limit | $200 | 超过后新服务创建被拒 |
| 告警阈值 | $50/天, $150/月 | 触发 Ops 通知 |

#### 对 Cleanup Daemon 设计的影响

Daemon 的职责不变（仍然要主动删除到期服务），但心态变了：**Daemon 是"尽力而为"的清理者，不是"必须成功"的单点守护者。** 这意味着：

1. Daemon 失败时的 dead-letter 处理可以适当放宽重试频率（从"每 5 分钟重试"改为"每 30 分钟重试"），因为 Auto-Sleep 已经兜住成本。
2. Daemon 宕机的告警级别从 `CRITICAL` 降为 `HIGH`——仍然需要尽快修复，但不会立即导致财务损失。

#### 对 Acceptance Criteria 的影响

**WS-42-03 追加 AC：**

> Render 服务创建时使用支持 Auto-Sleep 的 plan。验证方式：创建一个 preview，15 分钟内不访问，确认 Render 控制台显示 "Sleeping" 状态，且该时段不产生计算费用。

**WS-42-04 追加 AC：**

> Render 账户配置了月度 Spend Limit（$200）。验证方式：在 Render Dashboard 确认 Spend Limit 配置存在。

**Q9 测试用例追加：**

> 测试场景：手动停止 Cleanup Daemon，等待一个 preview TTL 过期 + 15 分钟。预期结果：Render 服务进入 Sleeping 状态，不产生额外计算费用。

---

## 问题 3：防抖机制下的快速状态跳转

**关联章节：** Section 4.5 (Debounced Message Editing)

### 问题本质

当前设计用 3 秒防抖窗口合并高频事件，只发送窗口内最后一个状态。这对大多数场景没问题，但存在一个边界情况：

如果一个 step 在防抖窗口内完成了完整的 `started → completed` 周期（例如一个耗时 200ms 的内部校验 step），防抖只会发送 `completed`，跳过 `started`。

**视觉层面**：这其实没问题，用户看到的进度从 "Step 3/7" 直接跳到 "Step 5/7"，感知上只是"跑得快"。

**逻辑层面**：这里有风险。如果防抖器内部的状态合并逻辑写得不小心，可能出现以下 bug：

1. **状态机卡死**：如果代码假设每个 step 都会经历 `started` 状态然后才能转到 `completed`，跳过的 `started` 可能导致状态机无法推进。
2. **计数器错误**：如果 step 计数器依赖 `started` 事件来递增，跳过的 `started` 会导致"Step 3/7"直接跳到"Step 5/7"，虽然视觉上可接受，但 total_steps 可能对不上。
3. **心跳丢失**：如果心跳更新只在 `started` 事件中触发，一个被跳过的 `started` 可能导致心跳看起来过时。

### 修正方案：防抖器基于"最终快照"而非"事件队列"

#### 核心设计原则

**防抖器不缓存事件序列，而是维护一个"当前状态快照"。** 每次事件到达时，更新快照；防抖窗口到期时，将快照渲染为 Discord 消息。

这意味着：
- 状态快照始终反映最新真实状态，不依赖事件顺序。
- 即使 `started` 被防抖窗口吞掉，`completed` 更新了快照，渲染出来的消息仍然正确。

#### 防抖器内部状态结构

```javascript
// 每个 run_id 维护一个快照对象
const snapshot = {
  run_id: "abc-123",
  current_step: "qa_review",        // 来自最新事件的 step_name
  current_step_index: 5,            // 来自最新事件的 step_index
  total_steps: 7,                   // 来自最新事件的 total_steps
  current_status: "RUNNING",        // 枚举，来自事件类型推导
  action_summary: "Running tests",  // 来自最新 step.progress 或 step.started
  last_heartbeat_ts: 1710000000,    // 每个事件都更新此字段
  dirty: true,                      // 标记是否有未发送的变更
};
```

#### 事件处理规则

| 事件类型 | 快照更新规则 |
|---|---|
| `step.started` | 更新 `current_step`, `current_step_index`, `total_steps`, `action_summary`, `last_heartbeat_ts`, 设 `current_status = RUNNING`, 标记 `dirty = true` |
| `step.progress` | 更新 `action_summary`, `last_heartbeat_ts`, 标记 `dirty = true` |
| `step.completed` | 更新 `current_step` (标为已完成), `current_step_index`, `last_heartbeat_ts`, 标记 `dirty = true`。**不**改 `current_status`（下一个 `step.started` 会改）。 |
| `step.failed` | 更新 `current_status = RETRYING 或 FAILED`, `action_summary` 包含错误摘要, 标记 `dirty = true` |
| `workflow.completed` | 设 `current_status = COMPLETED`, 标记 `dirty = true`, **立即发送**（绕过防抖） |
| `workflow.failed` | 设 `current_status = FAILED`, 标记 `dirty = true`, **立即发送**（绕过防抖） |

#### 关键规则

**规则 1：终态事件绕过防抖。**  
`workflow.completed` 和 `workflow.failed` 是终态事件，必须立即发送，不进入防抖窗口。用户不应该在工作流已经结束后还等 3 秒才看到结果。

**规则 2：心跳在每个事件上更新。**  
无论事件是否被防抖吞掉，`last_heartbeat_ts` 都在快照中更新。下次防抖窗口到期渲染时，心跳时间是准确的。

**规则 3：快照渲染不依赖事件历史。**  
渲染函数只读取当前快照的字段值，不需要知道"到达过哪些事件"。这从根本上消除了因跳过中间事件导致的状态不一致。

**规则 4：`dirty` 标记防止无变更编辑。**  
如果防抖窗口到期时 `dirty = false`，不发送编辑请求。这避免在无状态变化时浪费 Discord API 配额。

#### 边界场景验证

| 场景 | 事件序列（1 秒内） | 防抖行为 | Discord 显示 |
|---|---|---|---|
| 快速 step | `step_4.started` → `step_4.completed` → `step_5.started` | 快照更新 3 次，窗口到期发送 1 次 | "Step 5/7: Running next step" |
| 超快连续 steps | `step_3.completed` → `step_4.started` → `step_4.completed` → `step_5.started` → `step_5.completed` | 快照更新 5 次，窗口到期发送 1 次 | "Step 5/7: Completed" |
| 正常速度 step | `step_3.started` ...(5 秒后)... `step_3.progress` | 两次分别触发渲染 | 先 "Step 3/7: Starting..."，后 "Step 3/7: Writing file X" |
| 工作流结束 | `step_7.completed` → `workflow.completed` | `workflow.completed` 立即发送 | 立即显示最终结果 + URL |

#### 对 Acceptance Criteria 的影响

**WS-39-02 追加 AC：**

> 1) 一个 step 在防抖窗口（3s）内完成 started → completed 的完整周期时，不导致状态机卡死或计数器错误。验证方式：构造一个耗时 100ms 的 mock step，确认后续 steps 正常推进且最终 Discord 消息正确。  
> 2) `workflow.completed` 和 `workflow.failed` 事件绕过防抖窗口，立即触发 Discord 消息编辑。验证方式：在防抖窗口中间触发 `workflow.completed`，确认消息在 < 1s 内更新（而非等待防抖窗口到期）。  
> 3) 防抖窗口到期时如无状态变更（`dirty = false`），不发送 Discord 编辑请求。验证方式：连续触发两次相同内容的 `step.progress` 事件，确认只产生 1 次 API 调用。

**Q3 测试用例追加：**

> 测试场景：在 500ms 内连续触发 step_4.started → step_4.completed → step_5.started。预期结果：Discord 消息在下一个防抖窗口显示 "Step 5/7"，不出现"Step 4/7"中间态，且后续事件处理正常。

---

## 合入建议

以上 3 个补丁建议以如下方式合入主文档 v2.0：

| 补丁 | 合入位置 | 影响的 AC / 测试用例 |
|---|---|---|
| 数据库依赖静态扫描 | Section 3 新增扫描规则子节；Section 5.2 Phase 1 扩展描述 | WS-40-03, Q4 |
| Render Auto-Sleep 兜底 | Section 7.2 新增 L2/L3 防线描述 | WS-42-03, WS-42-04, Q9 |
| 防抖快照机制 | Section 4.5 替换为快照模型描述 | WS-39-02, Q3 |

合入后主文档版本号升级为 **v2.1**。
