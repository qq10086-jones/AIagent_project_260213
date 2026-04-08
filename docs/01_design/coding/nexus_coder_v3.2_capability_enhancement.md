# Nexus Coder v3.2 能力增强设计文档

> **状态**: PROPOSED (架构委员会评审中)
> **日期**: 2026-04-08
> **作者**: 架构团队
> **基线**: 基于 `OpenClaw_Nexus_Worker_Coding_Design_v4.2.md` (2026-03-11)
> **定位**: v4.2 Worker-Coding Design 的实施增强方案，不替换基线文档

---

## 变更日志

| 版本 | 变更说明 |
|------|---------|
| v3.2 | 初稿。基于 v4.2 基线，规划三阶段增强：轻量微操层、语义上下文管道、调度弹性化 |
| v3.2.1 | 新增 Phase 1.5 迭代修复回路 (Refinement Re-entry)，补齐"做完后回来改"的核心缺失 |
| v3.2.2 | 新增 Phase 0 Redis Caching 层 + gpt-tokenizer；新增附录 B Harness 对标分析与结论 |
| v3.2.3 | Phase 0 实施完成 (P0-1/P0-2/P0-3 全部落地，测试全绿)，更新状态标记 |
| v3.2.4 | Phase 1 + Phase 1.5 实施完成 (P1-1/P1-2/P1-3/P1.5-1/P1.5-2/P1.5-3/P1.5-5 全部落地，27 测试文件全绿含 33 新用例)，feature flags 默认 false |

---

## 一、背景与定位

### 1.1 本文档解决什么问题

v4.2 设计文档定义了完整的 beta-ready 编码能力框架：task class 分类、context envelope 硬约束、failure attribution 四维归因、cohort validation 矩阵。这些**基础契约层**是正确的方向，但在实施过程中暴露了三个工程瓶颈：

1. **微小修复成本过高**：当 `static_checks.js` 检测到"第15行少个括号"这类语法错误时，当前路径是触发完整的 `delegateTask` 重试循环（重新构建 prompt → 调用 adapter → 等待 LLM 响应），耗时 30-120 秒。理想状态是 < 5 秒的确定性热修复。

2. **上下文注入依赖人工**：v4.2 Layer E 预留了 `ContextRequest/ContextResponse` 接口，但当前阶段 context 完全由 operator 手动填充 `context_packet` 和 `repo_map`。对于 TC-02 (FE Modify) 和 TC-04 (Bug Fix) 这类高上下文依赖的 task class，人工组装上下文是 beta 用户的首要痛点。

3. **产品交付后无迭代回路**：当前执行模型是单次射击——任务进来、8 步跑完、产出代码、结束。用户审查成品后发现"按钮颜色不对"或"API 返回格式要改"，只能重新发一个全新任务，**丢失上次执行的全部上下文**（failure_memory、diff、artifact、代码理解）。这是从"工具"到"生产力"的最大断裂点。v4.2 两次将此列为 Non-Goal（Section 4 / Section 12），但实际上这恰恰是 beta 用户最核心的工作流。

4. **步骤间无并行能力**：8 步 workflow 严格串行。当 `impl_be` 和 `impl_fe` 无数据依赖时（如纯新增的独立模块），仍需排队等待，浪费调度时间。

### 1.2 本文档不做什么

- **不替换** v4.2 的 task class 定义、context envelope 契约、cohort validation 框架
- **不删除** 任何现有 JSON Schema 或治理契约（`step_artifact_contract.js`、`task_contract.js` 等保持不变）
- **不引入** 新的 provider 依赖或运行时基础设施（Redis/Postgres 拓扑不变）
- **不改变** 现有的 deny-by-default 安全模型和 `scope_guard.js` 路径约束
- **不实现** 完整的 RAG 系统（遵循 v4.2 Section 13 的 phasing criteria）

### 1.3 与 v4.2 的关系

```
v4.2 Worker-Coding Design (契约层 — 定义 WHAT)
  └── v3.2 Capability Enhancement (实施层 — 定义 HOW)
        ├── Phase 0:   Redis Caching 层 (零风险前置优化)
        ├── Phase 1:   轻量微操层
        ├── Phase 1.5: 迭代修复回路 (Refinement Re-entry)
        ├── Phase 2:   结构化上下文管道
        └── Phase 3:   调度弹性化
```

每个 Phase 的交付物必须通过 v4.2 定义的 cohort validation 验证，方可启用。

---

## 二、现有系统能力审计

在规划增强前，先客观盘点当前系统已有的基础设施。后续设计必须**复用而非重建**这些组件。

| 模块 | 文件 | 能力 | 代码行数 |
|------|------|------|---------|
| 路径安全守卫 | `scope_guard.js` | `validateAllowedTargetPaths` / `validateRequestedWrite` / `validateChangedFilesWithinScope`：三层写入校验，protected root 拦截 | 77 |
| 隔离沙盒 | `isolation_workspace.js` | scaffold/shadow/promote 三模式隔离执行，target_paths 自动 scope | 193 |
| 变更推广 | `promotion_workspace.js` | 隔离沙盒到主 workspace 的受控合并 | 249 |
| 静态检查 | `static_checks.js` | JS syntax (`node --check`)、JSON parse、Python compile，逐文件增量检查 | 102 |
| 分层验证 | `verification_runner.js` | `runVerificationPlan`：multi-tier (lint/type_check/unit_test/build/legacy)，逐层执行、首错中断 | 286 |
| 失败记忆 | `failure_memory.js` | JSONL 持久化、failure_attribution 四维归因 (`coding_logic` / `context` / `verification` / `infrastructure`) | 186 |
| 重试策略 | `retry_policy.js` | 同错去重 (`sameErrorRepeatLimit`)、phase 感知重试门控 | 63 |
| Prompt 契约 | `prompt_contract.js` | `buildExecutionPromptContract`：结构化约束注入（target_paths / required_outputs / rules） | 117 |
| 任务分类 | `task_contract.js` + `worker_coding_task_classes.js` | task_class 归一化、context_envelope 归一化、failure_attribution 推导 | 53 + 39 |
| 命令白名单 | `verification_runner.js` | `ALLOWED_CMD_PREFIXES` + shell metachar 拦截 + `&&` chain 拆分校验 | (内含) |
| 敏感信息脱敏 | `verification_runner.js` | OpenAI key / GitHub token / AWS key / Slack token / 通用 secret 正则脱敏 | (内含) |

**关键事实**：当前系统的 5472 行代码中，约 40% 是治理和安全基础设施。这些是资产，不是负债。

---

## 二点五、Phase 0 — Redis Caching 层 (Zero-Risk Prelude) ✅ COMPLETED (2026-04-08)

### 2.5.1 目标

消除当前管线中明确可见的重复计算浪费。利用已有的 Redis（ioredis）基础设施，在不改变任何业务逻辑的前提下，通过 content-hash 驱动的缓存减少 I/O 和 CPU 开销。

### 2.5.2 设计原则

- **纯加速，零语义变更**: 缓存 miss 时 fallback 到原始计算路径，结果与无缓存时完全一致
- **content-hash 驱动**: key 中包含文件内容或 git SHA 的 hash，文件变更自动 invalidate
- **不缓存 LLM 响应**: prompt 微小差异导致 cache miss 率极高，投入产出不合理
- **TTL 保底**: 即使 hash 未变，缓存也有 TTL 上限防止内存膨胀

### 2.5.3 任务清单

#### P0-1: `repo_map` 缓存 ✅ DONE

**变更文件**: `orchestrator/src/domain/repo_context_service.js`（+ `workflow_step_builder.js`、`workflow_engine.js`、`index.js` redis 透传 + async 迁移）

**现状**: `buildRepoMap()` 每次 task 都重新扫描文件树 + 提取 symbol_hints + 推断 entrypoints，对同一 workspace 的连续任务是纯浪费。

**实现**:
```javascript
// cache key = workspace_root + git_HEAD_sha + sorted(target_paths).join(",")
// 同一个 commit 上的相同 target_paths → 100% cache hit
const cacheKey = `repo_map:${sha256(workspaceAbs + gitHeadSha + targetPaths.sort().join(","))}`;
const cached = await redis.get(cacheKey);
if (cached) return JSON.parse(cached);
const fresh = buildRepoMap({ targetPaths, recentChangedFiles });
await redis.set(cacheKey, JSON.stringify(fresh), "EX", 3600); // 1h TTL
return fresh;
```

**git HEAD SHA 获取**: `git rev-parse HEAD`（复用 `git_side_effects.js` 的 `execFileImpl` 模式）

**验收标准**:
- 同 commit + 同 target_paths 的连续 2 次调用，第 2 次 < 10ms（vs 当前 200-800ms）
- 新 commit 后 cache 自动 miss，结果与无缓存完全一致
- Redis 连接失败时 graceful fallback 到原始路径
- 新增测试 3 case（hit / miss / redis-down fallback）

#### P0-2: `static_check` 结果缓存 ✅ DONE

**变更文件**: `worker-coder/static_checks.js`（+ `coding_service.js`、`worker.js` redis 透传）

**现状**: `runStaticChecks()` 对 `filesChanged` 中的每个文件逐一执行 `node --check` / `JSON.parse` / `py_compile`。Refinement 场景（Phase 1.5）中大部分文件未变更，重复检查无意义。

**实现**:
```javascript
// cache key = check_kind + file_path + sha256(file_content)
// 文件内容未变 → 100% cache hit
const contentHash = sha256(fs.readFileSync(absPath, "utf8"));
const cacheKey = `static_check:${checkKind}:${relPath}:${contentHash}`;
const cached = await redis.get(cacheKey);
if (cached === "pass") {
  records.push({ file: rel, kind: checkKind, ok: true, exit_code: 0, stderr: "", cached: true });
  continue;
}
// 执行原始检查...
if (result.ok) await redis.set(cacheKey, "pass", "EX", 86400); // 24h TTL
```

**约束**: 仅缓存 `ok: true` 的结果。失败结果不缓存（避免文件修复后仍 cache hit 到失败）。

**验收标准**:
- 同文件未变更时跳过 `node --check` 子进程调用
- 文件修改后 cache 自动 miss
- 现有 `static_checks.test.js` 零回归
- 新增缓存命中 / 失效 / redis-down 测试 3 case

#### P0-3: `gpt-tokenizer` 集成 — Context Envelope Token 计数 ✅ DONE

**变更文件**: `worker-coder/task_contract.js`（+ `prompt_contract.js` 截断集成）

**现状**: `context_envelope.max_tokens` 约束存在，但没有准确的 token 计数器。Prompt 是否超限只能靠粗略估算。

**实现**:
```bash
npm install gpt-tokenizer  # 纯 JS, 零 native 依赖
```

```javascript
import { encode } from "gpt-tokenizer";

export function countTokens(text) {
  return encode(String(text || "")).length;
}

export function validateContextTokenBudget({ contextText, maxTokens }) {
  if (!maxTokens || maxTokens <= 0) return { ok: true, usage: 0 };
  const usage = countTokens(contextText);
  return {
    ok: usage <= maxTokens,
    usage,
    max: maxTokens,
    overflow: Math.max(0, usage - maxTokens),
  };
}
```

**集成点**: `prompt_contract.js` 的 `buildExecutionPromptContract` 中，在最终 prompt 拼接后调用 `validateContextTokenBudget`，超限时截断 `context_packet` 的低优先级部分（`memory_hints` → `symbol_hints` → `toolchain_facts` 按序删减）。

**验收标准**:
- 对已知文本的 token 计数与 OpenAI tiktoken 结果一致（±1%）
- 超限时正确截断并记录 `context_envelope_truncated: true` 到 diagnostics
- 新增测试 3 case

### 2.5.4 Phase 0 度量

| 指标 | 当前基线 | 目标 | 数据来源 |
|------|---------|------|---------|
| `buildRepoMap` 连续调用耗时 | 200-800ms | < 10ms (cache hit) | 性能日志 |
| `runStaticChecks` 未变更文件耗时 | 每文件 50-200ms (子进程) | < 1ms (cache hit) | 性能日志 |
| Context envelope token 计数精度 | 无（粗估） | ±1% vs tiktoken | 单元测试 |

### 2.5.5 风控

- **feature flag**: `redis_cache_enabled`，默认 `true`（这是纯加速，无语义变更，可以默认开启）
- **Redis 连接失败**: graceful fallback，log warning，不阻塞执行
- **内存保护**: 每个 cache entry 设 TTL，repo_map 条目额外设 `MAXLEN` 限制总数

---

## 三、Phase 1 — 轻量微操层 (Surgical Patch Layer) ✅ COMPLETED (2026-04-09)

### 3.1 目标

在 `delegateTask` 的重试循环内部，增加一个**轻量级确定性修复路径**：当 `static_checks.js` 或 `verification_runner.js` 报告的错误满足"可确定性修复"条件时，跳过完整的 LLM 重试，直接执行极速修补。

### 3.2 设计约束

- 微操路径**必须**经过 `scope_guard.js` 的 `validateRequestedWrite` 校验
- 微操路径**必须**在 `isolation_workspace` 的沙盒内执行（不直接操作主 workspace）
- 微操的产出**必须**记录到 `failure_memory.js` 的 JSONL 中，failure_attribution 标记为 `surgical_fix`
- 微操失败**必须**回落到标准的 `delegateTask` 重试路径
- 微操每次 delegate 最多触发 **1 次**（防止循环微操消耗时间）

### 3.3 任务清单

#### P1-1: `surgical_patch.js` — 确定性微修复引擎

**模块位置**: `worker-coder/surgical_patch.js`

**输入**:
```javascript
{
  workspaceRoot: string,          // 执行 workspace（isolation 或 main）
  taskDir: string,                // 当前 run 的 artifact 目录
  staticCheckResult: object,      // static_checks.js 的返回值
  verificationResult: object,     // verification_runner.js 的返回值
  allowedTargetPaths: string[],   // scope_guard 约束
}
```

**输出**:
```javascript
{
  attempted: boolean,             // 是否尝试了微修复
  success: boolean,               // 微修复是否成功
  patches_applied: Array<{file: string, kind: string, detail: string}>,
  static_recheck: object | null,  // 微修复后的 static_check 结果
  failure_reason: string | null,  // 失败原因
}
```

**可修复条件 (v1 scope)**:
| 错误类型 | 修复策略 | 置信度 |
|---------|---------|--------|
| JSON parse error (missing comma, trailing comma) | 正则定位 + 确定性插入/删除 | HIGH |
| JS syntax: unexpected token at EOF (缺少闭合括号/花括号) | AST 括号平衡检查 + 追加 | HIGH |
| Python `IndentationError` | 上下文行对比 + 缩进对齐 | MEDIUM |

**不可修复条件 (直接回落到 delegateTask)**:
- 语义错误（逻辑 bug）
- 多文件联动错误
- 错误信息无法确定性定位到具体行号
- 修复涉及 scope_guard 不允许的路径

**验收标准**:
- 27 个现有测试全绿（零回归）
- 新增测试覆盖：JSON 修复 3 case、JS 括号修复 3 case、Python 缩进修复 2 case、scope_guard 拦截 2 case、回落到 delegate 2 case
- 修复耗时 < 2 秒（排除磁盘 I/O 的纯计算时间）

**与现有模块的集成点**:

```
coding_service.js delegateTask 循环
  └── attempt N: adapter 执行完毕
        ├── static_checks.js → 失败
        │     └── surgical_patch.canFix(staticCheckResult)?
        │           ├── YES → surgical_patch.apply() → static_checks 复检
        │           │     ├── 复检通过 → 继续到 verification
        │           │     └── 复检失败 → 标准 retry (attemptIndex++)
        │           └── NO → 标准 retry (attemptIndex++)
        └── verification_runner.js → 失败 (Phase 1 暂不处理 verification 级微修复)
```

#### P1-2: 集成到 `coding_service.js` 重试循环

**变更文件**: `worker-coder/coding_service.js`

**变更范围**: 在 `delegateTask` 的 attempt 循环内，`runStaticChecks` 返回失败后、`shouldRetryAutoFix` 判断前，插入微修复尝试。

**关键约束**:
- `surgical_patch.js` 作为**可选增强**引入，通过 `runtime_config.js` 的 `surgical_patch_enabled` 开关控制
- 默认 `false`（deny-by-default 原则），需显式启用
- 开关关闭时，行为与当前完全一致

**failure_memory 集成**:
- `failure_attribution` 新增枚举值 `surgical_fix_success` / `surgical_fix_failed`
- `task_contract.js` 的 `deriveFailureAttribution` 增加对应分支
- cohort validation 可单独统计微修复的尝试率/成功率

#### P1-3: Native File Read 工具 (只读)

**模块位置**: `worker-coder/native_file_tools.js`

**scope**: 仅提供 `readFile` 和 `listDir`，**不提供 writeFile**。写入路径统一走 `surgical_patch.js` + `scope_guard.js`。

**设计理由**: 将"看"和"改"分开治理。读操作的安全风险远低于写操作，可以更宽松地暴露给 adapter 的上下文构建阶段。

**实现**:
```javascript
export function nativeReadFile({ workspaceRoot, relPath, maxBytes = 64 * 1024 }) {
  // 1. normalizeRelPath (复用 scope_guard.js)
  // 2. isProtectedRoot 检查（.git、infra、configs 等不可读）
  // 3. 路径必须在 workspaceRoot 内（防止 path traversal）
  // 4. 读取并截断到 maxBytes
  // 5. redactSensitiveText (复用 verification_runner.js)
  // 返回: { ok, content, truncated, bytes_read, error }
}

export function nativeListDir({ workspaceRoot, relPath, maxEntries = 200 }) {
  // 1. normalizeRelPath
  // 2. isProtectedRoot 检查
  // 3. 路径必须在 workspaceRoot 内
  // 4. fs.readdirSync + stat (复用 isolation_workspace.js 的 hashlessStatEntry 模式)
  // 5. 截断到 maxEntries
  // 返回: { ok, entries: [{path, type, size}], truncated, error }
}
```

**验收标准**:
- protected root 拦截测试 3 case
- path traversal 拦截测试 2 case
- 正常读取 + 截断测试 3 case
- 敏感信息脱敏测试 2 case

---

## 三点五、Phase 1.5 — 迭代修复回路 (Refinement Re-entry) ✅ COMPLETED (2026-04-09, worker-coder side; orchestrator P1.5-4 pending)

### 3.5.1 问题陈述

当前 worker-coder 的致命断裂：**做完即失忆**。

```
任务 A (原始需求)
  └── 8步执行 → 产出 v1 代码
        └── 用户审查: "API 返回格式要改成 array"
              └── 任务 B (全新任务)
                    └── 8步执行 → 从零开始，不知道 v1 做了什么
                          └── 大概率重写而非修补，或修错位置
```

理想状态：

```
任务 A (原始需求)
  └── 8步执行 → 产出 v1 代码
        └── 用户反馈: "API 返回格式要改成 array"
              └── 任务 A-R1 (refinement round 1，继承 A 的全部上下文)
                    └── 仅执行必要的步骤 → 产出 v1.1 代码 (增量修改)
                          └── 用户反馈: "加个分页参数"
                                └── 任务 A-R2 (refinement round 2)
                                      └── ...
```

### 3.5.2 设计目标

让已完成的 coding 任务可以**低成本地接受用户反馈并迭代修复**，而不丢失前次执行建立的上下文。

### 3.5.3 核心概念: Task Lineage (任务血缘)

引入一个轻量的血缘追踪机制，让 refinement 任务能找到并复用其"祖先"的产出。

```javascript
// 新增字段，扩展到 Redis stream 的 task payload
{
  // ...existing fields...
  "lineage": {
    "parent_task_id": "task_abc123",      // 被 refine 的原始任务
    "parent_run_id": "run_xyz789",        // 原始任务的 run_id
    "refinement_round": 1,                // 第几轮 refinement
    "refinement_instruction": "把 API 返回从 object 改成 array，加 total_count 字段",
    "inherited_context": {
      "target_paths": ["workspace/src/api/"],   // 继承自 parent
      "files_changed": ["src/api/customers.js", "src/api/types.ts"],  // parent 改了什么
      "artifact_root": "artifacts/runs/run_xyz789/task_abc123/",       // parent 的 artifact 位置
    }
  }
}
```

### 3.5.4 执行路径: 轻量 Refinement vs 完整 Workflow

refinement 任务**不需要跑完整 8 步**。根据反馈类型，系统选择最短路径：

| 反馈类型 | 跳过的步骤 | 执行的步骤 | 理由 |
|---------|-----------|-----------|------|
| UI 微调 ("按钮颜色改蓝") | pm_spec, arch_design, impl_be, release_pack, deploy_preview | impl_fe → smoke_test → qa_verify | 架构不变，仅前端实现变更 |
| API 修改 ("返回格式改 array") | pm_spec, arch_design, impl_fe*, release_pack, deploy_preview | impl_be → smoke_test → qa_verify | 后端逻辑变更 |
| Bug 修复 ("点击报500") | pm_spec, arch_design, release_pack, deploy_preview | impl_be/fe → smoke_test → qa_verify | 定位+修复 |
| 架构级变更 ("改用 WebSocket") | 无 | 完整 workflow | 架构变更必须从头走 |

*如果 be_to_fe.json 存在接口变更，`impl_fe` 不可跳过。

**步骤跳过的判断逻辑**: 不由 LLM 判断，而是由 **orchestrator 基于规则** 决定：
- 如果 `refinement_instruction` 被 task_class 分类为 `fe_modify` → 跳过 BE 步骤
- 如果分类为 `be_create` / `be_modify` → 跳过 FE 步骤（除非 handoff 有接口依赖）
- 如果分类为 `bug_fix` → 不跳过实现步骤，但跳过 pm_spec / arch_design
- 默认：不跳过任何步骤（安全回退）

### 3.5.5 上下文继承机制

refinement 任务从 parent 继承三类上下文：

#### (a) 文件快照上下文
通过 Phase 1 的 `native_file_tools.js`（P1-3）读取 parent 任务产出的**实际文件内容**。这比任何中间 JSON 契约都可靠——文件就是 Source of Truth。

```javascript
// refinement_context_builder.js
export function buildRefinementContext({ workspaceRoot, parentArtifactRoot, parentFilesChanged }) {
  const context = {
    previous_changes: [],         // parent 改了什么文件，每个文件的当前内容
    previous_diff_summary: null,  // parent 的 git diff summary
    previous_verification: null,  // parent 的验证结果
  };

  // 1. 读取 parent 改过的文件的当前内容 (用 nativeReadFile)
  for (const filePath of parentFilesChanged) {
    const content = nativeReadFile({ workspaceRoot, relPath: filePath });
    if (content.ok) {
      context.previous_changes.push({
        path: filePath,
        content: content.content,   // 已经 redact 过
        truncated: content.truncated,
      });
    }
  }

  // 2. 读取 parent 的 diff artifact
  const diffPath = path.join(parentArtifactRoot, "scoped_delta.json");
  // ...

  // 3. 读取 parent 的 verification 结果
  // ...

  return context;
}
```

#### (b) Prompt 注入上下文
将 refinement_instruction + previous_changes 注入到 `prompt_contract.js` 的 prompt 中：

```
[Refinement Context v1]
- refinement_round: 1
- parent_task_id: task_abc123
- refinement_instruction: 把 API 返回从 object 改成 array，加 total_count 字段
- previous_files_changed: src/api/customers.js, src/api/types.ts
- previous_verification_result: passed (lint + unit_test)
Rules:
1. 仅修改与 refinement_instruction 直接相关的代码
2. 不得回退 parent 任务已通过验证的变更（除非 instruction 明确要求）
3. 保持 target_paths 范围不变
```

#### (c) target_paths 继承
默认继承 parent 的 `target_paths`。refinement 任务**不允许扩大** target_paths 范围——如果反馈需要改新路径，必须发起全新任务。

### 3.5.6 任务清单

#### P1.5-1: `refinement_context_builder.js` — 血缘上下文构建器

**模块位置**: `worker-coder/refinement_context_builder.js`

**职责**: 给定 parent_task_id 和 parent_run_id，从 artifact 目录中提取前次执行的文件变更、diff、验证结果，构建注入到 prompt 的 refinement context。

**依赖**: Phase 1 的 `native_file_tools.js`（P1-3）

**输入**:
```javascript
{
  workspaceRoot: string,
  parentRunId: string,
  parentTaskId: string,
  refinementInstruction: string,
  refinementRound: int,
}
```

**输出**:
```javascript
{
  ok: boolean,
  context: {
    previous_changes: Array<{path, content, truncated}>,
    previous_diff_summary: object | null,
    previous_verification: object | null,
  },
  prompt_injection: string,     // 可直接拼入 prompt 的文本块
  token_estimate: int,
  error: string | null,
}
```

**验收标准**:
- parent artifact 存在时正确提取 2+ 文件上下文
- parent artifact 不存在时优雅降级（返回空 context，不阻塞执行）
- token_estimate 准确度 ≤ ±20%
- 新增测试 5 case

#### P1.5-2: Task Payload 扩展 — `lineage` 字段

**变更文件**: `worker-coder/worker.js` (payload 解析), `worker-coder/task_contract.js` (归一化)

**变更范围**: 在 Redis stream payload 解析中识别 `lineage` 字段，传递给 `delegateTask`。

**关键约束**:
- `lineage` 为可选字段，缺失时行为与当前完全一致（非 refinement 任务）
- `lineage.parent_task_id` 必须对应 artifact 目录中存在的 run（不存在时降级为普通任务，记录 warning）
- `lineage.refinement_round` 有硬上限（默认 **5 轮**），防止无限迭代

#### P1.5-3: `coding_service.js` 集成 — Refinement 路径

**变更文件**: `worker-coder/coding_service.js`

**变更范围**: 在 `delegateTask` 的 prompt 构建阶段，如果 `lineage` 存在：
1. 调用 `refinement_context_builder.js` 构建上下文
2. 将 `prompt_injection` 追加到 `prompt_contract.js` 生成的 prompt 中
3. `target_paths` 继承自 parent（不允许扩大）

**Prompt 构建流程**:
```
原始 prompt (task_prompt)
  + Execution Contract v1 (现有 prompt_contract.js)
  + Refinement Context v1 (NEW — 仅当 lineage 存在)
  = 最终 prompt
```

**feature flag**: `refinement_reentry_enabled`，默认 `false`

#### P1.5-4: Orchestrator 侧 — Refinement 步骤跳过逻辑

**变更范围**: Orchestrator（不在 worker-coder 内，此处仅定义接口契约）

**Orchestrator 需实现**:
- 接收 refinement 任务请求（用户反馈 + parent_task_id）
- 基于 task_class 分类决定跳过哪些 workflow 步骤（见 3.5.4 表格）
- 构建带有 `lineage` 字段的 task payload 发布到 Redis stream
- 在 cohort 数据中标记 `is_refinement: true` 以区分首次执行和迭代修复

#### P1.5-5: failure_memory 扩展 — 血缘追踪

**变更文件**: `worker-coder/failure_memory.js`

**新增字段**:
```javascript
{
  // 现有字段不变，新增：
  lineage: {
    parent_task_id: string | null,
    refinement_round: int | null,
    inherited_target_paths: string[] | null,
    context_reuse_status: "full" | "partial" | "failed" | null,
  }
}
```

**目的**: cohort validation 可以分别统计"首次执行"和"refinement 迭代"的成功率，验证迭代回路是否真正提升了最终交付质量。

### 3.5.7 用户体验流程 (通过 Discord)

```
用户: /code 创建一个客户管理 API
  ↓
Nexus: [执行完毕] 已创建 customers.js, types.ts, customers.test.js
       验证结果: lint ✓, unit_test ✓
       [📎 查看 diff] [🔄 请求修改]
  ↓
用户: 🔄 把返回格式从 object 改成 array，加 total_count
  ↓
Nexus: [Refinement R1] 基于上次产出，仅修改 customers.js
       验证结果: lint ✓, unit_test ✓
       [📎 查看 diff] [🔄 继续修改] [✅ 完成]
  ↓
用户: ✅ 完成
```

### 3.5.8 Phase 1.5 度量

| 指标 | 当前基线 | 目标 | 数据来源 |
|------|---------|------|---------|
| Refinement 任务耗时（中位数） | N/A (需全新任务，3-10 min) | < 首次执行的 40% | task 时间戳 |
| Refinement 首次通过率 | N/A | ≥ 70% | cohort validation (is_refinement=true) |
| 用户发起全新任务做微调的频率 | 100% | < 20% | operator 统计 |
| Refinement 引入的回归 bug | N/A | 0 (parent 已通过的测试不可 break) | verification_runner |

### 3.5.9 风控

- **target_paths 不可扩大**: refinement 只能在 parent 的 scope 内修改，防止范围蔓延
- **轮数硬上限**: 默认 5 轮，超过后强制要求发起全新任务（重新走完整 workflow）
- **regression guard**: refinement 执行前记录 parent 的 verification 结果；refinement 完成后重跑相同的 verification plan，如果 parent 通过的 tier 在 refinement 后失败 → 标记为 `regression_detected`，阻止 promotion
- **parent artifact 不可变**: refinement 读取但不修改 parent 的 artifact 目录

---

## 四、Phase 2 — 结构化上下文管道 (Structured Context Pipeline)

### 4.1 目标

实现 v4.2 Layer E 预留的 `ContextRequest → ContextResponse` 接口的**结构层 (Layer 1)** 部分，为 TC-02/TC-03/TC-04 提供自动化的文件级上下文发现，减少 operator 手动组装负担。

### 4.2 启动前置条件 (Gate)

遵循 v4.2 Section 13.3 的 phasing criteria：

> Phase 2 仅在以下条件全部满足后启动：
> 1. Phase 1 + Phase 1.5 全部任务通过 cohort validation
> 2. 至少完成 **1 轮完整的 cohort validation** (覆盖 TC-01 ~ TC-05，含 refinement 场景)
> 3. cohort 数据中 `context_failure` 归因占比 **≥ 25%**（v4.2 阈值为 30%，此处适度下调以留出实施 buffer）
> 4. 或 beta 用户反馈中"上下文不足/错误"出现频率 **≥ 3 次 / 周**

如果条件不满足，Phase 2 保持 deferred，手动 context 模式继续运作。

### 4.3 任务清单

#### P2-1: `context_resolver.js` — 结构层上下文解析器

**模块位置**: `worker-coder/context_resolver.js`

**职责**: 给定 target_paths 和 task_class，自动发现结构相关的文件集合。

**实现策略 (不依赖 tree-sitter)**:
- **JS/TS**: 正则扫描 `import`/`require` 语句，递归解析到 `dependency_depth` 层
- **Python**: 正则扫描 `import`/`from...import` 语句
- **通用**: `package.json` 的 `dependencies` / `tsconfig.json` 的 `paths` 作为辅助索引

**输入**: v4.2 定义的 `ContextRequest`
```javascript
{
  task_class: string,
  target_paths: string[],
  max_files: int,
  max_tokens: int,
  dependency_depth: int,
}
```

**输出**: v4.2 定义的 `ContextResponse`
```javascript
{
  status: "complete" | "partial" | "failed",
  files: Array<{path: string, role: string, token_estimate: int}>,
  token_usage: int,
  missing_context: string[],
  confidence: float,           // 0-1, 基于解析成功率
  resolution_method: "import_graph",
  resolution_time_ms: int,
}
```

**与 context_envelope 的关系**:
- `max_files` 和 `max_tokens` 是硬约束，解析器到达上限时停止并返回 `status: "partial"`
- `dependency_depth` 控制递归层数
- 这些参数复用 v4.2 Layer A 的 task class 默认值（见 v4.2 Section 7 表格）

**验收标准**:
- 对 TC-02 场景：给定一个 React 组件路径，正确发现 2 层 import 依赖
- 对 TC-03 场景：给定一个 API route 路径，正确发现 model/service 依赖
- context_envelope 硬约束测试：超过 max_files 时返回 `partial`
- 解析失败优雅降级测试：import 路径不存在时不崩溃，记录到 `missing_context`
- 性能：1000 文件 workspace 内 < 3 秒

#### P2-2: 集成到 `coding_service.js` 的 delegateTask

**变更文件**: `worker-coder/coding_service.js`

**变更范围**: 在 `delegateTask` 的 adapter 调用前，如果 `context_packet` 为空且 `context_envelope.context_source` 为 `"automated"`，调用 `context_resolver.js` 自动填充。

**关键约束**:
- 通过 `runtime_config.js` 的 `context_resolver_enabled` 开关控制，默认 `false`
- 如果 operator 已手动提供 `context_packet`，不覆盖（manual 优先）
- `ContextResponse` 记录到 task artifact 目录，供 cohort 分析

#### P2-3: Cohort 数据采集扩展

**变更文件**: `worker-coder/failure_memory.js`

**新增字段**:
```javascript
{
  // 现有字段不变，新增：
  context_resolution: {
    method: "manual" | "automated" | "hybrid",
    files_provided: int,
    token_usage: int,
    confidence: float | null,
    missing_context: string[],
  }
}
```

**目的**: 为 v4.2 Section 13.3 的 RAG phasing decision 提供数据基础——如果结构层解析已经将 `context_failure` 占比降到 < 15%，则语义层 RAG 的投资可推迟。

---

## 五、Phase 3 — 调度弹性化 (Scheduling Flexibility)

### 5.1 目标

在现有 8 步 workflow 框架内，允许**无数据依赖的步骤并行执行**，同时保持严格的 artifact 契约和治理模型。

### 5.2 启动前置条件 (Gate)

> Phase 3 仅在以下条件全部满足后启动：
> 1. Phase 1 + Phase 2 全部任务通过 cohort validation
> 2. 至少 **2 轮 cohort validation** 在 TC-01 ~ TC-04 上达标（首次通过率 ≥ 60%）
> 3. Orchestrator 侧的 `dynamic_routing_enabled` 完成 M7 评审并启用

### 5.3 设计原则

**不是**重写执行引擎，而是在现有 workflow 编排中引入**受控的并行窗口**。

当前 8 步 workflow:
```
pm_spec → arch_design → impl_be → impl_fe → smoke_test → qa_verify → release_pack → deploy_preview
```

可并行的窗口（需 arch_design 的 handoff 明确声明无跨端依赖）：
```
pm_spec → arch_design → ┌ impl_be ┐ → smoke_test → qa_verify → release_pack → deploy_preview
                         └ impl_fe ┘
```

### 5.4 任务清单

#### P3-1: Workflow Dependency Graph 声明

**变更文件**: `worker-coder/step_artifact_contract.js`

**新增**: 每个 step 的 handoff 定义增加 `depends_on` 和 `parallel_eligible` 字段。

```javascript
{
  step_id: "impl_fe",
  depends_on: ["arch_design"],    // 而非 ["impl_be"]
  parallel_eligible: true,        // 可与同层其他 step 并行
  parallel_group: "implementation", // 并行分组标识
}
```

**约束**:
- `parallel_eligible` 默认 `false`（deny-by-default）
- 仅当 arch_design 的 handoff 中明确声明 `be_fe_independent: true` 时，orchestrator 才启用并行
- 如果 `be_to_fe.json` 的 typed_handoff 存在非空内容，强制串行（前后端有接口依赖）

#### P3-2: Worker 多实例并行调度

**变更文件**: Orchestrator 侧（不在 worker-coder 范围内）

**设计要点**:
- Orchestrator 检测到 parallel_group 内的 steps 可并行时，向 Redis stream 同时发布多个 task
- 每个 task 有独立的 isolation_workspace
- 所有 parallel tasks 完成后，orchestrator 按序执行 promotion（先 BE 后 FE，或按 handoff 顺序）
- 任何一个 parallel task 失败 → 所有同组 task 标记为 `parallel_peer_failed`，回滚全部 isolation workspace

**验收标准**:
- 无依赖的 BE/FE 任务并行执行，总耗时 < max(BE_time, FE_time) × 1.2
- 一端失败时另一端正确回滚，主 workspace 无污染
- 有依赖时（be_to_fe.json 非空）正确退化为串行

#### P3-3: 动态子任务生成 (Scoped)

**scope**: 仅允许 `impl_be` 和 `impl_fe` 在执行过程中生成**最多 1 个**子任务，且子任务类型限制为：
- `install_dependency`: 安装缺失的 npm/pip 包
- `create_config`: 创建缺失的配置文件（在 target_paths 范围内）

**不允许**:
- 子任务修改 arch_design 的产出
- 子任务生成新的子任务（深度 = 1）
- 子任务访问 target_paths 之外的路径

**设计理由**: 在受控范围内解决"执行中发现缺依赖"的最常见问题，而不是构建通用的动态任务树。通用动态任务树的复杂度远超当前需求，可作为 v4.0 的课题。

---

## 六、风控与治理

### 6.1 不变量 (Invariants)

以下约束在三个 Phase 中**绝对不可违反**：

1. **scope_guard 三层校验不可绕过**: 所有文件写入（含 surgical_patch）必须经过 `validateRequestedWrite`
2. **isolation_workspace 沙盒不可跳过**: 所有代码变更必须在隔离沙盒中执行，通过 `promoteIsolatedChanges` 合并
3. **failure_memory 必须记录**: 任何新增的执行路径（微修复、自动上下文、并行调度）的失败必须写入 `coding_failures.jsonl`
4. **command 白名单不可扩展**: `ALLOWED_CMD_PREFIXES` 的变更需要单独的安全评审
5. **敏感信息脱敏不可跳过**: 所有日志输出必须经过 `redactSensitiveText`
6. **context_envelope 硬约束不可软化**: 超限必须返回 `partial`/`failed`，不可静默截断

### 6.2 回滚方案

| Phase | 回滚方式 | 回滚代价 |
|-------|---------|---------|
| Phase 0 | `redis_cache_enabled: false` → fallback 到原始计算路径，结果完全一致 | 零，纯性能回退 |
| Phase 1 | `surgical_patch_enabled: false` → 完全跳过微修复路径，行为回归 v3.1 | 零，feature flag 即时生效 |
| Phase 1.5 | `refinement_reentry_enabled: false` → lineage 字段被忽略，所有任务按首次执行处理 | 零，用户仍可发起全新任务做修改 |
| Phase 2 | `context_resolver_enabled: false` → 回归手动 context 模式 | 零，手动模式始终可用 |
| Phase 3 | `parallel_scheduling_enabled: false` → 回归严格串行 | 零，串行始终是默认行为 |

每个 Phase 的 feature flag 通过 `runtime_config.js` 控制，无需重启 worker 进程。

### 6.3 现有测试回归保证

当前 27 个测试文件覆盖了 worker-coder 的全部核心路径。每个 Phase 的实施必须：
- 现有 27 个测试零回归
- 新增测试覆盖率 ≥ 80%（按新增代码行计）
- feature flag 关闭状态下的行为等价性测试

---

## 七、成功度量

### 7.0 Phase 0 度量

| 指标 | 当前基线 | 目标 | 数据来源 |
|------|---------|------|---------|
| `buildRepoMap` 连续调用耗时 | 200-800ms | < 10ms (cache hit) | 性能日志 |
| `runStaticChecks` 未变更文件耗时 | 每文件 50-200ms | < 1ms (cache hit) | 性能日志 |
| Context envelope token 计数精度 | 无 | ±1% vs tiktoken | 单元测试 |

### 7.1 Phase 1 度量

| 指标 | 当前基线 | 目标 | 数据来源 |
|------|---------|------|---------|
| 语法错误修复耗时（中位数）| 30-120 秒 (full retry) | < 5 秒 | `failure_memory.jsonl` |
| 语法错误自动修复成功率 | 0% (无此能力) | ≥ 70% | `surgical_fix_success` 归因计数 |
| 整体首次通过率（TC-01）| TBD (cohort 数据) | 不低于基线 | cohort validation |

### 7.2 Phase 1.5 度量

| 指标 | 当前基线 | 目标 | 数据来源 |
|------|---------|------|---------|
| Refinement 任务耗时（中位数）| N/A (需全新任务 3-10 min) | < 首次执行的 40% | task 时间戳 |
| Refinement 首次通过率 | N/A | ≥ 70% | cohort validation (is_refinement=true) |
| 用户发起全新任务做微调的频率 | 100% | < 20% | operator 统计 |
| Refinement 引入的回归 bug | N/A | 0 | verification_runner regression check |

### 7.3 Phase 2 度量

| 指标 | 当前基线 | 目标 | 数据来源 |
|------|---------|------|---------|
| context_failure 归因占比 | TBD (需 cohort 数据) | 降低 ≥ 50% (相对) | `failure_memory.jsonl` |
| Operator 手动组装 context 的频率 | 100% (TC-02/04) | < 30% | operator 反馈 + context_source 字段 |
| 自动上下文解析置信度（中位数）| N/A | ≥ 0.7 | `ContextResponse.confidence` |

### 7.4 Phase 3 度量

| 指标 | 当前基线 | 目标 | 数据来源 |
|------|---------|------|---------|
| BE+FE 并行任务总耗时 | T(BE) + T(FE) | < max(T(BE), T(FE)) × 1.2 | task 时间戳 |
| 并行执行中的 workspace 污染事件 | N/A | 0 | isolation_workspace 审计日志 |

---

## 八、实施排期与依赖关系

```
   ┌──────────────────────────────┐
   │  Phase 0 (P0-1~3)           │  ← 立即可做，无任何前置依赖
   │  Redis Caching + tokenizer  │
   └──────────┬───────────────────┘
              │
              ▼
                          ┌─────────────────────────────────┐
                          │  v4.2 Cohort Validation Round 1  │
                          │  (TC-01 ~ TC-05 基线数据采集)    │
                          └──────────┬──────────────────────┘
                                     │
                    ┌────────────────┼────────────────────┐
                    ▼                                      │
        ┌───────────────────┐                             │
        │  Phase 1 (P1-1~3) │                             │
        │  轻量微操层        │                              │
        └──────┬────────────┘                              │
               │                                           │
               ▼                                           │
   ┌─────────────────────────┐                             │
   │  Phase 1.5 (P1.5-1~5)  │                             │
   │  迭代修复回路            │                              │
   └──────┬──────────────────┘                              │
          │                                                │
          ▼                                                ▼
   ┌────────────────────────────┐     ┌───────────────────────────┐
   │  Phase 1 + 1.5 Cohort      │     │  context_failure 占比      │
   │  Validation (含 refinement) │     │  分析 (Gate for Phase 2)   │
   └──────┬─────────────────────┘     └──────────┬────────────────┘
          │                                       │
          │      ┌────────────────────────────────┘
          │      │ (仅当 context_failure ≥ 25%)
          ▼      ▼
   ┌───────────────────┐
   │  Phase 2 (P2-1~3) │
   │  结构化上下文管道   │
   └──────┬────────────┘
          │
          ▼
   ┌───────────────────────┐
   │  Phase 2 Cohort        │
   │  Validation × 2 轮     │
   │  + M7 Routing 评审     │
   └──────┬────────────────┘
          │ (仅当首次通过率 ≥ 60% + M7 启用)
          ▼
   ┌───────────────────┐
   │  Phase 3 (P3-1~3) │
   │  调度弹性化         │
   └───────────────────┘
```

**关键路径**: Phase 0 (立即) → v4.2 Cohort Round 1 → Phase 1 → Phase 1.5 → Cohort Validation → Phase 2 (conditional) → Phase 3 (conditional)

---

## 九、与原 v4.0 提案的差异说明

| 原 v4.0 提案 | 本文档 (v3.2) | 变更理由 |
|-------------|-------------|---------|
| 自称"v4.0"，定位为架构重写 | v3.2，定位为 v4.2 基线的实施增强 | 版本号应反映实际变更范围，v4.2 契约层尚未完全落地 |
| 未覆盖迭代修复/用户反馈回路 | Phase 1.5 Refinement Re-entry：task lineage + 上下文继承 + 步骤跳过 + regression guard | 这是从"工具"到"生产力"的核心断裂点，v4.2 两次列为 Non-Goal 但实际是 beta 用户最高频需求 |
| 以"打赢 Claude Code"为目标 | 以"提升 Nexus 自身的自动化可靠性"为目标 | Nexus 是无人值守编排平台，与交互式 CLI 的设计约束不同 |
| 引入 Codebase Indexer (tree-sitter AST) | 正则 import graph 解析，不引入 tree-sitter 依赖 | 先用最小依赖验证价值，cohort 数据不足以证明需要重量级工具 |
| 废弃"死板的 JSON Schema" | 保留全部 Schema，按 cohort 数据逐条评估 | 现有 Schema 是多轮 bug 修复沉淀，不可盲目删除 |
| Streaming Shell 定为 P0 | 不在 scope 内 | 无人值守场景下流式交互优先级低，同步 exec + timeout 已足够 |
| Dynamic Task Tree 通用引擎 | 受限子任务生成（深度=1，类型白名单） | 通用动态任务树复杂度远超需求，过早引入会污染治理模型 |
| Swarm Mode 多 Worker 并行 | 受控的 impl_be/impl_fe 并行窗口 | 先证明双轨并行的价值和安全性，再考虑通用 Swarm |
| "删除 50% 冗余 Schema" | 不删除任何 Schema | 无具体删减清单的量化承诺是工程红旗 |
| 修辞性语言（"瞎眼的指挥官"、"时空回溯"） | 工程规格文档 | 架构文档应以接口定义、migration path、验收标准为主体 |
| 未考虑 caching | Phase 0 Redis Caching：repo_map / static_check content-hash 缓存 + gpt-tokenizer | 利用已有 Redis 基础设施，零风险立即见效，是所有优化的前置基础 |
| 未考虑 RAG 的渐进路径 | Phase 2 正则 import graph → cohort 数据验证 → 再决定是否上 tree-sitter / vector search | 开源 RAG 框架（LangChain/LlamaIndex）为交互式 QA 设计，不适配 Nexus 的无人值守多 Worker 架构 |
| 未考虑 harness 统一化 | 附录 B 对标分析：Nexus 治理组件已超越 CC harness，统一中间件层留待 v4.0 | 当前散落但能跑且安全达标，重构收益不支撑投入 |

---

## 十、明确的非目标 (Explicit Non-Goals)

本文档不授权以下工作：

1. 完整的 RAG 系统实施（遵循 v4.2 Section 13 的 phasing criteria）
2. tree-sitter / LSP 等重量级代码分析工具的引入
3. 通用动态任务树 / Agentic Loop 引擎
4. 现有 JSON Schema 的删减或弱化
5. Streaming Shell / 长连接交互式终端
6. 多 Provider 市场化（当前仅 auto/opencode/codex）
7. 跨 workflow 的任务链 / 多步编排（注：Phase 1.5 的 refinement 是**单 workflow 内的迭代**，不是跨 workflow 链接）
8. `scope_guard.js` 保护根列表的变更
9. `ALLOWED_CMD_PREFIXES` 命令白名单的扩展
10. 开源 RAG 框架的整体引入（LangChain / LlamaIndex / Greptile 等——架构不匹配，详见附录 B）
11. 治理中间件层统一重构（harness 化——当前散落但功能达标，留待 v4.0 评估）
12. Vector DB / Embedding 服务部署（在正则 import graph 验证价值前不引入）
13. LLM 响应缓存（prompt 差异导致 cache miss 率过高，投入产出不合理）

---

## 十一、审批要求

- Phase 0 实施：无需审批（纯性能优化，零语义变更），但需 code review
- Phase 1 实施：需架构 Lead 审批
- Phase 1.5 实施：需架构 Lead + PM 联合审批（涉及用户交互流程变更）
- Phase 2 启动：需 cohort 数据审查 + PM 审批
- Phase 3 启动：需 M7 Routing 评审通过 + 架构委员会审批
- 任何不变量 (Section 6.1) 的例外：需安全委员会一票否决权

---

## 附录 A: 现有模块依赖图

```
worker.js
  └── coding_service.js
        ├── coding_executor_runtime.js  (adapter 委托)
        ├── patch_manager.js            (edit block 应用)
        ├── git_side_effects.js         (auto-commit, push)
        ├── failure_memory.js           (失败记忆持久化)
        │     └── task_contract.js      (task_class / context_envelope / failure_attribution)
        ├── artifact_scaffold.js        (expected artifacts 确保)
        ├── scoped_delta.js             (snapshot / diff / git summary)
        ├── isolation_workspace.js      (沙盒隔离)
        ├── promotion_workspace.js      (沙盒推广)
        ├── static_checks.js            (语法检查)
        ├── prompt_contract.js          (prompt 构建 + 约束注入)
        ├── retry_policy.js             (重试判断 + 同错去重)
        ├── verification_runner.js      (分层验证 + 命令校验 + 脱敏)
        ├── scope_guard.js              (路径安全)
        └── step_artifact_contract.js   (workflow handoff 校验)
```

**Phase 0 变更**:
```
  orchestrator/src/domain/repo_context_service.js  (MODIFY — Redis cache wrapper)
  worker-coder/static_checks.js  (MODIFY — content-hash cache)
  worker-coder/task_contract.js  (MODIFY — gpt-tokenizer token 计数)
  worker-coder/prompt_contract.js  (MODIFY — token budget 校验 + 超限截断)
  worker-coder/package.json  (MODIFY — +gpt-tokenizer)
```

**Phase 1 新增**:
```
  coding_service.js
        ├── surgical_patch.js  (NEW — 确定性微修复)
        └── native_file_tools.js  (NEW — 只读文件访问)
```

**Phase 1.5 新增**:
```
  coding_service.js
        └── refinement_context_builder.js  (NEW — 血缘上下文构建)
              └── native_file_tools.js  (依赖 Phase 1 的 P1-3)
  worker.js
        └── task_contract.js  (MODIFY — lineage 字段归一化)
  failure_memory.js  (MODIFY — lineage 追踪字段)
```

**Phase 2 新增**:
```
  coding_service.js
        └── context_resolver.js  (NEW — 结构层上下文解析)
```

**Phase 3 变更**:
```
  step_artifact_contract.js  (MODIFY — 新增 depends_on / parallel_eligible)
  (Orchestrator 侧变更不在此图范围)
```

---

## 附录 B: Harness 对标分析 — Nexus vs Claude Code

### B.1 组件级对照

| Claude Code Harness 组件 | Nexus 等价物 | 评估 |
|---|---|---|
| Permission Model (per-tool allow/deny) | `tool_permission_guard.js` — 角色×工具权限矩阵 + JSON Schema 校验 | **Nexus 更强**: 多角色 RBAC vs CC 的单用户 allow/deny |
| Hooks (pre/post tool execution) | `audit_hooks.js` — 5 个生命周期钩子，持久化到 Postgres | **Nexus 更强**: DB 持久化可审计 vs CC 的 shell 命令钩子 |
| Tool Schema & Risk Profile | `tool_schema.js` — CORE_TOOL_SCHEMAS + 5 级 risk_profile | **Nexus 更强**: 量化 risk_score vs CC 的无量化 |
| Safety Council | `permission_council.js` — SafetyAuditor + ContextValidator + RiskScorer 三层决策 | **Nexus 更强**: 多 agent 组合评审 vs CC 的规则匹配 |
| Agent Loop | `workflow_engine.js` + `coding_service.js` delegateTask 循环 | **架构不同**: CC 单会话 REPL，Nexus 多步 workflow + retry |
| Sandbox | `isolation_workspace.js` — scaffold/shadow/promote 三模式 | **Nexus 更强**: 文件级隔离 + promotion vs CC 的 Docker 整体隔离 |
| Command Validation | `verification_runner.js` — 白名单 + metachar 拦截 | **对等** |
| Credential Redaction | `verification_runner.js` — 6 种 token 模式脱敏 | **对等** |
| Single Agent Guardrails | `single_agent_guardrails.js` — evidence_id + replay_tag + bounded_validation | **Nexus 独有**: CC 无此概念 |

### B.2 Nexus 缺失的一点

CC harness 有**统一的工具执行中间件管道**——任何工具调用都经过同一条 permission → audit → execute → audit 链路。Nexus 的等价组件散落在 orchestrator (`tool_permission_guard`, `permission_council`) 和 worker-coder (`scope_guard`, `verification_runner`, `audit_hooks`) 之间，由各调用方自行编排。

### B.3 结论

1. **不 copy**: Nexus 的治理能力已超越 CC harness，copy 是降级
2. **不重构**: 当前散落但功能达标、27 个测试覆盖、安全约束完整，重构收益不支撑投入
3. **v4.0 课题**: 当 Phase 3 落地、系统复杂度进一步上升后，再评估统一中间件层的必要性
4. **可借鉴的设计思想**: CC 的 "每个工具调用都经过同一条管道" 是好的架构卫生习惯，值得在 v4.0 中参考

---

## 附录 C: RAG/Caching 开源方案评估

### C.1 Caching — 不需要框架

| 考虑的方案 | 结论 | 理由 |
|-----------|------|------|
| keyv / catbox | **不引入** | Nexus 已有 ioredis，场景只需 key-value + TTL，框架是过度工程 |
| Redis 原生 SET/GET + EX | **采用** | 零新依赖，几十行代码，与现有 `worker.js` 的 Redis 连接复用 |

### C.2 RAG — 开源框架不可直接 copy

| 方案 | 语言 | 不适配原因 |
|------|------|-----------|
| LangChain / LlamaIndex | Python | Nexus 是 Node.js 管线，跨语言调用增加复杂度；面向单次 QA 而非代码生成 workspace 索引 |
| Cursor / Continue | TS | 为 IDE 交互式设计，依赖 LSP + 用户实时操作；Nexus 是无人值守管线 |
| Aider repo map | Python | 最接近需求（tree-sitter repo map），但单用户 CLI，无分布式/安全层 |
| Greptile / Bloop | Various | 需独立部署 embedding 服务 + vector DB，基础设施成本过高 |
| ChromaDB / Qdrant | Python/Rust | 通用 vector store，需自写 code chunking + embedding pipeline，工作量 ≥ 自建 |

### C.3 可复用的组件（非框架）

| 组件 | 用途 | 引入时机 |
|------|------|---------|
| `gpt-tokenizer` (npm) | context_envelope token 计数 | **Phase 0 立即** |
| `node-tree-sitter` (npm) | AST 级 import graph 解析 | Phase 2 cohort 数据证明正则不够后 |
| Embedding model (local) | 语义搜索 | v4.2 RAG phasing criteria 触发后（context_failure ≥ 30%）|

### C.4 渐进路径

```
Phase 0: gpt-tokenizer (token 计数) ← 已规划
Phase 2: 正则 import graph (零依赖) ← 已规划
  └── cohort 数据: 正则够用? (context_failure < 15%)
        ├── YES → 停在正则，不引入更多
        └── NO → 评估 node-tree-sitter (AST 级)
              └── cohort 数据: AST 够用? (context_failure < 15%)
                    ├── YES → 停在 tree-sitter
                    └── NO → 评估 embedding + vector search (重大投资决策)
```

每一步都需要 cohort 数据验证，不做未经数据证明的升级。
