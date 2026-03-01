# OpenClaw Nexus 学习机制保护层 Patch 设计文档

## 1. 背景与问题

当前学习闭环为：

- `trace` 入库（对话/任务结果）
- 用户反馈（👍/👎）写入 `traces.feedback_json`
- 正反馈自动写 `mem_items`
- 负反馈自动写 `rules`

已验证该闭环可工作，但存在上线风险：

1. 反馈接口缺少鉴权，存在被伪造调用风险。
2. 正反馈“整包写记忆”，长期会引入噪声与重复。
3. 负反馈规则缺少去重、衰减与冲突管理，容易规则膨胀。
4. 缺少可观测指标（命中率、回滚、失效率），难以安全迭代。

---

## 2. 目标

本 Patch 目标：

1. 给学习入口加安全闸门（鉴权 + 速率限制 + 幂等）。
2. 给记忆写入加质量门控（去重、评分、TTL/衰减）。
3. 给规则系统加冲突管理（优先级、抑制、归档）。
4. 提供可回滚和灰度开关，避免影响主对话链路稳定性。

非目标：

- 不改动你现有核心任务编排模型（task/workflow 主流程不重构）。
- 不引入复杂向量数据库（先用 PostgreSQL 轻量实现）。

---

## 3. 总体方案（Guardrails Layer）

在 `orchestrator` 增加 `Learning Guardrails`：

1. **Ingress Guard**（入口保护）
   - 反馈 API token 校验
   - 反馈事件幂等键（`trace_id + user_id + feedback_type`）
   - 每用户速率限制（Redis）

2. **Memory Guard**（记忆保护）
   - 内容标准化 + 指纹（SHA256）
   - 相似/重复检测（完全指纹 + 文本归一 hash）
   - 质量评分阈值（短文本、空洞文本拒绝）
   - 生命周期（`active/expired` + 衰减）

3. **Rule Guard**（规则保护）
   - 规则分类（safety/style/routing）
   - 规则优先级 + 置信度
   - 冲突检测（同域互斥规则）
   - 自动降权和归档

4. **Observability**
   - 指标：accepted/rejected/conflict/expired/hit_rate
   - 审计日志：每次写入原因 + 触发源

---

## 4. 变更清单

## 4.1 配置开关（env）

新增（建议默认值）：

- `LEARNING_GUARDRAILS_ENABLED=true`
- `LEARNING_FEEDBACK_TOKEN=<strong-random-token>`
- `LEARNING_RATE_LIMIT_PER_MIN=20`
- `LEARNING_MIN_MEM_SCORE=0.60`
- `LEARNING_MEM_MAX_PER_PROJECT=500`
- `LEARNING_RULE_MAX_PER_PROJECT=300`
- `LEARNING_DECAY_HALF_LIFE_DAYS=30`
- `LEARNING_SHADOW_MODE=true`（先只记录不生效，灰度）

---

## 4.2 数据库变更（PostgreSQL）

### A. 新增表：`learning_events`

用途：统一审计每次学习写入行为。

字段建议：

- `event_id TEXT PK`
- `project_id TEXT`
- `source TEXT` (`discord_reaction|api_feedback|system`)
- `trace_id TEXT`
- `event_type TEXT` (`mem_accept|mem_reject|rule_accept|rule_conflict|rate_limited`)
- `payload_json TEXT`
- `created_at TIMESTAMPTZ DEFAULT NOW()`

### B. 扩展 `mem_items`

新增列：

- `fingerprint TEXT`
- `quality_score DOUBLE PRECISION DEFAULT 0`
- `status TEXT DEFAULT 'active'` (`active|suppressed|expired`)
- `last_hit_at TIMESTAMPTZ`
- `hit_count INT DEFAULT 0`
- `expires_at TIMESTAMPTZ`

索引建议：

- `(project_id, fingerprint)` 唯一索引（去重）
- `(project_id, status, created_at DESC)`

### C. 扩展 `rules`

新增列：

- `fingerprint TEXT`
- `priority INT DEFAULT 50`
- `confidence DOUBLE PRECISION DEFAULT 0.5`
- `status TEXT DEFAULT 'active'` (`active|suppressed|archived`)
- `conflict_with TEXT`（可空）
- `last_hit_at TIMESTAMPTZ`
- `hit_count INT DEFAULT 0`

索引建议：

- `(project_id, status, priority DESC, updated_at DESC)`
- `(project_id, fingerprint)` 唯一索引

### D. 新增表：`feedback_idempotency`

字段建议：

- `idem_key TEXT PK`
- `trace_id TEXT`
- `user_id TEXT`
- `created_at TIMESTAMPTZ DEFAULT NOW()`

---

## 4.3 API 变更

### `POST /traces/:trace_id/feedback`

新增校验：

1. Header 必带：`X-Learning-Token`，必须等于 `LEARNING_FEEDBACK_TOKEN`。
2. Body 必带：`user_id`、`feedback`。
3. 幂等键：`sha256(trace_id + user_id + feedback)`，重复提交直接返回 `ok + deduplicated=true`。
4. Redis 限流：`learning:feedback:{user_id}:{minute_bucket}`。

返回结构补充：

- `guardrails`: `{ accepted, deduplicated, reason, shadow_mode }`

---

## 4.4 规则/记忆写入策略

### Memory Accept 条件（正反馈）

仅在以下条件全部满足时写入 `mem_items`：

1. 文本长度 >= 40
2. 非模板废话（黑名单短语）
3. 质量评分 >= `LEARNING_MIN_MEM_SCORE`
4. 指纹未命中重复
5. 项目内 `active` 条目未超上限

否则写 `learning_events` 记录拒绝原因。

### Rule Accept 条件（负反馈）

1. `reason` 长度 >= 10
2. 能归类到 `rule_type`（safety/style/routing）
3. 与已有规则无高冲突，或新规则优先级更高
4. 指纹未重复

冲突时：

- 新规则低优先级：拒绝并记录 `rule_conflict`
- 新规则高优先级：旧规则 `status=suppressed`，新规则生效

---

## 4.5 衰减与清理任务（cron）

每日任务：

1. 根据半衰期衰减 `alpha/beta` 或 `confidence`。
2. 长期未命中条目转 `expired/archived`。
3. 项目超上限时，优先清理低分低命中旧条目。

建议在 orchestrator 内增加 `node-cron` job：

- `0 3 * * *`（本地时区凌晨）

---

## 5. 代码落点建议

建议改动文件：

1. `orchestrator/src/index.js`
   - feedback API 鉴权、限流、幂等
   - guardrails 写入逻辑
   - cron 衰减任务
2. `orchestrator/src/learning/guardrails.js`（新增）
   - `scoreMemoryCandidate()`
   - `dedupeFingerprint()`
   - `resolveRuleConflict()`
3. `orchestrator/src/learning/sql.js`（新增）
   - DB DDL + query helpers

可选：

4. `worker-quant/worker.py`
   - 在输出中补 `trace_meta`（提高 feedback 可解释性）

---

## 6. 灰度与回滚

灰度阶段：

1. `LEARNING_SHADOW_MODE=true`：
   - 仅记录 guardrails 决策，不真正写入 rules/mem_items。
2. 观察 3-7 天指标后切换为 `false`。

回滚策略：

1. 关 `LEARNING_GUARDRAILS_ENABLED=false`，即时回退到旧逻辑。
2. 不删除历史数据，仅停用新策略。

---

## 7. 验收标准

功能验收：

1. 未带 token 的反馈请求返回 403。
2. 同一幂等键重复提交不产生重复写入。
3. 垃圾短反馈不会写入规则/记忆。
4. 冲突规则按优先级处理且可审计。
5. 日常对话延迟无明显上升（P95 增幅 < 10%）。

稳定性验收：

1. 不影响 `stream:task` / `stream:result` 主链路。
2. DB 异常时 guardrails fail-open（不阻断主回复）。

---

## 8. 测试计划（最小集）

1. 单元测试：
   - 评分函数、去重、冲突判定、衰减函数。
2. 集成测试：
   - feedback API token/限流/幂等。
3. 回归测试：
   - Discord 正常聊天、tool 结果回传、reaction 反馈链路。
4. 压测：
   - 高频 reaction 下 DB/Redis 负载。

---

## 9. 里程碑拆分

M1（1天）：

- 鉴权 + 幂等 + 限流 + 审计日志。

M2（1-2天）：

- 记忆评分与去重 + 规则冲突管理 + shadow mode。

M3（1天）：

- 衰减清理任务 + 指标报表 + 验收脚本。

---

## 10. 结论

该 Patch 在不重构主系统的前提下，能显著降低学习闭环的“污染风险”和“规则失控风险”，并提供可观测、可灰度、可回滚的上线路径，适合你当前项目节奏。

