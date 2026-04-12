# v3.7 Static Audit Gate — 设计文档

**Date**: 2026-04-12
**Author**: Architect
**Status**: Draft
**Goal**: 在 coding_team_v0 流水线中加入一个**确定性**合规 gate，用纯代码（非 LLM）执行安全/契约检查，替代当前依赖 LLM 自我审计的不可靠模式。

---

## 1. 问题定义

### 1.1 现状
- `qa_verify` 由 LLM 生成 QA 报告，不可靠（自相矛盾、scaffold 化、规则被忽略）
- `smoke_test` 只打 1-2 个 endpoint 看 200，无合约测试
- 已知漏洞（codex 在 v3.5.2 审视中发现）无法被现有流程捕获：stored XSS、CSS class 注入、BE 缺输入验证、DELETE 对不存在行返回 200
- Prompt 层规则边际效应递减 — 加 10 条规则模型只遵守 6 条，加 20 条还是只遵守 6 条

### 1.2 根因
**创意生成（generation）和合规审计（audit）没有分离**。LLM 既是作者又是审计员，无法提供独立判断。Prompt 规则是"软约束"，需要"硬门禁"兜底。

### 1.3 架构原则
> Creative generation (LLM) ≠ Compliance enforcement (code)

审计必须确定性，必须由非 LLM 代码执行，必须能 block 流水线。

---

## 2. 设计方案

### 2.1 流水线位置

```
pm_spec → arch_design → impl_be → impl_fe_skeleton → impl_fe_modules
  → smoke_test → [NEW] static_audit → qa_verify → release_pack → deploy_preview
                  ↑ 确定性 gate，失败阻塞后续步骤
```

**新增一个 step**，不是 4 个（澄清上一条消息）。在 smoke_test 之后、qa_verify 之前。

### 2.2 内部结构

单 step 内挂载 4 个独立 scanner，失败汇总：

```
orchestrator/scripts/static_audit/
  run_static_audit.mjs       # 入口：加载并执行所有 scanner
  scanners/
    xss_scanner.mjs          # FE XSS 检查
    class_injection.mjs      # FE CSS 类注入检查
    be_contract_checker.mjs  # BE HTTP 合约测试（实际发请求）
    delete_semantics.mjs     # BE DELETE 404 语义测试
  lib/
    ast_parser.mjs           # 轻量 JS AST (acorn)
    http_client.mjs          # 对本地 server 发测试请求
    report_writer.mjs        # 输出 verify/static_audit.json
```

### 2.3 Workflow 集成

`coding_team_v0.json` 新增步骤：
```json
{
  "id": "static_audit",
  "role": "qa",
  "tool": "coding.execute",
  "gate": "acceptance",
  "depends_on": ["smoke_test"]
}
```

`tool: coding.execute` — 和 smoke_test 一样直接跑 shell 命令，payload.command 由 step_builder 构建为 `node scripts/static_audit/run_static_audit.mjs --artifact-root <root> --server-port 13099`。

### 2.4 Scanner 契约

每个 scanner 返回 JSON：
```json
{
  "scanner_id": "xss_scanner",
  "status": "pass" | "fail" | "warning",
  "findings": [
    {
      "severity": "critical" | "high" | "medium" | "low",
      "code": "XSS_UNESCAPED_INNERHTML",
      "file": "impl/fe_changes/public/app.js",
      "line": 20,
      "snippet": "overlay.innerHTML = `<div>${message}</div>`;",
      "detail": "message passed to showConfirmDialog is user-controlled but not escaped",
      "fix_hint": "wrap message with escapeHtml() before innerHTML assignment"
    }
  ],
  "duration_ms": 45
}
```

汇总报告 `verify/static_audit.json`：
```json
{
  "overall_status": "fail" | "pass_with_warnings" | "pass",
  "scanners": { "xss_scanner": {...}, "class_injection": {...}, ... },
  "total_findings": { "critical": 1, "high": 0, "medium": 2, "low": 0 },
  "blocking": true
}
```

### 2.5 Gate 行为

- `overall_status: fail` = 任意 critical finding → step 失败，触发现有 validation retry 机制
- 失败时把 findings 写入 `meta/validation_feedback_impl_fe_modules.txt` 和 `meta/validation_feedback_impl_be.txt`（具体定位到源头步骤）
- Retry 时 step_builder 读取 feedback 注入到对应步骤的下次 prompt，LLM 获得精确的修复指示
- `pass_with_warnings` = 只有 medium/low → 记录，不阻塞

### 2.6 Scanner 实现细节

#### xss_scanner.mjs
- 用 acorn 解析 app.js 成 AST
- 找所有 `AssignmentExpression` 其左侧是 `MemberExpression { property: innerHTML | outerHTML }`
- 找所有 `CallExpression` 为 `insertAdjacentHTML`
- 对每个找到的节点，分析右侧 template literal：每个 `TemplateElement` 之间的 expression，检查是否被 `escapeHtml()` 包裹
- 豁免：如果表达式是字符串字面量或常量，不报警

#### class_injection.mjs
- 扫描 template literal 中的字符串，匹配 `class="..."` 或 `className="..."` 或 `classList.add(...)`
- 检查插值内容是否是 `identifier.status / .priority / .type / .kind / .state`
- 若是，检查是否经过查表（`MAP[value]` 或 `SOME_OBJECT[value]`）
- 未经查表 → warning

#### be_contract_checker.mjs
- 解析 `plan/interfaces.md` 提取所有 endpoint（METHOD /path + 必填字段 + enum）
- 启动 `impl/be_changes/server.js` 在端口 13100（避开 smoke 用的 13099）
- 对每个 endpoint 发送：
  - POST 缺必填字段 → 期望 400
  - POST enum 外值 → 期望 400
  - POST 引用不存在 FK → 期望 404
- 任何端点不返回期望状态码 → fail

#### delete_semantics.mjs
- 解析 interfaces.md 找所有 DELETE 端点
- 启动 server，对每个 DELETE 发送：
  - DELETE /resource/nonexistent-id → 期望 404
  - 创建资源再 DELETE → 期望 204
- 不符合 → fail

### 2.7 失败反馈回路

当前已有 `validation_feedback_<step>.txt` 机制（workflow_engine.js:699）。扩展该机制：

- `xss_scanner` / `class_injection` 失败 → feedback 写入 `impl_fe_modules` 的 feedback 文件
- `be_contract_checker` / `delete_semantics` 失败 → feedback 写入 `impl_be` 的 feedback 文件
- 每个 finding 转成一行具体描述（文件:行 + 修复提示），LLM 重试时能直接定位

---

## 3. 非目标 (Out of Scope)

- 不做 SAST 全面扫描（eslint-plugin-security、semgrep）— 范围太大，v3.8 再看
- 不替换 qa_verify — 保留语义层的 LLM 检查，作为双层兜底
- 不改动 impl_be/impl_fe 的代码生成逻辑 — scanner 只读不写
- 不做性能/压力测试 — 只做合规

---

## 4. 验收标准

| 指标 | 目标 |
|------|------|
| v3.5.2 XSS 漏洞能否被 xss_scanner 抓到 | YES，作为 benchmark |
| v3.5.2 BE 缺 404 能否被 delete_semantics 抓到 | YES |
| 加入 static_audit 后，新 E2E run 的安全问题数 | 0 critical |
| static_audit 步骤耗时 | < 30 秒 |
| False positive rate（把正确代码报成 fail） | < 5% |
| 退出 v3.5.2 产出评分 | 7/10 → 8/10 |

---

## 5. 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| AST 分析复杂，可能漏掉动态构造的 innerHTML | 漏报 | 只解决 90% 常见 case，疑难 case 留给人审；文档里明确 scanner 非完备 |
| BE 合约测试需要启动 server，失败定位复杂 | Gate 不稳定 | 复用 run_smoke_test.mjs 的 server 启动逻辑；失败时保留 server log |
| 第一次 E2E 会大量 fail 触发 retry，可能超 retry budget | 流水线失败率上升 | 先在 staging 跑 benchmark，调整严重性阈值；必要时提高 step_validation_retry_max |
| scanner 本身有 bug，误报导致流水线永远不过 | 阻塞开发 | 加 `--dry-run` 模式 + 记录所有 finding 但不 block；先观察 2-3 个 run，再启用 block 模式 |

---

## 6. 实施阶段

### Phase 1：基础设施 + 非阻塞模式 (1 天)
- 创建 scanner 目录结构和 run_static_audit.mjs 入口
- 实现 xss_scanner（AST 版本）
- 实现 delete_semantics（HTTP 版本）
- 新 step 插入 workflow，**模式为 audit_mode=dry_run**：只记录 finding 不 fail
- 在现有 v3.5.2 artifact 上回溯跑一次，确认能抓到已知漏洞

### Phase 2：补全 scanner (0.5 天)
- 实现 class_injection
- 实现 be_contract_checker
- 对 4 个 scanner 做单元测试

### Phase 3：启用 gate 模式 (0.5 天)
- 开关改为 `audit_mode=blocking`
- 接入 validation_feedback 反馈回路
- 跑一次 E2E 验证 retry 机制能修复 finding
- 更新 MEMORY.md 和进展报告

### Phase 4：benchmark 和调优 (持续)
- 把 v3.5.2 的 artifact 做成 regression fixture
- 每次改 scanner 逻辑都回跑
- 收集 false positive，调整启发式规则

---

## 7. 任务清单

### Phase 1 (P0)
- [ ] T1-01: 创建 `orchestrator/scripts/static_audit/` 目录结构
- [ ] T1-02: 实现 `run_static_audit.mjs` 入口（CLI 参数解析、加载 scanner、汇总结果、写 verify/static_audit.json）
- [ ] T1-03: 实现 `lib/ast_parser.mjs`（acorn 封装，暴露 innerHTML/outerHTML/insertAdjacentHTML 赋值节点查询）
- [ ] T1-04: 实现 `scanners/xss_scanner.mjs`
- [ ] T1-05: 实现 `lib/http_client.mjs`（带超时和错误重试的轻量 fetch 封装）
- [ ] T1-06: 实现 `scanners/delete_semantics.mjs`
- [ ] T1-07: 在 `capability_registry.json` 新增 `static_audit` 步骤
- [ ] T1-08: 在 `workflow_state.js` 新增 STEP_CONTRACTS.static_audit
- [ ] T1-09: 在 `workflow_step_builder.js` 为 static_audit 构建 payload.command
- [ ] T1-10: 加 audit_mode 配置到 `runtime_defaults.json`（默认 dry_run）
- [ ] T1-11: 在 v3.5.2 artifact `deb20d7f-8997-4dcb-af80-10420e8f1c38` 上回溯测试，确认抓到已知 XSS + 缺 404

### Phase 2 (P1)
- [ ] T2-01: 实现 `scanners/class_injection.mjs`
- [ ] T2-02: 实现 `scanners/be_contract_checker.mjs`
- [ ] T2-03: 为每个 scanner 写单元测试（worker-coder/tests 风格）
- [ ] T2-04: 在 v3.5.2 artifact 上回跑全部 4 个 scanner，对比 codex 当时发现的问题列表，确认召回率

### Phase 3 (P2)
- [ ] T3-01: audit_mode 切到 `blocking`
- [ ] T3-02: 在 `workflow_engine.js` 扩展 feedback 写入，按 finding 定向注入到 impl_be / impl_fe_modules 的 feedback 文件
- [ ] T3-03: 跑 E2E 验证：新 run 不应有 critical finding；若有，确认 retry 能修复
- [ ] T3-04: 更新 tests — `workflow_dag.test.js` 加 static_audit 到期望步骤列表
- [ ] T3-05: 更新 memory: `project_v35_plan.md` 加 v3.7 节；写进展报告

### Phase 4 (P3, 持续)
- [ ] T4-01: 把 v3.5.2 artifact 做成 fixture 放到 `orchestrator/test/fixtures/static_audit/v352_baseline/`
- [ ] T4-02: 加 CI 回归测试（每次改 scanner 都回跑 fixture）
- [ ] T4-03: 收集 false positive，迭代启发式

---

## 8. 和已有机制的关系

| 已有机制 | 保留/变更 |
|---------|----------|
| `fidelity_gate_mode: "blocking"` | 保留，和 static_audit 互补 |
| `qa_verify` LLM 语义检查 | 保留，作为第二层兜底 |
| 设计规则 (design_rules/*.json) | 保留，作为 prompt 预防 |
| `validation_feedback_*.txt` retry 机制 | 扩展，static_audit 复用 |
| `step_validation_retry_max: 2` | 可能需要提高到 3，视 retry 触发频率 |
| `lane_escalation_chain` | 保留，如 gemma4:26b 修不好 finding 可升到 cloud |

---

## 9. 不做的事

- 不引入外部 SAST 工具（eslint-plugin-security、semgrep） — 第一版保持零依赖
- 不替换 qa_verify 为 static_audit — 两者互补
- 不做 JS runtime 执行（jsdom 等）— 只静态分析
- 不做 DB schema 检查 — 只检查 API 行为
- 不管 CSS/HTML 注入之外的 web 安全（CSRF、JWT 等）— v3.8 再说
