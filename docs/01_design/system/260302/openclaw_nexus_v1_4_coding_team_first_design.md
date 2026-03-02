# OpenClaw Nexus v1.4 — Coding Team First 设计文档（更新版）

- 日期：2026-03-02
- 当前基线：你已经拥有可运行的 **控制平面（Orchestrator + Audit + Policy + Approval）** 与 **双执行面（worker-coder / worker-quant）**，并完成了 P0 全链路 E2E 验证（审批/拒绝/恢复/fallback/切模型均可用）。
- 本文目标：把“可跑系统”升级为“可交付的 **Coding Team**”，并将其作为后续电商/短视频助手的可复用模板。

---

## 0. 结论与优先级（Coding Team First）

### 0.1 一句话目标
在不引入新 OSS 依赖（Kimaki/OpenSwarm/Lobster 可后接）的前提下，用 **Capability Registry + Workflow Shell + Artifact Pack** 三件套，把 `/coder` 从“单次写代码”升级为“团队化交付流水线”。

### 0.2 本阶段（v1.4）成功标准（必须可验证）
1. **一条可重复的 Coding Team 流水线**：输入一个中等需求（webapp/crm 类），输出可合并的 PR patch + 测试报告 + 风险报告 + 运行方式。
2. **角色齐全且可替换**：PM / Architect / Frontend / Backend / QA（可选 DevOps），每个角色都有明确产物与验收门槛。
3. **Registry 是运行时契约**：新增/修改项目类型、角色、工具、policy、验收套件，不改核心代码（只改 registry + workflow 定义）。
4. **Artifact Pack 强制化**：任何“成功”结果必须包含规定字段，否则任务状态不能标记为 succeeded。
5. **可恢复与可追溯**：step-level checkpoint、resume token、workspace hash、审计日志与产物索引齐全。

---

## 1. 核心需求与设计原则

### 1.1 核心需求（来自你的原始目标）
- **中枢 AI（OpenClaw/Nexus Orchestrator）**：负责任务拆解、分配、治理（审批/审计/回滚/恢复）。
- **Coding Team（多角色 Agent 团队）**：PM/UI/FE/BE/QA 的交付闭环，能持续迭代、能复用模板。
- 后续扩展：电商助手、短视频助手等以“项目类型 + 流水线模板”方式加入，而不是再堆 if/else。

### 1.2 设计原则（硬约束）
1. **控制平面瘦身**：Orchestrator 只做接入、policy、审批、审计、派发、索引；不做业务工作流智能。
2. **工作流确定性**：多步流程必须走 Workflow Shell（typed steps + checkpoints + gates）。
3. **契约优先**：Registry/Schema/Validator 优先于 prompt 约定；系统拒绝“无契约能力”。
4. **产物可交付**：每一步必须沉淀产物；最终必须有 Artifact Pack。
5. **最小可用优先**：先落地 Coding Team v0（1 条 pipeline），再扩展更多项目类型/模板。

---

## 2. 系统总览（目标形态）

```
[Ingress: Discord/Web/UI/API]
          |
          v
  Orchestrator (Control Plane)
  - policy engine (risk gate)
  - approval gate (human-in-the-loop)
  - audit persistence (Postgres)
  - dispatch to workers (Redis Streams)
  - artifact index (MinIO/DB)
          |
          v
  Workflow Shell (Deterministic)
  - step graph, checkpoints, resume
  - step-level artifacts
          |
          v
  Worker Execution Planes
  - worker-coder: patch/execute/delegate(opencode/codex)
  - worker-quant: research/news/browser/artifact.archive
  - (future) worker-ecom / worker-video
```

> 关键变化：**Coding Team 不是“一个更强的 coder”**，而是被 Workflow Shell 编排的一组角色步骤，严格产出与验收。

---

## 3. Capability Registry（运行时契约）

### 3.1 Registry 的职责
Registry 是“系统的真理来源”，统一声明：
- project_types：例如 `webapp_crm`, `data_pipeline`, `quant_execution`
- roles：PM/Architect/Frontend/Backend/QA/DevOps
- tools：每个 worker 暴露的 tool_name 与参数 schema
- workflows：某 project_type 绑定的 step graph、gates、artifact requirements
- policies：路径/命令/数据访问风险等级与审批规则
- acceptance_suites：验收用例集合（lint/test/smoke/e2e）

### 3.2 Registry 文件划分（建议）
- `configs/registry/capability_registry.json`（主索引）
- `configs/registry/schemas/*.schema.json`（JSON Schema）
- `configs/registry/workflows/*.json`（工作流定义）
- `configs/registry/acceptance/*.json`（验收套件）
- `configs/registry/policy/*.json`（风险与白名单）

### 3.3 强制校验（必须实现）
1. **CI 校验**：PR 里 registry 改动必须通过 `validate_registry`。
2. **启动校验**：Orchestrator 启动时加载 registry；无效则 fail fast。
3. **运行时校验**：任务提交时校验 project_type/workflow/tool/role/params；无效则拒绝入队。

---

## 4. Workflow Shell（确定性多步引擎）

### 4.1 为什么需要 Workflow Shell
你现在的审批/恢复/fallback 已经证明控制平面能力强；但如果没有统一的 workflow shell：
- 新增项目类型会把逻辑堆回 orchestrator/supervisor
- 角色与产物变成“约定”，不可验证
- 难以 step-level 恢复与重跑

### 4.2 核心数据模型（建议）
- `workflow_id`：例如 `coding_team_v0`
- `step_id`：例如 `pm_spec`, `arch_design`, `fe_impl`, `be_impl`, `qa_verify`
- `checkpoint_id`：每步至少 1 个
- `resume_token`：绑定 workspace revision/hash + step state
- `idempotency_key`：跨重试/重跑去重
- `step_artifacts[]`：每步产物索引（MinIO + DB）
- `gates[]`：policy gate / approval gate / acceptance gate

### 4.3 Gate 类型
- **Policy Gate**：风险判定（路径/命令/数据）
- **Approval Gate**：高风险需要人工 approve/reject
- **Acceptance Gate**：测试套件必须通过才能进入下一步/标记成功

### 4.4 最小落地策略（v1.4）
不追求完整 Lobster/OpenSwarm；先实现：
- step graph runner（顺序 + 并行可选）
- step checkpoint + resume
- step output -> artifact pack aggregator
- 失败策略：retry / fallback provider / require approval / fail fast

---

## 5. Coding Team v0（最急需求：立刻可交付）

### 5.1 适用范围（首条流水线）
- project_type：`webapp_crm`（或通用 webapp）
- 输入：自然语言需求 + 约束（环境/技术栈/截止/风险偏好）
- 输出：可合并的 patch/PR、测试报告、运行说明、风险报告、变更摘要

### 5.2 角色与职责（必须产物化）

#### (1) PM（产品/项目经理）
- 目标：把需求变成可实现且可验收
- 产物：
  - `spec.md`（用户故事、范围、非目标、边界）
  - `acceptance.json`（验收标准：功能点/性能/安全/可用性）
  - `milestones.md`（里程碑与拆分）

#### (2) Architect（架构/技术负责人）
- 目标：方案、模块边界、技术风险
- 产物：
  - `arch.md`（模块图、数据流、接口、依赖）
  - `risk_report.json`（风险分级、缓解措施、审批点）
  - `workplan.md`（FE/BE/QA 切分）

#### (3) UI/UX（可选，若需求有界面）
- 产物：
  - `ui_wireframe.md` 或 `ui_spec.md`（页面、组件、交互）
  - （可选）`assets/` 原型图（通过 browser+archive）

#### (4) Frontend
- 产物：
  - `diff.patch`（实现）
  - `frontend_test_report.md`（lint/unit）
  - `run_frontend.md`（本地运行方式）

#### (5) Backend
- 产物：
  - `diff.patch`（实现）
  - `backend_test_report.md`（unit/integration）
  - `run_backend.md`

#### (6) QA
- 产物：
  - `test_plan.md`（用例/覆盖）
  - `smoke_report.md`（最小冒烟）
  - `verification.json`（验收映射：acceptance.json -> 通过/失败）

#### (7) DevOps（可选）
- 产物：`deploy.md`、`docker-compose.patch`、`ops_runbook.md`

### 5.3 Coding Team v0 的 Step Graph（推荐）
1. `pm_spec`（PM） -> gate: low risk auto
2. `arch_design`（Architect） -> gate: policy check
3. `impl_fe`（Frontend） || `impl_be`（Backend）并行 -> gate: policy + tests
4. `qa_verify`（QA） -> gate: acceptance suite
5. `release_pack`（Aggregator） -> 生成最终 Artifact Pack

> **强制规定**：`release_pack` 只要缺一个必需产物，就不能标记 succeeded。

---

## 6. 工具与执行（Worker-Coder 为主）

### 6.1 Worker-Coder 工具标准化
- `coding.patch`：SEARCH/REPLACE 或 unified diff patch
- `coding.execute`：白名单命令执行（npm test, pytest, lint 等）
- `coding.delegate`：委托 opencode/codex 生成 patch（受 policy 约束）

### 6.2 Worker-Quant 作为“研发助理”
在 Coding Team 中，quant worker 主要承担：
- 竞品/开源参考检索
- 文档/网页信息抓取（openclaw browser screenshot + archive）
- 生成引用材料/决策备忘录

---

## 7. Artifact Pack（交付包）规范（强制）

### 7.1 Pack 结构（建议）
`artifacts/release/<run_id>/`
- `plan/spec.md`
- `plan/arch.md`
- `patch/diff.patch`（或多文件 diff）
- `logs/stdout.txt`, `logs/stderr.txt`
- `tests/test_report.md`
- `qa/verification.json`
- `risk/risk_report.json`
- `summary/run_summary.md`（给人看的最终总结）
- `meta/run_manifest.json`（机器可读：hash、版本、依赖、耗时、provider、model）

### 7.2 Validator 规则（必须）
- succeeded 任务必须包含：
  - patch（可为空但必须声明原因）
  - tests + verification
  - risk_report
  - run_manifest
- 任意缺失 -> `error_code=ARTIFACT_INCOMPLETE`，状态 fail 或 require approval。

---

## 8. Policy / Approval / Audit（治理闭环）

### 8.1 风险分级（示例）
- Low：读写 workspace 非敏感路径；允许自动运行
- Medium：执行测试命令/安装依赖；允许自动但记录审计
- High：触达 `infra/`、`.env`、凭证、网络扫描、系统级命令；必须审批

### 8.2 审批对象与信息最小集
审批卡片必须展示：
- step_id + 变更摘要
- 触发的 policy 规则
- 将要执行的命令/将要修改的路径
- 回滚方案（或声明不可回滚）

---

## 9. 可观测性与 UI（最小闭环）

### 9.1 Dashboard 必须展示（v1.4）
- 任务列表：状态、risk、当前 step、owner（role）
- run 详情：step timeline、artifacts 列表、result_json、error_code
- 审批队列：approve/reject + 理由
- 产物浏览：按 run_id 下载 release pack（zip 可后做）

### 9.2 关键一致性修复
- `/chat` 或自然语言入口必须绑定 `run_id` 的最新结果，避免出现“unknown 模板回退”导致用户误判。
- 任何“running 超时残留”必须有 reclaim/timeout 策略（DLQ 或 fail with reason）。

---

## 10. 与 Kimaki/OpenSwarm/Lobster 的兼容策略（后接不侵入核心）

### 10.1 接入位（接口先行）
- Kimaki：替换 ingress bridge（Discord/Slack/Webhook），保持 Orchestrator API 不变
- OpenSwarm：替换 role pipeline 模板生成器，输出同样的 workflow steps
- Lobster：替换 workflow shell 实现，但保留 step/gate/artifact 契约

> 原则：OSS 只能“替换实现”，不能“绕过治理链路”。

---

## 11. 里程碑与版本策略

### v1.4（Coding Team v0 上线，1–2 周内）
- Registry schema + CI + runtime validator
- Workflow shell（顺序 + checkpoint + resume）
- Coding Team v0（webapp_crm）
- Artifact pack validator + dashboard 展示

### v1.5（模板化与扩展，2–4 周）
- 多 project_type 模板（data_pipeline、ecom_assistant、video_assistant）
- 并行步骤与资源调度（worker concurrency）
- 更强的 acceptance suites（e2e、security）

### v1.6（OSS Adopt 阶段）
- Kimaki/OpenSwarm/Lobster 的“实现替换式接入”
- 更强的多入口、多租户与权限模型（如需要）

---

## 12. Appendix：你团队落地建议（组织与流程）

- 每个角色都可以由“人 + agent”混合承担：你员工负责最终决策与验收，agent 提供产出草案/patch/测试。
- 用 **“验收套件 + Artifact Pack”** 约束质量，避免依赖某个 agent 的稳定性。
- 用 **step-level ownership** 分配工作：每步指定 owner（人），agent 作为加速器。

---

> 本文是“Coding Team First”的更新版设计：先把团队交付跑起来，再谈接入更多 OSS 与更多助手类型。
