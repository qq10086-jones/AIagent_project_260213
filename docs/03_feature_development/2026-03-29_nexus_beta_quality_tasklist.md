# Nexus Beta Quality - 任务清单

**关联设计文档**: `2026-03-29_nexus_beta_quality_design.md`
**版本**: v1.1
**日期**: 2026-03-29
**执行顺序**: F-1 -> Track 1 -> Track 2 与 Track 3 并行 -> 集成验证

---

## 本期执行原则

- 最终运行目录统一为 `impl/be_changes/`
- FE 可以独立生成，但发布前必须装配到 `impl/be_changes/public/`
- release_pack 的事实源优先级为：
  - `impl/be_changes/package.json`
  - `impl/be_changes/server.js`
  - `smoke/smoke_result.json`
  - `impl/be_notes.md`
- smoke_test 采用 L1/L2 分层验证
- Discord 本期验收口径为：`step_started` + 最终结果消息

---

## 独立修复（预计 30 分钟，先做）

- [ ] **F-1** 修复 `release.pack.v1` 的事实源约束
  - 两份 `prompt_scripts/registry.json` 同步更新
  - 目标：release_pack 写说明前，优先读取 `impl/be_changes/package.json` 与 `impl/be_changes/server.js`
  - `impl/be_notes.md` 仅作辅助上下文
  - 验收：release/README 或 release_notes 中技术栈与真实产物一致

---

## Track 1：可运行产物（预计 2.5 天）

### 1-A：impl_be 强制产出最小可运行 Node 包（半天）

- [ ] **1-A-1** 更新 `backend.impl.v2` system_prompt（两份 registry）
  - 增加要求：
    - 必须写 `impl/be_changes/package.json`
    - 自动包含 `server.js` 中使用的运行时依赖
    - `server.js` 必须支持 `process.env.PORT`
- [ ] **1-A-2** 更新 `workflow_state.js`
  - 在 impl_be `required_artifacts` 中加入 `"impl/be_changes/package.json"`
- [ ] **1-A-3** 更新 `worker-coder/artifact_scaffold.js`
  - 为 `impl/be_changes/package.json` 增加最小有效 scaffold
- [ ] **1-A-4** canary 验证
  - 检查 `impl/be_changes/package.json` 存在
  - 执行 `cd impl/be_changes && npm install`
  - 验收：安装成功，无依赖缺失
- [ ] **1-A-5** 兜底增强
  - 如果模型仍漏写 `package.json`，则在 `workflow_step_builder.js` workplan 注入中补充显式任务

### 1-B：明确 FE/BE 装配策略（1 天）

- [ ] **1-B-1** 更新 `frontend.impl.v2` system_prompt
  - FE 继续写入 `impl/fe_changes/public/`
  - API 必须使用同源相对路径
- [ ] **1-B-2** 更新 `backend.impl.v2` system_prompt
  - 要求 `server.js` 从本地 `public/` 提供静态文件
  - `GET /` 返回 `public/index.html`
- [ ] **1-B-3** 更新 `workflow_state.js`
  - impl_fe `required_artifacts` 改为：
    - `"impl/fe_changes/public/index.html"`
    - `"impl/fe_changes/public/app.js"`
    - 如需要，再加 `"impl/fe_changes/public/styles.css"`
- [ ] **1-B-4** 更新 `worker-coder/artifact_scaffold.js`
  - FE scaffold 路径调整到 `impl/fe_changes/public/`
- [ ] **1-B-5** 检查 `ensureExpectedArtifacts`
  - 如存在旧路径假设，更新为 `impl/fe_changes/public/*`
- [ ] **1-B-6** 增加装配动作
  - 在 release 或单独 assembly 环节中，将 `impl/fe_changes/public/*` 复制到 `impl/be_changes/public/`
- [ ] **1-B-7** canary 验证
  - 启动 `cd impl/be_changes && node server.js`
  - `curl http://localhost:3000/` 返回 HTML
  - 页面 API 请求不包含硬编码 `localhost`

### 1-C：release_pack 产出可执行说明（半天）

- [ ] **1-C-1** 更新 `release.pack.v1` system_prompt
  - 先读真实产物，再生成 `release/README.md`
  - 额外产出 `release/start.sh`
- [ ] **1-C-2** 更新 `workflow_state.js`
  - release_pack `required_artifacts` 增加：
    - `"release/README.md"`
    - `"release/start.sh"`
- [ ] **1-C-3** 验证 README 的最小完整性
  - 必须包含：
    - 进入目录命令
    - 安装命令
    - 启动命令
    - 浏览器访问地址
- [ ] **1-C-4** 文案一致性检查
  - 禁止出现与 `package.json`、`server.js` 矛盾的技术栈描述

---

## Track 2：真实烟雾测试（预计 2 天）

### 2-A：确认 smoke_test 运行前提（半天）

- [ ] **2-A-1** 确认 `coding.execute` 的环境能力
  - 检查 worker-coder 是否具备：
    - Node.js
    - npm
    - curl
- [ ] **2-A-2** 确认测试端口策略
  - 使用固定端口 `13099`
  - 确认端口未与 orchestrator 或其他服务冲突
- [ ] **2-A-3** 确认后台进程清理方案
  - smoke_test 失败或超时时必须杀掉测试进程

### 2-B：实现 L1/L2 分层 smoke_test（1 天）

- [ ] **2-B-1** 在两份 `capability_registry.json` 中增加 `smoke_test`
  - 插入位置：`impl_fe` 后，`qa_verify` 前
  - tool 使用 `coding.execute`
- [ ] **2-B-2** 在 `workflow_state.js` 添加 `smoke_test` STEP_CONTRACT
  - `required_artifacts` 至少包含 `"smoke/smoke_result.json"`
  - 指令中明确：
    - 安装依赖
    - 用 `PORT=13099` 启动服务
    - 先做根路径检查
    - 再尝试主 API 检查
    - 清理进程
- [ ] **2-B-3** 在 `workflow_step_builder.js` 增加 payload 构建逻辑
  - L1：
    - `GET /`
  - L2：
    - 若能从产物中可靠识别主 API，再请求对应端点
    - 若无法识别，则标记 `api_check.skipped=true`
- [ ] **2-B-4** 在 `workflow_step_builder.js` 中写入 smoke 结果结构
  - 字段建议：
    - `install_ok`
    - `server_started`
    - `root_check`
    - `api_check`
    - `errors`
    - `verdict`
    - `evidence_level`
- [ ] **2-B-5** 失败策略接入 `workflow_engine.js`
  - L1 失败：步骤 `failed`，workflow 继续，最终倾向 `NO_GO`
  - L2 失败但 L1 成功：步骤 `partial_failure`
- [ ] **2-B-6** 更新 `qa_verify` system_prompt
  - 明确要求读取 `smoke/smoke_result.json`
  - 必须引用真实 HTTP 状态码
  - 必须区分 L1 与 L2
- [ ] **2-B-7** 新增 `orchestrator/contracts/smoke_result.schema.json`
  - 与上述结构一致
- [ ] **2-B-8** canary 验证
  - `smoke/smoke_result.json` 存在
  - 至少能稳定拿到 L1 证据
  - `qa_report.json` 引用了真实 smoke 结果

---

## Track 3：Discord 进度可见性（预计 1 天）

### 3-A：核查 channelId 传递链路（2 小时）

- [ ] **3-A-1** 阅读 `discord_message_handler.js`
  - 确认 Discord 消息触发 workflow 时，是否执行了 `workflowRunToContext.set(workflow_run_id, { channelId })`
- [ ] **3-A-2** 检查 `/vnext/dispatch` 入口
  - 确认 channelId 或等价上下文是否被透传
- [ ] **3-A-3** 用真实 Discord 消息触发一次 workflow
  - 验收：至少收到 1 条步骤开始通知

### 3-B：最小可用步骤通知（2 小时）

- [ ] **3-B-1** 阅读 `discord_gateway.js` 的 `handleWorkflowEvent`
  - 确认当前事件类型与消息模板
- [ ] **3-B-2** 统一本期消息口径
  - 仅发送 `step_started`
  - 不把 `step_completed` 作为本期强制项
- [ ] **3-B-3** 优化通知内容
  - 包含：
    - 当前步骤编号
    - 步骤名称
    - 当前状态
    - 已用时
    - 下一步预期

### 3-C：最终结果消息附带运行摘要（1 小时）

- [ ] **3-C-1** 在 workflow 完成消息中追加运行摘要
  - 读取 `release/README.md` 或从 release artifacts 中抽取：
    - 运行目录
    - `npm install`
    - `node server.js`
    - 浏览器访问地址
- [ ] **3-C-2** 控制长度
  - 结果消息应可直接复制执行，但不粘贴整份 README

---

## 集成验证（所有 Track 完成后）

- [ ] **V-1** 连跑 3 次 canary
  - 建议目标：
    - 图书馆系统
    - 电商购物车
    - 待办事项
  - 每次检查：
    - `impl/be_changes/package.json` 存在
    - `cd impl/be_changes && npm install && node server.js` 可启动
    - `smoke/smoke_result.json` 存在
    - L1 根路径检查通过
    - L2 主 API 检查至少 2/3 通过
    - `qa_report.json` 引用真实状态码
    - `release/README.md` 与真实产物一致

- [ ] **V-2** Discord 端到端验证
  - 从真实 Discord 发送一条 `/coder: <goal>`
  - 验收：
    - 至少收到每步开始通知
    - 最终结果消息包含运行摘要

- [ ] **V-3** 运行测试套件
  - `npm --prefix orchestrator test`
  - `npm --prefix worker-coder test`

- [ ] **V-4** 更新 `MEMORY.md`
  - 记录：
    - 事实源优先级
    - FE/BE 装配策略
    - smoke_test 的 L1/L2 设计
    - Discord 本期通知口径

---

## 快速参考：本期涉及的重复文件

每次改 `prompt_scripts/registry.json`，必须同步：

- `configs/prompt_scripts/registry.json`
- `orchestrator/configs/prompt_scripts/registry.json`

每次改 `capability_registry.json`，必须同步：

- `configs/registry/capability_registry.json`
- `orchestrator/configs/registry/capability_registry.json`

改完后按实际需要执行：

- `docker restart nexus-orchestrator`
- `docker compose -f infra/docker-compose.yml up -d --build worker-coder`

---

## 工作量汇总

| Track | 核心改动 | 预计工时 |
|---|---|---|
| F-1 | release_pack 事实源修正 | 30 分钟 |
| Track 1-A | package.json + PORT 约束 | 4 小时 |
| Track 1-B | FE/BE 装配策略 + public 路径 | 6 小时 |
| Track 1-C | README + start.sh | 2 小时 |
| Track 2 | smoke_test + schema + QA 接入 | 10 小时 |
| Track 3 | channelId 链路 + started 通知 + 结果摘要 | 4 小时 |
| 集成验证 | canary x3 + Discord E2E | 3 小时 |
| **合计** | | **约 5-7 天** |
