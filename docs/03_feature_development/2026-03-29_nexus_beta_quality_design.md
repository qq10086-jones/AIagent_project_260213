# Nexus Beta Quality - 设计方案

**版本**: v1.1
**日期**: 2026-03-29
**角色**: PM + QA 联合评审
**状态**: 待确认

---

## 问题陈述

Nexus 目前已经能在 5 分钟内产出结构完整、文档较全的代码，fidelity 分类稳定达到 `demo_usable`。但从内测用户视角看，当前体验仍有 3 个核心断点：

1. **拿到代码仍不一定能跑**
   - `impl/be_changes/server.js` 可能依赖 express/cors，但没有 `package.json`
   - FE 和 BE 虽然都能生成，但没有稳定的装配策略，用户不知道如何一键启动
2. **QA 结论缺少执行证据**
   - 现有 `qa_verify` 主要读文件做静态判断
   - “通过 QA”不等于“服务真的能启动、接口真的能返回”
3. **执行过程对用户是黑盒**
   - Discord 用户在 5 分钟内几乎看不到进展，只能等待最终结果

次要问题：

4. **无法在已有产物上稳定增量修改**
5. **release_pack 会误判技术栈**
   - 例如 release_notes 写成 “Python-based implementation”，实际产物却是 Node.js

---

## 产品目标

本期目标不是“多产出几个文件”，而是把 Beta 体验推进到：

**用户在 3 分钟内理解如何启动，并在 5 分钟内大概率跑起产物。**

对应验收标准如下：

| 目标 | 验收标准 |
|---|---|
| 用户拿到产物可直接运行 | 执行 `cd impl/be_changes && npm install && node server.js` 能启动服务 |
| 服务存活有真实证据 | smoke_test 至少能证明服务监听成功，且 `GET /` 返回 200 或 HTML |
| API 质量有真实证据 | 若识别出主 API，则 smoke_test 记录实际 HTTP 状态码和响应样例 |
| QA 不再靠猜测 | `qa_report.json` 引用 `smoke/smoke_result.json` 的真实结果 |
| Discord 用户看到实时进度 | 每个步骤启动时收到通知，workflow 完成时收到结果摘要 |

---

## 非目标

- 不做真实云端 deploy
- 不做增量修改工作流
- 不做大规模 prompt 重构
- 不改变整体 7 步骤流水线思路，只做必要插入和约束增强

---

## 总体策略

本期仍按 3 个 Track 推进，但先修正执行口径：

```
Track 1: 可运行产物（P0）        -> impl + assembly + release
Track 2: 真实烟雾测试（P1）      -> smoke_test + qa_verify
Track 3: 进度可见性（P1）        -> Discord 通知
```

另外增加一个前置原则：

**产物事实源优先级**

1. `impl/be_changes/package.json`
2. `impl/be_changes/server.js`
3. `smoke/smoke_result.json`
4. `impl/be_notes.md`

说明：
- `be_notes.md` 只能作为辅助说明，不能作为 release_pack 推断技术栈和运行命令的唯一依据
- 任何用户可见说明，都应优先从真实产物反推

---

## Track 1：可运行产物（P0）

### 核心结论

当前最大的结构性问题不是“少了一个 prompt”，而是：

**FE 和 BE 的产物目录、运行目录、最终交付目录没有统一。**

如果继续让 FE 产出到 `impl/fe_changes/public/`，而 BE 从 `impl/be_changes/public/` 提供静态文件，那么最终启动链路一定不稳定。

因此本期必须明确采用一个装配策略。

### 推荐方案：显式装配到 BE 目录

保留 FE 独立生成目录，但在发布前把 FE 静态产物装配到：

`impl/be_changes/public/`

这样最终用户只需要进入一个目录运行：

```bash
cd impl/be_changes
npm install
node server.js
```

### 1-A：impl_be 强制产出最小可运行 Node 包

**目标**

- `impl_be` 必须产出 `server.js`
- 同时必须产出最小有效 `package.json`
- `server.js` 必须支持 `process.env.PORT`

**改动位置**

- `configs/prompt_scripts/registry.json`
- `orchestrator/configs/prompt_scripts/registry.json`
  - 在 `backend.impl.v2` 中追加：
  ```
  You MUST also write impl/be_changes/package.json with name, version, main entry, and all runtime dependencies detected in server.js. If server.js uses express, include "express". If it uses cors, include "cors". server.js MUST listen on process.env.PORT when provided. Never omit package.json.
  ```
- `orchestrator/src/domain/workflow_state.js`
  - impl_be `required_artifacts` 增加 `"impl/be_changes/package.json"`
- 如存在能力契约文件，保持同步

**验收**

- `impl/be_changes/package.json` 存在
- `cd impl/be_changes && npm install` 成功

### 1-B：FE/BE 装配策略明确化

**设计原则**

- FE 仍可由前端步骤单独生成
- 但最终运行形态必须收敛到单目录交付
- 发布时 FE 静态文件必须位于 `impl/be_changes/public/`

**建议做法**

前端步骤继续写：

`impl/fe_changes/public/`

然后新增或合并到 release/assembly 阶段，把以下文件复制到：

`impl/be_changes/public/`

包括：
- `index.html`
- `app.js`
- `styles.css`

**为什么不建议直接改成 FE 写入 be_changes**

- 会弱化 FE/BE 步骤边界
- 后续要做增量修改或独立 FE 评估时更难拆解
- 显式装配更符合流水线思维，也更易 debug

**改动位置**

- `backend.impl.v2`
  - 增加：
  ```
  Configure server.js to serve static files from a local 'public/' subdirectory using express.static. The root GET / must return public/index.html.
  ```
- `frontend.impl.v2`
  - 保持前端产出到 `impl/fe_changes/public/`
  - 同时要求 API 使用同源相对路径
- `workflow_state.js`
  - impl_fe `required_artifacts` 使用 `impl/fe_changes/public/index.html` 等路径
- 发布阶段新增装配要求
  - 将 `impl/fe_changes/public/*` 复制到 `impl/be_changes/public/`

**验收**

- `node server.js` 启动后，`GET /` 返回 HTML
- 页面中的 API 请求使用相对路径
- 用户无需额外启动 FE 服务

### 1-C：release_pack 从真实产物生成运行说明

**核心修正**

`release_pack` 不能只读 `impl/be_notes.md`。

它应按以下顺序读取事实源：

1. `impl/be_changes/package.json`
2. `impl/be_changes/server.js`
3. `smoke/smoke_result.json`（如果存在）
4. `impl/be_notes.md`

**输出要求**

- `release/README.md`
- `release/start.sh`
- 如有 Windows 用户场景，可后续补 `release/start.ps1`，但本期不是必须

**README 最少要回答 3 个问题**

1. 进入哪个目录运行
2. 安装命令是什么
3. 启动后访问哪个 URL

**验收**

- `release/README.md` 的技术栈、命令、端口与真实产物一致
- 不能出现与产物矛盾的 “Python-based” 之类描述

---

## Track 2：真实烟雾测试（P1）

### 核心问题

当前 QA 最大的问题不是“不够智能”，而是**没有执行证据**。

因此本期要新增 `smoke_test`，但测试目标应分层，而不是把“猜中主业务接口”作为唯一成功条件。

### 步骤顺序

```
0. pm_spec
1. arch_design
2. impl_be
3. impl_fe
4. smoke_test
5. qa_verify
6. release_pack
7. deploy_preview
```

### 分层验证模型

#### L1：服务存活验证

目标：
- 依赖安装成功
- 服务成功启动
- `GET /` 返回 200、HTML 或至少有可解析响应

这是最低层、最重要的验证。

#### L2：主 API 验证

目标：
- 若能可靠识别主 API 端点，则执行真实 curl
- 记录状态码和响应片段

说明：
- 若主 API 无法可靠识别，不应因为这个原因把整个 smoke_test 判成完全失败
- 否则会出现“服务明明跑起来了，但因为猜错端点而误报失败”的情况

### smoke_result.json 建议结构

```json
{
  "install_ok": true,
  "server_started": true,
  "root_check": {
    "status": 200,
    "content_type": "text/html",
    "passed": true
  },
  "api_check": {
    "endpoint": "/api/books",
    "status": 200,
    "response_sample": "{\"books\":[]}",
    "passed": true,
    "skipped": false
  },
  "errors": [],
  "verdict": "pass",
  "evidence_level": "l1_only | l1_l2"
}
```

### 技术约束

- `server.js` 必须支持 `process.env.PORT`
- smoke_test 使用固定测试端口，例如 `13099`
- 超时应受控，失败后必须清理进程

### 失败策略

建议明确区分：

- **L1 失败**：服务未启动或根路径不可访问
  - 记为 `failed`
  - workflow 继续，但最终 go/no-go 倾向 `NO_GO`
- **L2 失败**：主 API 失败，但 L1 成功
  - 记为 `partial_failure`
  - workflow 继续
  - QA 必须明确引用该失败

这比单一 `pass/fail` 更利于用户理解问题层级。

### QA 接入要求

`qa_verify` 不是重复跑测试，而是消费 `smoke_result.json` 并做解释：

- 直接引用真实状态码
- 区分 L1 和 L2
- 不得伪造证据

---

## Track 3：Discord 进度可见性（P1）

### 核心目标

本期不要一上来做“全量状态流广播”，先解决用户最关键的不确定感：

**用户要知道 workflow 已启动、当前运行到哪一步、最后结果是什么。**

### 验收口径统一

本期统一采用：

- 每个步骤发 `step_started`
- workflow 完成时发 1 条结果摘要

暂不把 `step_completed` 设为本期强制验收项。

原因：

- 能显著降低消息噪音
- 用户最关心“现在在哪一步”，不是每一步结束都要被提醒
- 工程实现更简单，Beta 更稳

### 3-A：先核查 channelId 传递链路

重点确认：

- Discord 入口触发 workflow 时，`channelId` 是否被写入 `workflowRunToContext`
- `/vnext/dispatch` 是否透传了该上下文

### 3-B：步骤消息格式

推荐格式：

```text
[Nexus] 步骤 2/7：架构设计
状态：运行中
已用时：45s
下一步：后端实现
```

设计原则：

- 不追求花哨
- 重点降低等待中的不确定感
- “当前步骤 + 下一步预期”比单纯状态更有价值

### 3-C：最终结果消息

workflow 完成时，Discord 应附带简短运行指令摘要，例如：

- 进入目录：`impl/be_changes`
- 安装依赖：`npm install`
- 启动服务：`node server.js`
- 打开：`http://localhost:<PORT>`

不需要整份 README 全贴，只要用户能立即行动即可。

---

## 独立修复：release_pack 技术栈误判

这是一个可立即处理的小修复，但修复口径要调整。

### 错误修复思路

原思路：
- “让 release_pack 读取 `impl/be_notes.md`”

修正后：
- “让 release_pack 优先读取真实产物，仅把 `impl/be_notes.md` 作为辅助说明”

### 建议 prompt 要点

可在 `release.pack.v1` 中加入类似约束：

```text
Before writing release artifacts, inspect the actual backend deliverables first:
1. impl/be_changes/package.json
2. impl/be_changes/server.js
3. smoke/smoke_result.json if present
Use impl/be_notes.md only as supporting context. Never infer the tech stack from the goal description.
```

---

## 执行顺序

```text
独立修复（F-1）
    ->
Track 1-A
    ->
Track 1-B
    ->
Track 1-C
    ->
Track 2
    || Track 3 可并行
    ->
集成验证
```

原因：

- Track 2 依赖 Track 1 提供稳定运行入口
- Track 3 与运行产物弱耦合，可并行推进

---

## 成功验收标准（整体）

连跑 3 次 canary，不同 goal，至少满足：

| 检查项 | 标准 |
|---|---|
| `impl/be_changes/package.json` 存在 | 每次 |
| `npm install && node server.js` 可启动 | 每次 |
| `smoke/smoke_result.json` 存在 | 每次 |
| L1 服务存活验证通过 | 每次 |
| L2 主 API 验证通过 | 至少 2/3 |
| `qa_report.json` 引用真实状态码 | 每次 |
| `release/README.md` 与真实产物一致 | 每次 |
| Discord 收到步骤开始通知和最终结果 | 验证 1 次 |
| go_no_go verdict = GO | 至少 2/3 |

---

## 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| FE 与 BE 目录再次分叉 | 中 | 明确“最终运行目录只能是 `impl/be_changes/`”，并在 release/assembly 阶段强制装配 |
| smoke_test 因主 API 识别不稳误报失败 | 高 | 改为 L1/L2 分层验证，先保证服务存活证据 |
| `server.js` 不支持外部端口注入 | 中 | 在 `backend.impl.v2` 明确要求支持 `process.env.PORT` |
| release_pack 继续误判技术栈 | 中 | 以 package.json/server.js 为第一事实源 |
| Discord 通知过多造成打扰 | 低 | 本期只强制 `step_started` + 最终结果 |

---

## 总工期估算

整体仍可控制在 **约 5-7 天**，但前提是：

- 先冻结装配策略
- 不在本期扩展到“增量修改”
- Discord 只做最小可用通知闭环
