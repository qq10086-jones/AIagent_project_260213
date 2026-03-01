# Coding Agent 系统设计文档 v2（基于 OpenClaw：可落地工程版）
> 目标：用 **宏观拆解 + 微观工具化编辑 + Docker 沙盒 + 分层 QA** 的方式，把自然语言需求稳定转为可执行、可回滚、可评测的代码工程。  
> 核心原则：**确定性编排器（Orchestrator）掌控执行权**；Agent 只产出“计划/补丁/请求”，不直接改文件、不直接跑危险命令。

---

## 0. 设计目标与非目标

### 0.1 设计目标（必须）
- **可控**：每一步有状态、有产物、有回滚点，失败可定位、可复现。
- **低上下文成本**：大文件不进上下文，依赖工具定位与增量补丁。
- **安全**：最小权限、命令白名单、资源限额，避免破坏宿主机/泄露敏感信息。
- **质量闭环**：分层验证（静态/单测/验收），失败分类，支持回放评测与持续改进。
- **可扩展**：支持多语言/多框架，支持接入 Aider/SWE-agent 的编辑与检索思想。

### 0.2 非目标（暂不做）
- 自动把系统训练成 RL agent（先把数据闭环跑起来再谈学习）。
- 端到端全自动交付复杂产品（先保证稳定、可回滚、可诊断）。

---

## 1. 总体架构

### 1.1 两层 + 一中枢
- **宏观控制层（Macro）**：PM / Architect / Project Manager（规划与拆解）。
- **微观执行层（Micro）**：Coder / QA（按 Task 契约进行增量编辑与验证）。
- **确定性中枢（Orchestrator）**：状态机 + 产物记录 + 执行权限控制 + 回滚机制。

> 关键：Orchestrator 是“程序”，不是 Agent。Agent 永远不直接写盘或执行未经许可的命令。

### 1.2 数据流（高层）
1) 用户输入（自然语言/粗略设计）
2) PM -> PRD（结构化）
3) Architect -> 技术方案 + 目录树 + 接口/数据模型
4) Project Manager -> TaskList（严格 schema + DoD）
5) Orchestrator 循环执行 Task：
   - 读文件/检索（工具）
   - Coder 输出计划 + patch
   - Orchestrator dry-run 应用 patch -> 提交版本点
   - QA 分层验证 -> 通过则 Done，否则回传结构化失败并重试（有熔断）

---

## 2. Orchestrator（确定性编排器）

### 2.1 核心职责
- **状态机**：管理 run_id / task_id / 尝试次数 / 当前阶段 / 熔断。
- **产物与审计**：保存 PRD/架构/Task/patch/日志/测试结果；支持回放。
- **执行权限**：统一执行“白名单命令”、统一应用补丁、统一写盘。
- **版本控制与回滚**：每次成功应用 patch 后进行提交（git commit 或等价快照）。
- **资源限额**：超时/CPU/内存/磁盘上限；避免死循环烧算力。

### 2.2 状态机（建议）
- `INIT`
- `PRD_READY`
- `ARCH_READY`
- `TASKS_READY`
- `TASK_IN_PROGRESS`
- `PATCH_APPLIED`
- `QA_RUNNING`
- `TASK_DONE`
- `TASK_FAILED`（可重试）
- `ABORTED`（熔断/人工介入）
- `DONE`

每个 Task 内部建议有子状态：
- `READ_CONTEXT` -> `PLAN` -> `PATCH_PROPOSED` -> `PATCH_DRYRUN` -> `PATCH_APPLY` -> `QA` -> `DONE/FAIL`

### 2.3 产物目录规范（强制）
在 workspace repo 根目录下建立：
```
artifacts/
  prd.md
  arch.json
  tasks.json
  runs/
    <run_id>/
      state.json
      timeline.md
      task_<id>/
        attempt_01/
          plan.json
          patch.diff
          patch_apply.json
          qa_step_01_static.log
          qa_step_02_tests.log
          qa_step_03_acceptance.log
          verdict.json
        attempt_02/ ...
```

`state.json`（示例字段）：
- `run_id`
- `project_id`
- `current_task_id`
- `phase`
- `attempts_by_task`
- `last_commit_hash`
- `policy_version`
- `tool_versions`

---

## 3. Macro Pipeline（宏观流水线）

### 3.1 PM Agent -> PRD（结构化）
**输入**：用户描述/粗文档  
**输出**：`artifacts/prd.md`（必须含以下章节）

PRD 必须包含：
- 项目目标（1 段）
- 功能列表（P0/P1/P2）
- 约束（禁止修改核心算法模块等）
- 非功能需求（性能/安全/可维护性）
- 验收标准（高层）

### 3.2 Architect Agent -> 技术方案
**输入**：PRD  
**输出**：`artifacts/arch.json`（结构化）

arch.json 推荐字段：
- `tech_stack`：语言/框架/工具链/测试框架
- `directory_tree`：初始目录结构
- `api_specs`：接口定义（REST/CLI/内部模块 API）
- `data_models`：核心数据结构（可选）
- `build_run_commands`：建议的构建/运行/测试命令（进入白名单候选）

### 3.3 Project Manager Agent -> TaskList（严格契约）
**输入**：PRD + arch.json  
**输出**：`artifacts/tasks.json`

> 目标：把“写一个 XXX”变成“可验证的动作 + 清晰 DoD”。

---

## 4. Task Schema（关键：DoD 驱动）

### 4.1 tasks.json Schema（建议）
每个 Task 必须包含：

- `id`: string（如 "003"）
- `title`: string
- `goal`: string（一句话目标）
- `type`: enum（create_file / modify_file / refactor / add_test / wiring / docs）
- `inputs`:  
  - `files_required`: string[]（必须读取的文件）
  - `context_queries`: string[]（允许工具检索的关键词）
- `expected_changes`:
  - `files_touched`: string[]（允许修改/新增的文件）
  - `interfaces_affected`: string[]（模块/API）
- `constraints`:
  - `forbidden_paths`: string[]（例如 "core_algo/**"）
  - `forbidden_actions`: string[]（例如 "add new heavy deps"）
- `acceptance_tests`（DoD）:
  - `static_checks`: string[]（例如 "ruff check .", "black --check ."）
  - `unit_tests`: string[]（例如 "pytest -q" 或指定路径）
  - `smoke_tests`: string[]（例如 "python -m app.cli --help"）
  - `expected_signals`: string[]（输出/文件/接口行为特征）
- `rollback_plan`:
  - `strategy`: enum（git_reset / snapshot_restore）
- `max_retries`: int（默认 3）

### 4.2 Task 粒度准则
- 一个 Task 只做一种动作：**新增一个文件** 或 **修改一个模块** 或 **补一类测试**。
- DoD 必须可执行可验证（命令 + 信号）。
- 若 Task 需要修改多个文件，必须明确列出 `files_touched` 与理由；否则拆分。

---

## 5. Micro Execution（微观执行层）

### 5.1 工具库（ACI：Agent-Computer Interface）
> 工具由 Orchestrator 提供，集成成熟开源工具以确保鲁棒性。

必备工具（最低集）：
- **检索与定位**：
  - `list_directory(path)`：基础目录查看。
  - `fd_find(pattern)`：使用 `fd` 快速定位文件。
  - `rg_search(query)`：使用 `ripgrep` 进行全文高性能检索。
  - `read_file_lines(path, start, end)`：带缓存的分段读取。
  - `symbol_lookup(name)`：利用 `ctags` 或 `tree-sitter` 进行符号分析（如 Aider 的 repo-map 思想）。
- **编辑与应用**：
  - `apply_edit_block(edit_request)`：**核心修改入口**（见下文协议）。

### 5.2 “计划-执行分离”协议（强制）
Coder Agent 输出必须严格遵循 Aider 的 `Search/Replace` 模式。

---

## 6. 编辑协议（鲁棒性核心：Search/Replace Blocks）

### 6.1 协议格式
放弃对行号敏感的 `Unified Diff`，采用 Aider 风格的块匹配：

```python
<<<<<<< SEARCH
def old_function():
    print("hello")
=======
def new_function():
    print("hello world")
>>>>>>> REPLACE
```

### 6.2 Orchestrator 处理逻辑
- **模糊匹配**：即使 `SEARCH` 块中的缩进或微小字符差异，Orchestrator 应具备容错匹配能力。
- **原子性应用**：Orchestrator 负责将多个 Block 应用到文件。若任何一个 Block 匹配失败，则整组修改作废并回传失败上下文（提供当前文件的实际内容片段）。
- **自动提交**：修改成功后，Orchestrator 自动执行 `git commit -m "coding_agent: <task_id>"`。

---

## 7. QA Pipeline（基于成熟框架）

### 7.1 分层验证
1) **Static (pre-commit)**：集成 `pre-commit` 框架，统一运行 `ruff`, `black`, `eslint`, `mypy` 等。Agent 只需要请求执行 `pre-commit run`。
2) **Tests (pytest/jest)**：运行项目原生的测试套件。
3) **Acceptance**：根据 Task 定义的 smoke 命令进行最终验证。

### 7.2 QA 输出（结构化）
每轮 QA 产出 `verdict.json`：
- `status`: pass/fail
- `failed_stage`: static/tests/acceptance
- `error_category`: lint_error / import_error / test_fail / behavior_mismatch / patch_apply_fail / env_fail
- `top_errors`: string[]（摘要）
- `full_logs`: 指向 log 文件路径

### 7.3 反馈给 Coder 的格式
Orchestrator 将失败信息打包给 Coder：
- 失败阶段与类别
- 最关键的 20~50 行日志片段（避免塞满上下文）
- 指定“建议下一步读取的文件/函数”（可由 QA 辅助生成）

---

## 8. 安全模型（Docker ≠ 安全：必须最小权限）

### 8.1 Workspace 容器原则（最低配置）
- 非特权容器（no `--privileged`）
- 禁止挂载敏感宿主目录
- 禁止暴露 Docker socket（严禁 `/var/run/docker.sock`）
- 默认网络策略：可选  
  - MVP 可允许出网仅用于依赖安装（建议通过镜像缓存/代理控制）
- 资源限额（必须）：CPU、内存、单命令超时、磁盘配额

### 8.2 命令白名单（强制）
Orchestrator 只允许执行预先声明的命令集合，例如：
- 构建：`python -m pip install -r requirements.txt`
- 静态：`python -m compileall app`
- 测试：`pytest -q`
- 运行：`python -m app ...`（限定入口）

禁止：
- 任意 `curl|bash`
- 任意访问 repo 外路径
- 危险 shell 命令（rm/mv 到 repo 外等）

### 8.3 路径沙盒
所有读写路径必须在 repo root 内；超出直接拒绝并记录。

---

## 9. 失败处理与熔断（止损机制）

### 9.1 单 Task 熔断
- 默认 `max_retries = 3`
- 连续失败 -> `TASK_FAILED`，升级人工介入或调整 Macro 拆解

### 9.2 失败分类与处置策略（建议）
- `patch_apply_fail`：要求重新读取目标文件附近行号再生成 patch
- `lint_error/import_error`：快速修复，禁止引入大重构
- `test_fail`：定位失败断言，按 DoD 修逻辑或补测试
- `behavior_mismatch`：回到 PRD/Task 检查 DoD 是否定义不清或拆错

---

## 10. 可观测性与回放评测（让系统“越用越稳”的前提）

### 10.1 Timeline 记录（建议）
`timeline.md` 每步追加：
- 时间戳
- task id / attempt
- patch 摘要（文件/行数）
- QA 结果（pass/fail + 阶段）
- 错误摘要

### 10.2 指标（MVP 也要记）
- task 通过率、平均尝试次数
- 失败类别分布
- 平均 patch 大小
- 平均 QA 耗时

---

## 11. MVP 落地路线（建议）

### 11.1 MVP-1（先跑通闭环）
- Orchestrator：状态机 + artifacts + patch 应用 + git commit
- Macro：输出 PRD/arch/tasks
- Micro：Coder 输出 diff；QA 跑 tests + 1 个 smoke
- 安全：容器最小权限 + 白名单命令

### 11.2 MVP-2（提高稳定性）
- static stage（lint/format/typecheck）
- patch dry-run + 结构化失败回传
- symbol_lookup / find_references

### 11.3 MVP-3（可迭代）
- 指标与失败分类
- regression suite（小规模任务集回放）
- 多 tech_stack 的 policy 配置

---

## 12. 关键配置（Policy）

建议引入 `policy.yaml`（由 Orchestrator 读取）：
- `allowed_commands`
- `forbidden_paths`
- `max_patch_lines`
- `max_retries_default`
- `resource_limits`
- `tool_versions`

每次 run 记录 `policy_version`，确保可回放一致性。

---

## 13. 附录：示例 Task（片段）

```json
{
  "id": "003",
  "title": "Add SQLite connection helper",
  "goal": "Create db.py with a safe singleton connection factory for SQLite",
  "type": "create_file",
  "inputs": {
    "files_required": ["README.md", "app/config.py"],
    "context_queries": ["sqlite", "connect", "db path"]
  },
  "expected_changes": {
    "files_touched": ["app/db.py", "app/__init__.py"],
    "interfaces_affected": ["app.db:get_conn"]
  },
  "constraints": {
    "forbidden_paths": ["core_algo/**"],
    "forbidden_actions": ["add heavy new dependency"]
  },
  "acceptance_tests": {
    "static_checks": ["python -m compileall app"],
    "unit_tests": ["pytest -q"],
    "smoke_tests": ["python -c \"from app.db import get_conn; print(get_conn())\""],
    "expected_signals": ["no exception", "returns sqlite3.Connection"]
  },
  "rollback_plan": {
    "strategy": "git_reset"
  },
  "max_retries": 3
}
```

---

## 14. 总结（本版的“硬标准”）
- **没有 Orchestrator 状态机与产物目录 -> 不准开始写代码**
- **没有 Task DoD -> 不准进入微观执行**
- **没有 patch dry-run + 原子应用 + 回滚点 -> 不准落盘**
- **没有命令白名单 + 路径沙盒 + 资源限额 -> 不准在宿主机旁跑**

> v2 的重点不是“角色更全”，而是把系统从“概念流程”升级为“确定性可控的工程流水线”。
