# Superpowers Integration — 任务清单

**关联设计文档**: `2026-03-28_superpowers_integration_design.md`
**日期**: 2026-03-28
**执行顺序**: Track C → Track A → Track B → 集成验证

---

## Track C：模型分级（预计 2–3 小时）

- [ ] **C-1** 确认 `worker.js` 里 `model_override` 字段是否已被消费并传给 opencode_adapter
  - 若已有：记录字段路径，进入 C-2
  - 若缺失：在 worker.js payload 解析处补 `model_override` 透传

- [ ] **C-2** 在 `workflow_step_builder.js` 的 `runtimeByStep` 附近加 `modelByStep` 映射
  - 第一阶段只分级低风险步骤：`release_pack`, `deploy_preview` → `dashscope/qwen3-coder-plus`
  - impl_be / impl_fe 暂不切，等 Track A 验证后再决定

- [ ] **C-3** 跑一次 canary，确认 release_pack / deploy_preview 步骤 result_json 里 `model_used` 字段已变更
  - 验收：`model_used` = `qwen3-coder-plus` 且步骤 status = succeeded

---

## Track A：安装 superpowers 插件（预计 4–6 小时）

- [ ] **A-1** 确认当前 worker-coder 容器内 OpenCode 版本，查看是否支持 `plugins` 字段
  - 命令：`docker exec nexus-worker-coder opencode --version`
  - 查看 opencode.json.tpl 当前结构，确认 `plugins` key 的正确写法

- [ ] **A-2** 决定 superpowers 的引入方式
  - 选项 1（推荐）：`git submodule add https://github.com/obra/superpowers vendor/superpowers`，COPY 进镜像
  - 选项 2：Dockerfile 里 `RUN git clone --depth 1`（网络不稳定时有风险）
  - 若选项 1：在 `worker-coder/Dockerfile` 里加 `COPY ../vendor/superpowers /app/superpowers`

- [ ] **A-3** 在 `opencode.json.tpl` 里注册插件
  - 加 `"plugins": ["/app/superpowers"]`（或 opencode 实际支持的插件语法）
  - 本地验证：`envsubst` 渲染后检查生成的 opencode.json 格式正确

- [ ] **A-4** 更新 `configs/prompt_scripts/registry.json`（根目录）的 `backend.impl.v2` system_prompt
  - 末尾追加技能激活指令（见设计文档 Track A 段落）
  - 同步更新 `orchestrator/configs/prompt_scripts/registry.json`（两个文件必须一致）

- [ ] **A-5** 更新 `frontend.impl.v2` system_prompt（同上，两个文件各一份）

- [ ] **A-6** 重建 worker-coder 镜像，重启容器
  - `docker compose -f infra/docker-compose.yml up -d --build worker-coder`

- [ ] **A-7** 跑一次 canary，观察 impl_be 步骤日志，确认 superpowers 技能被引用
  - 验收标准：
    - impl_be / impl_fe status = succeeded
    - `product_fidelity_report.json` classification = demo_usable
    - `placeholder_free` = pass
  - 若 MiniMax 对技能激活无响应（无 TDD 行为）：记录结论，不 block，进入 Track B

---

## Track B：arch_design 微任务列表注入（预计 6–8 小时）

**前置条件**：Track A canary 结果稳定后开始

- [ ] **B-1** 更新 `architect.system_spec.v2` system_prompt，要求 workplan.md 产出结构化任务列表
  - 在两个 registry.json 里同步更新
  - 在 `workflow_state.js` 的 `arch_design` STEP_CONTRACTS instructions 里同步加说明

- [ ] **B-2** 跑一次 canary，只看 arch_design 步骤产出的 `plan/workplan.md`
  - 验收：workplan.md 包含 `## BE Tasks` 和 `## FE Tasks` 两节，每条带 `verify:` 字段
  - 若格式不符：调整 prompt 后重试，不超过 2 次
  - 不符合则记录，考虑在 B-4 里做 graceful fallback

- [ ] **B-3** 在 `workflow_step_builder.js` 里加 workplan 读取 + 注入逻辑
  - 见设计文档 Track B 代码段
  - 必须有 graceful fallback：workplan 不存在或解析失败时，原有 prompt 不受影响

- [ ] **B-4** 更新 `orchestrator/src/domain/workflow_state.js` 的 impl_be / impl_fe instructions
  - 加一条：`"Execute tasks from plan/workplan.md in order. After each task, self-check against its verify condition before proceeding."`

- [ ] **B-5** 跑一次完整 canary，观察 impl_be prompt 是否包含任务列表
  - 检查：`artifacts/runs/<run_id>/task_<id>/prompt_contract_*.json` 里 task_prompt 字段
  - 验收：task_prompt 包含注入的 `[Task List from plan/workplan.md]` 段落
  - 最终验收：`go_no_go_result.json` verdict = GO，classification = demo_usable

---

## 集成验证（所有 Track 完成后）

- [ ] **V-1** 连跑 3 次 canary（不同 goal），确认稳定性
  - 验收：3/3 verdict = GO，3/3 classification = demo_usable

- [ ] **V-2** 跑 `npm run metrics:compare_baseline`
  - 验收：`has_comparable_data: true`（需要足够的 fresh run 数据）
  - 记录 fidelity_pass_rate delta

- [ ] **V-3** 跑完整测试套件
  - `npm --prefix orchestrator test` → 全部 PASS
  - `npm --prefix worker-coder test` → 全部 PASS

- [ ] **V-4** 更新 MEMORY.md 中测试状态和运行配置段落

---

## 快速参考：需要同步更新的重复文件

每次改 prompt_scripts/registry.json 时，必须同时改两处：

| 用途 | 路径 |
|---|---|
| Docker 生产环境 | `configs/prompt_scripts/registry.json`（根目录） |
| 测试套件 | `orchestrator/configs/prompt_scripts/registry.json` |

每次改 capability_registry.json 时同理：

| 用途 | 路径 |
|---|---|
| Docker 生产环境 | `configs/registry/capability_registry.json`（根目录） |
| 测试套件 | `orchestrator/configs/registry/capability_registry.json` |

改完重启 orchestrator：`docker restart nexus-orchestrator`
