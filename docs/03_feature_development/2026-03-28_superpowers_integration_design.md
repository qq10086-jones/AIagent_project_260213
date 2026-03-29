# Superpowers Integration — 设计文档

**版本**: v1.0
**日期**: 2026-03-28
**状态**: 草稿，待确认

---

## 背景与问题陈述

当前 `coding_team_v0` 流水线在以下方面存在结构性短板：

1. **impl 步骤粒度过粗**：`impl_be` / `impl_fe` 将整个实现交给一次 LLM session，复杂任务容易产出占位代码或遗漏功能
2. **缺乏实现阶段内检**：QA 在流水线末尾才触发，发现问题时 impl 步骤已结束，只能报 fail，无法修
3. **无 TDD 约束**：impl 步骤不要求先写失败测试，qa_verify 验证的是文件存在性而非行为正确性
4. **模型一刀切**：设计判断型（pm/arch）和代码执行型（impl）使用同一个 MiniMax-M2.7，成本未优化

[obra/superpowers](https://github.com/obra/superpowers) 是一个 OpenCode 原生的 agentic 技能框架，提供 TDD、任务分解、自检、系统化调试等结构化工作流，与我们的 worker-coder 技术栈（OpenCode）直接兼容。

---

## 目标

| 目标 | 可测量标准 |
|---|---|
| impl 步骤减少占位代码 | `placeholder_free` 标准 pass rate ≥ 95% |
| impl 步骤内产出可运行代码 | `demo_usable` 分类率 ≥ 80%（当前首次有效测量为 1/1） |
| 降低 impl 步骤 token 成本 | impl 步骤切换至快模型后，单次运行 token 成本下降 ≥ 20% |
| qa_verify 产出真实证据 | `qa_evidence_not_scaffold_only` 持续 pass（已修复，保持） |

## 非目标

- 不引入 human-in-the-loop 审批门（P1-05 单独决策）
- 不改变流水线步骤数量或顺序
- 不替换 MiniMax-M2.7 作为设计型步骤的主力模型
- 不修改 orchestrator 层 DAG 调度逻辑

---

## 方案概述

三个独立 Track，可并行推进，互不阻塞。

```
Track A: 安装 superpowers 插件       → worker-coder 层
Track B: arch_design 产出微任务列表   → orchestrator prompt 层
Track C: 模型分级                    → workflow_step_builder 层
```

---

## Track A：安装 superpowers 为 OpenCode 插件

### 原理

superpowers 原生支持 OpenCode 插件接口（`.opencode/plugins/superpowers.js`）。worker-coder 容器内运行 OpenCode，注册插件后，在 system_prompt 里 mention 技能名即可激活对应工作流。

激活的技能（优先级排序）：

| 技能名 | 激活步骤 | 解决的问题 |
|---|---|---|
| `test-driven-development` | impl_be, impl_fe | 无 TDD 约束 |
| `verification-before-completion` | impl_be, impl_fe, qa_verify | 实现结束前无自检 |
| `systematic-debugging` | impl_be, impl_fe | 遇到错误乱猜不找根因 |
| `subagent-driven-development` | impl_be（可选） | 复杂 BE 任务可进一步分解 |

### 改动范围

**文件 1：`worker-coder/Dockerfile`**

```dockerfile
# 在 npm install 之后加
RUN git clone --depth 1 https://github.com/obra/superpowers.git /app/superpowers
```

**文件 2：`opencode.json.tpl`**（根目录）

```json
{
  "plugins": [
    "/app/superpowers"
  ]
}
```

**文件 3：`configs/prompt_scripts/registry.json`**（根目录 + orchestrator 目录各一份）

在 `backend.impl.v2` system_prompt 末尾追加：
```
Use test-driven-development: write a failing test for each function before implementing it.
Use verification-before-completion: run each implemented endpoint and verify output before moving on.
If blocked more than twice on the same issue, use systematic-debugging.
```

在 `frontend.impl.v2` system_prompt 末尾追加：
```
Use verification-before-completion: check each UI component renders correctly before proceeding.
Use test-driven-development where applicable.
```

### 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| superpowers 技能为 Claude 优化，MiniMax 激活效果差 | 中 | canary 验证后再决定是否 keep |
| 容器构建时 github clone 网络失败 | 中 | 备选：vendoring 到 repo 里 |
| 插件注册语法与当前 opencode 版本不兼容 | 低 | 查 opencode 版本，确认插件 API |

---

## Track B：arch_design 产出可执行微任务列表

### 原理

arch_design 已经产出 `plan/workplan.md`，但内容是叙述性文字。改为结构化 checklist 格式，让 impl 步骤按任务逐条执行，每完成一个 task 自检一次，而不是一次性生成所有文件。

### 改动范围

**文件 1：`configs/prompt_scripts/registry.json`** — `architect.system_spec.v2` system_prompt

在现有内容末尾追加：

```
In plan/workplan.md, emit a structured task list for BE and FE teams. Format exactly as:

## BE Tasks
- [ ] T-BE-1: <one-sentence description> | verify: <exact shell command or assertion>
- [ ] T-BE-2: ...

## FE Tasks
- [ ] T-FE-1: <one-sentence description> | verify: <exact assertion>

Rules:
- Each task must be completable in one function or one file section (2–5 minutes)
- verify field must be a concrete, runnable check — no "ensure it works" language
- No more than 8 tasks per team
- Tasks must be ordered by dependency (upstream first)
```

**文件 2：`orchestrator/src/domain/workflow_step_builder.js`**

在 impl_be / impl_fe 步骤的 prompt 构建逻辑里，读取 `plan/workplan.md` 并注入任务列表：

```js
if (["impl_be", "impl_fe"].includes(String(stepDef.id || ""))) {
  const workplanPath = path.resolve(workspaceRoot, artifactRoot, "plan/workplan.md");
  if (fs.existsSync(workplanPath)) {
    const workplan = fs.readFileSync(workplanPath, "utf8");
    const section = stepDef.id === "impl_be" ? "BE Tasks" : "FE Tasks";
    const match = workplan.match(new RegExp(`## ${section}([\\s\\S]*?)(?=## |$)`));
    if (match) {
      payload.task_prompt += `\n\n[Task List from plan/workplan.md — ${section}]\n${match[1].trim()}\n\nExecute tasks in order. After each task, self-check against its verify condition before proceeding.`;
    }
  }
}
```

### 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| arch_design 不遵循新格式，workplan 解析失败 | 中 | 注入逻辑做 graceful fallback（无任务列表时按原逻辑继续） |
| 任务列表过细导致 impl 步骤 token 超限 | 低 | 限制 ≤ 8 tasks，workplan 注入做截断保护 |

---

## Track C：模型分级

### 原理

设计判断型步骤（pm_spec、arch_design、qa_verify）需要强模型；代码执行型步骤（impl_be、impl_fe）更适合快模型，节省成本同时降低 latency。

### 改动范围

**文件：`orchestrator/src/domain/workflow_step_builder.js`**

在 `runtimeByStep` 附近加：

```js
const modelByStep = {
  impl_be: "dashscope/qwen3-coder-plus",
  impl_fe: "dashscope/qwen3-coder-plus",
  release_pack: "dashscope/qwen3-coder-plus",
  deploy_preview: "dashscope/qwen3-coder-plus",
};
```

在 payload 构建逻辑里：

```js
if (modelByStep[stepDef.id] && !payload.model_override) {
  payload.model_override = modelByStep[stepDef.id];
}
```

确认 `worker-coder/worker.js` 里 `model_override` 字段已被消费并传给 opencode_adapter（检查现有逻辑，若已有则直接复用）。

### 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| qwen3-coder-plus 在 impl 步骤质量明显下降 | 中 | 先只对 release_pack / deploy_preview 分级（低风险步骤），再看 impl 步骤数据 |
| model_override 字段未被 worker 消费 | 低 | 查 worker.js，若缺则补一行 |

---

## 依赖与执行顺序

```
Track C（最小改动，半天）
    ↓ 可独立验证
Track A（安装插件，一天）
    ↓ 插件生效后
Track B（微任务注入，两天）
    ↓ 最终集成验证
canary 3–5 次，metrics:compare_baseline
```

Track A 和 Track C 可并行开始，Track B 依赖 arch_design 格式稳定后再上。

---

## 成功验证方法

每个 Track 完成后跑一次 canary，检查：

```
curl -s "http://localhost:3000/workflow-runs/<run_id>"
cat artifacts/release/<run_id>/qa/product_fidelity_report.json
cat artifacts/release/<run_id>/qa/go_no_go_result.json
npm --prefix orchestrator run metrics:compare_baseline
```

整体达标标准：
- 连续 5 次 canary：verdict=GO，classification=demo_usable
- `fidelity_pass_rate` > 70%（替换 backfill baseline 后）
- impl 步骤无 `placeholder_free: false`

---

## 总工作量估算

| Track | 改动文件数 | 预计工时 |
|---|---|---|
| Track C（模型分级） | 1 | 2–3 小时 |
| Track A（superpowers 插件） | 4 | 4–6 小时 |
| Track B（微任务注入） | 2 | 6–8 小时 |
| 验证 + canary | — | 2–3 小时 |
| **合计** | | **约 2 天** |
