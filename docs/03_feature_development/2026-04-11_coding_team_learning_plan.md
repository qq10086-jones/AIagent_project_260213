# v3.5 Design Quality Intelligence — Coding Team Learning Plan

**Date**: 2026-04-11
**Author**: Architect/PM
**Status**: Draft
**Version**: v3.5
**Goal**: 让 coding team (LLM agents) 掌握通用资深项目开发经验，提升交付质量从 3/10 到 7/10+

---

## 1. 问题定义

当前 coding team 产出的系统"能跑但不能用"：
- 缺少删除功能（CRUD 不完整）
- 无操作反馈（保存后无视觉确认）
- 空状态无引导
- 前端只覆盖部分模块
- 无 loading/error 状态处理

**根因**：prompt 和 scaffold 中没有注入设计质量标准。LLM 本身具备这些知识，但我们的工作流没有要求它使用。

---

## 2. 核心策略：Atomic Task Decomposition with Scoped Context

### 用户提出的关键洞察

> "最小化单元任务指令输入，分多次输入和执行"

这在学术界已有成熟方法论：

| 方法 | 来源 | 核心思想 |
|------|------|---------|
| **Decomposed Prompting** | Khot et al. 2023 | 复杂 prompt 拆分为子 prompt，每个由专门 solver 处理 |
| **MetaGPT SOP** | Hong et al. 2023 | 每个角色只接收上一角色的结构化产出，不接收原始需求 |
| **ChatDev Chat Chains** | Qian et al. 2023 | 角色隔离——每个 agent 只看它需要的上下文 |
| **Aider Repo Map** | paul-gauthier/aider | 压缩索引 + 按需拉取详情 |
| **AlphaCodium** | Ridnik et al. 2024 | 多 pass：先粗后细，每个 pass 只关注失败点 |

### 我们的实现方式

**不是把所有设计知识塞进一个 prompt，而是**：

```
PM step     → 注入 [产品质量合约] → 输出含 UX 要求的 spec.md
                                      ↓ (编译后的产出，不是原始规则)
Arch step   → 读 spec.md → 输出含交互状态的 interfaces.md
                                      ↓
FE impl     → 读 interfaces.md → 只需实现已定义的交互
                                      ↓
QA verify   → 读 acceptance.json → 验证 UX 要素是否存在
```

每一层只接收**上一层的编译产出**（structured artifact），不接收原始规则库。设计知识通过 PM 层消化后逐层传递，每步 prompt 保持小体量。

---

## 3. 当前 Context Budget 分析

| 步骤 | 当前 prompt 大小 | 阈值 | 可用空间 |
|------|-----------------|------|---------|
| pm_spec | ~1.3K chars | 80K | **98% 空闲** |
| arch_design | ~1.5K chars | 100K | **99% 空闲** |
| impl_be | ~5K chars | 80K | **94% 空闲** |
| impl_fe | ~4.5K chars | 80K | **94% 空闲** |

**结论**：每步都有巨大的注入空间。但策略不是填满它，而是在 PM 层注入 ~2KB 的设计合约，让后续步骤通过 artifact handoff 自然继承。

---

## 4. 学习来源：开源项目参考

### 4.1 通用 UX 模式学习（不限于 CRM）

| 项目 | GitHub | 学什么 |
|------|--------|--------|
| **Shadcn/ui** | shadcn-ui/ui | 组件交互模式标准：dialog, toast, form validation, empty state |
| **Radix UI** | radix-ui/primitives | 无障碍交互原语：键盘导航、焦点管理、ARIA |
| **Tremor** | tremorlabs/tremor | Dashboard/图表/指标展示的设计模式 |

### 4.2 完整 SaaS 应用参考

| 项目 | GitHub | 学什么 |
|------|--------|--------|
| **Twenty** | twentyhq/twenty | CRM 全流程：CRUD + 关联 + 搜索 + 批量操作 |
| **Plane** | makeplane/plane | 项目管理：看板 + 状态机 + 拖拽 |
| **Cal.com** | calcom/cal.com | 预约系统：多步表单 + 时区 + 通知 |
| **Hoppscotch** | hoppscotch/hoppscotch | API 工具：极简但交互打磨极好 |
| **Formbricks** | formbricks/formbricks | 表单/调查：条件逻辑 + 实时预览 |

### 4.3 LLM Agent 架构参考

| 项目 | GitHub | 学什么 |
|------|--------|--------|
| **MetaGPT** | geekan/MetaGPT | SOP 驱动的多角色 agent，artifact handoff |
| **ChatDev** | OpenBMB/ChatDev | Chat chain + 角色隔离 |
| **Aider** | paul-gauthier/aider | Repo map + scoped context |
| **SWE-agent** | princeton-nlp/SWE-agent | 聚焦文件级操作 |

### 4.4 怎么"学"

不是人读代码，是**提炼模式注入 prompt**：

1. 分析 Twenty/Cal.com 的 CRUD 页面，提炼通用交互模式
2. 编码为 `design_quality_contract.json`（结构化规则）
3. 注入 PM step 的 prompt，让 spec 自动包含 UX 要求
4. QA step 用规则做验收检查

---

## 5. 实现计划：分阶段任务清单

### Phase 1: Design Quality Contract（1 天）

创建 `orchestrator/configs/design_quality_contract.json`：

```json
{
  "version": "1.0",
  "source_references": ["twentyhq/twenty", "shadcn-ui/ui", "calcom/cal.com"],
  "crud_completeness": {
    "rule": "Every entity MUST have Create, Read, Update, DELETE operations",
    "delete_requires": "confirmation dialog before destructive action"
  },
  "operation_feedback": {
    "rule": "Every write operation MUST show success/failure feedback within 2 seconds",
    "mechanism": "toast notification or inline status message",
    "states": ["success", "error", "loading"]
  },
  "empty_states": {
    "rule": "Every list/collection view MUST have an empty state with guidance text and primary action button"
  },
  "form_validation": {
    "rule": "Required fields MUST show inline validation errors on submit",
    "submit_button": "MUST show loading state during submission, disable to prevent double-submit"
  },
  "error_handling": {
    "network_error": "Show retry option with error description",
    "not_found": "Show 'not found' message with navigation back",
    "server_error": "Show generic error with retry"
  },
  "navigation": {
    "rule": "Multi-module apps MUST have visible navigation between modules"
  }
}
```

**任务**：
- [ ] T1-01: 分析 Twenty/Shadcn 源码，验证上述规则的普适性
- [ ] T1-02: 创建 `design_quality_contract.json`
- [ ] T1-03: 在 PM step prompt 中注入合约（workflow_state.js）
- [ ] T1-04: 在 acceptance criteria 生成中加入 UX 验证项（artifact_scaffold.js）
- [ ] T1-05: 跑一次完整 CRM E2E，对比产出差异

### Phase 2: Per-Role Scoped Injection（1 天）

将设计知识按角色分层，每步只注入相关切片：

- [ ] T2-01: 创建 `configs/design_rules/pm_product_rules.json`（产品要求：功能完整性）
- [ ] T2-02: 创建 `configs/design_rules/arch_interaction_rules.json`（交互状态：loading/error/empty）
- [ ] T2-03: 创建 `configs/design_rules/fe_component_rules.json`（组件模式：toast/dialog/form）
- [ ] T2-04: 创建 `configs/design_rules/qa_ux_checklist.json`（验收检查清单）
- [ ] T2-05: 在 `workflow_step_builder.js` 中按角色加载对应规则文件
- [ ] T2-06: 验证每步 prompt 增长 < 3KB

### Phase 3: Two-Phase FE Generation（2 天）

解决"FE 只实现一个模块"的问题——拆分 FE 步骤：

- [ ] T3-01: 设计方案——将 impl_fe 拆为 skeleton pass + per-module pass
- [ ] T3-02: Skeleton pass：只生成导航结构 + 模块占位 + 共享组件（toast/dialog）
- [ ] T3-03: Per-module pass：每个模块独立生成（只注入该模块的 API 合约）
- [ ] T3-04: 合并 pass 产出到最终 FE 输出
- [ ] T3-05: E2E 验证多模块产出

### Phase 4: Continuous Learning Loop（持续）

- [ ] T4-01: 每次 run 的 fidelity report 记录 UX 维度得分
- [ ] T4-02: 分析哪些规则有效、哪些被忽略，迭代合约
- [ ] T4-03: 建立 "design pattern library" 目录，积累经过验证的模式
- [ ] T4-04: 评估 RAG 方案——规则超过 20 条时从注入切换为检索

---

## 6. Context 扩展问题的解决方案

### 当前方案（够用到 Phase 2）

**编译式传递**：设计规则只注入 PM 层（~2KB），PM 消化后输出含 UX 要求的 spec.md，后续步骤通过 artifact handoff 继承，不重复注入。

```
design_quality_contract.json (2KB)
    → 注入 PM prompt (total ~3.3KB, 远低于 80K)
    → PM 输出 spec.md (含 UX acceptance criteria)
    → Arch 读 spec.md (不需要原始规则)
    → FE 读 arch handoff (不需要原始规则)
```

### 中期方案（Phase 3+）

**Per-Role Scoped Packs**：每个角色有自己的小规则文件（~1KB），只加载自己的。

### 远期方案（规则 > 20 条）

**Rule Retrieval (RAG)**：规则存入索引，每个 micro-task 只检索 top-3 相关规则。prompt 体量不随规则总量增长。

---

## 7. 验证标准

| 阶段 | 目标评分 | 验证方法 |
|------|---------|---------|
| Phase 1 完成 | 5/10 | CRUD 完整（含删除）+ 操作有反馈 |
| Phase 2 完成 | 6/10 | 交互状态完整（loading/error/empty） |
| Phase 3 完成 | 7/10 | FE 覆盖全部模块 |
| Phase 4 持续 | 8/10+ | 稳定产出可用级别的系统 |

---

## 8. 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| LLM 忽略注入的设计规则 | 产出不改善 | 在 QA 步骤加硬门禁（缺删除 = fail） |
| Per-module FE 拆分后模块间不一致 | 样式/交互风格碎片化 | Skeleton pass 先统一共享组件 |
| 规则过多导致 prompt 膨胀 | 注意力稀释 | 严格控制每层 < 3KB 注入量 |
| Gemma4:26b 能力天花板 | 复杂 FE 做不出来 | 评估切换更强模型或 few-shot |
