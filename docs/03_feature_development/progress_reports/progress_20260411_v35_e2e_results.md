# v3.5 Design Quality Intelligence — E2E 验证报告

**Date**: 2026-04-11
**Workflow Run**: `86a050ad-c635-4dc3-8062-33e79abf20a3`
**Run ID**: `f61ad424-8f27-47ea-873e-850b9acf1ffc`
**Artifact Dir**: `runtime/artifacts/release/f61ad424-8f27-47ea-873e-850b9acf1ffc`

---

## 1. 流水线执行结果

| Step | Status | Model | Notes |
|------|--------|-------|-------|
| pm_spec | SUCCEEDED | gemma4:26b | validator 修复后一次通过 |
| arch_design | SUCCEEDED | gemma4:26b | DELETE 端点 + 3 模块覆盖 |
| impl_be | SUCCEEDED | gemma4:26b | 完整 CRUD (customers+tickets+dashboard+files) |
| impl_fe | SUCCEEDED | gemma4:26b | 只实现了 Customer 模块 |
| smoke_test | SUCCEEDED | N/A | GET / → 200, /api/customers → 200 |
| qa_verify | SUCCEEDED | gemma4:26b | scaffold 自动生成，非真正 QA |
| release_pack | SUCCEEDED | gemma4:26b | |
| deploy_preview | SKIPPED | N/A | |

## 2. UX 质量评分

| 维度 | v3.4 基线 | v3.5 结果 | 状态 |
|------|----------|----------|------|
| BE DELETE 端点 | 缺失 | 3 个 DELETE handler | PASS |
| BE CRUD 完整性 | 部分 | 4 个资源完整 CRUD | PASS |
| FE Empty State | 无 | "No customers matched this view yet." | PASS |
| FE Form Validation | 无 | name+email required + inline error | PASS |
| FE Operation Feedback | 无 | error feedback div (无 toast success) | PARTIAL |
| FE Navigation/Sidebar | 无 | 无 | FAIL |
| FE Multi-module Coverage | 1/3 模块 | 1/3 模块 (Customer only) | FAIL |
| FE Loading State | 无 | 无 | FAIL |

**总评: 4/10** (v3.4 为 3/10，提升 1 分)

## 3. 根因分析

### 改善项（v3.5 注入生效）
- BE 端完全遵循 Design Quality rules: DELETE 端点、多模块 API、完整 CRUD
- FE 基础 UX 改善: empty state、form validation
- Arch workplan 包含 DELETE tasks 和导航 task

### 未改善项（gemma4:26b 能力瓶颈）
- **FE 多模块**: gemma4 在 impl_fe 步骤无视 two-phase 指令，只实现第一个模块
- **FE 导航**: 无 sidebar/nav，因为只有一个模块
- **FE Loading**: 未实现 loading indicator
- **QA 步骤**: gemma4 未真正执行 QA，产出为 scaffold 模板

### 瓶颈确认
gemma4:26b (26B 参数) 对 BE 端简单指令遵循良好，但对 FE 端复杂指令（two-phase generation、multi-module coverage、shared components）遵循度不足。

## 4. v3.5.1 Model Escalation 已实现

- `lane_escalation_chain`: gemma4 → MiniMax-M2.7 (验证失败时自动升级)
- pm_spec 步骤验证: 升级机制正常触发 (`[workflow] model escalation: stable_gemma4_lane → stable_cloud_lane`)
- 但 impl_fe 步骤没有触发升级（因为它没有 validation failure，只是产出质量低）

## 5. v3.5 期间修复的 Bug

1. **Validator GOAL_FIDELITY_VIOLATION 误报**: goal 解析用 `with` split 产生碎片短语，改为用 `(N)` 编号模式做 module-level 分割
2. **Architect prompt delete 冲突**: "Do not add delete" 与 Design Quality rules 矛盾，移除旧限制
3. **Minimal-scope FE cap 冲突**: FE tasks 改用 N*2+3 公式，不再硬限 5 个
4. **QA rule injection 缺 metadata**: severity/grep_pattern 现在注入 prompt
5. **Docker volume mount**: design_rules/ 目录未 mount 到容器，已添加

## 6. 下一步

### 优先: impl_fe 预防性模型升级
对 `impl_fe` 步骤预防性使用 `stable_cloud_lane`（MiniMax-M2.7），不等失败再升级。
方案: 在 `workflow_step_builder.js` 中按 step complexity 预选 lane：
```
pm_spec, arch_design, impl_fe → stable_cloud_lane (复杂指令需要强模型)
impl_be, smoke_test, qa_verify, release_pack → stable_gemma4_lane (指令较简单)
```

### 其次: QA 步骤真正执行 UX 检查
当前 gemma4 在 QA 步骤只产出 scaffold。需要：
- QA 也用 cloud model
- 或在 smoke_test 步骤增加 UX 自动检测（grep-based）

### 目标
- impl_fe 用 M2.7 → 预期 6-7/10
- QA 真正执行 → 预期 UX gate 能拦住单模块产出
