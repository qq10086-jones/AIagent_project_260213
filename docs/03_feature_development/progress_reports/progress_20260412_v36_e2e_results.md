# v3.6 FE Decomposition — E2E 验证报告

**Date**: 2026-04-12
**Workflow Run**: `413b8805-9a77-44b0-9f0a-87cfbbaca274`
**Run ID**: `4e512806-d317-420f-9235-83785705da30`

---

## 关键结论

**gemma4:26b (全本地) 达到 7/10 — 与 M2.7 cloud 持平，甚至在某些维度更优。**

问题根因不是模型能力，是工作流指令粒度。把 impl_fe 拆成 skeleton + modules 两个步骤后，gemma4:26b 完全可以胜任。

## 流水线执行结果

| Step | Status | Model |
|------|--------|-------|
| pm_spec | SUCCEEDED | gemma4:26b |
| arch_design | SUCCEEDED | gemma4:26b |
| impl_be | SUCCEEDED | gemma4:26b |
| impl_fe_skeleton | SUCCEEDED | gemma4:26b |
| impl_fe_modules | SUCCEEDED | gemma4:26b |
| smoke_test | SUCCEEDED | N/A (verdict=pass) |
| qa_verify | SUCCEEDED | gemma4:26b |
| release_pack | SUCCEEDED | gemma4:26b |
| deploy_preview | QUEUED | (no creds) |

**全链路 gemma4:26b，无任何 cloud model escalation。**

## FE 产物对比

| 指标 | v3.5 gemma4 (4/10) | v3.5.2 M2.7 (7/10) | **v3.6 gemma4 decomposed** |
|------|-------------------|---------------------|------------------------------|
| app.js 行数 | ~180 | 616 | **682** (最多) |
| 模块覆盖 | 1/3 | 3/3 | **3/3** |
| sidebar nav | 无 | 有 | **有** (5 nav-item) |
| showToast | 无 | 有 | **有** |
| showConfirmDialog | 无 | 有 | **有** |
| renderLoading | 无 | 有 | **有** |
| renderEmptyState | 无 | 有 | **有** |
| escapeHtml | 基本 | 有 | **有** |
| renderErrorState | 无 | 无 | **有** (新增) |
| renderFormField helper | 无 | 无 | **有** (新增) |
| validateForm helper | 无 | 无 | **有** (新增) |

app.js 结构（682 行）：
- Shared utils: escapeHtml, apiFetch, showToast, showConfirmDialog, showLoading/hideLoading, renderLoading, renderEmptyState, **renderErrorState**, **renderFormField**, setActiveNav, navigateTo, **validateForm**
- DASHBOARD MODULE (lines 162-227)
- CUSTOMERS MODULE (lines 228-431): list/create/edit/delete, form validation
- TICKETS MODULE (lines 432-673): list/create/edit/delete, form validation
- NAVIGATION (lines 674-682)

## Smoke Test

```json
{
  "verdict": "pass",
  "root_check": {"status": 200, "passed": true},
  "api_check": {"endpoint": "/api/customers", "status": 200, "passed": true},
  "errors": []
}
```

## QA 报告

- overall_status: pass_with_warnings
- gemma4:26b 生成的 QA 是 scaffold 风格（"Auto-generated QA scaffold pending human review"），不像 M2.7 生成的 400+ 行详细分析
- **这是 v3.6 唯一明显不如 M2.7 的地方** — QA 深度检查能力差一些

## 评分轨迹

```
v3.4: 3/10 → v3.5 gemma4: 4/10 → v3.5.2 M2.7: 7/10 → v3.6 gemma4 decomposed: 7/10
```

## 关键发现

### 1. 最小单元指令输入策略验证成功
把 "build everything for 3 modules" 这种 gemma4 无法完成的复杂指令，拆成：
- skeleton: 只做 app shell + 共享组件 + nav
- modules: 在 skeleton 基础上填充每个模块的 CRUD

每个步骤的指令粒度都在 gemma4 的能力范围内。

### 2. gemma4:26b 能力天花板被高估
之前认为是模型能力问题，实际是工作流设计问题。同样的 gemma4:26b，换一种交互方式，产出甚至超过 M2.7。

### 3. gemma4:31b 不适合 24GB 显卡
超过显存导致 CPU offload，速度断崖式下降。已从 escalation chain 移除。

### 4. MiniMax 可以不续费
v3.6 证明 gemma4:26b 本地化即可达到 7/10。剩余 3 分差距在 activity feed、read-only detail、dashboard quick actions —— 这些是 spec 的完整性问题，和模型无关。

## 剩余差距（与满分 10/10）

同 v3.5.2 一致：
- 无统一 activity feed（dashboard 只有 Recent 分离列表）
- Customer 无 read-only detail（edit form 兼做 detail）
- Dashboard 无独立 quick action buttons

这些需要 v3.7 继续优化（PM spec 阶段的 scope 更明确，或 arch 层输出更细的 workplan）。

## 下一步

1. ~~升级 impl_fe 到 cloud model~~ — **不再需要**
2. QA step 深度不足，可以考虑再拆分 QA 步骤或用更详细的 qa prompt template
3. 继续优化 PM/Arch 让 spec 覆盖 activity feed / read-only detail / quick actions
4. 安全修复（XSS in showConfirmDialog 等，沿用 v3.5.2 的 codex 审视发现）
