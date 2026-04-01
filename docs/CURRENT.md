# Nexus — 当前文档导航

> **新员工从这里开始。** 本页列出所有"当前有效"文档的直接链接。
> 历史版本在各子目录中保留，但不需要阅读。

**最后更新**: 2026-04-01

  - latest live workflow: `runtime/artifacts/orchestrator/canary/live_m9_workflow/live_m9_workflow_report.json`
  - latest verified status: `PASS / GO`, `superpowers_configured_steps = 6`, `superpowers_steps_used = 6`

---

## 立即需要读的三份文档

| 文档 | 路径 | 说明 |
|------|------|------|
| **系统设计 v3.1** | [`01_design/system/260401/Nexus_System_Design_v3.md`](01_design/system/260401/Nexus_System_Design_v3.md) | 架构全貌、Shared Contracts + Observability 层、Permission Council（advisory）、single_agent micro-workflow |
| **任务清单 v3.1** | [`01_design/system/260401/Nexus_Tasklist_v3.md`](01_design/system/260401/Nexus_Tasklist_v3.md) | 当前所有待办任务、优先级、验收标准 |
| **执行治理文档 v3.1** | [`01_design/system/260401/Nexus_Execution_Governance_v3.md`](01_design/system/260401/Nexus_Execution_Governance_v3.md) | 开发规范、禁止事项、Permission Council advisory 规则、防跑偏指南 |

---

## 垂类模块设计（当前有效）

| 模块 | 文档 | 说明 |
|------|------|------|
| **Coding Worker** | [`01_design/coding/coding_agent_design_latest.md`](01_design/coding/coding_agent_design_latest.md) | worker-coder 定位与流程 |
| **Quant Worker** | [`01_design/quant/quant_design_latest.md`](01_design/quant/quant_design_latest.md) | worker-quant 设计 |
| **Web Chat** | [`01_design/web/web_augmented_chat_design_latest.md`](01_design/web/web_augmented_chat_design_latest.md) | Web 增强对话设计 |
| **Learning** | [`01_design/learning/learning_design_latest.md`](01_design/learning/learning_design_latest.md) | 学习层设计 |

---

## 接口合同文档（当前有效，260306）

这些文档定义了各模块之间的接口协议，修改接口前必须先更新对应合同。

| 合同 | 路径 |
|------|------|
| Brain Router / TaskEnvelope | [`01_design/system/260306/Brain_Router_TaskEnvelope_Contract.md`](01_design/system/260306/Brain_Router_TaskEnvelope_Contract.md) |
| Agent Contract Layer | [`01_design/system/260306/Agent_Contract_Layer_Contract.md`](01_design/system/260306/Agent_Contract_Layer_Contract.md) |
| Tool Adapter Interface | [`01_design/system/260306/Tool_Adapter_Interface_Contract.md`](01_design/system/260306/Tool_Adapter_Interface_Contract.md) |
| Artifact Model | [`01_design/system/260306/Artifact_Model_Contract.md`](01_design/system/260306/Artifact_Model_Contract.md) |
| Backend Execution Adapter | [`01_design/system/260306/Backend_Execution_Adapter_Contract.md`](01_design/system/260306/Backend_Execution_Adapter_Contract.md) |
| Coding Team Handoff | [`01_design/system/260306/Coding_Team_Handoff_Contract.md`](01_design/system/260306/Coding_Team_Handoff_Contract.md) |
| QA Verifier | [`01_design/system/260306/QA_Verifier_Contract.md`](01_design/system/260306/QA_Verifier_Contract.md) |
| Observability | [`01_design/system/260306/Observability_Contract.md`](01_design/system/260306/Observability_Contract.md) |

---

## 进行中的功能开发

| 功能 | 设计文档 | 任务清单 | 状态 |
|------|---------|---------|------|
| Superpowers 集成 | [`03_feature_development/2026-03-28_superpowers_integration_design.md`](03_feature_development/2026-03-28_superpowers_integration_design.md) | [`03_feature_development/2026-03-28_superpowers_integration_tasklist.md`](03_feature_development/2026-03-28_superpowers_integration_tasklist.md) | 60%，Track A/B/C 待完成 |
| Beta 质量提升 | [`03_feature_development/2026-03-29_nexus_beta_quality_design.md`](03_feature_development/2026-03-29_nexus_beta_quality_design.md) | [`03_feature_development/2026-03-29_nexus_beta_quality_tasklist.md`](03_feature_development/2026-03-29_nexus_beta_quality_tasklist.md) | 已并入 v3 任务清单 |

---

## 最新进度报告

- **当前状态**: [`03_feature_development/PROGRESS_LATEST.md`](03_feature_development/PROGRESS_LATEST.md)
- **最近一次验证**: [`03_feature_development/progress_reports/progress_20260401_discord_supported_beta_live.md`](03_feature_development/progress_reports/progress_20260401_discord_supported_beta_live.md)
  - verdict = GO ✅ | superpowers_configured_steps = 0 ⚠️

---

## 历史文档

历史设计版本保存在 `01_design/system/` 各日期子目录中（260301 → 260401），无需阅读，仅供溯源。

归档文档在 [`90_archive/`](90_archive/) 目录。

---

## 文档维护规则

- 新设计文档放在 `01_design/system/YYMMDD/` 对应日期目录
- 更新本页 `CURRENT.md` 中的链接指向最新版本
- 历史文档**不删除**，只更新导航指针
- 每次里程碑完成后更新"最新进度报告"行
