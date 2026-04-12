# v3.7 Static Audit — E2E 验证报告

**Date**: 2026-04-12
**Workflow Run**: `7b815c6a-61ac-4dc1-bd0c-9a9894656c20`
**Run ID**: `1f10ed0a-5449-4cee-bda2-155a2986576a`
**Mode**: dry_run (不阻塞，仅记录)

---

## 流水线执行结果

10 步全部成功（deploy_preview 因无 creds 跳过），全链路 gemma4:26b，含新增 `static_audit` 步骤：

| Step | Status |
|------|--------|
| pm_spec | SUCCEEDED |
| arch_design | SUCCEEDED |
| impl_be | SUCCEEDED |
| impl_fe_skeleton | SUCCEEDED |
| impl_fe_modules | SUCCEEDED |
| smoke_test | SUCCEEDED |
| **static_audit** | **SUCCEEDED (dry_run, overall=fail)** |
| qa_verify | SUCCEEDED |
| release_pack | SUCCEEDED |

## Scanner 结果对比 (v3.5.2 vs v3.7)

| Scanner | v3.5.2 baseline | v3.7 本次 | 变化 |
|---------|----------------|-----------|------|
| xss_scanner | 5 critical + 12 medium | **0 critical + 17 medium** | critical 消失 |
| class_injection | 6 medium | **3 medium** | 减半 |
| delete_semantics | 未能测 (npm 没装) | **2 high** | 真实 E2E 可测，抓到真漏洞 |
| be_contract_checker | 未能测 | **1 high** | 真实 E2E 可测，抓到真漏洞 |

## 关键发现

### v3.6.1 prompt 层安全规则**实际起作用了**
- `showConfirmDialog` 的 message 这类高危 XSS 点已经被 gemma4 记得用 escapeHtml
- Critical XSS 从 5 个降到 0 个
- 剩余 17 medium 主要是数字字段（stats.total_customers 这类），误报风险高

### BE 侧仍有真实漏洞
- `DELETE /api/customers/:id` 和 `DELETE /api/tickets/:id` 对不存在 id 返回 2xx（应 404）
- `POST /api/tickets/:id/comments` 接受空 body 返回 2xx（应 400）
- 这些是 v3.6.1 prompt 规则没完全覆盖的盲区，验证了"prompt pressure ≠ enforcement"的论断

### static_audit 集成工作正常
- 10 步流水线顺畅执行
- 4 个 scanner 全部运行，耗时均在秒级（xss 3ms, class 2ms, delete ~8s, be_contract ~8s）
- `verify/static_audit.json` + `meta/static_audit_feedback_*.txt` 正常输出

## 下一步选项

| 选项 | 说明 |
|------|------|
| A. 切 blocking 模式 | 让 `DELETE_MISSING_404` + `BE_MISSING_REQUIRED_VALIDATION` 的 high findings 阻塞流水线 |
| B. 调优 xss 数字字段噪音 | 进一步降低数字字段的 severity 或豁免 |
| C. 接入 retry 反馈回路 | 让 impl_be / impl_fe_modules 的下次 retry 读取 static_audit_feedback_*.txt |
| D. 剩余 3 分差距 | activity feed, read-only detail, dashboard quick actions |

推荐顺序: C → A → B → D。C 是最有杠杆的一步 — 让 static_audit 的发现真正反馈到源步骤 retry，形成闭环。
