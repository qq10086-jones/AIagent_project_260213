# Progress Report: Worker-Coder E2E 审计与加固 (2026-04-10~11) [UPDATED]

## 概要

对 worker-coder 全链路产物质量进行了系统性审计和加固，修复了 8 个测试失败、12 个产物质量缺陷，完成了首次真实 E2E CRM 任务（7/8 步骤通过），并针对产物质量评分 2/10 的问题实施了 8 项系统级修复。

## 完成的工作

### 1. 测试修复（311/311 全绿）
- DAG/finalization tests: mock pool 缺少 `claimStepForDispatch` CAS 处理 + `milestones` 关键字
- Context integration test: header 文本变更后 regex 更新
- Permission council: `coding.execute` 不应获得 sandbox 豁免
- Canary E2E: 补全 `claimStepForDispatch` mock + 更新 fixture 数据

### 2. 基础设施搭建
- Docker Compose 启动 Redis + PostgreSQL + Orchestrator + Worker-Coder
- 清理 Redis 积压队列（948 条历史任务）
- 通过 HTTP API 模拟 Discord 派发任务
- 配置 Gemma4:26b/31b 本地 Ollama 模型 + opencode.json 注册

### 3. 产物质量审计 (crm-test-004)
独立审计 + Codex 对抗审计，评分 2/10，发现：
- **Critical**: package.json `"type":"module"` 与 server.js `require()` 冲突
- **Critical**: Goal 4 功能仅实现 Customer CRUD
- **High**: PM spec 泛化、acceptance.json scaffold 默认、smoke fail 不阻断
- **High**: release_notes 虚报功能、arch/impl 持久化不一致

### 4. 系统级修复（8 项）
| Fix | 文件 | 效果 |
|-----|------|------|
| P0 ESM/CJS 检测 | artifact_scaffold.js | 检测 require() 后不加 type:module |
| P1 PM feature 枚举 | workflow_state.js | 强制展开 goal 每个功能 |
| P2 criteria 提取增强 | artifact_scaffold.js | 5 层提取策略（AC/US/Feature/Scope/As-a） |
| P3 smoke 硬门禁 | workflow_step_validator.js | smoke verdict=fail 阻断后续步骤 |
| P4 goal fidelity | workflow_step_validator.js | PM 产物必须覆盖 goal 全部功能 + Non-Goals 矛盾检测 |
| P5 release 真实性 | workflow_state.js | 禁止虚报未实现功能 |
| P6 模块系统校验 | workflow_step_validator.js | package.json type vs server.js 语法一致性 |
| P7 架构/实现一致 | workflow_step_validator.js | SQLite 声明 vs Map 实现检测 |
| P8 arch scope 注入 | workflow_step_builder.js | 从 spec.md 提取模块列表注入 arch prompt |
| P9 acceptance.json 兼容 | coding_team_validators.js | acceptance.json 有 criteria 时免除 spec heading 要求 |

### 5. LLM 模型对比
| 模型 | PM Spec 质量 | Arch 完整性 | 速度 |
|------|-------------|-------------|------|
| MiniMax M2.7 | 4 模块展开但 acceptance 弱 | 仅 Customer 端点，无 DELETE | 每步 5-15min |
| Gemma4:26b | 结构简洁但缺子模块 | 待验证（PM 阶段被 fidelity 拦截） | 每步 3-8min |

## 验证器效果验证
- `GOAL_FIDELITY_VIOLATION`: 正确拦截了 Gemma4 缺失功能的 PM spec
- `SMOKE_TEST_VERDICT_FAIL`: 正确拦截了 crm-test-004 的不可运行代码
- `STEP_MODULE_SYSTEM_MISMATCH`: 就绪但本轮未触发（P0 scaffold 修复已先行修正）

## 当前状态
- 测试: 311/311 orchestrator + 全 PASS worker-coder
- Docker: Redis + PG + Orchestrator + Worker-Coder 运行中
- LLM: Gemma4:26b 已配置为默认，MiniMax M2.7 作为 fallback
- 主要瓶颈: **LLM 在 arch_design 步骤的 scope 覆盖度不足**

## 下一步建议
1. **arch prompt 进一步强化**: 用 few-shot 示例展示完整的 4 模块 interfaces.md 格式
2. **尝试 Gemma4:31b**: 更大参数量可能改善 scope 覆盖
3. **refinement re-entry**: 当 arch fail 时自动用反馈重试，而非直接终止
4. **多模型 pipeline**: PM 用 MiniMax（scope 展开好），arch/impl 用 Gemma4（速度快）

## Phase 2 更新 (2026-04-11 凌晨)

### 新增修复
| Fix | 文件 | 效果 |
|-----|------|------|
| Goal-aware scaffold | artifact_scaffold.js | PM/Arch/workplan 模板从 goal 提取功能模块，动态生成全覆盖骨架 |
| Validation retry | workflow_engine.js + workflow_step_builder.js | 步骤验证失败时自动带反馈重试（最多 2 次） |
| acceptance_criteria 兼容 | coding_team_validators.js | acceptance.json 有 criteria 时免除 spec heading 要求（修复 re-add bug） |
| goal fidelity 精准匹配 | workflow_step_validator.js | Non-Goals 矛盾检测改为全短语匹配（避免单词误报） |
| Gemma4 opencode 注册 | opencode.json.tpl | 加入 gemma4:26b/31b 模型 |
| execution lane 切换 | runtime_defaults.json + docker-compose.yml | 默认 lane 切到 stable_gemma4_lane |

### 关键突破
**crm-gemma4-007 (wf: ef518d19)**: PM + Arch 双双 SUCCEEDED
- PM spec: 4 模块全覆盖（Customer/Ticket/File/Dashboard），acceptance.json 22 条结构化 AC
- Arch interfaces.md: **13 个端点含 DELETE**，覆盖全部 4 模块
- impl_be: 正在用 Gemma4:26b 生成完整后端代码...

### 根因分析修正
**scaffold 模板是最大的 saboteur**：
- 旧模板 hardcoded customer-only 内容，LLM 看到后不扩展
- 新模板从 goal 提取模块，动态生成覆盖全部模块的骨架
- LLM 从"生成完整结构"变为"填充已有结构"，利用 completion bias

### 当前运行状态
- crm-gemma4-007: pm_spec SUCCEEDED, arch_design SUCCEEDED, impl_be running
- 测试: 311/311 orchestrator + 全 PASS worker-coder
- Docker: Redis + PG + Orchestrator + Worker-Coder (Gemma4:26b) 运行中

### 下次启动指令
1. 读取本报告
2. 检查 crm-gemma4-007 运行结果: `docker exec infra-db-1 sh -c "psql -U nexus -d nexus -t -c \"SELECT step_id||' '||status||coalesce(' '||substring(error_code,1,80),'') FROM workflow_steps WHERE workflow_run_id LIKE 'ef518d19%' ORDER BY step_index\""`
3. 如果全部 succeeded，做产物审计并调 Codex 评分
4. 如果 impl_be fail，检查原因（STEP_MODULE_SYSTEM_MISMATCH / STEP_ARCH_IMPL_MISMATCH / other）并修复
5. 持续目标: 产物评分 8/10+

## Phase 3 更新 (2026-04-11 v3.4 Patch)

### 实施的改进（学习 MetaGPT/ChatDev/Cursor BugBot）
| Fix | 来源 | 文件 | 效果 |
|-----|------|------|------|
| be_to_fe goal-aware | MetaGPT SOP | artifact_scaffold.js | api_contracts 覆盖全部模块端点 |
| impl_to_qa goal-aware | MetaGPT SOP | artifact_scaffold.js | modules_to_verify + verification_endpoints |
| Handoff coverage guard | MetaGPT validation | coding_team_handoff_validators.js | be_to_fe 端点 >= arch 的 30% |
| Smoke 多端点 probe | Cursor BugBot | run_smoke_test.mjs + workflow_step_builder.js | 全部 GET 端点 probe |
| acceptance auto verify_command | MetaGPT executable feedback | artifact_scaffold.js | curl 命令自动生成 |
| isMinimalScope 修正 | 自有 | coding_team_validators.js | 多模块项目不受 5-task 限制 |
| scaffold "reviewable" 移除 | 自有 | artifact_scaffold.js | 不触发 minimal scope 检测 |
| Patch bundle placeholder skip | 自有 | workflow_step_artifacts.js | scaffold placeholder 不阻断 |

### 当前运行
crm-gemma4-011 (wf: cea0125b) 正在执行中

### 下次启动指令
1. 检查 crm-gemma4-011: `docker exec infra-db-1 sh -c "psql -U nexus -d nexus -t -c \"SELECT step_id||' '||status||coalesce(' '||substring(error_code,1,80),'') FROM workflow_steps WHERE workflow_run_id LIKE 'cea0125b%' ORDER BY step_index\""`
2. 如果全部 succeeded，读取全部 artifact 做审计 + Codex 评分
3. 目标: 8/10+
4. 测试: 311/311 orchestrator + 全 PASS worker-coder
