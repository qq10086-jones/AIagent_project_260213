# OpenClaw Nexus vNext: M6 Summary & M7 Draft Plan

## Date: 2026-03-09
## Type: Engineering Task List & Phase Summary
## Author: PM / Project Lead

---

## 1. Milestone 6 (M6) Delivery Summary: Infrastructure Complete (STAY_GATED)
*M6 核心目标是验证并行化执行（BE/FE）的安全性并建立受控发布的护栏，目前所有底层设施均已完成且测试通过。*

### 1.1 已交付核心内容 (Completed)
- [x] **WS-23 重放语料库与契约 (Replay + Contracts)**
  - 构建了包含50个真实场景（pm_heavy, arch_heavy, FE-safe等）的脱敏重放语料库。
  - 完成并发布了 FE-Safe 并行边界契约与异常处理降级契约。
  - 制定了生产重放数据的脱敏与保留规范（Governance）。
- [x] **WS-24 运行时并行控制桥 (Runtime Bridge)**
  - 移除硬编码串行锁，引入 **三层策略驱动的并行控制网关**。
  - 实现了基于白名单（workflow/project/input_class）的资格筛选。
  - 引入了“结构化防呆保护层”以防止非 FE-Safe 的任务意外进入并行分支。
- [x] **WS-23.5 Staging 验证与对比执行**
  - 部署了支持并行/串行对比执行的自动化 Runner（支持 `--mode compare`）。
  - 生成了全量的对比报告，所有 M6 仿真验证测试全数通过。
- [x] **WS-25 发布治理与熔断器 (Rollout Governance)**
  - 实现了自动熔断器（Circuit-Breaker），异常阈值超标时自动降级为串行。
  - 交付了运维状态查询工具（`< 30s` 响应）和紧急回滚预案（演练完成耗时 `8s`）。
- [x] **WS-26 指标基线建立**
  - 完成 Context Budget、Diff-first 命中率等基础指标采集。
  - 针对并行化进行了代码仓库层的基线测量。

### 1.2 当前项目状态
- **整体状态**: `STAY_GATED`（基础设施已就绪，安全锁默认处于拦截状态）。
- **拦截原因**: 目前所有的指标基线均基于仿真（Simulation）环境生成，系统架构师要求在正式开放曝光之前，必须拿到真实 LLM 调用的指标证明。

---

## 2. Immediate Next Steps: 解锁受限曝光 (Path to `GO_LIMITED_EXPOSURE`)
*为了将 M6 从 `STAY_GATED` 推进到小规模试运行，需紧急完成以下任务。*

- [ ] **Task 2.1: 执行真实 LLM Staging 压测**
  - 针对现有的 50 个 Replay 测试用例，关闭仿真（Stub），接入真实的 LLM 调度层进行完整运行。
- [ ] **Task 2.2: 刷新并发布生产基线指标**
  - 利用 Task 2.1 产生的数据，更新 `metrics/` 目录下的四组指标数据（Context Budget, Diff-first reliability, Patch mismatch, Parallel eligibility）。
- [ ] **Task 2.3: Go/No-Go 阈值二次评估**
  - 对照架构师设定的可用性阈值重新评估真实指标，出具最终的《M6 Exposure Go/No-Go Report》。
- [ ] **Task 2.4: 获得架构师授权并开启灰度开关**
  - 获取架构师最终审批签字。
  - 在生产配置 `production_parallel_rollout.json` 中将 `master_enabled` 设为 `true`，正式开启基于白名单的并行灰度。
- [ ] **Task 2.5: 灰度运行期监控护栏值班**
  - 监控并确认系统熔断器没有被误触发。
  - 监控回退到 `forced_sequential` 的比例。

---

## 3. Milestone 7 (M7) Draft Task List: 智能路由与全面推开 (Adaptive Routing & GA Rollout)

*M7 的核心目标是将系统从目前的“受控策略配置级并行”推进到“基于智能意图分类的全量智能路由”，进一步压缩系统延迟并提高大规模任务的资源利用率。*

### Phase 1: 智能路由设计与架构重构 (Brain Router Adaptive Engine)
- [ ] **WS-27-01: Brain Router 意图分析层设计**
  - 设计 LLM 任务早期分类器（Classification Classifier），分析前端需求属于 `fe_led`, `be_led` 还是 `arch_heavy`。
  - 定义模型选择下发接口（e.g., 简单任务选用较快/便宜模型，复杂推理走高成本推理模型）。
- [ ] **WS-27-02: 自适应路由策略协议 (Adaptive Routing Policy Contract)**
  - 确立模型路由下发、超时控制、Token 限额动态分配的控制协议。
  - 评审并引入动态并行评估（动态替代现有的基于静态配置文件 `parallel_exposure_policy.json` 的白名单）。
- [ ] **WS-27-03: M7 基础架构审批与立项**
  - 完成《OpenClaw_Nexus_Design_Document_v4.md》的编写与审批。

### Phase 2: 全面推开并行机制 (Parallel Rollout GA)
- [ ] **WS-28-01: M6 小规模受限并行数据的回溯分析**
  - 审阅 M6 `GO_LIMITED_EXPOSURE` 阶段产生的线上实际报错、重试与人工干预率。
- [ ] **WS-28-02: 取消 M6 并行配置白名单 (Whitelist Deprecation)**
  - 清理硬编码/静态配置的 `workflow/project` 白名单，将评估权交给 Task Validator 与智能路由层。
- [ ] **WS-28-03: FE/BE 双向合并自动处理机制优化**
  - 针对偶发性的双分支代码合并冲突，引入 AI Merge Resolve Agent，实现合并冲突的自愈与自修复。

### Phase 3: 观测增强与性能对标 (Observability & Efficiency)
- [ ] **WS-29-01: 端到端延迟耗时归因诊断工具**
  - 在现有的 `exposure_state_query` 基础上，增加按 `Node`（节点）维度的时延瀑布流展示分析（Waterfall Trace）。
- [ ] **WS-29-02: 多 LLM 提供商动态切换 (Multi-Vendor Fallback)**
  - 根据各云厂商接口的实时速率和配额限制，自动进行大模型供应商之间的路由降级和请求重拨。

### Phase 4: M7 验收与发布 (Release & Closure)
- [ ] **WS-30-01: 生产负载容量与稳定性演练 (Load Testing)**
  - 在完全自适应网络的情况下对服务吞吐量进行极限压测。
- [ ] **WS-30-02: M7 Go/No-Go 闭环与指标发布**
  - 提交基于全面并行的成功率、效率提升百分比指标。
  - 整理 M7 验收架构清单并推动 `CLOSED` 节点。