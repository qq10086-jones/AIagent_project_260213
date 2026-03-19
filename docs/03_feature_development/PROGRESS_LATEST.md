# Nexus Project Progress Report - 2026-03-17 (QA Stress Test Milestone)

## 1. 核心进展：worker-coder 强度验证 (Internal Beta Ready)
- **修复确认**：`E_AUTH_FAILED` 问题已彻底根除。系统已从 `dashscope/qwen` 平稳迁移至 `minimax-coding-plan/MiniMax-M2.5`。
- **配置规范**：已完成 `infra/.env` 环境配置规范化，移除了 `docker-compose.yml` 中的硬编码。
- **压力测试结果**：
    - **测试套件**：Gate B (Real E2E)
    - **负载模型**：Runs: 15, Concurrency: 3, Warmup: 2
    - **执行情况**：完成 17/17 任务（含 Warmup），全链路成功。
    - **成功率**：**100% (100% Workflow Success Rate)**。
    - **稳定性指标**：在 3 并发下，Redis 队列和 LLM 适配器表现稳定，无长尾延迟波动。
- **状态**：**代码任务链路 (Coding Track) 已达成发布标准**。

## 2. 进展：worker-quant OpenBB 集成
- **已完成**：代码层面集成了 `openbb` v4 SDK，实现了新闻采集逻辑及降级采集管道。
- **当前瓶颈**：尚未进行容器构建验证和真实 API 联调。
- **状态**：**开发完成，待验证 (Dev Done, Pending Validation)**。

## 3. 遗留问题与风险记录 (Risk & Issues)

### A. 环境一致性 (P1)
- **.env 维护**：目前的 `infra/.env` 包含敏感 API Key。由于该文件通常不提交，需确保在生产/测试环境部署时，CI/CD 或运维手册中有明确的 Key 注入步骤。
- **Docker 运行环境**：由于测试是在本地 Docker 中通过挂载卷 (`/app`) 进行的，需确保最终镜像构建 (`docker build`) 时，代码修复也被正确包含进去（目前代码挂载是动态的）。

### B. 验证缺口 (P1)
- **OpenBB 依赖冲突风险**：`openbb` 的依赖树较为复杂，可能与现有 `worker-quant` 的 `pandas<3.0.0` 或 `numpy<2.0.0` 产生冲突。
- **OpenBB 降级稳健性**：需要模拟 OpenBB API 超时或不可用场景，确认 Yahoo/Google RSS 采集能平滑接管，不影响 `worker-quant` 的整体吞吐。

## 4. 下一步动作路线图 (Next Steps)

1.  **[Validation] worker-quant 容器构建**：运行 `docker-compose build worker-quant` 验证依赖一致性。
2.  **[Test] OpenBB 回退逻辑测试**：编写针对 `_merge_recent_news` 的离线单元测试。
3.  **[Final Gate] Gate B (Full Load)**：执行一次涵盖 Coding 和 Quant 两个 Worker 的全量混合负载测试。
4.  **[Release] 代码提交与内部测试宣告**：整理 Git Commits，编写 Release Note，正式开启 Internal Beta。
