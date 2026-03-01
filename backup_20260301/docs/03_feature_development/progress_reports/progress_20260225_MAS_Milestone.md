# OpenClaw Nexus 进展报告 (2026-02-25) - MAS 里程碑

## 1. 当前状态：v1.2.3 MAS (Multi-Agent System) 架构已跑通
项目已完成从传统“工具调用”到“多智能体协作”的核心升级。目前系统具备云端决策逻辑与本地硬件执行的完美协同。

## 2. 已实现的核心组件
- **Supervisor Agent (Brain)**: 
  - 基于 **LangGraph** 构建，运行在独立容器中。
  - 负责任务拆解、节点调度（Quant <-> Browser）以及最终结果审核。
  - 实现了 **Polling (轮询)** 机制，确保只有拿到 Worker 的真实数据才会生成报告。
- **Message Gateway (Orchestrator)**:
  - Node.js 实现，负责 Discord 实时交互、NLP 指令初步解析。
  - 对接 **Qwen-Plus API** (新加坡节点)，提供极速意图识别与翻译。
  - 实现了 `run_id` 全链路追踪与 `idempotency_key` 幂等性校验。
- **Execution Plane (Worker-Quant)**:
  - 复用了原有的 `Project_optimized` 量化核心。
  - 增加了 **Facts Store 写入** 能力，能将股价、指标、截图路径存入 Postgres。
- **Evidence Plane (OpenClaw)**:
  - 实现了基于意图的自动截图功能，支持从 MinIO 到 Discord 的自动文件转发。

## 3. 数据库核心结构 (MAS Ready)
- `runs`: 记录每次对话的生命周期。
- `fact_items`: 存储 Worker 产生的原子事实数据。
- `evidence`: 存储原始证据（URL、截图引用、文本片段）。
- `tasks`: 任务队列状态追踪（支持 `run_id` 关联）。

## 4. 关键文件路径
- **MAS 逻辑**: `brain/supervisor.py` (LangGraph 定义)
- **网关逻辑**: `orchestrator/src/index.js` (Discord & API)
- **NLP 路由**: `orchestrator/src/nlp/router.js` (Qwen API 封装)
- **量化工具**: `worker-quant/worker.py` (真实工具映射)
- **基础设施**: `infra/docker-compose.yml` (自定义网络 `nexus-net`)

## 5. 开发调试注意 (Important for next session)
- **API Key**: 已存入 `infra/.env` 中的 `QWEN_API_KEY`。
- **模型路由**: Supervisor 默认优先尝试 Qwen Cloud，认证失败或明确要求时回退到本地 `deepseek-r1:1.5b`。
- **日志观察**: 
  - `docker logs brain`: 观察 Agent 之间的规划和轮询。
  - `docker logs nexus-orchestrator`: 观察 Discord 消息和任务分发。

## 6. 下一步计划
1. 启用 `AUTO_REPORT_CHANNEL_ID` 实现定时任务推送。
2. 按照 MAS 设计手册完善“权威源优先级”仲裁逻辑。
3. 增加更多垂直领域的 Agent（如：专攻社媒舆情的 Sentiment Agent）。
