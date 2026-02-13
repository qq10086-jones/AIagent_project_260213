# Project Nexus: 基于 OpenClaw 的本地自主商业闭环系统

> [!abstract] 项目概述
> **核心目标**：利用 OpenClaw 作为中枢调度代理，整合多源 LLM（Ollama 本地 + 云端 API），驱动六大业务触角，实现从量化交易、电商运营到内容变现的全自动化商业闭环。
> **硬件基础**：AMD RX 7900 XTX (利用 ROCm/DirectML 进行本地推理 & ComfyUI 绘图)。
> **系统核心**：OpenClaw (Node.js/TS) + Python Agent Skills。

---

## 1. 系统宏观架构 (System Architecture)

该系统采用 **Hub-and-Spoke (轮毂-辐条)** 结构。

```mermaid
graph TD
    User[用户指挥官] -->|指令/自然语言| Core[OpenClaw 中枢]
    
    subgraph "大脑池 (Brain Pool)"
        B1[GLM 4.7-flash (Ollama/本地)] -->|逻辑推理/代码| Core
        B2[Claude 4.6 (API)] -->|复杂逻辑推理/代码| Core
        B3[Gemini 3.0pro (API)] -->|复杂逻辑推理/多模态| Core
        B4[deepseek-r1:8b (Ollama/本地)] -->|日常任务| Core
    end
    
    subgraph "六大触角 (The Tentacles)"
        T1[量化交易 Agent]
        T2[电商自动驾驶 Agent]
        T3[社媒运营 Agent]
        T4[安全审计官 Agent]
        T5[小说漫改流水线 Agent]
        T6[数字人带货 Agent]
    end
    
    Core -->|任务分发| T1
    Core -->|任务分发| T2
    Core -->|任务分发| T3
    Core -->|任务分发| T4
    Core -->|任务分发| T5
    
    T4 -.->|监控与阻断| T1
    T4 -.->|监控与阻断| T2
    T4 -.->|内容审核| T3