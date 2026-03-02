# Nexus Decoupled Multi-Skill Architecture Design

## 1. 核心目标
解决 OpenClaw Nexus 随技能（Skills）增加而产生的代码耦合与开发难度指数级上升的问题。通过“插件化工具集”与“解耦型智能体”架构，实现技能的快速横向扩展。

## 2. 总体架构图 (Conceptual)
```mermaid
graph TD
    Brain[Brain - Agent Orchestrator] -->|Unified Tool Protocol| Orch[Orchestrator - API Gateway]
    Orch -->|Redis Streams| W_Quant[Worker-Quant]
    Orch -->|Redis Streams| W_Coder[Worker-Coder]
    Orch -->|Redis Streams| W_Media[Worker-Media - Future]
    
    W_Quant -->|Shared State| DB[(PostgreSQL/Redis)]
    W_Coder -->|Shared State| DB
    W_Media -->|Shared State| DB
    
    W_Coder -->|File Access| WS[/Workspace/ - Shared Volume]
```

## 3. 核心解耦方案

### 3.1 Orchestrator 路由“去业务化” (Skill-as-a-Service)
- **现状**: `/coding/patch` 等业务逻辑硬编码在 Orchestrator。
- **目标**: Orchestrator 仅作为 **API 网关和任务调度器**。
- **设计**:
    - 将 `patch_manager.js` 等逻辑移至专门的 `Worker-Coder` 服务。
    - Orchestrator 只负责解析任务请求并将其分发到对应的 Redis Stream。
    - 统一工具调用接口：`/execute-tool` 替代 `/coding/patch`, `/quant/analysis` 等。

### 3.2 Brain 节点“插件化” (Plugin-based Nodes)
- **现状**: `supervisor.py` 手动硬编码所有 LangGraph 节点。
- **目标**: 引入 **动态技能注册机制**。
- **设计**:
    - 将技能定义封装在 `brain/skills/` 目录下（如 `coding_skill.py`, `quant_skill.py`）。
    - 技能模块导出 `node_fn`, `route_rules`, `prompts`。
    - `supervisor.py` 根据 `mode` 环境变量或初始状态动态加载对应的技能模块并构建图。

### 3.3 状态与上下文交换协议 (Common Context Protocol)
- **协议内容**:
    - 所有技能产生的输出（Facts）必须符合统一的 `FactItem` 格式：`{agent: str, data: dict, metadata: dict}`。
    - 引入 `GlobalMemory`，允许跨技能读取。
    - **示例**: `CoderAgent` 可以读取 `QuantAgent` 标记的“策略回测失败”事实，从而主动检查相关代码。

### 3.4 资源与环境隔离
- 每个技能（Worker）采用独立的 Docker 镜像。
- **Quant**: 依赖 TA-Lib, Pandas 等。
- **Coder**: 依赖 Git, Compilers, Aider-logic 等。
- **Media**: 依赖 FFmpeg, PyTorch 等。
- 只有需要修改文件的技能才挂载 `/workspace` 卷。

## 4. 开发规范 (Standard Operating Procedures)
1. **新增技能**:
    - 在 `infra/docker-compose.yml` 中新增对应的 Worker 服务。
    - 在 `brain/skills/` 中创建技能描述文件。
    - 在 `tools.json` 中注册该技能提供的原子化 API 接口。
2. **测试验证**: 必须提供该技能的独立 `test_skill.py` 脚本，且需在容器环境内运行通过。

## 5. 实施路线图 (Milestones)
- [ ] **Phase 1 (Short-term)**: 重构 Orchestrator，将 Coding 逻辑独立到内部模块，剥离硬编码路径。
- [ ] **Phase 2 (Medium-term)**: 建立第一个独立的 `Worker-Coder` 容器。
- [ ] **Phase 3 (Long-term)**: 实现 Brain 端的动态插件加载机制，彻底解耦 `supervisor.py`。
