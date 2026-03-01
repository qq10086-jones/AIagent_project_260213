# OpenClaw Nexus Progress Update

- Date: 2026-02-28
- Scope: Discord 本地模型切换稳定性与回复质量修复

## 今日完成

1. 修复 `/model` 重复回复风险
- 在 `orchestrator/src/index.js` 增加基于 Redis 的消息去重锁（按 `msg.id`）。
- 目标：避免同一消息被重复处理时出现双回复。

2. 新增并修复本地模型切换命令
- 支持：`/model-local`、`/model-local:<model>`、`/model-cloud`。
- 修复模型名解析问题（保留 `deepseek-r1:32b` 中的冒号，不再被拆错）。

3. 打通本地模型链路并增强兼容
- 本地聊天调用增加 `chat -> generate` 兼容回退。
- 增加超时控制与失败提示，避免无响应时只返回通用错误。

4. 本地回复质量治理
- 增加本地系统提示，统一助手身份为 NEXUS。
- 增加 THINK 内容清洗：
  - 过滤 `<think>...</think>`。
  - 过滤未闭合 `<think>` 残留。
  - 过滤常见推理草稿句式并启用兜底回复。

5. 性能参数对齐
- 本地调用显式传入参数：
  - `num_ctx=4096`
  - `num_predict=128`
  - `keep_alive=30m`
- 目的：改善速度与稳定性，降低超时概率与 GPU 抖动。

6. 服务重启与验证
- 已重启 `nexus-orchestrator` 多次以加载补丁。
- 核验关键容器处于运行状态，核心 HTTP 入口可访问。

## 当前状态

- 云端/本地模型切换功能已可用。
- 本地模型回复链路已可工作。
- 仍需继续观察 deepseek-r1:32b 在高负载下的超时概率与思考内容泄露边缘样例。

## 下一步（待办）

1. 增加本地调用重试（仅超时场景 1 次重试）。
2. 将本地参数（`OLLAMA_CHAT_TIMEOUT_MS`、`OLLAMA_NUM_CTX`、`OLLAMA_NUM_PREDICT`）写入配置并文档化。
3. 补充回归测试清单：
- `/model`
- `/model-local:deepseek-r1:32b`
- `/model-local:glm-4.7-flash:latest`
- 普通问答连续 5 轮稳定性。
