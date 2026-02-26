# OpenClaw Nexus 进度报告

- 报告时间: 2026-02-26 12:08:58
- 项目版本: `OpenClaw_Nexus_v1_2_4`
- 基线文档: `AIagent_project/Project_OpenClaw_Nexus_v1_2_3_MAS.md`
- 后续补丁: `docs/progress_20260226_135851_OpenClaw_Nexus_v1_2_5_PATCH.md`

## 本次里程碑结论

当前核心链路已跑通：`Discord -> orchestrator -> brain -> worker-quant/browser -> fact_items -> narrative 回传`。  
并完成你要求的两项关键目标：

1. 千问模型固定为 `qwen3.5-122b-a10b`
2. OpenClaw 浏览器真实抓取链路可用（可 `start/open/screenshot`）

## 已完成内容（按模块）

1. 模型与配置
- `infra/.env` 固定 `QWEN_MODEL=qwen3.5-122b-a10b`
- `infra/docker-compose.yml` 将 `QWEN_MODEL` 注入 `orchestrator` 与 `brain`
- 运行中容器环境已验证为该模型值

2. OpenClaw 运行稳定化
- `infra/docker-compose.yml` 调整 OpenClaw 配置挂载路径到 `/app/config/openclaw.json`
- 新增 `infra/openclaw.json`（默认 profile/headless/noSandbox）
- `infra/openclaw-adapter/server.py` 修正 browser 子命令调用链路与 profile 初始化逻辑

3. 分析结果“去幻觉/事实化”
- `worker-quant/worker.py` 的 `quant.deep_analysis` 改为基于 yfinance 的结构化事实输出
- `brain/supervisor.py` 改为“事实优先”叙事生成，避免自由发挥
- `orchestrator/src/nlp/router.js` 改善 ticker 提取（含 `.T`）

4. Discord 长消息报错修复
- `orchestrator/src/index.js` 增加分段发送（chunk）逻辑，避免超过 2000 字符限制

## 验证结果（实测）

1. OpenClaw adapter 接口
- `browser.start`: 成功
- `browser.open https://finance.yahoo.co.jp/quote/9432.T`: 成功
- `browser.screenshot`: 成功（返回截图路径）

2. 9432.T 端到端
- 通过 `POST /chat` 触发分析，返回 symbol 为 `9432.T`，公司为 `NTT, Inc.`
- 运行 `run_id=44d4ee99-3efe-4674-bcd3-d1c266252ea4` 的任务状态：
  - `quant.deep_analysis = succeeded`
  - `browser.screenshot = succeeded`
- `fact_items` 中存在 browser `visual_evidence` 且有 `object_key`

## 当前状态评估

- 结论: 版本 `v1.2.4` 可进入 Discord 联调与日常使用阶段
- 风险项: `openclaw-adapter /healthz` 中 `gateway_http` 仍显示不可达（当前链路走 docker exec，不影响实际截图）

## 下一步建议

1. 启用并验证“按天自动报告”定时推送（指定频道 + 失败重试）
2. 增加回归脚本（9432.T、AAPL、新闻日报）形成一键 smoke test
3. 补充 `fact_items/evidence` 的质量检查规则（缺失字段告警）
