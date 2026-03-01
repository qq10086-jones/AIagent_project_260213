# Progress Report - 2026-03-01

## Scope
- 文档基线：`system_design_vNext` + `EXEC_PLAN_20260301_2day_coder_go_live`
- 执行范围：48小时收官清单中的 P0（不扩项）

## Completed Today
- 完成 `P0-1`：新增 OpenCode 适配器 `worker-coder/adapters/opencode_adapter.js`
  - 对齐统一输出契约
  - 统一错误码映射：`E_PROVIDER_UNAVAILABLE` / `E_TIMEOUT` / `E_APPLY_FAILED` / `E_INTERNAL`
- 完成 `P0-2`：`worker-coder/coding_service.js` 路由升级
  - `provider=auto -> opencode`
  - 仅在 OpenCode 不可用时 fallback 到 `codex`
  - 增加审计字段：`provider_requested` / `fallback_from` / `command_source`
- 完成 `P0-2` 配套：`worker-coder/worker.js` 支持透传 `opencode_command`
- 完成 `P0-3`：`orchestrator/src/index.js` 最小改造
  - `/coder` 默认 provider 改为 `opencode`
  - 默认 model 为 `minimax-m2.5`
  - 支持显式 `@gpt-5.3` 覆盖
  - 保持 risk-based approval，不回退全量审批
  - 保持 coder 专用结果模板
- 完成 `P0-4`：配置一致性修复
  - `configs/tools.json`：`coding.delegate` 与风险审批逻辑对齐（去除 blanket approval 配置）
  - `infra/docker-compose.yml`：补齐 `OPENCODE_BIN` / `CODER_PROVIDER_DEFAULT` / `CODER_MODEL_DEFAULT`
- 完成 `P0-5`：上线前安全硬门（代码层）
  - `worker-coder/coding_service.js` 增加 artifacts/logs 输出脱敏
  - `.gitignore` 补齐本地凭据与运行产物覆盖项
  - secrets 扫描（tracked + staged diff）结果：`0` 命中

## OpenCode Environment Fix
- 初次尝试在 `node:20-alpine` 安装 `opencode-ai` 失败（`spawnSync .../.opencode ENOENT`）
- 处理方案：`worker-coder/Dockerfile` 基础镜像改为 `node:20-bookworm-slim`
- 重建并重启 `worker-coder` 成功

## Validation
- 适配器测试（本地）：
  - `npm run test:adapter` 通过（受限环境下部分 spawn 用例自动 skip）
- 容器内验证（重点）：
  - `opencode --version` => `1.2.15`
  - `codex --version` => `codex-cli 0.106.0`
  - `node tests/opencode_adapter.test.js` => 全部通过

## Risks / Gaps
- `codex_adapter` 的“缺失鉴权应失败”测试在容器内会受挂载认证文件影响，需改造成可配置测试场景
- 目前尚未完成 Day1/Day2 的完整 E2E 与 canary（A/B/C 场景 + 20条灰度任务）

## Next Actions
- 执行容器级 E2E 三场景：
  - 场景A：低风险自动执行
  - 场景B：高风险进入审批
  - 场景C：`minimax-m2.5` 与 `gpt-5.3` 模型切换均成功
- 汇总上线指标：成功率、审批命中率、失败码分布
- 输出 go-live 报告与回滚验证记录
