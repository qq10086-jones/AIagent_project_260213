# Nexus Coder 48小时收官任务清单（vNext执行版）

## 目标（必须达成）
1. `/coder` 生产可用上线（默认走 OpenCode）。
2. 核心模型双路可用：`minimax-m2.5` + `gpt-5.3`。
3. 风险审批生效：低风险自动执行，高风险进入审批门。
4. 结果可追溯：run_id/task_id/files_changed/artifacts 全链路可查。
5. 不再继续扩张 monolithic orchestrator（只做必要拆分，不做大重构）。

---

## 非目标（48小时内禁止扩项）
1. 不做 Brain 插件化重构。
2. 不做 Kimaki/OpenSwarm/Lobster 全量接入。
3. 不做多新技能（`/ui` `/db`）扩展。
4. 不做“新平台”开发，只做上线闭环。

---

## P0任务清单（按优先级）

### P0-1 OpenCode Adapter上线
- [ ] 新增 `worker-coder/adapters/opencode_adapter.js`
- [ ] 统一输出契约（与 codex adapter 一致）：
  - `ok/provider_used/model_used/summary/files_changed/diff_stats/artifacts/diagnostics/error`
- [ ] 失败码统一映射：
  - `E_PROVIDER_UNAVAILABLE/E_TIMEOUT/E_APPLY_FAILED/E_INTERNAL`

### P0-2 Delegate双模型路由
- [ ] `worker-coder/coding_service.js` 增加 provider 路由：
  - `auto -> opencode`
  - fallback: `codex`（仅当 opencode 不可用）
- [ ] 支持模型透传：
  - `minimax-m2.5`
  - `gpt-5.3`
- [ ] 支持超时与命令来源记录（便于审计）

### P0-3 Orchestrator策略收口（最小改造）
- [ ] `orchestrator/src/index.js`：
  - `/coder` 默认 payload 改为 `provider=opencode`
  - 默认 model 策略：`minimax-m2.5`
  - 可在任务中显式指定 `gpt-5.3`
- [ ] 保留现有 risk-based approval 逻辑，不再全量审批
- [ ] Coder结果渲染固定为专用模板（禁止回退 quant 文案）

### P0-4 配置一致性修复
- [ ] `configs/tools.json` 与运行逻辑一致（消除漂移）：
  - `coding.delegate` 不做 blanket approval
  - 审批由风险规则决定
- [ ] `infra/docker-compose.yml` 补齐运行环境变量：
  - OpenCode所需provider配置
  - 默认模型配置
  - 运行超时配置

### P0-5 上线前安全硬门
- [ ] secrets 扫描（tracked files + staged diff）
- [ ] `.gitignore` 覆盖本地产物和凭据文件
- [ ] 禁止密钥写入 artifacts/logs（必要时加redaction）

---

## 48小时排程（实际执行）

## Day 1（0h - 24h）
### 0h - 2h：冻结范围 + 开工
- [ ] 锁定分支与上线范围
- [ ] 创建任务看板（P0 only）

### 2h - 8h：Adapter与Delegate核心改造
- [ ] 完成 `opencode_adapter.js`
- [ ] 完成 `coding_service.js` provider/model 路由
- [ ] 完成 `worker.js` payload透传校验

### 8h - 12h：Orchestrator最小改造
- [ ] `/coder` 默认 provider/model 策略落地
- [ ] 结果渲染检查（coder-only模板）

### 12h - 16h：环境与配置统一
- [ ] `tools.json` / `docker-compose.yml` 同步
- [ ] 启动容器并完成健康检查

### 16h - 24h：首轮E2E
- [ ] 场景A（低风险）：自动执行成功
- [ ] 场景B（高风险）：进入 waiting_approval
- [ ] 场景C（模型切换）：minimax 与 gpt-5.3 均成功

---

## Day 2（24h - 48h）
### 24h - 30h：稳定性修复
- [ ] 修复首轮E2E问题（仅P0）
- [ ] 修复 files_changed / artifacts 展示偏差

### 30h - 36h：回归测试（必须全绿）
- [ ] 低风险自动执行
- [ ] 高风险审批
- [ ] approve后恢复执行
- [ ] reject后正确终止
- [ ] provider不可用时fallback策略正确

### 36h - 42h：灰度上线
- [ ] 指定1个Discord频道做canary
- [ ] 连跑20条任务（简单/中等/高危混合）
- [ ] 记录成功率、审批命中率、失败码分布

### 42h - 48h：正式切换
- [ ] 切换默认provider到 opencode
- [ ] 发布上线报告 + 回滚命令
- [ ] 打版本tag并归档文档

---

## 验收标准（上线门槛）
1. `/coder` 非高危任务成功率 >= 90%
2. 高危任务100%进入审批门
3. 双模型均可完成真实改代码任务
4. 每个run都有完整追溯字段与artifact
5. 无明文密钥泄漏

---

## 回滚方案（必须提前准备）
1. 环境切回：`CODER_PROVIDER_DEFAULT=codex`
2. 恢复上一镜像tag并 `docker compose up -d`
3. 保留risk-policy，不回退到全量审批

---

## 关键文件改动清单（精确到路径）
1. `worker-coder/adapters/opencode_adapter.js`（新增）
2. `worker-coder/coding_service.js`
3. `worker-coder/worker.js`
4. `orchestrator/src/index.js`
5. `configs/tools.json`
6. `infra/docker-compose.yml`
7. `docs/03_feature_development/PROGRESS_LATEST.md`（上线记录）

---

## 战术要求（执行纪律）
1. 只做P0，不接新需求。
2. 每6小时出一次“可运行结果”，不是口头进展。
3. 任一环节失败，优先保证 `/coder` 可用与可回滚。
