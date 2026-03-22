# Nexus Project Progress Report - 2026-03-23 (worker-quant v1.1 交付，M12 Final Gate 准备中)

## 1. 核心进展：worker-quant v1.1 专属量化AI员工（2026-03-23 完成）

### 1a. 交付范围
本次迭代将 worker-quant 从"基础行情工具集"升级为具备完整工作日流程的专属量化AI员工。21/22 任务完成（1项设计上延迟）。

**工具数量**：19 → 29 个（新增10个）

| 新增工具 | 功能 |
|----------|------|
| `portfolio.position_review` | 持仓审查：成本价、P&L、风控标记、LLM叙事 |
| `portfolio.midday_pnl` | 盘中P&L快速查询（< 10秒，无LLM） |
| `portfolio.morning_brief` | 早盘简报：大盘概览 + 持仓状况 + 今日TDnet公告 |
| `portfolio.post_close` | 收盘复盘：当日P&L、快照归档、持仓建议 |
| `portfolio.set_preference` | 用户偏好持久化（止损比例、仓位上限等） |
| `quant.event_alert` | TDnet公告事件预警（与持仓交叉比对） |
| `quant.signal_backfill` | 信号IC/ICIR回测（Spearman秩相关，无scipy依赖） |
| `quant.portfolio_risk` | 相关性矩阵 + 组合波动率 + 集中度预警 |
| `quant.watchlist` | 观察名单管理（add/remove/list） |
| `news.tdnet_announcements` | TDnet公告直查（Kabutan RSS → Google News → GDELT降级） |

### 1b. 核心能力升级

**新闻信号改造（Epic B）**
- TDnet公告替代GDELT作为日股alpha主源，抓取延迟 < 1小时（原GDELT：1-2天）
- `deep_analysis` 现整合新闻情绪因子（`news_sentiment`）和TDnet催化剂（`today_announcements`）
- 上调修正/增配/回购等8类事件的alpha权重映射已配置

**持仓管理基础（Epic A）**
- WAVG成本价计算（BUY/SELL序列精确核算）
- `positions_snapshot` 表每日归档，支持日间P&L对比
- 风控标记：止损触发 / 信号反转 / 持仓超期（> 20天）

**统计因子工程（Epic D）**
- `signal_log` 表自动记录每次信号，5d/20d收益率异步回填
- 横截面相对强弱因子（D-03）接入 `discovery_workflow`，对4分位分档调整 `selection_score`（±0.30/±0.10）
- IC/ICIR计算就绪，待4周历史积累后启用自适应权重（D-05）

**个性化与修复（Epic E）**
- `user_profile.json` 持久化用户偏好，`portfolio.set_preference` 可在线修改
- `watchlist_alias.json` 支持29只日股的中英文别名识别
- `auto_adjust=True` 修复除权日虚假信号（E-03）

### 1c. 新增文件
- `worker-quant/quant_trading/Project_optimized/watchlist_alias.json` — 29只日股别名映射
- `worker-quant/quant_trading/Project_optimized/user_profile.json` — 用户偏好（资金40万JPY，止损8%，中等风险）
- `worker-quant/quant_trading/Project_optimized/docs/design/QUANT_AI_EMPLOYEE_DESIGN_v1.md` — 完整架构设计文档
- `worker-quant/quant_trading/Project_optimized/docs/design/QUANT_AI_EMPLOYEE_TASKS_v1.md` — 22项任务清单（21/22完成）

---

## 2. 历史里程碑回顾

### M12 Internal Beta 准入状态（2026-03-22 Gate A 关闭）
- **Gate B Real E2E**：17/17 PASS，并发3，成功率100%
- **Gate A**：治理决策关闭（dispatch 100%验证，workflow质量由Gate B覆盖，见 `docs/governance/m12_gate_a_closure_note.md`）
- **worker-coder**：19/19单元测试PASS，已迁移至MiniMax-M2.5

### M10 负载测试（2026-03-15 PASS）
- `stable_local_lane` 单run 6/6步骤全通过，GO verdict

### M8 Go/No-Go（2026-03-09 APPROVED）
- `master_enabled=true` 激活，M6 GO_LIMITED_EXPOSURE 生效

### worker-quant OpenBB验证（2026-03-19 PASS）
- 容器构建无冲突，`_merge_recent_news` 10/10离线测试通过，外层异常捕获bug修复

---

## 3. 当前生产配置

```
master_enabled: true
dynamic_routing_enabled: true
router_mode: dynamic_routing_enforced
execution_lane_default: stable_cloud_lane
force_sequential: false
```

**测试基线**（2026-03-23）
- `npm --prefix orchestrator test` → 150/150 PASS
- `npm --prefix worker-coder test` → 19/19 PASS
- worker-quant syntax check → OK，容器运行正常

---

## 4. 遗留问题与风险

| 项目 | 优先级 | 说明 |
|------|--------|------|
| D-05 信号加权合成 | P2（延迟） | 需≥4周 signal_log 历史才有统计意义 |
| C-03 cron调度 | P2 | 工具已就绪，Discord bot侧定时触发尚未配置 |
| TDnet/J-Quants真实API验证 | P1 | 需真实网络环境运行 `validate_sources.py` |
| Golden Set扩充 | P2 | 当前100条，目标≥200条 |
| opencode.json MiniMax认证 | P1 | Final Gate前需 `git diff opencode.json` 确认 |

---

## 5. 下一步行动

1. **[Final Gate]** 全量混合负载测试：Coding + Quant 双 Worker 同时压测
2. **[worker-quant]** 运行 `validate_sources.py` 验证TDnet/J-Quants真实网络解析
3. **[Release]** 编写 M12 Internal Beta Release Note，正式宣告上线
4. **[长期]** D-05 信号加权合成（4周后）、C-03 cron调度配置
