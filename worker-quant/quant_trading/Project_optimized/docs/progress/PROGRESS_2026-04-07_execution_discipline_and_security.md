# Progress: 2026-04-07 执行纪律系统 + 安全加固 + 全面审计修复

## Status

六段式盘中调度系统已上线，Discord 推送验证通过，安全漏洞已修复，
全量测试 78/78 PASS。

## 背景

04-06/04-07 实盘复盘暴露执行纪律问题：
- 模型 04-06 建议 SELL 7267.T + 9432.T，用户未及时执行
- 延迟一天导致 Honda -1,200 JPY 额外亏损
- 盘中 7 小时无系统介入（仅 08:30 晨报 + 16:30 收盘）
- 模型 3/3 判断正确，但操作者净亏 -760 JPY

## 一、执行纪律系统（Phase 0 + Phase 1）

### 新增模块

| 文件 | 用途 |
|------|------|
| `action_plan_builder.py` | 生成 `reports/action_plan_today.json`，汇总待执行操作 + regime + 持仓状态 |
| `check_pending_actions.py` | 对比 action_plan vs fills，未执行信号推送 Discord 提醒 |
| `intraday_monitor.py` | 盘中监控：止损/止盈检查 + 提醒升级 (open_watch/midday/pre_close 三种模式) |
| `compliance_tracker.py` | 决策日志 + 模型偏差检测 + 遵从率统计 |

### 六段式调度

| 时间 (JST) | 任务名 | 脚本 | Task Scheduler |
|------------|--------|------|----------------|
| 07:30 | QuantMorningBriefing | `scheduled_morning_briefing.cmd` | Ready |
| 09:00 | QuantOpenWatch | `open_watch.bat` | Ready |
| 09:30 | QuantOpenConfirm | `open_confirm.bat` | Ready |
| 11:30 | QuantMiddayCheck | `midday_check.bat` | Ready |
| 14:00 | QuantPreClose | `pre_close.bat` | Ready |
| 16:30 | QuantDailyRun | `scheduled_daily_run.cmd` | Ready |

全部 6 个任务已注册并手动触发验证：
- OpenWatch / OpenConfirm / MiddayCheck / PreClose: Last Result=0
- MorningBriefing: Last Result=0（修复了 pause 卡住问题）
- DailyRun: 未重跑（避免重复全链路），注册正常

### 提醒升级策略

| 时段 | 级别 | 措辞 | Discord |
|------|------|------|---------|
| 09:30 | WARNING (黄色) | "信号未执行，请确认" | 推送 |
| 11:30 | WARNING (黄色) | "仍未执行，预计额外亏损 ¥X" | 推送 |
| 14:00 | ERROR (红色) | "最后执行窗口，今日不执行则失效" | 推送 |
| 16:30 | AUDIT | 记录为 miss/override/follow | decision_journal |

### daily_run.py 集成

在 make_decision 完成后新增：
- 调用 `build_action_plan()` 预生成次日 action plan
- 调用 `record_daily_compliance()` 记录当天遵从情况
- 推送 `action_plan_generated` 事件到 Discord
- 全部包在 try/except 中，失败不影响主链路

### morning_briefing.bat 扩展

新增 Step 3: 运行 `action_plan_builder.py` 生成 action plan 并推送。

## 二、安全加固

| 问题 | 修复 |
|------|------|
| Discord webhook URL 明文在 config.yaml (已入 git history) | config.yaml `webhook_url` 清空，新 URL 写入 `.env`（已 gitignore），`daily_run.py` 新增 `_load_dotenv()` |
| reports/ 运行时文件被 git 追踪 | `.gitignore` 新增 13 项运行时输出 + `.env` |
| 旧 webhook URL 泄露 | 用户已创建新 webhook (ID: 1491092751926956225)，旧 webhook 需在 Discord 后台删除 |

### .env 加载机制

`daily_run.py` 新增 `_load_dotenv()` 函数：
- 读取项目根目录 `.env` 文件
- 仅设置尚未存在的环境变量（不覆盖已有值）
- 无外部依赖（不需要 python-dotenv）
- `configure_alert_env()` 启动时自动调用

## 三、全面审计修复

### 数据库修复

| 问题 | 修复 |
|------|------|
| positions.entry_date 全部为 NULL | 回填为 2026-04-06（实际建仓日） |
| positions.high_since_entry 部分 NULL | 回填为 market_price |
| decision_journal 4 条重复 | 清空 + 加幂等性守卫（按 asof+strategy_id 去重） |
| factor_registry 低样本因子权重异常 | vol_stability(n=2), sharpe_20(n=2), sortino_60(n=2) 权重归零 |
| 9 个 excluded 因子仍 is_active=1 | 设为 is_active=0 |

### 代码修复

| 问题 | 修复 |
|------|------|
| test_sell_clears_tracking 失败 | 改为跨天 sell（day2 sell day1 position） |
| daily_close.bat 路径错误 + 引用不存在脚本 | git rm 删除 |
| CLAUDE.md signals 表列名错误 | date→asof, signal_mode→reason+version |
| CLAUDE.md 生产信号模式写 ridge | 更正为 sprint_momentum/shadow_hybrid_ic |
| CLAUDE.md + NEXUS_WORKER_CONTRACT.md 路径 | C:\Users\linweiye → E:\AIagent_project_260213 |
| intraday_monitor.py FutureWarning | 修复 pandas Series float() 弃用调用 |

### Schema 新增

`trade_schema.py` 新增 `ensure_decision_journal_table()`:

```sql
CREATE TABLE decision_journal (
  journal_id TEXT PRIMARY KEY,
  asof TEXT NOT NULL,
  ts TEXT NOT NULL,
  strategy_id TEXT DEFAULT 'sprint',
  action_type TEXT NOT NULL,     -- model_follow | model_override | manual_entry
  model_signal TEXT,
  actual_action TEXT NOT NULL,
  override_reason TEXT,
  outcome_pnl REAL,
  outcome_filled_at TEXT,
  compliance_score REAL
);
```

## 四、测试

- **新增**: `tests/test_execution_discipline.py` — 13 个用例
  - TestActionPlanBuilder: 3 tests
  - TestCheckPendingActions: 3 tests
  - TestComplianceTracker: 5 tests
  - TestDecisionJournalTable: 2 tests
- **修复**: `tests/test_risk_management.py::test_sell_clears_tracking`
- **全量**: 78/78 PASS (1.5s)

## 五、Discord 联动验证

发送了 4 条真实 webhook 消息：
1. action_plan_generated (WARNING, 黄色) — 操作计划摘要
2. pending_action_alert (WARNING, 黄色) — 模拟未执行提醒
3. intraday_stop_loss_triggered (ERROR, 红色) — 模拟止损触发
4. webhook_rotation_test (WARNING, 黄色) — 新 webhook 验证

全部成功收到。

## 六、设计文档

- `docs/design/PATCH_2026-04-07_Execution_Discipline_and_Regime_Optimization.md`
  - 六段式交易日调度设计（含时间轴、各时段详细规格、提醒升级策略）
  - 4 个设计方向：执行纪律 / 入场优化 / Regime 分级 / Risk-On 窗口最大化
  
- `docs/tasks/TASKS_2026-04-07_Execution_Discipline_and_Regime_Optimization.md`
  - 阶段零~五任务清单，共 T0~T10
  - 阶段零 (调度) + 阶段一 (纪律) 已完成
  - 阶段二~五 待实施（入场优化、Regime 分级、Kelly 加速、Dashboard）

## 七、新增文件清单

```
action_plan_builder.py          (195 行)
check_pending_actions.py        (191 行)
compliance_tracker.py           (247 行)
intraday_monitor.py             (235 行)
open_confirm.bat
midday_check.bat
pre_close.bat
open_watch.bat
scripts/register_all_tasks.ps1
scripts/register_intraday_tasks.bat
tests/test_execution_discipline.py (261 行)
docs/design/PATCH_2026-04-07_Execution_Discipline_and_Regime_Optimization.md
docs/tasks/TASKS_2026-04-07_Execution_Discipline_and_Regime_Optimization.md
```

## 八、当前持仓状态

| 标的 | 数量 | 成本 | 现价 | 浮盈 | 止损线 | 止损距离 |
|------|------|------|------|------|--------|----------|
| 4005.T | 100 | 528.8 | 533.7 | +0.9% | ~465 (ATR 12% cap) | 12.7% |

NAV: 399,680 | Cash: 346,360 | Regime: off (MA gap -1.9%)

## 九、遗留项

- [ ] 阶段二: 限价建议引擎 (execution_advisor.py)
- [ ] 阶段三: 三级 Regime (benchmark_regime v2)
- [ ] 阶段四: 窗口质量评分 + Kelly 快速积累
- [ ] 阶段五: Compliance Dashboard (Streamlit)
- [ ] 旧 Discord webhook (ID 1490007156299399374) 需在 Discord 后台手动删除
- [ ] quant_briefing.py 补测试（1,217 行零测试）

---

*Git commit: c934af6 — 32 files, +2,577/-276*
