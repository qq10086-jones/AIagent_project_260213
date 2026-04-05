# PATCH: Accelerated Simulated Forward Run

## Implementation Status

Status as of `2026-04-05`:

- implemented
- unit-tested
- validated with compressed-time end-to-end runs

Implemented surfaces:

- `simulation_clock.py`
- `simulate_forward_run.py`
- `daily_run.py` logical-date injection and simulation event tagging
- isolated simulation `state/`, `reports/`, and `artifacts/` roots
- strict-PIT news filtering for Sprint shadow evaluation
- `evaluate_promotion.py` evidence-type labeling

## 1. 背景

当前 quant 项目里有两类明确依赖自然时间累积的门槛：

- `30` 天 news shadow 观察窗口
- `30` 天 Sprint paper evidence

这两类门槛已经有代码和产物跟踪，但按自然日推进会拖慢验证节奏。为缩短验证周期，同时保留“逐日累积状态”的特征，本 patch 提议引入一套 `模拟时钟 + 压缩时间推进` 机制。

目标不是把历史回放伪装成真实生产，而是提供一套更强的工程验证模式：

- 比一次性 backfill 更接近真实逐日运行
- 能验证状态累积、冷却期、审计记录、日报和告警链路
- 能在较短墙钟时间内形成 `30+` 个逻辑交易日证据

---

## 2. 目标与非目标

### 2.1 目标

- 允许 quant 假定“今天”是配置指定的逻辑日期
- 允许逻辑日期按交易日推进，而不是按真实系统时间推进
- 支持 `1 分钟 = 1 个交易日` 或 `0 秒 = 1 个交易日` 的压缩执行
- 让 `daily_run.py` 链路在每个逻辑交易日真实执行一次
- 将运行产物、账户状态、审计状态与生产目录隔离
- 产出可供治理层阅读的 `simulated_forward` 证据

### 2.2 非目标

- 不修改现有 alpha 逻辑、选股逻辑和风险控制核心定义
- 不把压缩时间运行直接宣称为“自然时间前瞻证据”
- 不默认改变当前生产运行路径
- 不在第一版里处理实盘券商接入

---

## 3. 总体方案

引入统一的“逻辑时钟”层。当 simulation 开启时，所有日期敏感逻辑都不再直接读取真实系统时间，而是读取 `simulation_clock` 当前给出的逻辑交易日。

执行模式如下：

1. 配置一个逻辑起点，例如 `2026-02-01`
2. 根据交易日历推导出下一个有效交易日
3. 每个 tick 执行一次完整日常链路
4. tick 完成后，把逻辑日期推进到下一个交易日
5. 持续推进直到达到 `end_asof` 或达到目标观测天数

这个模式本质上是：

- `逐日推进`
- `状态真实累积`
- `时间被压缩`

而不是一次性把整段历史批量算完。

---

## 4. 核心设计原则

### 4.1 单一时钟原则

simulation 开启后，所有使用“今天”“最近一天”“距上次 review 几天”的逻辑，都必须统一使用逻辑时钟。

禁止出现以下混用：

- `daily_run.py` 用模拟日期
- `compute_ic.py` 仍调用 `date.today()`
- `paper_execute.py` 用真实系统日期写入 paper 账本

只要混用，这份证据就不可信。

### 4.2 PIT 原则

逻辑日期为 `D` 时，系统只能看到 `D` 当日应当可见的数据。

最低要求：

- 价格数据：`trade_date <= D`
- 基本面数据：只能使用 `available_ts <= D` 的版本
- 新闻数据：只能使用在 `D` 的策略决策截点之前已经发布且已抓到的数据
- 评估结果：若引用前序产物，只能引用 `< D` 的产物

否则就会变成带未来信息污染的 replay。

### 4.3 生产隔离原则

simulation 产物不能默认覆写生产目录。

建议使用独立目录：

- `reports/simulated_forward/`
- `artifacts/simulated_forward/`
- `state/simulated_forward/`
- 可选独立 DB：`japan_market_sim.db`

### 4.4 可恢复原则

simulation 一定要支持断点续跑。

如果逻辑时间跑到第 `12` 个交易日时中断，系统应能从最近完成日继续，而不是从头重跑。

---

## 5. 配置扩展

建议在 `config.yaml` 中新增：

```yaml
simulation:
  enabled: false
  mode: "accelerated_forward"
  start_asof: "2026-02-01"
  end_asof: "2026-03-31"
  tick_seconds: 60
  trading_day_per_tick: 1
  strict_pit: true
  resume: true
  use_cloned_db: true
  cloned_db_path: "japan_market_sim.db"
  reports_dir: "reports/simulated_forward"
  artifacts_dir: "artifacts/simulated_forward"
  state_dir: "state/simulated_forward"
  stop_on_error: true
```

说明：

- `tick_seconds=60` 表示 `1 分钟推进 1 个交易日`
- `tick_seconds=0` 表示无等待快速推进
- `use_cloned_db=true` 用于保护生产 DB
- `strict_pit=true` 表示未满足 PIT 条件时直接报错终止

---

## 6. 新增模块

### 6.1 `simulation_clock.py`

职责：

- 维护当前逻辑交易日
- 基于交易日历推进到下一个交易日
- 支持状态持久化和恢复

建议接口：

```python
class SimulationClock:
    def __init__(self, start_asof, end_asof, trading_dates, state_path):
        ...

    def current_asof(self) -> str:
        ...

    def advance(self, n: int = 1) -> str | None:
        ...

    def is_finished(self) -> bool:
        ...

    def save(self) -> None:
        ...
```

### 6.2 `simulate_forward_run.py`

职责：

- 读取 `simulation` 配置
- 准备独立输出目录
- 视需要克隆 DB
- 初始化/恢复 `SimulationClock`
- 循环调用日常运行主链
- 记录每个逻辑交易日的运行结果

### 6.3 `simulation_state.json`

建议状态格式：

```json
{
  "mode": "accelerated_forward",
  "started_at_utc": "2026-04-05T01:00:00Z",
  "last_completed_asof": "2026-02-18",
  "completed_days": 12,
  "failed_days": 0,
  "reports_dir": "reports/simulated_forward",
  "artifacts_dir": "artifacts/simulated_forward",
  "db_path": "japan_market_sim.db"
}
```

---

## 7. 需要修改的现有模块

### 7.1 `daily_run.py`

需要支持：

- 接受 `asof_override`
- 接受输出目录重定向
- 在运行事件中写入：
  - `simulation`
  - `simulation_mode`
  - `simulation_asof`

建议做法：

- 新增参数 `--asof_override`
- 若存在 simulation 配置，则优先使用逻辑日期而不是自动探测“今天”

### 7.2 `compute_ic.py`

当前已有 `review_frequency_days` 执行逻辑，但 review 日期依赖“当前日期”的语义仍需要和逻辑时钟对齐。

需要支持：

- tier review 基于逻辑日期推进
- `learning_audit` 中记录模拟日期
- 输出里区分 `simulated_forward` 与自然时间

### 7.3 `sprint_signal.py`

需要支持：

- news shadow 观察天数按逻辑日期推进
- shadow 观察记录写入 simulation 隔离状态
- prior regime / prior evaluation 优先读取 simulation 目录

### 7.4 `paper_execute.py`

需要支持：

- paper 账户状态使用 simulation 独立账本
- 交易记录、资金曲线、持仓快照与生产隔离
- 支持用逻辑日期作为执行日写入

### 7.5 `evaluate_promotion.py`

需要支持把证据明确分层：

- `natural_time`
- `simulated_forward`

建议输出字段：

```json
{
  "paper_evidence_type": "simulated_forward",
  "simulated_forward_days": 30
}
```

### 7.6 告警链路

simulation 运行也应保留告警，但必须显式标识不是生产运行。

建议新增 payload 字段：

- `simulation: true`
- `simulation_mode: accelerated_forward`
- `simulation_asof: 2026-02-18`

---

## 8. 数据与 PIT 约束

这是本 patch 是否可信的关键部分。

### 8.1 价格数据

simulation 日 `D` 运行时：

- 仅允许读取 `<= D` 的价格
- 不允许在 `D` 的运行里使用 `D+1` 数据

若本地 DB 已包含未来日期数据，则读取层必须加裁剪。

### 8.2 新闻数据

如果 news overlay 作为 gate 或权重输入，则需至少同时约束：

- `published_ts`
- `ingested_ts`

也就是：

- 在逻辑日 `D` 的决策截点前发布
- 且在决策截点前已经被系统抓到

否则 shadow 观察和 gating 评估会被未来新闻污染。

### 8.3 基本面数据

基本面最容易出 PIT 问题。

需要保证：

- 使用带 `available_ts` 或等价口径的数据
- 若当前管道无法保证 PIT，则 simulation 模式下应把相关因子标为只读历史分析，不参与 promotion 证据

---

## 9. 产物设计

建议新增以下产物：

- `reports/simulated_forward/simulation_state.json`
- `reports/simulated_forward/simulation_summary.json`
- `reports/simulated_forward/news_shadow_evaluation.json`
- `reports/simulated_forward/runtime_alert_status.json`
- `artifacts/simulated_forward/<asof>/<run_id>/decision_snapshot.json`

建议在 `simulation_summary.json` 汇总：

- 起止逻辑日期
- 已完成逻辑交易日数
- shadow 已累计天数
- paper 已累计天数
- 是否达到 gating review 门槛
- 是否出现失败日

---

## 10. 治理口径

这套模式不能直接写成“已完成真实 30 天生产前瞻验证”。

建议在治理文档里明确分成两类：

- `simulated_forward_evidence`
- `natural_time_evidence`

推荐用途：

- `simulated_forward_evidence`
  - 用于工程验证
  - 用于加速发现逻辑缺陷
  - 用于提前形成阶段性证据

- `natural_time_evidence`
  - 用于最终 promotion / live readiness 盖章

如果治理层同意，也可以定义一条折中规则：

- 压缩时间证据可视为“预审通过”
- 仍需补足少量自然时间运行天数后，才可正式晋升

---

## 11. 实施阶段

### Phase 1

- 新增 `simulation` 配置
- 新增 `simulation_clock.py`
- `daily_run.py` 支持 `asof_override`
- 输出目录可重定向

### Phase 2

- `simulate_forward_run.py` 驱动脚本落地
- `paper_execute.py` 隔离 simulation 账本
- `sprint_signal.py` 切换到 simulation 状态目录

### Phase 3

- 完成价格/新闻/基本面 PIT 裁剪校验
- `evaluate_promotion.py` 区分证据类型
- 增加 simulation 汇总报告

### Phase 4

- 增加端到端测试
- 增加 `30` 天压缩运行 smoke test
- 在治理文档中写明 acceptance 口径

---

## 12. 验收标准

当以下条件满足时，可认为本 patch 达到可用状态：

- 能从指定 `start_asof` 启动 simulation
- 能按交易日推进直到 `end_asof`
- 中断后能从最近完成逻辑日恢复
- `daily_run` 每个逻辑交易日只使用 `<= 当前逻辑日` 的数据
- paper、reports、artifacts 不污染生产目录
- `news shadow` 和 `paper days` 能在 simulation 中逐日累计
- 告警和运行事件能明确标记 `simulation=true`
- 汇总报告能清楚区分 `simulated_forward` 与 `natural_time`

---

## 13. 风险与限制

- 若 PIT 裁剪不严格，这套模式会退化成“带未来信息的历史回放”
- 若仍有模块偷偷读取系统时间，30 天累计结果不可信
- 若 simulation 与 production 共用状态目录，会污染现有审计口径
- 若治理文档不区分证据类型，容易造成“误把压缩运行当真实前瞻”的管理风险

---

## 14. 建议结论

建议把本 patch 定位为：

- `工程验证加速器`
- `压缩时间前瞻模拟`
- `不替代自然时间最终盖章`

这能在不破坏现有生产逻辑的前提下，显著缩短 30 天类门槛的验证时间，并为后续 promotion 评审提供更强的工程证据。
