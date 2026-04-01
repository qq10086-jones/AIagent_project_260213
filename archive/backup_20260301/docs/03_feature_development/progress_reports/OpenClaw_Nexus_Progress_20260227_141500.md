# OpenClaw Nexus 项目进展报告 (Execution & Discovery Upgrade)
> **报告时间**: 2026-02-27 14:15
> **项目版本**: `OpenClaw_Nexus_v1.3.2_Execution_Patch`
> **前置里程碑**: `OpenClaw_Nexus_v1.3.1_Hotfix`

## 1. 本次迭代摘要 (Milestone Summary)
本次迭代完成了从“分析中枢”向“执行中心”的关键跨越，重点解决了小额本金（40w日元）下的实战建仓难题，并实现了全球多市场灵活切换。

## 2. 核心改进清单 (Core Enhancements)

### 2.1 执行定价层 (Execution Layer) - Patch v1.2.7
- **ATR 动态定价**: 引入 ATR (Average True Range) 指标，根据市场波动自动计算 Aggressive/Balanced/Patient 三档买卖价格。
- **Tick Size 对齐**: 实现了对日本股市 (TOPIX) 步长规则的自动适配，确保挂单价格合法。
- **价格安全护栏**: 增加了 10% 的偏移量保护，防止算法在极端行情下给出非理性建议。

### 2.2 资金约束层 (Capital Constraint) - Patch v1.2.7/v1.2.9
- **一手/Lot Size 意识**: 深度适配日股“100股一手”规则，计算建议持仓时自动向下取整。
- **本金优先算法**: 重构了股票搜索逻辑，优先过滤掉单价过高（一手即超预算）的标的，确保“所见即买得起”。
- **实时汇率转换**: 增加了 USD/JPY 自动转换逻辑，支持用日元本金实时检索美股机会。

### 2.3 市场切换与搜索增强 - Patch v1.2.8/v1.2.9
- **多市场支持**: `discovery_workflow` 增加 `market` 参数，支持 `JP`, `US`, `ALL` 灵活切换。
- **搜索深度翻倍**: 扫描范围从 Top 20 扩展至 Top 100，并放宽了初始 Alpha Score 过滤门槛，提高了在小本金下的候选股覆盖率。

### 2.4 系统健壮性与交互 (System Robustness)
- **调度记忆 (Session Memory)**: Orchestrator 现在能记住频道内最后讨论的股票代码，支持“那挂多少卖比较好？”等模糊上下文提问。
- **模型全面线上化**: 移除了对本地 Ollama 的依赖，全链路平滑切换至 Qwen (DashScope) 线上 API。
- **Docker 优化**: 增加了构建缓存清除机制 (Cache Busting)，确保本地代码更改能即时部署。

## 3. 验证与验收
- **NTT (9432.T) 测试**: 成功生成了 40w 日元本金下的卖出定价建议，ATR 计算与 0.1 日元步长对齐无误。
- **全市场扫描测试**: 验证了在 40w 日元约束下，系统能自动避开昂贵美股（如 COST），精准定位至低单价优质日股。

## 4. 下一步计划
- [ ] 接入更多日股成份股（如 TOPIX 500），进一步丰富选股池。
- [ ] 优化报告模板，将 `execution_suggestions` 更加直观地展示在 Discord 嵌入卡片中。
- [ ] 探索基于 Redis 的简单 Cash 账户管理，实现初步的模拟实盘跟踪。

---

## 5. 2026-02-27 17:16 增量更新 (Auto-Evolve Discovery)
> Patch: OpenClaw_Nexus_v1.3.4_discovery_adaptive

### 5.1 目标与问题
- 现象：Scanned 27 stocks, found 0 candidates，在 40W 日元约束下经常直接失败。
- 目标：让系统在一次任务中自动多轮尝试，逐步放宽约束，并输出完整尝试轨迹，而不是一次失败即结束。

### 5.2 本次实现
- 在 worker-quant/worker.py 重构 discovery_workflow 为多轮自适应搜索。
- 新增 auto_evolve（默认开启）、max_attempts、min_candidates 控制。
- 尝试策略从 strict 到 relax/fallback，逐轮调整。
- max_position_pct 逐步上调（受上限保护）。
- alpha_floor 逐步放宽。
- 可选自动扩市场（auto_expand_market，如 JP -> ALL）。
- 引入轻量学习记忆（本地 JSON）。
- 新增 DISCOVERY_LEARNING_PATH。
- 记录最近成功参数组合，下一次优先作为 learned_seed 尝试。

### 5.3 输出与可观测性增强
- discovery 输出新增字段。
- search_attempts：每轮参数与候选数量。
- selected_attempt：最终采用第几轮。
- min_candidates_target、auto_evolve、learning_enabled 等状态字段。
- analysis 文本新增“自适应档位被选中”的说明，便于在聊天端解释“为何这次找到或找不到”。

### 5.4 验证与部署
- 语法验证：python -m py_compile worker-quant/worker.py 通过。
- 部署：docker compose -f infra/docker-compose.yml up -d --build worker-quant 完成。
- 运行状态：worker 已正常启动并 ready。

### 5.5 已知事项
- Compose 日志仍提示 DASH_SCOPE_API_KEY / OPENAI_API_KEY 未注入（环境变量链路问题）。
- 该问题不影响 discovery 逻辑执行，但会影响长分析质量与新闻总结深度，需下一补丁统一校验 .env -> compose -> container 注入路径。

---

## 6. 2026-02-27 18:20 增量更新 (Multi-Armed Bandit Learning & UX Fixes)
> Patch: OpenClaw_Nexus_v1.3.5_MAB_Learning

### 6.1 本次优化内容
1. **算法升级 (Thompson Sampling MAB)**
   - 痛点：原先的学习机制为“单点记忆”，容易对最近一次的市场噪音产生过拟合，导致策略在极端保守与宽松之间来回震荡。
   - 解决方案：在 `discovery_workflow` 引入基于 Beta 分布的汤普森采样（Thompson Sampling）多臂老虎机模型（Multi-Armed Bandit）。现在各搜索策略的优先级由历史胜率概率采样动态决定，兼顾探索 (Exploration) 与利用 (Exploitation)。
   - 记忆衰减：增加了衰减因子（0.98），让模型自动逐步遗忘陈旧的市场特征。

2. **UX 修复：报告易读性优化**
   - 痛点：Discovery 任务将系统后台的 metadata 日志当作最终分析报告输出，对非技术人员不友好。
   - 解决方案：重写了 `summary` 字段，现在能输出直观、排版友好的中文选股与配置报告，包含选股名单评分与仓位推演。

3. **UX 修复：新闻抓取本地化**
   - 痛点：盘尾情报与盘后闪讯常常因为 Freshness 过滤器找不到可用新闻，原因是 Google News API 搜索参数硬编码在了英语/北美区，无法抓取日本本地财报新闻。
   - 解决方案：修改 `_fetch_news_from_google_rss` 工具，增加 `lang="ja"` 参数支持切换至日区节点，并将盘后闪讯关键词修改为原生日本财经术语（如 `決算 OR 業績修正`）。

### 6.2 待解决的遗留问题 (待修复)
1. **本金与资产错配问题**：执行“用40W日元，目标3个月增值10%以上，帮我找JP可建仓标的并给分批建仓计划”任务时，依然找出了美股（纳斯达克）标的。货币未对齐，美股是美元计价，导致按日元预算根本买不起。
2. **新闻因子的闭环缺失**：目前抓取的新闻信息是否已经作为数据真正反馈到量化模型池子中以提供因子计算，逻辑链条尚需确认和完善。
3. **策略结果的 Quant 模型验证**：Discovery 给出的基于目标收益（3个月增值10%）和资金计划的股票，其结果是否已经经过了 Quant 核心模型的严格历史回测或量化指标验证，目前尚未明确落实。