# OpenClaw Nexus 项目进展报告 (Multi-Agent & Quant Upgrade)
> **报告时间**: 2026-02-26 18:25
> **项目版本**: `OpenClaw_Nexus_v1.3_MAS`

## 1. 本次里程碑摘要 (Milestone Summary)
本次迭代完成了项目的**核心架构蜕变**：从“基于正则的单线程工具箱”彻底升级为**“大模型驱动路由、多 Agent 协同拆解、包含前沿新闻门控量化模型 (SS7)”**的成熟 Multi-Agent 投研系统。

系统现在不仅具备闲聊能力，更能在接收到“推荐股票”指令时，自动执行宏观情报收集、NLP 新闻风险打分、Feature Store 特征持久化以及 SS7 资金/新闻自适应量化回测的完整流水线。

---

## 2. 核心架构重构与落地 (Core Architecture Upgrades)

### 2.1 Nexus Copilot (LLM 路由中枢) 落成
- **彻底废弃死板的正则分发 (`extractTicker`)**：重构了 `orchestrator/src/nlp/router.js`。
- **纯语义路由 (Semantic Router)**：大模型接管控制权。现在能精准区分 `CHAT`（日常问答，不调工具）和 `RUN`（明确的量化执行需求）。
- **工具描述赋能 (Tool Schema)**：将所有 Agent 技能注入到系统 Prompt 中，LLM 可自主决策调用哪个工具。
- **模型热切换指令**：新增 `/model <model_name>` 指令，支持在 Discord 无缝切换底层大模型（如 `qwen-max`），无需重启容器。

### 2.2 Brain (大脑) 任务动态拆解能力 (Task Decomposition)
- **多状态流转**：扩充了 `brain/state.py`，支持 `mode: discovery`（市场扫描）和 `mode: analysis`（单票分析）双模式。
- **动态 Agent 委派**：在 `discovery` 模式下，大脑能将任务自动拆解为三步：
  1. `Intelligence Agent` (新闻收集)
  2. `Screener Agent` (执行专业量化模型)
  3. `Writer Agent` (统筹证据撰写最终报告)
- **超时与兜底保护**：大幅上调了大脑对子 Agent 执行耗时的容忍度（Timeout 延长至 240 秒），确保复杂模型有充足时间跑完。

---

## 3. 专业量化引擎升级 (Quant Engine Enhancements v1.1 - v1.3)

### 3.1 引入 Feature Store 与账户状态 (Account Model)
- **特征持久化**：在底层 `japan_market.db` 中新建 `feature_daily` 表，确保所有量化特征（如新闻风险因子）可回溯、可复现。
- **资金本金管理**：新建 `account_state` 表，记录 `starting_capital` 和 `equity`。
- **手动录入接口**：在 Discord 侧和工具库中新增 `portfolio.set_account` 与 `portfolio.record_fill`，支持用户自然语言录入本金和购入动作（例：“帮我把本金设置为两千万日元”）。

### 3.2 落地新闻风险因子模型 (News Risk Factor & SS7)
- **双重兜底搜集**：当 GDELT 出现断流或全空时，系统自动 fallback 到 Google News RSS，确保分析不中断。
- **新闻时效性风控 (Freshness Guard)**：实现了 `lag_p50` 与 `staleness_flag`。对于过期陈旧新闻，自动实施权重衰减（0.5 或 0.2）。
- **LLM 多维风险打分**：系统每日自动对新闻文本进行大模型情感分析，生成 `sentiment`、`uncertainty` 和 `severity` 评分，并合成 `news_risk_raw`。
- **SS7 模型无缝接管**：量化管线底层的执行脚本已从 `ss6` 升级为用户提供的 `ss7_sqlite_news_overlay.py`。该模型以“外层门控 (Outer Gating Mechanism)”的方式安全地融合了新闻风险因子，在新闻情绪极度不稳或关注度过高时，能自动触发风控缩减仓位（g值衰减）。

### 3.3 日本市场特定时点情报 (TSE Pre-Close)
- 适配日本东京证券交易所（TSE）最新的延长交易时间（15:30 收盘）。
- **15:15 JST 盘尾主推** (`news.preclose_brief_jp`)：捕获当天最后窗口的资金面与消息面动向。
- **15:35 JST 盘后闪讯** (`news.tdnet_close_flash`)：专门盯防收盘后 TDnet 的重要突发公告。
- 这两项简报现在均由 Qwen 旗舰大模型进行专业中文润色，并在末尾附带原始新闻 URL，实现了专业度与可验证性（防幻觉）的统一。

---

## 4. UI 与可视化体验 (Dashboard & UI)
- **MinIO 资产预览**：重构了 Streamlit `ui/app.py` 的 Asset Vault 页面。现在生成的精美 HTML 报告可以利用预签名 URL（Presigned URL）在网页面板中直接以 IFrame 形式在线预览，做到了“打开就能看”。

---

## 5. 一键部署与启动优化 (DevOps)
- **Windows 一键拉起**：编写了 `start_nexus.bat` 脚本。它能自动检测 Docker 环境和宿主机 Ollama 状态，并在完成校验后一键拉起包括 Orchestrator、Brain 和 Worker 及其关联数据库在内的全链条容器网络。

---

## 6. 下一步演进建议 (Next Steps / Backlog)
1. **真实持仓 PnL 监控**：基于目前已建好的 `account_state` 和 `fills` 表，开发 `portfolio_daily_monitor` 每日自动推算真实浮亏，并在 Discord 发送组合日报。
2. **多模态取证增强**：完善 Browser Agent 在风险爆发时自动抓取目标公司 IR 页面或 K 线截图的能力，作为强视觉证据归档。
3. **SS7 模型参数调优**：观察新版“新闻门控量化模型”在未来几周的实盘模拟表现，优化衰减半衰期 (`half_life_days`) 和不确定性惩罚项 (`k_U`)。