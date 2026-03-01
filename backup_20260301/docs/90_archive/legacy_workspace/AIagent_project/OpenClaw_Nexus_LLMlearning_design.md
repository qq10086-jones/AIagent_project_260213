# 项目级“越用越匹配需求”的零预算学习闭环设计文档（低延迟优先）

> 目标：在**尽量不影响本地推理速度**的前提下，让你的本地 AI（deepseek-r1:32b 为主）随着使用持续变得更“懂你的项目”、更贴合你的交付习惯与硬约束。  
> 核心手段：**在线偏好学习（Preference Learning） + 规则/记忆更新（Rule & Memory Writeback） + 回放评测闭环（Replay Evaluation）**，并引入 **强化学习（Bandit/RL）** 与 **贝叶斯优化（BO）** 用于“系统层决策与超参调优”，而不是训练大模型本体。

---

## 1. 设计约束与原则

### 1.1 刚性约束
- **零预算**：不使用付费云 GPU。
- **硬件受限**：本地能推理 32B，但不保证能训练 32B。
- **低延迟底线**：不能因为学习机制导致生成明显变慢或卡顿。
- **项目匹配优先**：学习对象必须围绕“项目交付成功率/一致性”，而不是泛聊天人格。

### 1.2 关键原则（速度优先）
1. **学习发生在“系统层”**（模板选择、工具链、记忆注入、检索策略、输出结构），避免对 32B 做在线训练。
2. **注入短上下文**：Profile/规则摘要控制在 100–300 tokens；检索片段 top_k 3–5，每段截断。
3. **写入门控**：只有“验收通过/高价值”内容才进入长期记忆；避免 RAG 垃圾膨胀导致变慢。
4. **记录与更新不阻塞**：实现上可“先交付、后写回”，但对用户体验表现为“即时交付”。

---

## 2. 总体架构（四层）

```text
User
  |
  v
[Project Router]  —— 选择项目、任务类型、动作集合（模板/策略/工具）
  |
  v
[Context Builder]
  - 硬规则注入（短）
  - Profile Memory 注入（短）
  - Episodic RAG 检索（少量）
  |
  v
[LLM Inference]
  - deepseek-r1:32b (主)
  - 可选小模型：用于“风格/格式润色”或“快速草案”
  |
  v
[Post-Processing]
  - 规则校验（硬约束）
  - 失败自动重写（最多1次，避免拖慢）
  |
  v
Deliverable（命令/补丁/MD/邮件/方案）
  |
  v
[Feedback Collector] -> ✅/❌/评分/A-B选择 + 一句原因
  |
  v
[Learning & Writeback]
  - 在线偏好学习（轻量）
  - 规则/记忆更新
  - 回放评测 + 指标看板
  - BO 周期性调参
```

---

## 3. 核心对象：按“项目”组织学习（而非按人设）

### 3.1 项目（Project）与作用域（Scope）
- **Global**：全局偏好（如“不要反复确认”“给可执行步骤”）
- **Project**：仅对某项目生效（quant/openclaw/comfyui/商务日语/光伏…）
- **Task**：仅对当前任务/子任务生效（一次性约束）

### 3.2 Project Profile（结构化、非向量化）
**目标**：把你的“项目习惯/硬约束/验收标准”变成可执行配置，做到“自动注入且极短”。

字段建议：
- `project_id`
- `goal`：项目目标（交付物类型）
- `hard_constraints[]`：硬约束（不可违反）
- `environment`：OS/路径/容器名/端口/显卡/关键版本
- `deliverable_templates[]`：可选输出模板集合（动作空间的一部分）
- `acceptance_criteria[]`：验收标准（用于奖励与回放评测）
- `known_pitfalls[]`：踩坑与规避（只收敛记录“确认过”的）
- `style_prefs`：输出偏好（简洁、一步一条命令、先结论…）

**注入策略**：每次对话只生成 `profile_digest`（100–300 tokens），把硬约束放最前。

---

## 4. 在线偏好学习（Preference Learning）设计（低成本）

> 目标：让系统在同一个项目里越来越会“选对模板、给对步骤粒度、少犯你讨厌的错”。

### 4.1 反馈形式（越轻越好）
优先级从高到低：
1. **二元反馈**：✅/❌ + 原因（最省事）
2. **1–5 分**：满意度
3. **A/B 选择**：你选更好的候选（最适合偏好学习，但成本略高）

### 4.2 偏好模型（轻量，不训练 LLM）
使用 Bradley–Terry / Logistic 偏好学习：
- 特征：`φ(x, y)`（上下文 x 与回答 y 的“可解释特征”）
- 参数：`w`（小维度向量，几十到几百维）

A/B 场景：

\[
P(y^+ \succ y^-|x) = \sigma \Big(w^\top(\phi(x,y^+) - \phi(x,y^-)) \Big)
\]

在线更新（SGD）：

\[
w \leftarrow w + \eta \cdot (1-\hat{P})\cdot(\phi(x,y^+) - \phi(x,y^-))
\]

✅ **为什么不拖慢**：特征抽取与打分都在毫秒级，不触碰 32B 训练。

### 4.3 特征 φ（必须项目化）
示例（可直接落地成规则检测/正则/轻量解析）：
- **约束遵守特征**
  - 是否出现 PowerShell（openclaw 项目禁止）
  - 是否修改“核心算法文件”（quant 禁止）
  - 是否提到 AMD 不兼容风险/替代方案（需要时加分）
- **可执行性交付特征**
  - 是否包含命令行代码块
  - 是否包含“预期输出”
  - 是否包含失败分支（错误时下一步怎么做）
- **上下文控制特征**
  - 输出长度（过长扣分）
  - 是否按模板结构（标题/步骤/回滚）
- **检索质量特征**
  - 引用的记忆片段是否来自“已验收 SOP”
  - 引用片段数量是否超标（超标扣分）

> 关键点：**能被机器快速判定**，不需要再调用 LLM。

---

## 5. 规则/记忆更新（Writeback）机制：让系统“越用越干净”

### 5.1 规则类型
- **硬规则（Hard Rules）**：违反直接判失败并触发重写  
  - 例：openclaw 输出命令必须是 cmd；quant 禁止改核心算法
- **软规则（Soft Rules）**：用于偏好打分与模板选择  
  - 例：更喜欢“一步一步命令”；更喜欢“先结论再解释”

### 5.2 记忆分层（推荐）
1. **Profile Memory（结构化）**：规则与偏好（SQLite/JSON）
2. **Episodic Memory（向量库）**：可复用 SOP、最终命令、最终模板（Chroma/FAISS）
3. **Working Memory（会话）**：当前任务状态（短）

### 5.3 写入门控（Memory Gating）
核心：不是“记得越多越好”，而是“只记得该记的”。

对每条候选记忆 m 维护“有用概率”：
- 若反馈为 ✅ 且 m 被用于本次交付：提升 m 的可信度
- 若反馈为 ❌ 或 m 导致偏差：降低权重或拒绝写入

最简实现（Beta-Bernoulli）：
- \(p_m \sim \text{Beta}(\alpha_m, \beta_m)\)
- ✅：\(\alpha_m += 1\)
- ❌：\(\beta_m += 1\)
- 只有当 \(\mathbb{E}[p_m] = \alpha/(\alpha+\beta) > \tau\) 才晋升为长期记忆（SOP）

✅ **性能好处**：长期记忆库不会被噪声污染，从源头避免“越用越慢、越用越偏”。

---

## 6. 回放评测闭环（Replay Evaluation）：用数据让它持续变好

### 6.1 为什么必须回放
你要“匹配项目需求”，就需要指标化：
- 是否一次成功（One-shot success）
- 是否违反硬约束（Hard violation count）
- 平均响应时延（Latency）
- 检索注入 tokens（Context bloat）
- 你满意度（Reward）

### 6.2 回放数据结构（Trace）
每次交付记录：
- `project_id`, `task_type`
- `context_digest`（短摘要）
- `actions`（选择了哪个模板/检索策略/是否重写）
- `retrieval_refs`（用了哪些记忆片段）
- `output_metrics`（长度、是否含命令块、是否含预期输出）
- `feedback`（✅/❌/评分/原因）
- `latency_ms`, `tokens_in`, `tokens_out`

### 6.3 回放方式
- **离线回放**：每 N 次交互，对过去样本进行“策略对比”（不同模板/不同 top_k）
- **反事实评估（轻量）**：不需要再跑 32B，只评估“规则/特征层”的差异（速度快）
- **小规模真实回放**：仅对少量关键任务重跑（控制成本）

---

## 7. 强化学习怎么用（不训练大模型）：Bandit/RL 控制“系统决策”

> 你的 RL 不是让 32B 变聪明，而是让系统越来越会**选择正确的交付策略**。

### 7.1 首选：上下文老虎机（Contextual Bandit）
适用：每次交互主要是“选一个策略/模板/工具链”，反馈马上到。

- 上下文 \(x\)：项目+任务类型+约束+历史偏好
- 动作 \(a\)：选择模板、步骤粒度、是否先给命令、是否检索、top_k 桶
- 奖励 \(r\)：✅/❌ 或评分 - 延迟惩罚

目标：

\[
\max \mathbb{E}[r|x,a]
\]

**策略**：Thompson Sampling（贝叶斯 bandit，样本效率高）
- 每个动作在特定上下文簇下维护后验
- 采样后验选动作
- 根据反馈更新后验

### 7.2 何时升级到“真正的 RL（MDP）”
适用：多步流程（debug/排障/复杂规划）需要“先查什么后做什么”的序列决策。

- 状态 \(s_t\)：已获得的信息、上一步命令输出摘要
- 动作 \(a_t\)：下一步收集哪类信息/给哪条命令/是否询问一个关键问题
- 奖励：最终是否解决 + 步数惩罚 + 延迟惩罚

实践建议（低成本）：
- 先用 **规则流程图** 做基线
- RL 只做“分支选择/下一步选择”，不生成内容
- 训练数据来自 Trace（离线 RL / imitation + bandit）

---

## 8. 贝叶斯优化怎么用：调“系统超参”，提升成功率并压住延迟

### 8.1 适用对象（强烈建议）
- RAG：`top_k`, `chunk_size`, `similarity_threshold`, `max_inject_tokens`
- 记忆门控：`τ`（晋升 SOP 的阈值）
- 重写策略：是否允许重写、重写次数（最多 1）
- 模板长度约束：`max_steps`, `max_tokens_out`
- 升级策略：何时需要额外上下文/更多日志

### 8.2 目标函数（包含延迟惩罚）
定义：

\[
J(\theta) = \text{SuccessRate}(\theta) - \lambda \cdot \text{Latency}(\theta) - \mu \cdot \text{ContextTokens}(\theta)
\]

- SuccessRate：✅ 比例（可按项目加权）
- Latency：P95 延迟（更贴合体验）
- ContextTokens：注入 tokens（控制 KV cache 压力）

### 8.3 运行频率（不影响日常）
- 每 50～200 次交互跑一次 BO，或每周一次
- 优先在离线回放样本上评估，少量真实重跑（可选）

---

## 9. 低延迟实现细则（默认建议）

### 9.1 强制上限
- `profile_digest_tokens <= 300`
- `rag_top_k <= 5`
- `rag_inject_tokens <= 800`（按机器再调）
- `rewrite_max = 1`（规则违规才触发）
- 工具调用尽量少；debug 类任务再开

### 9.2 预检与后检（不依赖 LLM）
- 输出后做 **regex/静态检查**：
  - 是否出现禁用命令（PowerShell 等）
  - 是否出现“修改核心算法文件”的建议（按路径/文件名单）
  - 是否缺少“预期输出”（openclaw）
- 违规则触发**一次重写**：提示中加入“必须修正点”（短）

### 9.3 缓存策略（本地、可控）
- 缓存 `profile_digest`（按 project_id）
- 缓存向量检索结果（按 query hash，短时有效）
- 缓存“常用模板骨架”

---

## 10. 数据存储设计（SQLite + 向量库）

### 10.1 SQLite 表（建议）
- `projects(project_id, name, profile_json, updated_at)`
- `rules(rule_id, project_id, scope, rule_type, rule_json, weight, updated_at)`
- `mem_items(mem_id, project_id, type, content, tags, alpha, beta, created_at)`
- `traces(trace_id, project_id, task_type, context_digest, action_json, metrics_json, feedback_json, created_at)`
- `bandit_stats(project_id, context_bucket, action_id, alpha, beta, updated_at)`
- `bo_runs(run_id, project_id, theta_json, score, created_at)`

### 10.2 向量库（Chroma/FAISS）
仅存：
- “验收通过的 SOP”
- “最终可复用命令序列/补丁说明”
- “确认过的邮件模板/报告模板”
并带元数据：
- `project_id`, `task_type`, `sop_level`, `trust_score`（可由 alpha/beta 计算）

---

## 11. 在线学习流程（端到端）

### 11.1 生成前
1. Router 判定 `project_id`、`task_type`、`context_bucket`
2. Bandit/偏好模型选择动作 `a`：模板 T、检索策略 R、步骤粒度 G
3. Context Builder 组装短上下文：硬约束 + profile_digest +（可选）rag_snippets

### 11.2 生成后
4. 规则校验：违规 -> 触发一次重写（带“修正点”）
5. 输出交付物

### 11.3 反馈后（学习与写回）
6. 记录 trace
7. 更新：
   - bandit 后验（动作选择更准）
   - 偏好模型参数 w（软偏好更准）
   - 规则库（把 ❌原因 转为硬/软规则）
   - 记忆门控（alpha/beta 更新，决定是否晋升 SOP）
8. 周期性（离线）：回放评测 + 贝叶斯优化调参（θ）

---

## 12. 最小可行版本（MVP）建议

### MVP-1：规则 + 写回 + trace（立刻见效）
- 项目 Profile（结构化）
- 硬规则校验（regex）
- trace 记录
- ✅/❌ 写回（把原因写成规则）

### MVP-2：加入 bandit（Thompson Sampling）
- 动作：模板选择（2–5 个模板）
- 奖励：✅/❌
- 后验：Beta(α,β)

### MVP-3：加入 BO（离线）
- 优化 RAG 超参/注入长度/阈值
- 指标：成功率 - 延迟惩罚

---

## 13. 最省事的反馈协议（建议你默认就这么回）
- `✅ 保存为SOP`
- `✅ 但更偏好：先给结论`
- `❌ 原因：用了PowerShell（openclaw 禁止）`
- `❌ 原因：建议改核心算法（quant 禁止）`
- `A更好` / `B更好`

系统据此完成：规则更新 + 记忆门控 + bandit 更新。

---

## 14. 风险与对策
- **记忆膨胀导致变慢**：写入门控 + 注入上限 + 只存“验收通过”
- **跨项目串味**：严格作用域（Global/Project/Task）+ project_id 绑定
- **学习噪声**：Beta 后验鲁棒；规则升级可要求“连续两次同类 ❌”
- **重写拖慢**：只在硬规则违规触发；最多 1 次；提示短

---

## 15. 你会得到的最终效果
- 同一项目下：输出模板越来越贴合（步骤粒度、结构、语言风格）
- 一次跑通率提高，重复踩坑显著减少
- 通过短注入 + 门控 + BO，延迟稳定，不随使用时间显著恶化
- 学习对象是“项目交付”，不是泛聊天人格

---

## 16. 实施细则与优化补充（三步走战略）

### A. 实施建议
*   **第一阶段（MVP-1：确定性规则）**：
    *   在 `orchestrator` 中引入 `Project Profile` (JSON)。
    *   实现**硬规则校验（Post-Processing）**：比如检测到 OpenClaw 任务中出现 `powershell` 字样，直接拦截并触发一次重写提示。
    *   开始在 SQLite 中记录 `trace`（原始请求、LLM 输出、用户反馈）。
*   **第二阶段（MVP-2：经验记忆与 SOP）**：
    *   集成 `Memory Gating`：只有被用户标记为 ✅ 的交付物才存入向量库。
    *   优化 `Context Builder`：根据任务类型自动注入最相关的 3-5 条 SOP。
*   **第三阶段（MVP-3：策略选择优化）**：
    *   引入 `Contextual Bandit`：当你有多个输出模板时，让系统根据历史成功率自动选择（比如针对“报错修复”任务，系统会自动选择“先分析日志再给补丁”的模板）。

### B. 细节优化建议：
1.  **增加“环境感知”自动写回**：除了人工 ✅/❌，可以增加**执行反馈写回**。例如：LLM 给出了一条命令，用户在终端运行成功了（或者 `worker-quant` 任务成功完成），自动标记该条目为高权重记忆。
2.  **利用“思考过程”进行自我修正**：既然 DeepSeek-R1 有 `<think>` 过程，可以在 `Post-Processing` 中让小模型（如 GLM-4.7-Flash）快速二次检查思考逻辑是否违反了硬约束，如果违反，在正式回答前拦截。
3.  **冷启动方案**：针对新项目，预置一批“种子规则”（Seed Rules），防止学习初期系统表现太随机。