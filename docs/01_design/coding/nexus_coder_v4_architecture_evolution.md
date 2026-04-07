# Nexus Coder v4.0 架构演进设计文档：从“流水线”到“自主智能体”

> **状态**: PROPOSED (架构委员会评审中)  
> **日期**: 2026-04-07  
> **作者**: 资深 PM & 架构团队  
> **目标**: 完全对标并超越 Claude Code，实现动态反应式、语义感知、微观调控的 AI 原生编码引擎。

---

## 一、 核心痛点与演进哲学 (The Paradigm Shift)

基于 v3.1 版本的深度审计与 `claude_code_src` 源码的全面对标，当前 Nexus Coder 的核心痛点在于**“过度管控导致的智力降级”**。我们构建了一个严密的工业流水线（8步写死的工作流 + 43个JSON Schema），但丧失了 AI Agent 原生的灵动性与自愈能力。

如果把 Claude Code 比作**自动驾驶的特斯拉**，Nexus Coder 现状更像是一个**带有自动导航功能的传送带工厂**。

| 维度 | Nexus Coder v3.1 (现状) | Claude Code (对标) | Nexus Coder v4.0 (超越目标) |
| :--- | :--- | :--- | :--- |
| **执行模型** | 瀑布流 (PM->Arch->Impl->QA) | 动态树 (目标导向，按需拆解) | **反应式图计算 (Reactive DAG) + 专家网络** |
| **上下文传递** | 传真机模式 (强类型 JSON Schema) | 终端会话上下文 + 隐式记忆 | **全息语义索引 (AST/LSP RAG) + 跨态图谱** |
| **工具交互** | 黑盒委托 (等 10 分钟看全量结果) | 微操交互 (频繁 ls, grep, 读写) | **高频微操 + 宏观委托混合 (Hybrid)** |
| **错误恢复** | 事后质检 + 盲目整体重试 | 运行时 REPL 报错即修 | **毫秒级极速热修 (Hot-fix) + 沙盒快照回滚** |

---

## 二、 核心架构重构设计 (Core Architecture Design)

为了实现上述跃迁，打赢 Claude Code，Nexus 必须在底层架构上动刀，引入三大核心子系统：

### 1. 神经中枢：动态任务树 (Dynamic Task Graph Engine)
打破现有的硬编码工作流，将编排器从“发号施令者”变为“自适应调度中心”。
*   **设计**: 引入 `Agentic Loop`。系统启动时只有一个宏观目标。
*   **机制**: Orchestrator 具备 `SpawnTask` 和 `Yield` 的能力。在执行 `impl_be` 时，如果 Agent 发现依赖缺失，它不是直接报错失败，而是**动态生成一个 `install_dependency` 的子节点**，完成后继续原任务。
*   **超越点**: Nexus 原生支持多 Worker 分布式架构。当任务树发生分叉时（如前后端互不依赖），Nexus 可**并行调度多个微 Agent 同时开工**（这是单机版 Claude Code 难以做到的）。

### 2. 感官系统：语义感知引擎 (Semantic Context Engine)
废弃死板且极易引发幻觉的 `be_to_fe.json` 握手协议，实现“代码即上下文”。
*   **设计**: 在 Workspace 旁路挂载轻量级的 **Codebase Indexer**（如基于 `tree-sitter`）。
*   **机制**: 后端写完接口后，索引引擎自动提取 AST 生成最新签名。前端 Agent 开工时，直接通过工具 `semantic_search("Customer API")` 实时获取物理硬盘上最新的代码状态，而非读取过时的 JSON 契约。
*   **超越点**: 永远消除“文档与代码不一致”的幻觉，做到 100% Grounded Context。

### 3. 手术刀工具箱：微观控制面 (Surgical Control Plane)
赋予 Nexus (Worker-Coder/Orchestrator) 直接读写沙盒文件的能力，打破“只能通过重型适配器黑盒修改代码”的局限。
*   **设计**: 为 Worker 增加内建 Native Tools (`read_file`, `replace_exact_string`, `run_shell_stream`)。
*   **机制**: 当 `static_check` 定位到“第15行少个括号”这种明确微小错误时，Nexus 内部的轻量诊断模型直接调用 `replace` 热修复（耗时 < 2秒），而不是重新打包 prompt 让 OpenCode 跑几分钟的重试。
*   **超越点**: 结合现有的 `Isolation Workspace`，实现安全、极速的微操。一旦出错，立即 `git reset --hard` 到微操前的快照（时空回溯）。

---

## 三、 任务执行清单 (Epic & Task Backlog)

演进分为三个阶段（M7, M8, M9），旨在渐进式替换现有逻辑。

### 阶段一：感知与微操强化 (Epic-M7: Senses & Hands)
**目标**：让 Nexus 具备自己看代码、改代码的能力，摆脱对重型黑盒委托的绝对依赖。

| 任务 ID | 模块 | 任务描述 | 优先级 | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **M7-1** | Worker-Coder | **实现 Native File I/O Tools**：在 worker 内部直接暴露 `read_file`, `list_dir`, `replace_text`, `write_file` 工具链，允许 Agent 在委托之外直接操作沙盒。 | P0 | Nexus 可通过内部逻辑毫秒级修改 `isolation_workspace` 内的文件。 |
| **M7-2** | Worker-Coder | **流式终端交互 (Streaming Shell)**：重构 `executeCommand`，支持长连接流式输出和交互。 | P0 | Agent 能运行阻塞进程并根据流式输出（如 `npm install` 进度）决定后续动作或提前终止。 |
| **M7-3** | Orchestrator | **热修复仲裁器 (Hot-fix Arbiter)**：当 `static_check` 抛出极小范围的语法错误时，使用 Native Tools 尝试极速自我修复，拦截全量 Delegate 重试。 | P1 | 修复拼写/语法小错误无需重跑长上下文（耗时缩短 90%）。 |

### 阶段二：打破 JSON 枷锁，构建语义网 (Epic-M8: Semantic Context)
**目标**：用真实代码库的语义状态取代人为编造的 JSON 上下文。

| 任务 ID | 模块 | 任务描述 | 优先级 | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **M8-1** | Shared | **Codebase Indexer 基础引擎**：引入轻量级静态分析工具，在任务沙盒中快速提取函数、类、接口的 AST 签名图谱。 | P0 | 能根据 `workspaceRoot` 瞬间生成当前代码的符号图谱。 |
| **M8-2** | Worker-Coder | **新增工具 `search_symbols` & `grep_code`**：将其作为核心能力暴露给底层执行大模型。 | P0 | Agent 在执行 `impl_fe` 前，能自主搜索并精准阅读后端的接口定义，无需依赖外部注入。 |
| **M8-3** | Orchestrator | **重构 SP-03 Handoff 机制**：将强依赖的 `be_to_fe.json` 降级为架构指导（Hint），真实依赖改为强制触发语义检索 (M8-2)。 | P1 | 删除 50% 冗余的强类型检查 Schema，架构容错率大幅提升。 |

### 阶段三：反叛瀑布流，进化为自适应网 (Epic-M9: Reactive Autonomy)
**目标**：这是真正超越 Claude Code 的终极形态，引入动态调度与并发协作。

| 任务 ID | 模块 | 任务描述 | 优先级 | 验收标准 (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **M9-1** | Orchestrator | **Dynamic Task Tree 引擎重构**：升级 `DAG` 引擎，允许正在执行的 Step 动态抛出新的 `Sub-Task` 或请求 `Rollback` 到上游节点。 | P0 | `impl_be` 发现架构不合理时，能动态派生 `update_arch` 子任务。 |
| **M9-2** | Orchestrator | **持续验证循环 (Continuous REPL)**：废弃步骤末尾的“一次性 Smoke Test”。在每次 Native 工具写文件后，后台自动增量运行相关检查。 | P0 | 代码错误在生成后 5 秒内被拦截，而非等待 10 分钟的委托结束。 |
| **M9-3** | Orchestrator | **多轨并行 (Swarm Mode)**：当 Dynamic Tree 拆解出无依赖模块时，Orchestrator 自动调度多个 Worker-Coder 实例并发执行。 | P1 | 复杂项目的整体生成与验证时间比单线程 Claude Code 缩短 40% 以上。 |
| **M9-4** | Permission Council | **微观干预审计 (Micro-Audit)**：将审计从“审整包”变为“实时风险拦截”。对敏感文件（`.env`）的微操触发立即熔断。 | P0 | 在获得原生灵活性的同时，确保企业级的绝对安全。 |

---

## 四、 架构师的最后警示 (Architect's Warning)

执行这份清单，意味着你们必须**抛弃引以为傲的部分过度设计的 JSON Schema**。

对于企业级研发团队，编写繁杂的 JSON Schema 容易产生“虚假的安全感”。我们自以为定义了完美的接口契约，AI 就会乖乖执行。但现实是，**物理代码本身才是唯一的真相 (Source of Truth)**。

Nexus 目前的优势在于**多 Agent 编排和强大的基础设施 (Redis, Postgres, 高级隔离沙盒)**。Claude Code 是单兵作战的特种兵，而 Nexus 必须成为能指挥海陆空三军的数字化司令部。

**下一步行动建议**：
立刻着手实施 **M7-1 (Native File I/O)** 和 **M7-2 (流式终端)**。给瞎眼且手脚被缚的指挥官装上雷达和微创手术刀，打破黑盒委托。做完 M7，整个系统将迎来质的飞跃。