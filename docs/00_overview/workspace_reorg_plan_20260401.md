# Workspace Reorg Plan (2026-04-01)

## Executive Summary

当前仓库的主要问题不是“目录太多”，而是不同生命周期的内容被放进了同一个主仓库：

- 产品源码
- 上游依赖源码镜像
- 实验/沙盒项目
- 运行产物与验证快照
- 历史备份与归档
- 本地数据库与报表

结果是：

- 顶层目录语义不清，难以判断哪些是主路径
- 运行产物与源码混放，Git 历史被噪音淹没
- 备份目录和归档目录长期占位，工作区认知负担高
- `openclaw` 作为大型嵌套仓库占据主视野，但并非当前业务代码主体
- `orchestrator/artifacts/**` 和 `worker-quant/.../reports/**` 中有大量应归档而非长期跟踪的内容

结论：

- 这个仓库需要按“职责”和“生命周期”重划边界
- 不建议直接大搬家
- 建议按 3 个阶段完成：先收口入口，再隔离产物，最后拆分历史负担

## Current Structure Assessment

目前顶层大致分为 6 类：

### 1. 主业务代码

- `orchestrator/`
- `worker-coder/`
- `worker-quant/`
- `brain/`
- `shared/`
- `infra/`
- `configs/`
- `scripts/`
- `docs/`
- `ui/`

这是主仓库真正应该围绕组织的内容。

### 2. 上游或外部依赖源码

- `openclaw/`
- `vendor/superpowers/`

这类目录不应与主业务代码并列成为“默认阅读入口”。需要明确它们是：

- 子模块/镜像依赖
- 二次开发 fork
- 临时参考代码

否则新人和未来的你都会误判系统边界。

### 3. 运行产物和验证结果

- `artifacts/`
- `orchestrator/artifacts/`
- `metrics/`
- `worker-quant/quant_trading/Project_optimized/reports/`
- `worker-quant/quant_trading/Project_optimized/*.db`

这部分是当前结构污染最严重的区域。它们的价值是“回放、审计、追踪”，不是“源码协作”。

### 4. 沙盒和实验区

- `sandbox/`
- `scratch/`
- `coder_test/`

这类目录应该保留，但必须明确是非生产路径，否则会持续向根目录扩散。

### 5. 历史备份和归档

- `backup_20260301/`
- `backup_20260301_v2_coding_final/`
- `docs/90_archive/`

备份可以存在，但不应该长期占据仓库根目录。

### 6. 模糊职责目录

- `claude_code_src/`
- `claude_code_study/`
- `project` 类实验内容散落在 `sandbox/`

这些目录命名没有把“用途、来源、保留期限”表达清楚，是后续继续失控的高风险点。

## Main Structural Problems

### Problem 1: Source and Runtime Data Share the Same Repository Surface

最典型的是：

- `orchestrator/artifacts/**`
- `artifacts/**`
- `worker-quant/.../reports/**`
- `worker-quant/.../*.db`

这导致仓库同时扮演：

- 源码仓库
- 审计仓库
- 数据仓库
- 运行缓存仓库

这是结构层面的角色冲突。

### Problem 2: Top-Level Directories Lack a Strong Information Hierarchy

当前顶层把：

- 主业务模块
- 依赖仓库
- 备份快照
- 沙盒实验

全部平铺。根目录没有表达“什么最重要”。

### Problem 3: Backup Strategy Is Filesystem-Based Instead of Repo-Based

`backup_20260301*` 这类目录说明你在用“复制目录”替代：

- Git tag
- release branch
- archive bundle
- docs archive

这在短期上安全，但长期会造成：

- 重复文件
- 重复认知路径
- 难以判断哪个版本才是当前真相

### Problem 4: Nested Repositories Are Present but Not Clearly Governed

`openclaw/` 当前以 gitlink 形式存在，但在主仓库中缺少明确说明：

- 为什么放这里
- 谁负责升级
- 是否允许在本仓库直接改
- 升级策略是锁版本还是跟主线

这会在后续升级、回滚、排障时反复出问题。

### Problem 5: Sandbox/Test Fixtures Are Valuable but Under-Named

`sandbox/` 本身是合理的，但里面同时存在：

- 演示项目
- cohort fixture
- live validation 输入

这类内容应进一步分层，否则会逐渐演化成第二个根目录。

## Recommended Target Layout

建议目标不是“最漂亮”，而是“稳定、可维护、能迁移”。

```text
AIagent_project_260213/
  apps/
    orchestrator/
    worker-coder/
    worker-quant/
    brain/
    ui/

  packages/
    shared/

  platform/
    configs/
    infra/
    scripts/

  docs/
    00_overview/
    01_design/
    02_patch/
    03_feature_development/
    90_archive/

  external/
    openclaw/
    vendor/

  workspace/
    sandbox/
    scratch/
    fixtures/

  runtime/
    artifacts/
    metrics/
    reports/
    state/

  archive/
    backup_20260301/
    backup_20260301_v2_coding_final/
```

## Directory Rules

### `apps/`

只放真正可运行、可部署、需要协作开发的服务。

### `packages/`

只放被多个应用共享的代码，不放运行产物。

### `platform/`

放平台级配置、编排脚本、基础设施定义。

### `external/`

只放上游仓库、vendor 代码、第三方镜像，不允许混入主业务实现。

### `workspace/`

放实验、临时实现、sandbox、fixture。默认不作为生产入口。

### `runtime/`

放运行结果、缓存、快照、报表、临时数据库、验证输出。默认不进 Git，或只保留极少量基线样例。

### `archive/`

放必须随仓库存档的历史快照，但不应继续参与主工作流。

## What Should Stay in Git

建议继续保留：

- 应用源码
- 配置模板
- 合同/协议定义
- 小体积测试 fixture
- 必须用于回归的基线样例
- 文档和架构设计

## What Should Leave Git or Be Greatly Reduced

建议逐步移出版本库或只保留最小样例：

- `orchestrator/artifacts/**` 的大批量历史运行记录
- `artifacts/release/**` 和 `artifacts/runs/**` 的完整执行输出
- `worker-quant/.../reports/**` 中自动生成报表
- `worker-quant/.../*.db` 本地数据库文件
- 高频生成的 `canary` / `validation` 时间戳目录
- 临时调试文件和 `scratch/*.txt`

## Recommended First-Step Reorg Without Breaking Runtime

### Phase 1: Clarify Ownership and Entry Points

目标：不搬核心代码，只先把结构说清楚。

动作：

- 在根 `README.md` 增加仓库导航，只保留主入口
- 明确标注 `openclaw/` 是外部依赖还是 fork
- 明确 `sandbox/`、`scratch/`、`backup_*` 属于非主路径
- 为 `runtime` 类目录写统一约定

这是最小风险、最高收益的一步。

### Phase 2: Isolate Runtime Outputs

目标：把源码和产物切开。

动作：

- 新建统一运行根目录，例如 `runtime/`
- 将以下内容逐步迁移：
  - `artifacts/` -> `runtime/artifacts/`
  - `metrics/` -> `runtime/metrics/`
  - `worker-quant/.../reports/` -> `runtime/reports/quant/`
  - 本地数据库 -> `runtime/state/quant/`
- 对历史产物只保留：
  - 1 份 happy-path 样例
  - 1 份 failure-path 样例
  - 少量 contract fixture

### Phase 3: Move Non-Product Material Out of the Root

目标：让根目录只剩“产品结构”。

动作：

- `openclaw/`、`vendor/` 迁移到 `external/`
- `sandbox/`、`scratch/`、`coder_test/` 迁移到 `workspace/`
- `backup_*` 迁移到 `archive/`
- `claude_code_*` 归并进 `workspace/research/` 或直接归档

## Concrete Recommendations by Directory

### `openclaw/`

建议：

- 保留，但迁移到 `external/openclaw/`
- 在根文档中明确说明升级策略
- 只通过固定流程更新子模块指针

不建议：

- 继续和主业务代码并列放在根目录

### `orchestrator/artifacts/`

建议：

- 只保留少量回归 fixture
- 其余全部转移到 `runtime/` 或外部对象存储

不建议：

- 继续在源码目录下长期累积时间戳运行结果

### `worker-quant/quant_trading/Project_optimized/`

建议：

- 拆成 `src/`, `configs/`, `runtime/`, `reports/` 的内部分层
- 数据库和报表外移到统一运行目录

当前问题：

- 实现代码、数据库、结果文件、业务分析输出放在同一层级

### `sandbox/`

建议拆成：

- `workspace/sandboxes/`
- `workspace/fixtures/`
- `workspace/live-validation/`

这样能保留价值，同时让非正式内容退出主路径。

### `backup_*`

建议：

- 迁到 `archive/`
- 每个备份目录补一个 `README.md`，说明备份原因和可删除条件

### `scratch/`

建议：

- 默认本地使用，不入库
- 若必须留样例，只保留结构化样本，不保留大量临时文本

## Git Hygiene Rules to Add

建议补充以下规则：

1. 任何时间戳命名目录默认不应提交
2. 任何自动生成的 `report`, `summary`, `run_manifest`, `state`, `*.db` 默认不应提交
3. `sandbox/` 和 `workspace/` 下只有显式标记为 fixture 的内容允许入库
4. 第三方代码统一放 `external/`
5. 根目录禁止新增无说明目录

## Suggested Migration Order

### Step 1

完成文档和规则治理，不改运行路径。

### Step 2

把新生成产物统一输出到 `runtime/`，停止新增结构债务。

### Step 3

清理 Git 中大批量历史产物，只保留回归需要的基线。

### Step 4

移动 `external/`, `workspace/`, `archive/` 三类目录。

### Step 5

最后再评估是否把顶层重构为 `apps/ + packages/ + platform/`。

## Recommended Immediate Actions

如果现在就开始整理，我建议优先做这 5 件事：

1. 给根目录建立正式导航文档，定义哪些是主路径
2. 新建 `runtime/` 约定，并停止把新产物写回源码树
3. 把 `backup_*` 从根目录移到 `archive/`
4. 把 `openclaw/` 和 `vendor/` 归为 `external/`
5. 对 `orchestrator/artifacts/` 制定“只保留最小 fixture”的清理策略

## Final Judgment

这个仓库不是不能维护，而是缺少“仓库治理层”。

从架构角度看，当前最需要的不是继续加目录，而是建立三个边界：

- 源码边界
- 运行产物边界
- 外部依赖边界

只要这三个边界立住，后续目录整理就会变成可持续的演进，而不是反复大扫除。
