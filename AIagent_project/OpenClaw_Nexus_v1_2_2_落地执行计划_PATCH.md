# OpenClaw Nexus v1.2.2 落地执行计划（Obsidian）

> 最终目标：一个中枢 Agent（Orchestrator）统一调度多类 Worker（量化/媒体/写作/电商），具备 Guard 风控、人审、可观测、可恢复、可扩展，并能沉淀任务与资产（DB/MinIO）。

---

## A. 项目启动与规范（Foundation）

### A1. 需求边界与成功指标

-  定义 MVP 范围：先选 **量化闭环** or **媒体闭环** or **电商闭环**（建议量化优先）
    
-  定义成功指标（必须量化）
    
    -  e2e 成功率 ≥ 99%
        
    -  重启恢复成功率 ≥ 99%
        
    -  单任务全链路可追踪（event_log 完整）
        
-  定义风险等级与“必须人审”的清单（资金动作 / 对外发布 / 删除资产等）
    

**交付物**

- `docs/PRD.md`
    
- `docs/SuccessMetrics.md`
    

---

### A2. Repo & 工程规范

-  建仓库结构（orchestrator/worker-*/infra/shared/docs）
    
-  统一配置管理：`.env` / `config/*.yaml`
    
-  统一日志格式：JSON log（含 task_id、trace_id、run_id）
    
-  统一编码风格：eslint/ruff/pre-commit
    

**交付物**

- `docs/Architecture.md`
    
- `docs/RUNBOOK.md`
    

---

## B. 核心平台层（Platform Core）——先把“系统骨架”做成产品

> 这一层做完，你才有资格接任何业务能力。

### B1. Infra（可运行环境）

-  docker-compose：redis + db + orchestrator + worker-dummy
    
-  健康检查 / 启停脚本
    
-  本地开发一键启动（makefile / npm scripts）
    

**交付物**

- `infra/docker-compose.yml`
    
- `infra/.env.example`
    
- `scripts/up.sh`, `scripts/down.sh`
    

---

### B2. Queue（Redis Streams 任务系统）

-  Streams 命名与 consumer group 固化
    
-  Task 状态机（queued/running/succeeded/failed/waiting_approval）
    
-  Retry 策略（指数退避/最大重试/幂等键）
    
-  DLQ（死信）与再投递工具
    
-  Reclaim（超时未 ack 任务回收继续执行）
    

**交付物**

- `shared/task_schema.*`
    
- `orchestrator/src/queue/*`
    
- `worker-*/consumer.*`
    
- `tools/dlq_replay.*`
    

---

### B3. DB（事件溯源 & 当前状态）

-  `tasks`（当前状态）
    
-  `event_log`（不可变事件流）
    
-  `runs`（一次执行 run 的聚合视角，可选）
    
-  索引/查询（按 task_id、tool_name、时间）
    

**交付物**

- `infra/init.sql`
    
- `orchestrator/src/db/*`
    
- `docs/DB_Schema.md`
    

---

### B4. Guard（风险门禁 + 人审流程）

-  Guard 作为 Orchestrator 内部模块（MVP 不拆服务）
    
-  风险规则：资金/对外写入/删除/高成本 GPU 等 → waiting_approval
    
-  人审接口：approve/reject（写入 event_log）
    
-  审批超时策略（可选）
    

**交付物**

- `orchestrator/src/guard/*`
    
- `orchestrator/src/api/approval.*`
    
- `docs/Guard_Policy.md`
    

---

### B5. Observability（可观测与调试）

-  全链路 trace_id（task_id + run_id）
    
-  每个 task 输出执行时间、重试次数、错误栈
    
-  指标：队列长度、pending、成功率、DLQ 数量
    
-  最小仪表盘（先用简单 HTML/CLI 也行）
    

**交付物**

- `docs/Monitoring.md`
    
- `orchestrator/src/metrics/*`
    

---

### B6. e2e 测试与验收脚本

-  “空任务闭环” e2e
    
-  “失败→重试→DLQ” e2e
    
-  “重启→reclaim” e2e
    
-  回归测试脚本（每次改动必跑）
    

**交付物**

- `tests/e2e/*`
    
- `scripts/e2e_run.*`
    

---

---


### B7. Skill Registry（技能注册表：装/搜/禁用）

**目的**：让“能力接入”从“改代码”变成“安装技能 + 导入工作流”。

**交付物**
- DB 表：`skills(name, version, manifest_json, checksum, source, enabled, installed_at)`
- Manifest 校验器：校验 schema / risk_tag / runtime / resources
- CLI：  
  - `skill install <git-url>@<tag>`（或本地路径）  
  - `skill enable/disable <name>@<version>`  
  - `skill search <capability>`  

**验收**
- 能安装 3 个示例技能（read_only / write_draft / write_external 各 1）
- 禁用后不可被路由选中（强制生效）

---

### B8. Workflow Runner（DAG 执行：像 n8n，但 SSOT 在 OpenClaw）

**交付物**
- Workflow Spec（YAML/JSON）：`trigger + nodes + edges + mapping + gates + retries`
- Runner：节点级重试、节点级重跑（从任意节点重放）、节点级审批 Gate
- 事件：所有节点执行状态写入 `event_log`；产物入 MinIO（DB 只存引用）

**验收**
- 跑通 1 条 DAG（建议量化闭环）并可在节点失败后从中间节点重跑
- 高风险节点自动插入 Guard 审批


## C. 能力层（Capabilities）——以 Skill 形式接入（可组合/可版本化）

> 这一层开始按“工具=任务”扩展，每个能力都是一个 tool。

### C1. 量化能力（建议先做，最容易闭环）

-  数据采集（行情、指数、财报等） → `fetch_market_data`
    
-  特征/信号 → `compute_features`
    
-  策略计算 → `run_strategy`
    
-  回测 → `backtest`
    
-  报告产出 → `backtest_report`
    
-  建议下单 → `propose_orders`
    
-  人审确认 → `approve_orders`（Guard 触发）
    
-  交易录入/对账 → `record_fills`
    
-  盘后复盘 → `execution_report`
    

**交付物**

- `worker-quant/tools/*`
    
- `docs/Quant_Tools.md`
    

---

### C2. 媒体能力（视频/图片/配音/投稿）

-  Prompt → 图（ComfyUI API） → `gen_image`
    
-  文本 → 音频（TTS） → `gen_voice`
    
-  音视频合成（FFmpeg） → `compose_video`
    
-  自动字幕/翻译（可选） → `subtitle_translate`
    
-  发布（YouTube/小红书等） → `publish_video`（必人审）
    

**交付物**

- `worker-media/tools/*`
    
- `docs/Media_Pipeline.md`
    

> 注意：这条线要在 **MinIO + 外部 ComfyUI** 稳定后再做，不要一开始 GPU 把你拖死。

---

### C3. 写作能力（小说/脚本/企划）

-  结构化大纲 → `write_outline`
    
-  分章生成 → `write_chapter`
    
-  校对润色 → `edit_polish`
    
-  版本管理与差异对比（可选）
    

**交付物**

- `worker-writing/tools/*`
    
- `docs/Writing_Flow.md`
    

---

### C4. 电商能力（商品上架/文案/图片处理）

-  商品信息抓取/整理 → `extract_product_info`
    
-  日文/中文文案生成 → `generate_listing_copy`
    
-  图片修复/去水印/翻译覆盖 → `edit_images`（高风险看场景）
    
-  上架（平台 API） → `publish_listing`（必人审）
    

**交付物**

- `worker-commerce/tools/*`
    
- `docs/Commerce_Flow.md`
    

---

## D. 资产层（Assets）——把“产物”变成可管理资产

### D1. MinIO 对象存储

-  报告/图片/音频/视频统一入库
    
-  DB 存 object_key + sha256 去重 + metadata
    
-  生命周期（TTL）与归档策略
    

**交付物**

- `infra/minio/*`
    
- `shared/asset_schema.*`
    

---

### D2. 模板与可复用组件

-  报告模板（markdown/html）
    
-  FFmpeg 脚本模板
    
-  ComfyUI workflow 模板（作为资产版本化）
    

**交付物**

- `assets/templates/*`
    
- `docs/Templates.md`
    

---

## E. 产品化（Productization）——把它变成你每天都能用的东西

### E1. UI（Streamlit / Web）

-  任务列表 / 任务详情（事件时间线）
    
-  手动触发工具（参数表单）
    
-  审批中心（waiting_approval）
    
-  DLQ 管理（查看/重放/丢弃）
    
-  资产浏览（MinIO 文件）
    

**交付物**

- `ui/streamlit_app/*` 或 `ui/web/*`
    

---

### E2. 运行策略（Schedules & Triggers）

-  定时任务：每日收盘后跑 quant pipeline
    
-  触发任务：新素材入库自动启动 media pipeline
    
-  手动任务：一键生成报告/一键复盘
    

**交付物**

- `orchestrator/src/scheduler/*`
    
- `docs/Scheduling.md`
    

---

## F. 可靠性与安全（Hardening）

### F1. 权限与审计

-  审批操作写入 event_log（不可篡改）
    
-  对外 API key 管理（env + vault 思路）
    
-  危险动作双确认（publish/delete）
    

### F2. 灾备与数据一致性

-  DB 备份策略
    
-  MinIO 备份策略
    
-  Redis 持久化与恢复策略