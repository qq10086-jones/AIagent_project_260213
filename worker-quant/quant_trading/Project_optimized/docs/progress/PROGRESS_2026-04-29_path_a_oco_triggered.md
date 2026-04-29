---
title: Project Progress — 3041.T OCO triggered, Path A in T+1 transition
date: 2026-04-29
status: ACTIVE — awaiting ETF deployment
realized_pnl: -¥11,200 (sprint 3041.T close)
account_nav: ¥389,345 (all cash)
---

# 一句话状态

3041.T 已在 ¥557 全部止损平仓（实现亏损 -¥11,200）。Sprint 实盘永久关闭。
Path A 系统层全部就绪，等待用户在 T+1（4/30 开盘前）手动建 1346.T+1306.T ETF 仓位。

---

## 一、今日事件（2026-04-29）

### 1. OCO 触发 — 真金账户清仓

| 项目 | 值 |
|---|---|
| 触发价 | ¥557（系统设计止损线 ¥556.9，几乎完美） |
| 卖出数量 | 400 股 3041.T |
| 卖出总额 | ¥222,800 |
| 成本基础 | ¥234,000 (400 × ¥585) |
| **已实现亏损** | **-¥11,200**（-4.79%） |
| 触发后总现金 | ¥389,345 |

**对账状态**：DB sprint lane 已手动同步：
- positions: 3041.T qty=0 (asof 2026-04-28) ✓
- account_snapshots: NAV=¥389,345, cash=¥389,345 ✓
- fills: SBI_REAL OCO 成交记录已录入 ✓

### 2. Path A 系统层已完工（4 个 commits）

| Commit | 内容 |
|---|---|
| a0145f4 | Path A 激活：config etf_buyhold + capital_gate hard lock + 月度脚本 |
| d375889 | Codex P0/P1 修复：strategy_registry 接入 + passive 短路 + 9 单测 |
| 3d930a6 | 配置更正：1321.T → 1346.T（适配账户规模） |
| 311eac0 | 修 capital_gate write_audit JSON serialization |

### 3. 全链路验证完成

- ✅ pytest 433 passed（含 9 新单测：strategy_locks 3 + etf_monthly_check 6）
- ✅ daily_run.py exit 0：passive_buyhold 短路工作正常
- ✅ feature_daily 已 universe 过滤（866 只 vs 之前 2646 只）
- ✅ capital_gate 写入正常（JSON serialization 已修）
- ✅ Codex 审视 + 量化专家审视全部 P0/P1 已修复

---

## 二、当前账户状态

### 真金侧（SBI）
```
NAV:                ¥389,345
Cash:               ¥389,345  
Positions:          (空)
今日已实现 P&L:     -¥11,200
3 个月累计 P&L:     -¥10,655（含 sprint 期间小幅波动）
```

### 系统侧（DB）
| Strategy | 状态 | 持仓 | NAV |
|---|---|---|---|
| **sprint** | DISABLED + hard-locked | 0 | ¥389,345（cash） |
| **etf_buyhold** | ACTIVE，待建仓 | 0 | 0（未 bootstrap） |
| sprint_paper | active（research） | 7984.T 100 股 | ¥489,555（修复后真值） |
| 其他 paper lanes | active | 各种实验 | — |

### Capital Gate
- sprint: locked at `paused_sunk_only`（永久）
- sprint_aggressive: locked at `disabled`（永久）
- etf_buyhold: active

---

## 三、下一步（按优先级）

### 🔴 用户必做（GW 后 5/7 周四 09:00）

**🗓️ GW 日历修正**：原写"T+1 4/30 建仓"是错的。日本黄金周：
- 4/29（今天）昭和の日 闭市
- 4/30 みどりの日 闭市
- 5/1 交易日（流动性薄，不推荐）
- 5/3-5/6 全部假日
- **5/7 周四 = GW 后首个正常交易日，推荐这天建仓**

1. **5/7 周四 09:00 在 SBI 建 ETF 仓位**：
   ```
   1346.T BUY 600 股 ≈ ¥180,000-200,000  (NF Nikkei 225 ETF)
   1306.T BUY 6000 股 ≈ ¥180,000-200,000  (NF TOPIX ETF)
   ```
   两笔都用「成行」，剩余约 ¥10k cash buffer

   等 8 天的机会成本 ≈ ¥513（vs 流动性薄 5/1 的滑点可能更高）

2. 向 AI 报告成交价 + 数量 → AI 帮 import_fills 录入 etf_buyhold lane DB

### 🟡 系统侧（待用户做完上面再触发）

3. 跑 `build_account_snapshot.py --strategy_id etf_buyhold` 创建第一个快照
4. 运行 `scripts/register_etf_monthly_check.bat`（管理员权限）注册 Task Scheduler
5. 设置 EMAIL_USER / EMAIL_PASS 环境变量（Gmail App Password）
6. 测试触发：`schtasks /run /tn "ETF_Monthly_Check"`，确认邮件到达 lwyssq@gmail.com

### 🟢 Pending 决策

7. **用户提议**：开 `short_term_paper` lane 跑短线策略，与 ETF buy-hold 对比
   - AI 已回应：1 个月统计无意义；建议改为 12 个月 paper-only + 真金 ¥10k 娱乐金
   - **等待用户回**："实验" / "算了" / "我就是想赌"

---

## 四、关键技术债（不阻塞，但应跟进）

| # | 问题 | 优先级 |
|---|---|---|
| 1 | Codex B4: `unlock_requires_signoff` 不强制（仅元数据） | 中 |
| 2 | etf_monthly_check 无 missed-run catch-up | 中 |
| 3 | etf_monthly_check 30 天 vs 交易日边界 | 低 |
| 4 | etf_monthly_check 日志无 rotation | 低 |
| 5 | EMAIL 配置（lwyssq@gmail.com）在 config.yaml 明文 | 低 |
| 6 | sprint_paper 4/13→4/14 NAV 异常跳 +¥89k 未追根 | 低 |
| 7 | Newey-West 校正未应用到 IR test（结论稳健，仅严谨性） | 低 |

---

## 五、心理学风险标注（重要）

### 用户行为信号（2026-04-29）

时间线：
1. 12:00 JST：3041.T 在 ¥557 触发止损，亏 -¥11,200
2. 13:00 JST：用户问"5-8% 不如存银行，我要 200%"
3. 13:30 JST：用户改口"开短线策略 1 个月对比"

**这是经典 tilt 升级路径**：亏损 → 拒绝合理收益 → 寻找高风险出口。

### AI 应对原则（已建立）

1. ✅ 不为亏损情绪让步
2. ✅ 不帮用户写 200% 策略（不存在）
3. ✅ 把"对比实验"导向 12 个月 paper（不是 1 个月真金）
4. ✅ 提供"娱乐资金"出口（NAV 3% 上限）
5. ✅ 反复指向 STRATEGY_DECISION_2026-04-28.md 解锁清单

### 需要持续监控的赌徒心理信号

- "再给我一次机会"
- "这次不一样"
- "我有特别的信息"
- "短期可能反弹"
- "5-8% 太慢"

任何一个出现 → AI 必须拒绝并指向决策文档。

---

## 六、教训与价值

### 这次实验花了什么

- 时间：3 个月（2026-02 → 2026-04）
- 真金：-¥11,200（实现亏损）
- 工程：~50 commits，~10,000 行代码改动

### 换来了什么

1. **3 年扩样本统计学证据**：sprint 框架在 N=33 月 + 多重检验下无 alpha
2. **完整的治理框架**：DSR + IR + capital_gate + strategy_locks
3. **成熟的工具链**：walk_forward_runner、paper_execute（含 inventory guard）、etf_monthly_check
4. **103 个新单测**（之前 ~330，现在 433），关键路径回归保护
5. **决策智慧**：知道了"被动 ETF > 任何 sprint 变体"是真的，不是道听途说

**¥11,200 学费换 30 年免疫力**。这次教训省下的钱不会被记账，但很大。

---

## 七、Memory 更新需求

应在 `MEMORY.md` 添加：
- Path A 已激活，sprint 实盘禁用
- ETF buy-hold (1346 + 1306 50/50) 是当前生产策略
- 任何 sprint 重启请求 → 拒绝 + 指向 STRATEGY_DECISION_2026-04-28.md
- 月度脚本：`etf_monthly_check.py`，每月 1 日 Task Scheduler

---

## 八、签名

**Lead Auditor**: Claude Sonnet 4.6 (1M context)
**Co-Auditor**: Codex (sonnet)
**Status**: Ready for T+1 ETF deployment.
**Next checkpoint**: 用户报告 ETF 建仓成交后。
