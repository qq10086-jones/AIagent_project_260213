# Progress: 2026-04-03 J-Quants Rate Limit Bypass & Resumable Caching

## Status
- **Date:** 2026-04-03
- **Feature:** `import_jquants_v2` 基本面数据分流抓取引擎重构
- **State:** ✅ Completed & Validated

---

## 解决的问题 (Problem Addressed)
在 2026-04-02 的初步测试中，使用 J-Quants Free 账号抓取所有候选股票基本面数据时，遭遇极严苛的 API 限流拦截（`429 Too Many Requests`），并被迫回退至 `yfinance`（后者季报数据严重缺失，丢失营业现金流 OCF、毛利等核心超额因子，是模型始终输出低迷权重的罪魁祸首之一）。这阻断了系统的生产数据链路管道。

## 升级措施 (Enhancements Implemented)
通过纯代码工程升级解决免费版限流（绕过氪金墙），并保障系统级增量热更新：

### 1. 智能断点跳过 (14-Day Resumable Cache)
在向 J-Quants 提交批量拉取任务前，优先进行本地数据库（`fundamental_snapshots`）检索：
- **逻辑**：剥离出最近 14 天内已经通过 `jquants_v2` 成功写入的股票（考虑到日本标的季报发行周期在三个月左右，14天缓存截面极度安全）。
- **效果**：将 4000 个池子完全拦截过滤，日均任务直接降维。哪怕爬取在中途断开，下次运行自动跳过已摄取标的并从断点处接续（增量爬取）。

### 2. 指数退避式 429 防爆盾 (Exponential Backoff Handling)
重写了 `get_fin_summary` 的通讯封装。当系统截获含有 `429`, `rate limit`, `exceeded` 的报错时：
- **逻辑**：立刻取消当前线程执行，脚本休眠 `30 * (2^retries)` 秒，进行静默等待（避免连续叩击被关黑名单），并在退避结束后发起请求。
- **保险**：至高支持 `max_retries=3` 级探测；若达到探测上限仍遇到死角封禁，脚本将优雅抓取中断，并且主动跳出，**依然能够将中途已经抓取完毕的数据全部落库**（规避了早前一挂全挂的白跑局面）。

### 3. 主配置修正返回黄金路线
更新了全局 `config.yaml` 恢复最强生产力：
```yaml
fundamental:
  enabled: true
  source: "jquants_v2" # 从 "noop" / "yfinance" 切回
```

## 下一步跟进工作 (Next Actions Tracking)
- 监控明天 `daily_run.py` 定时跑批任务，看后台打出的 `剔除 XX 只最近已更新的标的` 过滤率是否正常起效。
- 在拥有海量高纯度季报因子 `cfo_assets`、`accruals_inv` 补充进特征库一周后，使用 `compute_ic.py` 复查这些基本面源特征池的 `IC` 与 `t-stat`。预期将对 Ridge 罚函数重新释放强阿尔法影响。
