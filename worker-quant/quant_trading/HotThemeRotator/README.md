# HotThemeRotator

## Local Beta v0 Runbook (localhost only — Rule 15)

Single-user, single-machine research cockpit. **localhost only — never LAN / cloud / multi-user** (Rule 15.0). Output is research-only; calibration is K-fold **downgraded** to `uncalibrated_research_score` until Rule 8.2.2 / 9.4.1 are honestly met (Rule 15.1).

### Start the dashboard

```powershell
cd E:\AIagent_project_260213\worker-quant\quant_trading\HotThemeRotator
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000/`

Pull-only surface: theme heat, candidate panel, K-line, per-symbol profile / factors / outcomes / strategy card / LLM brief, manual portfolio recording, watchlist, proposal inbox, reflection observability.

### Remote personal access (optional — Rule 15.9)

Single-operator remote access to your own cockpit, ONLY over a private overlay network you control (Tailscale / WireGuard / SSH tunnel). Public internet exposure stays forbidden.

```powershell
# 1) install Tailscale on this machine + your phone/laptop (same tailnet)
# 2) start with the guarded runner (fail-closed: refuses non-loopback without token)
$env:HTR_BIND_HOST = "<your-tailscale-ip>"      # e.g. 100.x.y.z — NEVER a public IP
$env:HTR_ACCESS_TOKEN = "<random string, >=16 chars>"
python tools\serve_remote.py
# 3) on your device: open  http://<your-tailscale-ip>:8000/login?token=<token>
#    (sets a session cookie; API calls also accept X-HTR-Token / Bearer header)
```

Loopback without a token = Local Beta v0, unchanged. The token gates every request (pages + API, reads + the Rule 11.5 manual-record writes); rotate it by restarting with a new value.

### Daily smoke gate (pre-open — Rule 15.2 / 15.6)

```powershell
$lane = ".runtime\lanes\fast"
New-Item -ItemType Directory -Force -Path "$lane\tmp","$lane\cache","$lane\basetemp" | Out-Null
$env:TMP = "$lane\tmp"; $env:TEMP = "$lane\tmp"; $env:TMPDIR = "$lane\tmp"
$env:PYTHONNOUSERSITE = "1"
python -m pytest tests\ -m "not slow" -q -o "cache_dir=$lane\cache" --basetemp "$lane\basetemp"
```

Fast and deterministic; excludes the vectorbt research lane (numba arrives through vectorbt — nothing here imports it directly). A green smoke lane is **not** proof of model edge.

**Why the four lines before pytest (P37-03 step 5).** Setting `--basetemp` alone
moves pytest's own scratch but leaves every `tempfile` call in the system temp,
whose ACL defect on this machine produces mass collection ERRORs and false
hangs. And pytest does **not** create missing parents for `--basetemp`: pointing
it at an uncreated path errors every `tmp_path` test, which looks identical to
the ACL failure. Create first, then pin all three variables.
`tools/daily_routine.py` does exactly this via
`hot_theme_rotator.common.runtime_paths`, which is the single owner of these
paths — prefer it over retyping the block.

### Research regression lane (not a daily readiness signal)

```powershell
$lane = ".runtime\lanes\slow"
New-Item -ItemType Directory -Force -Path "$lane\tmp","$lane\cache","$lane\basetemp" | Out-Null
$env:TMP = "$lane\tmp"; $env:TEMP = "$lane\tmp"; $env:TMPDIR = "$lane\tmp"
$env:PYTHONNOUSERSITE = "1"
python -m pytest tests\ -m slow -q -o "cache_dir=$lane\cache" --basetemp "$lane\basetemp"
```

Separate lane, separate verdict, separate scratch directory — the two can run at
once without fighting over a basetemp.

### Installing from the locks

```powershell
python -m pip install --require-hashes -r requirements\bootstrap.txt   # build toolchain
python -m pip install --require-hashes -r requirements\fast.txt        # or slow.txt / runtime.txt
python -m pip install --no-deps --no-build-isolation --no-index .
```

The bootstrap step is not optional: a fresh CPython 3.13 venv has pip and **no
setuptools**, so a plain `pip install .` would fetch an unlocked build backend
from PyPI, straight past every hash. See `requirements/README.md`, and
`python tools/verify_clean_environments.py` to rebuild and re-verify all three
environments from scratch.

### After close — forward sample collection (Rule 15.5 step 4)

```powershell
python tools\emit_daily_predictions.py
python tools\sweep_pending_outcomes.py
```

`emit` writes live `PredictionRecord` rows from the day's selected tickers; `sweep` joins realized 1D/3D/5D outcomes as forward bars arrive. These accumulate toward the Rule 8.2.1 sunset / Rule 9.4 validation — they do **not** promote any label to a win rate.

### Automated daily routine (P10-28) — hands-off

The deterministic half of the rhythm runs unattended via two Windows scheduled tasks (Mon-Fri, JST):

- `HTR_Daily_Preopen` 08:30 → daily smoke gate + candidate freshness check
- `HTR_Daily_AfterClose` 16:00 → refresh candidates (deterministic screener, read-only) → `emit` → `sweep`

```powershell
scripts\register_daily_routine_tasks.bat      REM (re)register the tasks
schtasks /Run /TN "HTR_Daily_AfterClose"      REM run once on demand
python tools\daily_routine.py --mode afterclose --dry-run   REM preview without writing
```

Runs are logged to `reports\observability\daily_routine_log.jsonl`. It is deterministic, fail-closed, idempotent, and never touches a broker / order / LLM path. Two steps stay manual by design: recording a fill (only when you actually trade) and pulling LLM narrative briefs.

### Rollback

Beta baseline snapshot 2026-06-02 (readiness + automation + write-path wiring + calibration validation + forward-collection honesty + dashboard freshness decouple + market-session derivation, smoke 1294 green): `git stash apply f14da50` (also `git stash list` → stash@{0}). Designer-redesign fallback: restore `frontend_zerobuild_backup_2026-05-30/`. Committed baseline: HEAD `2ce7504`.

> Legacy P8 Streamlit panel (`streamlit run .\tools\streamlit_opportunity_app.py --server.port 8501`) is superseded by the FastAPI dashboard above and is no longer the primary UI.

日股为主、A股和美股为外部温度因子的热点龙头轮动工具。

本项目不是高频交易系统，也不是自动实盘下单系统。第一阶段目标是把主观的“市场温度、新闻热度、板块龙头、2%-5%止盈换仓”流程固化成可解释、可回测、可复盘的量化决策工具。

## 项目定位

- 主市场：日本股票。
- 外部温度：A股、美股、汇率、指数、半导体/AI/汽车/机械等跨市场链条。
- 核心策略：新闻催化 + 市场温度 + 热点主题 + 龙头强度 + 严格退出。
- 执行模式：只生成交易建议和风控提示，人工确认后执行。
- 继承资产：参考 `../Project_optimized` 的日股数据库、新闻 overlay、候选排序、盘中建议和报告经验；参考 `../Project_v5` 的 V7 News-Driven Hot Theme Hunter 设计。

## 入口文档

- `PROJECT_STATUS.md`：唯一项目更新文件，所有进度、决策状态、下一步都只更新这里。
- `docs/00_DESIGN.md`：产品和系统设计。
- `docs/01_TASKS.md`：任务清单。
- `docs/02_GOVERNANCE.md`：治理规则和修改规则。
- `docs/03_FOLDER_MAP.md`：目录职责。
- `docs/04_DATA_AND_OPEN_SOURCE.md`：数据源和开源项目选型。
- `docs/05_USER_GUIDE.md`：使用说明书 — 每天怎么操作、每张卡片显示什么/怎么读/有什么功能。

## 唯一推进规则

任何改动必须先对应到 `docs/01_TASKS.md` 中的任务，并在 `PROJECT_STATUS.md` 记录状态。禁止新增散落的 `PROGRESS_*.md`、临时总结文档或口头状态作为项目事实来源。
