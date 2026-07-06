# Bi-weekly DSR progress alert (OS-level, survives Claude restarts).
# Runs the digest (reads the daily-accumulated trace) and pops a desktop alert.
# Full report lands in reports/observability/DSR_PROGRESS.txt. Research-only.
$ErrorActionPreference = 'Continue'
$root = 'E:\AIagent_project_260213\worker-quant\quant_trading\HotThemeRotator'
Set-Location $root
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONPATH = "$root\src"
$out = & 'D:\python\python.exe' "$root\tools\dsr_progress_digest.py" --lookback-days 14 2>&1 | Out-String
$head = ($out -split "`n" | Where-Object { $_ -match '^HEADLINE:' } | Select-Object -First 1)
if (-not $head) { $head = 'DSR digest updated - see reports/observability/DSR_PROGRESS.txt' }
$head = ($head -replace '^HEADLINE:\s*', '').Trim()
# Desktop alert: msg is reliable on Win Pro; fall back to writing a flag file.
try { msg * "HTR price_reversal DSR progress: $head  (full: reports/observability/DSR_PROGRESS.txt)" } catch {
    "$((Get-Date).ToString('s'))  $head" | Out-File -FilePath "$root\reports\observability\DSR_ALERT_LASTRUN.txt" -Encoding utf8
}
