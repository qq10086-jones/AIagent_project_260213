$proj = "E:\AIagent_project_260213\worker-quant\quant_trading\Project_optimized"

# Use .cmd wrappers for bat files that have --no-pause logic
# Use .bat directly for files that don't pause

$tasks = @(
    @{ Name="QuantMorningBriefing"; Time="07:30"; TR="$proj\scheduled_morning_briefing.cmd" },
    @{ Name="QuantOpenWatch";       Time="09:00"; TR="$proj\open_watch.bat" },
    @{ Name="QuantOpenConfirm";     Time="09:30"; TR="$proj\open_confirm.bat" },
    @{ Name="QuantMiddayCheck";     Time="11:30"; TR="$proj\midday_check.bat" },
    @{ Name="QuantPreClose";        Time="14:00"; TR="$proj\pre_close.bat" },
    @{ Name="QuantDailyRun";        Time="16:30"; TR="$proj\scheduled_daily_run.cmd" }
)

foreach ($t in $tasks) {
    schtasks /Delete /TN $t.Name /F 2>$null
    schtasks /Create /TN $t.Name /TR $t.TR /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST $t.Time /F
}

Write-Host ""
Write-Host "=== Registered Tasks ==="
schtasks /Query /FO TABLE | Select-String "Quant"
