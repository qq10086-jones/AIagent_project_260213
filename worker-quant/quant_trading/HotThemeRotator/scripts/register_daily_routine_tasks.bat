@echo off
REM ============================================================================
REM Register the HTR Local Beta v0 daily routine scheduled tasks (Rule 15.5).
REM Task P10-28. Two current-user tasks, Mon-Fri (JST local time):
REM   HTR_Daily_Preopen    08:30  -> daily smoke gate + candidate freshness check
REM   HTR_Daily_AfterClose 19:30  -> refresh candidates -> emit -> sweep (forward samples)
REM   (19:30: JPX closes 15:30 JST; free EOD sources finalize the day's closes in
REM    the evening — the sibling price batch empirically writes ~19:04, so 16:00 was
REM    too early and collected 0. Codex 2026-06-01 review.)
REM
REM These are current-user tasks and normally need NO elevation. If schtasks
REM reports "Access is denied", re-run this from an elevated Command Prompt.
REM
REM Usage:   cd to project root, then:  scripts\register_daily_routine_tasks.bat
REM
REM Unregister:
REM   schtasks /Delete /TN "HTR_Daily_Preopen" /F
REM   schtasks /Delete /TN "HTR_Daily_AfterClose" /F
REM Inspect / run now:
REM   schtasks /Query /TN "HTR_Daily_AfterClose" /V /FO LIST
REM   schtasks /Run   /TN "HTR_Daily_AfterClose"
REM ============================================================================
set LAUNCH=%~dp0run_daily_routine.bat

schtasks /Create /TN "HTR_Daily_Preopen" /TR "\"%LAUNCH%\" preopen" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 08:30 /F
schtasks /Create /TN "HTR_Daily_AfterClose" /TR "\"%LAUNCH%\" afterclose" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 19:30 /F

echo.
echo Registered HTR_Daily_Preopen (08:30) and HTR_Daily_AfterClose (19:30), Mon-Fri.
schtasks /Query /TN "HTR_Daily_Preopen" 2>nul
schtasks /Query /TN "HTR_Daily_AfterClose" 2>nul
