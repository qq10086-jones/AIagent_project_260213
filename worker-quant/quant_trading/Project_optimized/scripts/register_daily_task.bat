@echo off
REM ============================================================
REM  Register Windows Scheduled Task: QuantDailyRun
REM  Runs daily_run.bat on weekdays at 16:30 JST
REM  Run this script ONCE as Administrator
REM ============================================================

set TASK_NAME=QuantDailyRun
set BAT_PATH=C:\Users\linweiye\AIagent_project_260213\worker-quant\quant_trading\Project_optimized\daily_run.bat

REM Delete old task if exists
schtasks /Delete /TN "%TASK_NAME%" /F 2>nul

REM Create task: Mon-Fri at 16:30 (JST = UTC+9, so 16:30 JST is 07:30 UTC)
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "\"%BAT_PATH%\"" ^
  /SC WEEKLY ^
  /D MON,TUE,WED,THU,FRI ^
  /ST 16:30 ^
  /RL HIGHEST ^
  /F

echo.
echo Task "%TASK_NAME%" registered. Runs Mon-Fri at 16:30 JST.
echo To verify: schtasks /Query /TN "%TASK_NAME%" /FO LIST
echo To run now: schtasks /Run /TN "%TASK_NAME%"
echo To delete:  schtasks /Delete /TN "%TASK_NAME%" /F
pause
