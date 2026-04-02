@echo off
REM ============================================================
REM  Register Windows Scheduled Task: QuantDailyRun
REM  Runs daily_run.bat on weekdays at 16:30 local time
REM ============================================================

set TASK_NAME=QuantDailyRun
for %%I in ("%~dp0..") do set PROJECT_DIR=%%~fI
set BAT_PATH=%PROJECT_DIR%\scheduled_daily_run.cmd

REM Delete old task if exists
schtasks /Delete /TN "%TASK_NAME%" /F 2>nul

REM Create task: Mon-Fri at 16:30 local time
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "\"%BAT_PATH%\"" ^
  /SC WEEKLY ^
  /D MON,TUE,WED,THU,FRI ^
  /ST 16:30 ^
  /F

echo.
echo Task "%TASK_NAME%" registered. Runs Mon-Fri at 16:30 local time.
echo BAT_PATH=%BAT_PATH%
echo To verify: schtasks /Query /TN "%TASK_NAME%" /FO LIST
echo To run now: schtasks /Run /TN "%TASK_NAME%"
echo To delete:  schtasks /Delete /TN "%TASK_NAME%" /F
pause
