@echo off
REM ============================================================
REM  Register Windows Scheduled Task: QuantMorningBriefing
REM  Runs morning_briefing.bat on weekdays at 08:30 local time
REM  (30 min before Tokyo market open)
REM ============================================================

set TASK_NAME=QuantMorningBriefing
for %%I in ("%~dp0..") do set PROJECT_DIR=%%~fI
set BAT_PATH=%PROJECT_DIR%\scheduled_morning_briefing.cmd

REM Delete old task if exists
schtasks /Delete /TN "%TASK_NAME%" /F 2>nul

REM Create task: Mon-Fri at 08:30 local time
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "\"%BAT_PATH%\"" ^
  /SC WEEKLY ^
  /D MON,TUE,WED,THU,FRI ^
  /ST 08:30 ^
  /F

echo.
echo Task "%TASK_NAME%" registered. Runs Mon-Fri at 08:30 local time.
echo BAT_PATH=%BAT_PATH%
echo To verify: schtasks /Query /TN "%TASK_NAME%" /FO LIST
echo To run now: schtasks /Run /TN "%TASK_NAME%"
echo To delete:  schtasks /Delete /TN "%TASK_NAME%" /F
pause
