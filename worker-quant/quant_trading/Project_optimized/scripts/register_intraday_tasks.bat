@echo off
REM ============================================================
REM  Register Windows Scheduled Tasks for Intraday Monitoring
REM  JST times (local machine is Beijing time = JST - 1h)
REM ============================================================

for %%I in ("%~dp0..") do set PROJECT_DIR=%%~fI

REM --- Open Confirm: 09:30 JST = 08:30 Beijing ---
set TASK_NAME=QuantOpenConfirm
set BAT_PATH=%PROJECT_DIR%\open_confirm.bat
schtasks /Delete /TN "%TASK_NAME%" /F 2>nul
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "\"%BAT_PATH%\"" ^
  /SC WEEKLY ^
  /D MON,TUE,WED,THU,FRI ^
  /ST 08:30 ^
  /F
echo Task "%TASK_NAME%" registered (08:30 local = 09:30 JST).

REM --- Midday Check: 11:30 JST = 10:30 Beijing ---
set TASK_NAME=QuantMiddayCheck
set BAT_PATH=%PROJECT_DIR%\midday_check.bat
schtasks /Delete /TN "%TASK_NAME%" /F 2>nul
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "\"%BAT_PATH%\"" ^
  /SC WEEKLY ^
  /D MON,TUE,WED,THU,FRI ^
  /ST 10:30 ^
  /F
echo Task "%TASK_NAME%" registered (10:30 local = 11:30 JST).

REM --- Pre-Close: 14:00 JST = 13:00 Beijing ---
set TASK_NAME=QuantPreClose
set BAT_PATH=%PROJECT_DIR%\pre_close.bat
schtasks /Delete /TN "%TASK_NAME%" /F 2>nul
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "\"%BAT_PATH%\"" ^
  /SC WEEKLY ^
  /D MON,TUE,WED,THU,FRI ^
  /ST 13:00 ^
  /F
echo Task "%TASK_NAME%" registered (13:00 local = 14:00 JST).

echo.
echo All intraday tasks registered.
echo To verify: schtasks /Query /TN "QuantOpenConfirm" /FO LIST
pause
