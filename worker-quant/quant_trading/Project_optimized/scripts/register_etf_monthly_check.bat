@echo off
REM ============================================================
REM Register ETF monthly check Task Scheduler entry
REM (2026-04-28 Path A: Long-Only ETF Buy-Hold)
REM
REM Triggers: 1st of every month, 09:30 JST
REM Action: run etf_monthly_check.py and email report
REM
REM Prerequisites:
REM   1. Set EMAIL_USER and EMAIL_PASS as user environment variables
REM      (Win + R → sysdm.cpl → Advanced → Environment Variables)
REM      EMAIL_USER = lwyssq@gmail.com
REM      EMAIL_PASS = <Gmail App Password from https://myaccount.google.com/apppasswords>
REM
REM   2. Run this .bat as Administrator (right-click → Run as administrator)
REM ============================================================

set TASK_NAME=ETF_Monthly_Check
set SCRIPT_DIR=E:\AIagent_project_260213\worker-quant\quant_trading\Project_optimized
set PYTHON_EXE=D:\python\python.exe

echo Removing any existing task with name %TASK_NAME%...
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

echo Registering Task Scheduler entry...
schtasks /create ^
  /tn "%TASK_NAME%" ^
  /tr "cmd /c cd /d %SCRIPT_DIR% && %PYTHON_EXE% etf_monthly_check.py >> logs\etf_monthly.log 2>&1" ^
  /sc monthly ^
  /d 1 ^
  /st 09:30 ^
  /ru "%USERNAME%" ^
  /rl highest ^
  /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Task registered successfully.
    echo   Trigger: 1st of every month at 09:30 JST
    echo   Logs:    %SCRIPT_DIR%\logs\etf_monthly.log
    echo.
    echo To verify: schtasks /query /tn "%TASK_NAME%"
    echo To test now: schtasks /run /tn "%TASK_NAME%"
) else (
    echo.
    echo ✗ Task registration failed. Run as Administrator?
)

pause
