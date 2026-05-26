@echo off
REM Register delayed price health report task with Windows Task Scheduler
REM Observability-only local job for P10-19 / P10-20 Stage 0 cockpit.
REM No notification. No broker. No order.
REM
REM Usage:
REM   1. Open Command Prompt AS ADMINISTRATOR
REM   2. cd to project root
REM   3. scripts\register_price_health_task.bat 6768.T,5074.T,6962.T
REM
REM To unregister:
REM   schtasks /Delete /TN "HTR_Price_Health_Report" /F

set TASK_NAME=HTR_Price_Health_Report
set PROJECT_ROOT=%~dp0..
set SCRIPT=%PROJECT_ROOT%\tools\write_price_health_report.py
set PYTHON=python
set SYMBOLS=%~1

if "%SYMBOLS%"=="" (
    echo ERROR: symbols argument is required, e.g. 6768.T,5074.T,6962.T
    exit /b 2
)

schtasks /Create ^
    /TN "%TASK_NAME%" ^
    /TR "\"%PYTHON%\" \"%SCRIPT%\" --symbols \"%SYMBOLS%\" --base-dir \"%PROJECT_ROOT%\"" ^
    /SC MINUTE ^
    /MO 15 ^
    /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Task "%TASK_NAME%" registered successfully.
    echo Will run every 15 minutes calling: %PYTHON% %SCRIPT% --symbols %SYMBOLS% --base-dir %PROJECT_ROOT%
    echo View: schtasks /Query /TN "%TASK_NAME%"
    echo Unregister: schtasks /Delete /TN "%TASK_NAME%" /F
) else (
    echo.
    echo Failed to register task. Make sure you are running as Administrator.
    exit /b 1
)
