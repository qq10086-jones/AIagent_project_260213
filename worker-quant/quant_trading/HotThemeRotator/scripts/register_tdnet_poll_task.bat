@echo off
REM Register TDnet RSS polling task with Windows Task Scheduler
REM Runs every 15 minutes (Rule 9.2 within-session refresh cadence)
REM Per P10-14 task acceptance + Rule 12.2 stale fail-closed contract
REM
REM Usage:
REM   1. Open Command Prompt AS ADMINISTRATOR
REM   2. cd to project root
REM   3. scripts\register_tdnet_poll_task.bat
REM
REM To unregister:
REM   schtasks /Delete /TN "HTR_TDnet_RSS_Poll" /F

set TASK_NAME=HTR_TDnet_RSS_Poll
set PROJECT_ROOT=%~dp0..
set SCRIPT=%PROJECT_ROOT%\tools\poll_tdnet_rss.py
set PYTHON=python

schtasks /Create ^
    /TN "%TASK_NAME%" ^
    /TR "\"%PYTHON%\" \"%SCRIPT%\" --latest" ^
    /SC MINUTE ^
    /MO 15 ^
    /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Task "%TASK_NAME%" registered successfully.
    echo Will run every 15 minutes calling: %PYTHON% %SCRIPT% --latest
    echo View: schtasks /Query /TN "%TASK_NAME%"
    echo Unregister: schtasks /Delete /TN "%TASK_NAME%" /F
) else (
    echo.
    echo Failed to register task. Make sure you are running as Administrator.
    exit /b 1
)
