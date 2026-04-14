@echo off
REM Scheduled task: run full multi-strategy pipeline after Japan market close.
REM Recommended cron: 17:00 JST every trading day.
REM
REM This replaces the legacy scheduled_daily_run.cmd when running multi-strategy
REM (sprint + high52w + amihud + all paper lanes).

cd /d "%~dp0"
set LOG_DIR=%~dp0logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set STAMP=%DATE:~-10,4%-%DATE:~-5,2%-%DATE:~-2,2%
set LOG=%LOG_DIR%\scheduled_all_strategies_%STAMP%.log

python run_all_strategies.py >> "%LOG%" 2>&1
echo [%DATE% %TIME%] rc=%ERRORLEVEL% >> "%LOG%"
