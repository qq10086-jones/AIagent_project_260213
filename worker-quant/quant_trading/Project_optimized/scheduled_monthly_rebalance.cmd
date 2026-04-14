@echo off
REM Scheduled task: monthly real-money rebalance for alpha_factor strategies.
REM Recommended cron: 10:00 JST, daily (script itself checks calendar).
REM The script no-ops on non-first-of-month days.

cd /d "%~dp0"
set LOG_DIR=%~dp0logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set STAMP=%DATE:~-10,4%-%DATE:~-5,2%-%DATE:~-2,2%
set LOG=%LOG_DIR%\monthly_rebalance_%STAMP%.log

python monthly_rebalance.py >> "%LOG%" 2>&1
echo [%DATE% %TIME%] rc=%ERRORLEVEL% >> "%LOG%"
