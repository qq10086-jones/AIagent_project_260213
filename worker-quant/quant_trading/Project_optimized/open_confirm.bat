@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist "logs" mkdir "logs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set "LOG=logs\open_confirm_%TS%.log"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

echo =============================================== > "%LOG%"
echo [INFO] OPEN CONFIRM START %date% %time% >> "%LOG%"
echo =============================================== >> "%LOG%"

REM Step 1: Quick intraday price update for held positions
echo [1/2] Updating intraday prices... >> "%LOG%"
python intraday_update.py --db japan_market.db --period 1d --interval 1m >> "%LOG%" 2>&1

REM Step 2: Check pending actions from action plan
echo [2/2] Checking pending actions... >> "%LOG%"
python check_pending_actions.py --db japan_market.db --strategy_id sprint --config config.yaml --check_label 09:30 --alert_level warning >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo =============================================== >> "%LOG%"
echo [INFO] OPEN CONFIRM END %date% %time% RC=%RC% >> "%LOG%"
echo =============================================== >> "%LOG%"

if not "%RC%"=="0" (
  echo [ERROR] open_confirm failed. See: "%cd%\%LOG%"
)
exit /b %RC%
