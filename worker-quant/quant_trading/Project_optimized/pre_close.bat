@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist "logs" mkdir "logs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set "LOG=logs\pre_close_%TS%.log"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

echo =============================================== > "%LOG%"
echo [INFO] PRE-CLOSE CHECK START %date% %time% >> "%LOG%"
echo =============================================== >> "%LOG%"

python intraday_monitor.py --mode pre_close --db japan_market.db --config config.yaml --strategy_id sprint >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo =============================================== >> "%LOG%"
echo [INFO] PRE-CLOSE CHECK END %date% %time% RC=%RC% >> "%LOG%"
echo =============================================== >> "%LOG%"

if not "%RC%"=="0" (
  echo [ERROR] pre_close failed. See: "%cd%\%LOG%"
)
exit /b %RC%
