@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "NO_PAUSE=0"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

if not exist "logs" mkdir "logs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set "LOG=logs\morning_briefing_%TS%.log"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

echo =============================================== > "%LOG%"
echo [INFO] MORNING BRIEFING START %date% %time% >> "%LOG%"
echo =============================================== >> "%LOG%"

REM Step 1: Quick data refresh (fetch latest prices only, no full model run)
echo [1/3] Refreshing market data... >> "%LOG%"
python db_update.py --db japan_market.db >> "%LOG%" 2>&1

REM Step 2: Generate full briefing report using yesterday's model output + today's fresh prices
echo [2/4] Generating briefing report... >> "%LOG%"
python quant_briefing.py --mode full --strategy_id sprint >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

REM Step 3: Generate action plan + push to Discord
echo [3/4] Building action plan... >> "%LOG%"
python action_plan_builder.py --db japan_market.db --strategy_id sprint >> "%LOG%" 2>&1

REM Step 4: Briefing ready
echo [4/4] Briefing ready. >> "%LOG%"
echo =============================================== >> "%LOG%"
echo [INFO] MORNING BRIEFING END %date% %time% RC=%RC% >> "%LOG%"
echo =============================================== >> "%LOG%"

if not "%RC%"=="0" (
  echo [ERROR] morning briefing failed. See: "%cd%\%LOG%"
  if not "%NO_PAUSE%"=="1" pause
  exit /b %RC%
)

echo.
echo [OK] Morning briefing ready.
echo   Report: reports\briefing_latest.md
echo   Log:    %LOG%
if not "%NO_PAUSE%"=="1" pause
exit /b 0
