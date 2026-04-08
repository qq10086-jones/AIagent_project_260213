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

REM Step 0: Cross-asset overnight signals (S&P500, USD/JPY, VIX, NK futures, Oil, Gold, Copper, SOX)
echo [0/5] Fetching cross-asset signals... >> "%LOG%"
python cross_asset_signals.py --db japan_market.db >> "%LOG%" 2>&1

REM Step 0.5: Macro event detection (rule engine — L1/L2 event check)
echo [0.5/6] Running macro event detector... >> "%LOG%"
python macro_event_detector.py --db japan_market.db >> "%LOG%" 2>&1

REM Step 0.7: LLM macro analysis (Gemma via Ollama — only runs if L1/L2 detected)
echo [0.7/6] Running macro digest (LLM)... >> "%LOG%"
python macro_digest.py --db japan_market.db >> "%LOG%" 2>&1

REM Step 1: Quick data refresh (fetch latest prices only, no full model run)
echo [1/4] Refreshing market data... >> "%LOG%"
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
