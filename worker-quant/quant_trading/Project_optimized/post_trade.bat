@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist "logs" mkdir "logs"

REM Robust timestamp
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set "LOG=logs\post_trade_%TS%.log"

echo =============================================== > "%LOG%"
echo [INFO] START %date% %time% >> "%LOG%"
echo [INFO] CWD=%cd% >> "%LOG%"
echo =============================================== >> "%LOG%"

REM 1) Read latest run_id & asof from DB
for /f "usebackq tokens=*" %%a in (`python -c "import sqlite3;db=sqlite3.connect('japan_market.db');r=db.execute('select run_id, asof from decision_runs order by ts desc limit 1').fetchone();db.close();print(r[0])"`) do set RUN_ID=%%a
for /f "usebackq tokens=*" %%a in (`python -c "import sqlite3;db=sqlite3.connect('japan_market.db');r=db.execute('select run_id, asof from decision_runs order by ts desc limit 1').fetchone();db.close();print(r[1])"`) do set ASOF=%%a

echo [INFO] Latest run_id=%RUN_ID% asof=%ASOF% >> "%LOG%"
echo Latest run_id=%RUN_ID%
echo Latest asof=%ASOF%
echo.

REM 2) Choose input mode
echo Choose fills input mode:
echo   [1] Import from file (xlsx/csv)
echo   [2] Manual entry (interactive)
echo.
set /p MODE=Enter 1 or 2: 

if "%MODE%"=="1" goto :FILEMODE
if "%MODE%"=="2" goto :MANUALMODE

echo [ERROR] Invalid mode: %MODE%
echo [ERROR] Invalid mode: %MODE% >> "%LOG%"
pause
exit /b 1


:FILEMODE
REM 3A) Ask user for fills file path
set /p FILLS_FILE=Enter fills file path (xlsx/csv, e.g. fills_2026-02-03.xlsx): 

if not exist "%FILLS_FILE%" (
  echo [ERROR] File not found: %FILLS_FILE%
  echo [ERROR] File not found: %FILLS_FILE% >> "%LOG%"
  pause
  exit /b 1
)

echo [INFO] Importing fills from file... >> "%LOG%"
python import_fills.py --db japan_market.db --run_id "%RUN_ID%" --asof "%ASOF%" --file "%FILLS_FILE%" >> "%LOG%" 2>&1
if errorlevel 1 goto :FAIL

goto :POSTSTEPS


:MANUALMODE
REM 3B) Manual entry
if not exist "manual_fills_entry.py" (
  echo [ERROR] manual_fills_entry.py not found in %cd%
  echo [ERROR] manual_fills_entry.py not found in %cd% >> "%LOG%"
  pause
  exit /b 1
)

echo [INFO] Manual fills entry... >> "%LOG%"
echo.
echo [MANUAL MODE] Please input fills in this window. (This step is NOT logged.)
echo.
python manual_fills_entry.py
if errorlevel 1 goto :FAIL

goto :POSTSTEPS


:POSTSTEPS
REM 4) Build positions
echo [INFO] Building positions... >> "%LOG%"
python build_positions.py --db japan_market.db --run_id "%RUN_ID%" --asof "%ASOF%" >> "%LOG%" 2>&1
if errorlevel 1 goto :FAIL

REM 5) Execution report
echo [INFO] Writing execution report... >> "%LOG%"
python execution_report.py --db japan_market.db --run_id "%RUN_ID%" --asof "%ASOF%" >> "%LOG%" 2>&1
if errorlevel 1 goto :FAIL

REM 6) Account snapshot (first time may need initial_cash)
echo.
set /p INIT_CASH=Enter initial_cash (JPY). If already have previous snapshots, just press Enter: 

if "%INIT_CASH%"=="" (
  echo [INFO] Building account snapshot (auto prev cash)... >> "%LOG%"
  python build_account_snapshot.py --db "japan_market.db" --run_id "%RUN_ID%" --asof "%ASOF%" >> "%LOG%" 2>&1
) else (
  echo [INFO] Building account snapshot (initial_cash=%INIT_CASH%)... >> "%LOG%"
  python build_account_snapshot.py --db "japan_market.db" --run_id "%RUN_ID%" --asof "%ASOF%" --initial_cash "%INIT_CASH%" >> "%LOG%" 2>&1
)
if errorlevel 1 goto :FAIL


echo =============================================== >> "%LOG%"
echo [OK] DONE %date% %time% >> "%LOG%"
echo =============================================== >> "%LOG%"

echo.
echo [OK] Post-trade steps complete.
echo run_id=%RUN_ID%
echo asof=%ASOF%
echo Log saved to:
echo   "%cd%\%LOG%"
pause
exit /b 0


:FAIL
echo =============================================== >> "%LOG%"
echo [ERROR] FAILED %date% %time% >> "%LOG%"
echo =============================================== >> "%LOG%"

echo.
echo [ERROR] Post-trade failed. See log:
echo   "%cd%\%LOG%"
pause
exit /b 1
