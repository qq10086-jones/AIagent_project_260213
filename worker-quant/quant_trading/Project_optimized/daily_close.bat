@echo off
setlocal enabledelayedexpansion

REM ==========================
REM Daily close runner (Mode A)
REM ==========================

REM 1) Go to project dir
cd /d E:\quantum\Project_optimized

REM 2) Basic checks
if not exist run_pipeline.py (
  echo [ERROR] run_pipeline.py not found in %cd%
  pause
  exit /b 1
)

if not exist config.yaml (
  echo [ERROR] config.yaml not found in %cd%
  pause
  exit /b 1
)

REM 3) Prepare logs dir
if not exist logs mkdir logs

REM 4) Timestamp for log filename (YYYYMMDD_HHMMSS)
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do (
  set YYYY=%%a
  set MM=%%b
  set DD=%%c
)
for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
  set HH=%%a
  set MN=%%b
  set SS=%%c
)
REM remove possible leading spaces in HH
set HH=%HH: =0%
set TS=%YYYY%%MM%%DD%_%HH%%MN%%SS%

set LOG=logs\daily_close_%TS%.log

echo =============================================== > "%LOG%"
echo [INFO] START %date% %time% >> "%LOG%"
echo [INFO] CWD=%cd% >> "%LOG%"
echo [INFO] Running: python run_pipeline.py --config config.yaml >> "%LOG%"
echo =============================================== >> "%LOG%"

REM 5) Run pipeline
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

python run_pipeline.py --config config.yaml >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo =============================================== >> "%LOG%"
echo [INFO] END %date% %time% RC=%RC% >> "%LOG%"
echo =============================================== >> "%LOG%"

if not "%RC%"=="0" (
  echo.
  echo [ERROR] Pipeline failed. See log:
  echo   %cd%\%LOG%
  pause
  exit /b %RC%
)

echo.
echo [OK] Pipeline finished successfully.

echo.
echo [INFO] Latest run from DB (decision_runs):

python -c "import sqlite3; db=sqlite3.connect('japan_market.db'); row=db.execute('select run_id, asof, status, snapshot_path from decision_runs order by ts desc limit 1').fetchone(); db.close(); print('  run_id      :', row[0]); print('  asof        :', row[1]); print('  status      :', row[2]); print('  snapshot    :', row[3])" >> "%LOG%" 2>&1

python -c "import sqlite3; db=sqlite3.connect('japan_market.db'); row=db.execute('select run_id, asof, status, snapshot_path from decision_runs order by ts desc limit 1').fetchone(); db.close(); print('  run_id      :', row[0]); print('  asof        :', row[1]); print('  status      :', row[2]); print('  snapshot    :', row[3])"

echo Log saved to:
echo   %cd%\%LOG%
pause
exit /b 0
