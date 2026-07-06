@echo off
REM Launcher for the HTR Local Beta v0 daily routine (Rule 15.5, task P10-28).
REM Invoked by the Windows scheduled tasks. Arg 1 = mode (preopen | afterclose).
REM Deterministic, advice-only, no LLM/GPU, no broker. cd to project root so the
REM orchestrator's relative report paths resolve.
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"
if "%HTR_PYTHON%"=="" set HTR_PYTHON=D:\python\python.exe
if not exist "reports\observability" mkdir "reports\observability"
"%HTR_PYTHON%" tools\daily_routine.py --mode %1 >> "reports\observability\daily_routine_task.out" 2>&1
endlocal
