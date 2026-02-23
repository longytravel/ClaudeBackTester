@echo off
REM ============================================================
REM  DEPLOY - Pull latest strategies and start all traders
REM
REM  SAFE: only starts NEW strategies. Running traders with
REM  open trades are left alone (no interruption).
REM
REM  Put a shortcut to this file on your desktop.
REM  Double-click it any time you push new strategies.
REM ============================================================

cd /d %~dp0

echo.
echo ============================================================
echo   DEPLOY: Pulling latest + starting NEW strategies
echo ============================================================
echo.

REM 1. Pull latest from GitHub
echo [1/2] Pulling from GitHub...
git pull origin master
if %errorlevel% neq 0 (
    echo ERROR: git pull failed. Check your internet connection.
    pause
    exit /b 1
)
echo      Done.
echo.

REM Install dependencies only on first run (creates .installed marker)
if not exist .installed (
    echo Installing dependencies (first time only, be patient)...
    where uv >nul 2>&1
    if %errorlevel% equ 0 (
        uv sync --quiet
    ) else (
        pip install -r requirements.txt
        pip install -e .
    )
    echo. > .installed
    echo      Done.
    echo.
)

REM 2. Launch strategies (skips already-running ones)
echo [2/2] Launching strategies...
python scripts/start_all.py practice
echo.
pause
