@echo off
REM ============================================================
REM  DEPLOY - Pull latest strategies and start all traders
REM
REM  Put a shortcut to this file on your desktop.
REM  Double-click it any time you push new strategies.
REM  Safe to run multiple times (stops old traders first).
REM ============================================================

cd /d %~dp0

echo.
echo ============================================================
echo   DEPLOY: Pulling latest + starting all strategies
echo ============================================================
echo.

REM 1. Pull latest from GitHub
echo [1/4] Pulling from GitHub...
git pull origin master
if %errorlevel% neq 0 (
    echo ERROR: git pull failed. Check your internet connection.
    pause
    exit /b 1
)
echo      Done.
echo.

REM 2. Install/update dependencies
echo [2/4] Installing dependencies (first time is slow, be patient)...
REM Try uv first (dev machine), fall back to pip (VPS)
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv sync --quiet
) else (
    pip install -r requirements.txt
    pip install -e .
)
echo      Done.
echo.

REM 3. Stop any existing traders
echo [3/4] Stopping old traders...
python scripts/stop_all.py
echo.

REM 4. Launch all strategies
echo [4/4] Launching strategies...
python scripts/start_all.py practice
echo.
pause
