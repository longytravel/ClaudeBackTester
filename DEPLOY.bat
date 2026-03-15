@echo off
REM ============================================================
REM  DEPLOY - Pull latest, stop old traders, deploy fresh
REM
REM  1. Stops any running traders
REM  2. Pulls latest code from GitHub
REM  3. Cleans old state (fresh start)
REM  4. Installs deps if needed
REM  5. Launches all strategies with backtest results
REM
REM  Just double-click. Nothing to configure. Ever.
REM ============================================================

cd /d %~dp0

echo.
echo ============================================================
echo   DEPLOY
echo ============================================================
echo.

REM Stop any running traders first
echo Stopping old traders...
python scripts/stop_all.py 2>nul
echo.

echo Pulling from GitHub...
git pull origin master
IF ERRORLEVEL 1 (
    echo ERROR: git pull failed.
    pause
    exit /b 1
)

REM Create .env from example on first deploy
IF NOT EXIST .env (
    echo.
    echo First deploy - creating .env from .env.example...
    copy .env.example .env
)

REM Clean old state for fresh start
echo.
echo Cleaning old state...
if exist state rmdir /s /q state
mkdir state

echo.
python scripts/ensure_deps.py

echo.
echo Launching strategies...
python scripts/start_all.py --testing
echo.
pause
