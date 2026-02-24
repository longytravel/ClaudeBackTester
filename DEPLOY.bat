@echo off
REM ============================================================
REM  DEPLOY - Pull latest strategies and start all traders
REM
REM  SAFE: only starts NEW strategies. Running traders with
REM  open trades are left alone (no interruption).
REM  AUTO: detects when dependencies change and reinstalls.
REM
REM  Just double-click. Nothing to configure. Ever.
REM ============================================================

cd /d %~dp0

echo.
echo ============================================================
echo   DEPLOY
echo ============================================================
echo.

echo Pulling from GitHub...
git pull origin master
IF ERRORLEVEL 1 (
    echo ERROR: git pull failed.
    pause
    exit /b 1
)

REM Check .env exists (needed for MT5 login)
IF NOT EXIST .env (
    echo.
    echo ERROR: .env file not found!
    echo Create .env with MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    echo See .env.example for format.
    pause
    exit /b 1
)

echo.
python scripts/ensure_deps.py

echo.
echo Launching strategies...
python scripts/start_all.py practice
echo.
pause
