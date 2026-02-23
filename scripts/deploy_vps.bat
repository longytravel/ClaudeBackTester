@echo off
REM ============================================================
REM  Deploy & Run Live Trader on VPS
REM
REM  What this does:
REM    1. Pulls latest code from GitHub
REM    2. Installs/updates dependencies
REM    3. Starts the trader in the background (survives terminal close)
REM
REM  Usage: double-click this file, or run from command prompt
REM  To stop: run stop_trader.bat
REM ============================================================

cd /d %~dp0\..

echo.
echo ========================================
echo   Pulling latest code from GitHub...
echo ========================================
git pull origin master
if %errorlevel% neq 0 (
    echo ERROR: git pull failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Installing dependencies...
echo ========================================
uv sync
if %errorlevel% neq 0 (
    echo ERROR: uv sync failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Starting trader in background...
echo ========================================

REM Start the trader with pythonw (no console window) via START /B
REM Logs go to state/trader.log so you can check progress
REM The trader saves state to disk so it survives crashes

if not exist state mkdir state

start "LiveTrader" /B cmd /c "uv run python scripts/live_trade.py --strategy rsi_mean_reversion --pair EURUSD --timeframe H1 --pipeline results/rsi_eurusd_h1/checkpoint.json --mode practice --state-dir state/rsi_eurusd_h1 > state/trader.log 2>&1"

echo.
echo ========================================
echo   Trader is running in background!
echo ========================================
echo.
echo   Mode:      PRACTICE (demo account)
echo   Strategy:  rsi_mean_reversion
echo   Pair:      EURUSD H1
echo   Logs:      state\trader.log
echo   State:     state\rsi_eurusd_h1\
echo   Heartbeat: state\rsi_eurusd_h1\heartbeat.json
echo.
echo   You can close this window now.
echo   To check status: type "check_trader.bat"
echo   To stop:         type "stop_trader.bat"
echo.
pause
