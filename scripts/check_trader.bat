@echo off
REM Check if the trader is running and show its status
echo.
echo ========================================
echo   Trader Status Check
echo ========================================

REM Check if process is running
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python" >nul
if %errorlevel% equ 0 (
    echo   Process: RUNNING
) else (
    echo   Process: NOT RUNNING
)

echo.

REM Show heartbeat (last update time)
set "hb=state\rsi_eurusd_h1\heartbeat.json"
if exist %hb% (
    echo   Last heartbeat:
    type %hb%
) else (
    echo   No heartbeat file found yet
)

echo.

REM Show last 20 lines of log
set "log=state\trader.log"
if exist %log% (
    echo   Recent log:
    echo   ----------
    powershell -command "Get-Content '%log%' -Tail 20"
) else (
    echo   No log file yet
)

echo.
pause
