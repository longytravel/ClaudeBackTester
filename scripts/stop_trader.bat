@echo off
REM Stop the running live trader process
echo Stopping live trader...
taskkill /F /FI "WINDOWTITLE eq LiveTrader" >nul 2>&1
REM Also kill any python running live_trade.py
wmic process where "commandline like '%%live_trade.py%%'" call terminate >nul 2>&1
echo Trader stopped.
pause
