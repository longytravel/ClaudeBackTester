@echo off
cd /d %~dp0
uv run python scripts/stop_all.py
pause
