@echo off
cd /d %~dp0
uv run python scripts/status_all.py
pause
