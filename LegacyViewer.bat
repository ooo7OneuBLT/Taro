@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Taro LegacyViewer (旧版・新版完成後に廃止)
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [error] %PY% not found.
  echo Run this file from the Taro project folder.
  pause
  exit /b 1
)
set PYTHONIOENCODING=utf-8
"%PY%" "run\viewer_tools\e_launch.py"
echo.
echo ----------------------------------------------------------------
pause
