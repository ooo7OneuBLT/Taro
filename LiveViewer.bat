@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Taro LiveViewer
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [error] %PY% not found.
  echo Run this file from the Taro project folder.
  pause
  exit /b 1
)
set PYTHONIOENCODING=utf-8
"%PY%" "run\viewer_tools\e_launch_qt.py"
echo.
echo ----------------------------------------------------------------
echo   this is a stage1 prototype (skeleton + scene panel only).
echo   the old viewer (Viewer wo hiraku.bat) still has everything else.
echo ----------------------------------------------------------------
pause
