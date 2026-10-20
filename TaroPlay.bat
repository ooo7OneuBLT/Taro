@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Taro - play as parent
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [error] %PY% not found.
  echo Run this file from the Taro project folder.
  pause
  exit /b 1
)
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
"%PY%" -m core_v2.tools.board_play --interactive
pause
