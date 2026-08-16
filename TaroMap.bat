@echo off
rem TaroMap - launches the "what's wired and what does it do" desktop app
rem (run\viewer_tools\wiring_viewer). PySide6 native app, not a browser page.
rem
rem Name note (2026-08-15): named "TaroMap" instead of yet another "...Viewer"
rem to avoid confusion with LiveViewer.bat / LegacyViewer.bat / replayViewer
rem (those replay/preview a running or recorded simulation). This app is a
rem static-ish inspector of what mechanisms exist and what's switched on for
rem a given experiment file, not a playback tool, so the name is deliberately
rem different.
rem
rem Why no "chcp 65001" here (same reasoning as 作業記録（非公開）
rem see that file's own long comment): switching codepage mid-batch-file has a
rem known read-ahead buffering bug. This file's own command line is entirely
rem ASCII (run.viewer_tools.wiring_viewer.app), so the risk is low, but the
rem existing convention (no chcp, just PYTHONIOENCODING=utf-8) is kept anyway
rem for consistency.
cd /d "%~dp0"
title TaroMap
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [error] %PY% not found.
  echo Run this file from the Taro project folder.
  pause
  exit /b 1
)
set PYTHONIOENCODING=utf-8
"%PY%" -m run.viewer_tools.wiring_viewer.app
if errorlevel 1 (
  echo.
  echo ----------------------------------------------------------------
  echo   TaroMap exited with an error. See the messages above.
  echo ----------------------------------------------------------------
  pause
)
