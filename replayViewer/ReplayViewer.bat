@echo off
rem Start the replay viewer via a local HTTP server (avoids file:// CORS errors).
rem Double-click to launch. A browser tab opens automatically. Close this window to stop.
cd /d "%~dp0application"
echo Starting replay viewer...
echo Browser will open automatically. Close this window to stop.
start "" "http://localhost:8777/index.html"
python -m http.server 8777
