@echo off
rem リプレイビューアをローカルHTTPサーバー経由で開く（file://のCORS回避）。
rem ダブルクリックで起動 → 既定ブラウザで自動的に開く。閉じるときはこの黒い窓を閉じる。
chcp 65001 >nul
cd /d "%~dp0application"
echo リプレイビューアを起動します...
echo ブラウザが自動で開きます。終了するにはこの窓を閉じてください。
start "" "http://localhost:8777/index.html"
python -m http.server 8777
