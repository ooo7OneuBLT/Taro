$ErrorActionPreference = "Continue"
Set-Location 'C:\claude\AI\Taro'
$py = ".\.venv\Scripts\python.exe"
& $py run\main.py "F\experiments\F2-140_壁あり消失発話_睡眠両方_学習_個体r1_2026-09-17.json" *> "F\experiments\F2-140_壁あり消失発話_睡眠両方_学習_個体r1_2026-09-17.json.log"; if ($LASTEXITCODE -ne 0) { "止まった: F2-140_壁あり消失発話_睡眠両方_学習_個体r1_2026-09-17.json" | Out-File -Append -Encoding utf8 "F\experiments\_F2-140_鎖.status"; exit 1 }
"済: F2-140_壁あり消失発話_睡眠両方_学習_個体r1_2026-09-17.json" | Out-File -Append -Encoding utf8 "F\experiments\_F2-140_鎖.status"
& $py run\main.py "F\experiments\F2-140b_壁あり消失発話_睡眠両方_テスト_個体r1_2026-09-17.json" *> "F\experiments\F2-140b_壁あり消失発話_睡眠両方_テスト_個体r1_2026-09-17.json.log"; if ($LASTEXITCODE -ne 0) { "止まった: F2-140b_壁あり消失発話_睡眠両方_テスト_個体r1_2026-09-17.json" | Out-File -Append -Encoding utf8 "F\experiments\_F2-140_鎖.status"; exit 1 }
"済: F2-140b_壁あり消失発話_睡眠両方_テスト_個体r1_2026-09-17.json" | Out-File -Append -Encoding utf8 "F\experiments\_F2-140_鎖.status"
"鎖 完了" | Out-File -Append -Encoding utf8 "F\experiments\_F2-140_鎖.status"
