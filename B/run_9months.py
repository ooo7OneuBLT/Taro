"""
9ヶ月シミュレーション — スタンドアロン実行スクリプト

Claude Codeのセッションとは独立して動く。
完了・進捗はログファイルに書き込む。
"""
import sys
import os
import time
import io

# パスを通す
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ログをファイルに書き出す（Windows terminalのエンコード問題回避）
log_path = os.path.join(os.path.dirname(__file__), "logs", "sim_progress.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

with open(log_path, "w", encoding="utf-8") as logf:
    def log(msg):
        logf.write(msg + "\n")
        logf.flush()

    try:
        from environment.parent_sim_b import run_simulation_b

        log(f"[開始] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        start = time.time()

        r = run_simulation_b(
            max_sim_seconds=23328000,  # 9ヶ月
            verbose=False,
            run_name="B9_9months",
        )

        elapsed = time.time() - start
        log(f"[完了] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"所要時間: {elapsed:.1f}秒 ({elapsed/60:.1f}分)")
        log(f"喃語: {r['babble_count']}回")
        log(f"睡眠: {r['sleep_count']}回")
        log(f"定着: {r['consolidate_count']}件")
        log(f"発話: {r['speak_count']}回")

    except Exception as e:
        import traceback
        log(f"[エラー] {e}")
        log(traceback.format_exc())
