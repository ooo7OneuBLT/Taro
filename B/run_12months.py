"""
12ヶ月（1年）シミュレーション — スタンドアロン実行スクリプト

Claude Codeのセッションとは独立して動く。
完了・進捗はログファイルに書き込む。

人間の初語出現の平均は生後12ヶ月（MacArthur-Bates CDI）。
9ヶ月（旧run_9months.py）は声道stage2の解禁時期を流用しただけで
初語の判定時期としては早すぎたため、12ヶ月に変更した。

複数試行対応：引数でtrial IDを渡すと、run_nameとログファイル名に
サフィックスを付ける。各プロセスは乱数シードが別々（OSのエントロピー
由来）なので、同じスクリプトを複数プロセスで同時起動するだけで
独立した複数試行になる（Pythonプロセスは起動時に自動でシードされる）。
1試行=スレッド2つに制限しているため、複数同時実行してもPCへの
負荷は緩やか。
  例: python run_12months.py trial1
      python run_12months.py trial2
"""
import sys
import os
import time
import io

trial_id = sys.argv[1] if len(sys.argv) > 1 else ""
suffix = f"_{trial_id}" if trial_id else ""

# パスを通す
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# CPU実行時、PyTorchがデフォルトで全論理コアを使おうとし、
# ブラウザ等の他アプリと取り合いになっていた。この処理は極小テンソルの
# 逐次演算が大半でスレッド数を増やしても速くならないため、スレッド数を
# 絞って他アプリにCPUを譲る（体感の重さの軽減が目的、速度への影響は軽微）。
import torch
torch.set_num_threads(2)

# プロセス優先度を下げ、他の操作（ブラウザ等）を優先させる
try:
    import psutil
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except ImportError:
    pass

# ログをファイルに書き出す（Windows terminalのエンコード問題回避）
log_path = os.path.join(os.path.dirname(__file__), "logs", f"sim_progress{suffix}.txt")
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
            max_sim_seconds=31536000,  # 12ヶ月（365日）
            verbose=False,
            run_name=f"B2-9_12months{suffix}",
        )

        elapsed = time.time() - start
        log(f"[完了] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"所要時間: {elapsed:.1f}秒 ({elapsed/60:.1f}分)")
        log(f"泣き: {r['cry_count']}回")
        log(f"食事: {r['feed_count']}回")
        log(f"要求語: {r['request_count']}回")
        log(f"喃語: {r['babble_count']}回")
        log(f"睡眠: {r['sleep_count']}回")
        log(f"定着: {r['consolidate_count']}件")
        log(f"発話: {r['speak_count']}回")

    except Exception as e:
        import traceback
        log(f"[エラー] {e}")
        log(traceback.format_exc())
