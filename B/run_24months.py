"""
24ヶ月（2年）シミュレーション — スタンドアロン実行スクリプト

run_12months.py の2年版。文脈一貫（空腹→まんま）が12ヶ月以降も
人間のように自然に強まるかを見るために期間を延ばす。報酬設計は
いじらず、育つ時間を増やすだけ（B2-9のコードのまま）。

声道・NEの成熟はstage3_time（12ヶ月）を基準にしており、2年目は
完全成熟（探索の天井0.3・声道stage3）のまま継続する。

複数試行対応：引数でtrial IDを渡すと、run_nameとログ名にサフィックスが付く。
各プロセスは乱数シードが別々なので、複数同時起動＝独立した複数試行。
1試行=2スレッド制限なので複数同時でも負荷は緩やか。
  例: python run_24months.py trial1
      python run_24months.py trial2
"""
import sys
import os
import time

trial_id = sys.argv[1] if len(sys.argv) > 1 else ""
suffix = f"_{trial_id}" if trial_id else ""

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
torch.set_num_threads(2)

try:
    import psutil
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except ImportError:
    pass

log_path = os.path.join(os.path.dirname(__file__), "logs", f"sim24_progress{suffix}.txt")
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
            max_sim_seconds=63072000,  # 24ヶ月（730日）
            verbose=False,
            run_name=f"B2-9_24months{suffix}",
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
