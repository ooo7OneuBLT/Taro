"""
診断実験 — hungerを強制固定して喃語内容への影響を直接測定する

12ヶ月学習させた直後、同一プロセス内でhunger=0.0とhunger=1.0を
それぞれ強制した状態で喃語を生成させ、「まんま」との類似度に
差が出るかを比較する。自然に変動するhungerとの相関はノイズが大きく、
「学習されているが弱すぎて埋もれている」のか「そもそも学習されて
いない」のかを区別できないため、この人工的な比較で切り分ける。
"""
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
torch.set_num_threads(2)
try:
    import psutil
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except ImportError:
    pass

log_path = os.path.join(os.path.dirname(__file__), "logs", "diagnostic_progress.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

with open(log_path, "w", encoding="utf-8") as logf:
    def log(msg):
        logf.write(msg + "\n")
        logf.flush()

    try:
        from environment.parent_sim_b import run_simulation_b

        log(f"[学習開始] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        start = time.time()

        r = run_simulation_b(
            max_sim_seconds=31536000,  # 12ヶ月（365日）
            verbose=False,
            run_name="B2-6_diagnostic_train",
        )

        elapsed = time.time() - start
        log(f"[学習完了] {time.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed/60:.1f}分)")
        log(f"泣き{r['cry_count']} 食事{r['feed_count']} 要求語{r['request_count']} "
            f"喃語{r['babble_count']} 睡眠{r['sleep_count']}")

        env = r["env"]

        log("[診断開始] hunger=0.0 と hunger=1.0 で喃語を1000回ずつ生成し比較")
        N = 1000
        sims_low = env.diagnostic_babble_at_hunger(0.0, target_word="まんま", n_samples=N)
        sims_high = env.diagnostic_babble_at_hunger(1.0, target_word="まんま", n_samples=N)

        def summarize(name, sims):
            if not sims:
                log(f"  {name}: 有効サンプルなし")
                return
            avg = sum(sims) / len(sims)
            sd = statistics.pstdev(sims) if len(sims) > 1 else 0.0
            log(f"  {name}: n={len(sims)} 平均={avg:.4f} 標準偏差={sd:.4f} "
                f"最大={max(sims):.4f} 最小={min(sims):.4f}")
            return avg

        avg_low = summarize("hunger=0.0（満腹想定）", sims_low)
        avg_high = summarize("hunger=1.0（空腹想定）", sims_high)

        if avg_low is not None and avg_high is not None:
            diff = avg_high - avg_low
            log(f"差（高hunger - 低hunger）: {diff:.4f}")

            # t検定相当の簡易判定（Welchのt検定）
            n1, n2 = len(sims_low), len(sims_high)
            m1, m2 = avg_low, avg_high
            v1 = statistics.pvariance(sims_low) if n1 > 1 else 0.0
            v2 = statistics.pvariance(sims_high) if n2 > 1 else 0.0
            se = (v1 / n1 + v2 / n2) ** 0.5 if n1 > 0 and n2 > 0 else 0.0
            t_stat = diff / se if se > 0 else 0.0
            log(f"簡易t統計量: {t_stat:.3f}（目安：|t|>2でおおむね有意）")

        log(f"[診断完了] {time.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        import traceback
        log(f"[エラー] {e}")
        log(traceback.format_exc())
