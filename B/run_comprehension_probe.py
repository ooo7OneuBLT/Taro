"""
理解テスト（A案）— 学習後の太郎に語を「聞かせて」内部を読む。

産出側（diagnostic_babble_at_hunger）は「空腹→まんまと言う」を測るが、それは
要求発声(conditioned mand)でも成立し意味理解を意味しない。こちらは逆に「聞く」側：
太郎に語を聞かせた直後の内部（critic価値・隠れ状態・聞いた後の発声傾向）を読み、
「まんまを特別な音として認識しているか(Rung1)」「ごはんの予期が立つか(Rung2)」を調べる。

使い方: python run_comprehension_probe.py [月数]   （既定12ヶ月。2年なら 24）
"""
import sys
import os
import time
import math

months = int(sys.argv[1]) if len(sys.argv) > 1 else 12
suffix = f"_{months}mo"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
torch.set_num_threads(2)
try:
    import psutil
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except ImportError:
    pass


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


log_path = os.path.join(os.path.dirname(__file__), "logs", f"comprehension_probe{suffix}.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

with open(log_path, "w", encoding="utf-8") as logf:
    def log(msg):
        logf.write(msg + "\n")
        logf.flush()

    try:
        from environment.parent_sim_b import run_simulation_b

        log(f"[学習開始] {time.strftime('%Y-%m-%d %H:%M:%S')} ({months}ヶ月)")
        start = time.time()
        r = run_simulation_b(
            max_sim_seconds=months * 2592000,
            verbose=False,
            run_name=f"B2-11_comprehension_train{suffix}",
        )
        log(f"[学習完了] {time.strftime('%Y-%m-%d %H:%M:%S')} ({(time.time()-start)/60:.1f}分)")
        log(f"泣き{r['cry_count']} 食事{r['feed_count']} 要求語{r['request_count']} 喃語{r['babble_count']}")

        env = r["env"]

        # 聞かせる語：まんま(標的) / ままん(同じ音・並び違いの非語) / あうあ(無関係な母音)
        words = ["まんま", "ままん", "あうあ"]
        hungers = [(0.1, "満腹"), (0.9, "空腹")]

        log("\n[理解テスト] 語を聞かせた直後の内部を読む（産出でなく受信側）")
        log("  ※ 指標訂正(2026-07-02)：理解＝『語→食べ物(授乳)の先取り』。生の予測値そのものは")
        log("     hungerで説明できる分を含むので理解の証拠にしない。hungerを固定し“語だけ”を")
        log("     変えたときの差（＝語の寄与）が本命指標（乳児研究の予期的注視に対応, 参考文献§9）。")
        results = {}
        for hv, hname in hungers:
            for w in words:
                res = env.comprehension_probe(w, hv, n_samples=200)
                results[(hname, w)] = res
                sat = res.get("satiety")
                sat_s = f"{sat:.4f}" if sat is not None else "N/A"
                log(f"  [{hname}] 「{w}」を聞く → 食べ物予期(生値)={sat_s} ／ critic価値={res['critic_value']:.4f} "
                    f"／ 聞いた後の発声のまんま類似={res['echoic_mama_sim']:.4f} (n={res['n']})")

        log("\n[本命指標：語の寄与] hunger一定で“語だけ”変えたとき、まんまが他語より食べ物を強く")
        log("  先取りするか。まんま−あうあ>0 なら『まんま→食べ物』の先取り＝初期・連合的理解の証拠。")
        log("  （同じhunger内で比べるので、体が既に知っている満腹度の分は相殺される）")
        for hv, hname in hungers:
            sm = results[(hname, "まんま")].get("satiety")
            ss = results[(hname, "ままん")].get("satiety")
            sa = results[(hname, "あうあ")].get("satiety")
            if sm is not None:
                log(f"  [{hname}固定] まんま={sm:.4f}  ままん={ss:.4f}  あうあ={sa:.4f}  "
                    f"→ 語の寄与 まんま−あうあ={sm-sa:+.4f}"
                    f"{'  ★語で食べ物を先取り' if sm-sa > 0.05 else '  （語の寄与ほぼ無し＝まだ理解せず）'}")

        log("\n[Rung1：認識] 聞いた直後の隠れ状態が語ごとに区別できるか（コサイン類似, 1に近い=似てる）")
        for hv, hname in hungers:
            hm = results[(hname, "まんま")]["hidden"]
            hs = results[(hname, "ままん")]["hidden"]
            ha = results[(hname, "あうあ")]["hidden"]
            log(f"  [{hname}] まんま↔ままん={cosine(hm,hs):.4f}  まんま↔あうあ={cosine(hm,ha):.4f}  "
                f"ままん↔あうあ={cosine(hs,ha):.4f}")

        log("\n[Rung2：意味] critic価値が“聞いた語”で変わるか（同じ空腹なら同値＝語→予期の読み出しが無い）")
        for hv, hname in hungers:
            vals = [results[(hname, w)]["critic_value"] for w in words]
            log(f"  [{hname}] critic価値 = {['%.4f'%v for v in vals]} → "
                f"{'語によらず同一（予期の場が無い）' if max(vals)-min(vals) < 1e-6 else '語で変化あり'}")

        log("\n[Rung2補足：状態依存] 「まんま」を聞いた後の発声まんま類似が、空腹と満腹で違うか")
        m_hungry = results[("空腹", "まんま")]["echoic_mama_sim"]
        m_full = results[("満腹", "まんま")]["echoic_mama_sim"]
        log(f"  まんまを聞いた後：空腹時={m_hungry:.4f} 満腹時={m_full:.4f} 差={m_hungry-m_full:+.4f}")

        log(f"\n[完了] {time.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        import traceback
        log(f"[エラー] {e}")
        log(traceback.format_exc())
