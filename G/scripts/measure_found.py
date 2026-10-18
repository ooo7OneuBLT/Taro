"""「見つけた」の閾値を決めるための測定（2026-10-15、統合担当）。

目的を持たせたまま、目を手で ボール → コップ → 妨害物 の順に向け（脳の発射は全部握りつぶす）、
そのとき gaze が出す「見つけた」の値を、「探し物を見ている回」と「探し物以外を見ている回」に分けて集める。
* 目的：4.1秒に「ボールはどこ？」または「コップはどこ？」→ 4.4〜9.4秒に目的を持つ
  （この測定の間は閾値を 1.1 にして、見つけても棚が空にならないようにする）
* 目：1.0秒 妨害物、4.6秒 ボール、6.6秒 コップ、8.6秒 妨害物（物の投影の方向へ）
* 「見ている」＝視線から10度以内の物のうち一番近いもの。跳んでいる途中の回（向けてから0.3秒）は除く
出力：G/logs/見つけた/found.csv（回ごと）と、分布の要約・絵 found.png
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, REPO)
import core_v2  # noqa: E402,F401  負荷の決まり（core_v2/perf.py：スレッド1本など）を numpy・torch より先に効かせる

import numpy as np

from core_v2 import run_v2
from core_v2.boxes import goal as goal_mod
from core_v2.brain import Brain
from core_v2.tools.register_appearance import project

OUT = os.path.join(REPO, "G", "logs", "見つけた")
SCENES = ["段階2b_3物体_ボール左_2026-10-15", "段階2b_3物体_ボール右_2026-10-15"]
WORDS = {"ボールはどこ？": "ball", "コップはどこ？": "cup"}
KICKS = {9: "distractor", 45: "ball", 65: "cup", 85: "distractor"}
SETTLE_TICKS = 3


def main(seeds=(0, 1)):
    os.makedirs(OUT, exist_ok=True)
    goal_mod.FOUND_THRESHOLD = 1.1          # 測定中は見つけても棚を空にしない
    brain = Brain()
    brain.boxes["eyein"].setup()
    brain.boxes["semantic"].load_table(os.path.join(REPO, "G", "models", "semantic_table.pt"))
    out_rows = []
    for scene in SCENES:
        for word, target in WORDS.items():
            env = run_v2.build_env(scene, word, seed=0)
            u = env.unwrapped
            last_kick = {"i": -99}

            def manual(i, _brain_cmd):
                if i in KICKS:
                    last_kick["i"] = i
                    px, py = project(u, run_v2.OBJECTS[KICKS[i]])
                    return ((px - 112.0) / 112.0, (112.0 - py) / 112.0)
                return None

            try:
                for seed in seeds:
                    rows = run_v2.run_trial(
                        brain, env, seed=seed, duration_s=10.0, manual_eye=manual,
                        on_tick=lambda row: row.update(run_v2.path_record(brain),
                                                       since_kick=row["tick"] - last_kick["i"]))
                    for r in rows:
                        if r["goal.age"] == "" or r["since_kick"] < SETTLE_TICKS:
                            continue
                        cands = [(r[f"ang_{k}"], k) for k in run_v2.OBJECTS if r[f"ang_{k}"] <= 10.0]
                        if not cands:
                            continue
                        look = min(cands)[1]
                        out_rows.append({"scene": scene, "word": word, "seed": seed, "t": round(r["t"], 2),
                                         "looking": look, "is_target": look == target,
                                         "angle": round(min(cands)[0], 2), "found": r["gaze>goal"]})
            finally:
                run_v2.close_env(env)
    with open(os.path.join(OUT, "found.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    tgt = np.array([r["found"] for r in out_rows if r["is_target"]])
    non = np.array([r["found"] for r in out_rows if not r["is_target"]])
    summ = {}
    for name, a in (("target", tgt), ("non_target", non)):
        summ[name] = {"n": int(a.size), "min": float(a.min()), "p05": float(np.percentile(a, 5)),
                      "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}
    by_obj = {}
    for r in out_rows:
        by_obj.setdefault(f"{r['word']}→{r['looking']}", []).append(r["found"])
    summ["by_word_object"] = {k: {"n": len(v), "min": min(v), "median": float(np.median(v)), "max": max(v)}
                              for k, v in by_obj.items()}
    with open(os.path.join(OUT, "found_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(non, bins=bins, color="#7f7f7f", alpha=0.8, label=f"探し物以外を見ている回（{non.size}）")
    ax.hist(tgt, bins=bins, color="#2ca02c", alpha=0.8, label=f"探し物を見ている回（{tgt.size}）")
    ax.set_xlabel("「見つけた」の値")
    ax.set_ylabel("回数")
    ax.legend()
    ax.set_title("目的を持っているときの「見つけた」の分布（2つの置き方×2語×2種）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "found.png"), dpi=110)
    print("絵:", os.path.join(OUT, "found.png"))


if __name__ == "__main__":
    main()
