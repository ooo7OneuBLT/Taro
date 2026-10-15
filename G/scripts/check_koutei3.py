"""段階2・工程3の確かめ（仕様書 §3 の終わり）：7つの仮部品を入れて1場面ぶん回し、
通り道の矢印ごとの中身を脳1回ごとに CSV へ書く（目で確かめる用）。

使い方：python G/scripts/check_koutei3.py [場面名] [語]
出力：G/logs/工程3/<場面>_<語>.csv
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, REPO)

from core_v2 import run_v2
from core_v2.brain import Brain

OUT = os.path.join(REPO, "G", "logs", "工程3")


def main(scene="段階2_3物体_ボール左_2026-10-15", word="ボールはどこ？", seed=0):
    os.makedirs(OUT, exist_ok=True)
    brain = Brain()
    brain.boxes["eyein"].setup()
    brain.boxes["semantic"].load_table(os.path.join(REPO, "G", "models", "semantic_table.pt"))
    env = run_v2.build_env(scene, word, seed=seed)
    try:
        rows = run_v2.run_trial(brain, env, seed=seed, duration_s=10.0,
                                on_tick=lambda row: row.update(run_v2.path_record(brain)))
    finally:
        run_v2.close_env(env)
    path = os.path.join(OUT, f"{scene}_{word.rstrip('？?')}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("保存:", path)
    for r in rows:
        if r["heard"] or r["cmd_h"] is not None or r["orienteye.reason"] in ("fire", "inhibited") \
                or r["goal.cleared_by"] or r["listen>semantic"] or r["semantic>goal"]:
            print(f"t={r['t']:.1f} 聞={r['heard']!r} l>s={r['listen>semantic']} s>g={r['semantic>goal']} "
                  f"g>gate={r['goal>gate3']} age={r['goal.age']} clr={r['goal.cleared_by']} "
                  f"found={r['gaze>goal']} seek={r['gaze>prio']}@{r['gaze>prio@']} "
                  f"prio={r['prio>orienteye']}@{r['prio>orienteye@']} inh={r['gate3>orienteye']} "
                  f"oe={r['orienteye.reason']} cmd={r['orienteye>eyeout']} "
                  f"角度 球{r['ang_ball']:.1f} コ{r['ang_cup']:.1f} 妨{r['ang_distractor']:.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
