"""通り道の確かめ（工程3の追加）：妨害物を見ているところから、語で探し物へ目が移るか。

今の場面では、目立ちの1位がボールなので発話の前からボールを見てしまい、またボールから見ると
コップが視野の外（約33度）にあるので、「目立つ物を見ている → 語を聞く → 抑えて探し物へ跳ぶ」
という段階2の本題の流れが、まだ一度も起きていない。場面を変えずにこれを確かめる：
* 5秒までは脳の発射を握りつぶし、1.0秒に手で妨害物へ目を向ける
* 5秒からは脳の発射をそのまま目へ通す
  （最初は4秒からにしたが、抑えが届く前の4.4秒に、目立ちの1位のボールへ普通に跳んでしまい、
  本題の流れを測れなかった。注意：握りつぶした発射も orienteye は「撃った」と数えるので、
  休み（1.2秒）の数え方は実際とずれる）
妨害物から見ると、ボールもコップも約23度で視野の中にある。

使い方：python G/scripts/check_path_from_distractor.py
出力：G/logs/工程3/妨害物から_<場面>_<語>.csv と要約
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, REPO)
import core_v2  # noqa: E402,F401  負荷の決まり（core_v2/perf.py：スレッド1本など）を numpy・torch より先に効かせる

from core_v2 import run_v2
from core_v2.brain import Brain
from core_v2.tools.register_appearance import project

OUT = os.path.join(REPO, "G", "logs", "工程3")
KICK_TICK = 9           # 1.0秒の回
BRAIN_FROM_T = 5.0      # 発話(4.1秒)→目的(4.4)→抑え(4.6)→探し物らしさの地図(4.7) が orienteye に揃った後


def main():
    os.makedirs(OUT, exist_ok=True)
    brain = Brain()
    brain.boxes["eyein"].setup()
    brain.boxes["semantic"].load_table(os.path.join(REPO, "G", "models", "semantic_table.pt"))
    scene = "段階2_3物体_ボール左_2026-10-15"
    for word, target in (("ボールはどこ？", "ball"), ("コップはどこ？", "cup")):
        env = run_v2.build_env(scene, word, seed=0)
        u = env.unwrapped

        def manual(i, brain_cmd):
            if i == KICK_TICK:
                px, py = project(u, run_v2.OBJECTS["distractor"])
                return ((px - 112.0) / 112.0, (112.0 - py) / 112.0)
            if u.data.time < BRAIN_FROM_T:
                return None
            return brain_cmd

        try:
            rows = run_v2.run_trial(brain, env, seed=0, duration_s=10.0, manual_eye=manual,
                                    on_tick=lambda row: row.update(run_v2.path_record(brain)))
        finally:
            run_v2.close_env(env)
        path = os.path.join(OUT, f"妨害物から_{scene}_{word.rstrip('？?')}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("保存:", path)
        for r in rows:
            if r["t"] < 4.0 and r["tick"] not in (KICK_TICK, KICK_TICK + 5, 35):
                continue
            if r["tick"] % 5 == 0 or r["cmd_h"] is not None or r["orienteye.reason"] in ("fire", "inhibited") \
                    or r["heard"]:
                print(f"t={r['t']:.1f} 聞={r['heard']!r} 目的age={r['goal.age']} clr={r['goal.cleared_by']} "
                      f"found={r['gaze>goal']} seek={r['gaze>prio']}@{r['gaze>prio@']} "
                      f"prio={r['prio>orienteye']}@{r['prio>orienteye@']} inh={r['gate3>orienteye']} "
                      f"oe={r['orienteye.reason']} 目へ={r['cmd_h']},{r['cmd_v']} "
                      f"角度 球{r['ang_ball']:.1f} コ{r['ang_cup']:.1f} 妨{r['ang_distractor']:.1f}")


if __name__ == "__main__":
    main()
