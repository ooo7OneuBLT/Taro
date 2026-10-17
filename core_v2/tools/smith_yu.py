"""段階6：スミスとユー（Smith & Yu 2008）の赤ちゃんの実験を太郎で回す道具（仕様書
`2026-10-17_段階6_仕様書_スミスとユーの再現.md` §3・§4）。

1人の太郎（種1つ）ごとに：
1. 耳だけで親の話しかけを聞かせて育てる（hear_parent と同じ文、種＝太郎の種）
2. 学ぶ画面30：物2つを左右に4秒、出てから0.5秒・1.0秒に語を1つずつ（どれの名前かは分からない）。
   意味の辞書と耳は覚え続ける。画面と画面のあいだは GAP_S 何も出さない（下の注）
3. テスト12試行：正解の物と別の物を8秒、語を4回。覚えるのは止める。語を最初に言ってから試行の終わりまでに、
   正解の物と別の物をそれぞれ見ていた時間を数える（視線と物の角度が LOOK_DEG 以内で、近い方）

注：画面と画面のあいだの空白（GAP_S）は論文に無い（元の実験は注意を引く画面を挟む）。物の名簿は物を
場所で追うので、同じ置き場所で物が続けて替わると、前の物と次の物の見た目が1つの行に混ざる。
KEEP_S（2秒）より長く何も出さなければ、前の画面の行は消える。
注2：テストの試行のあいだにも同じ空白を入れる（見始めが前の試行の向きに引っぱられないように）。
注3：注意を引く画面の代わりに、空白のあいだは目を最初の向き（2つの置き場所のあいだ）へ戻す（RECENTER）。
これまでの試行は10秒ごとに目を戻していたが、ここは約5分の1本で、戻さないと目が端へ寄っていき
（1回目の確かめ：眼の可動域の外を狙う警告、後半の画面とテストで物を一度も見ない）、物を見に行けなかった。
太郎がその画面を見て目を向けたのではなく、道具が目を動かす（実験者が真ん中に赤ちゃんの目を引くのと同じ役）。

使い方：
  python -m core_v2.tools.smith_yu [--seeds 0-19] [--modes B,A] [--ear-n 1000] [--gap 2.5] [--log-dir 段階6_スミスとユー] [--ticks]
    --modes … B（直前3秒の割合）・A（名前を結び付ける1つの物だけ）・none（覚えない太郎、比べる用）
    --ticks … 1回ごとの記録（CSV）も残す（1回だけの確かめ用）
出力：G/logs/<log-dir>/<覚え方>_seed<種>.json と、まとめ.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

from .. import run_v2
from ..boxes.semantic import RECALL_MIN
from ..tagging import word_tag
from .parent_speech import make_corpus

SCENE = "段階6_スミスとユー_6物体_2026-10-17"
LOOK_DEG = 10.0
GAP_S = 2.5
HALF_VIEW_DEG = 30.0     # 視野の片側（目への方向 ±1 がこの角度。run_v2.START_JITTER の説明）
RECENTER_DEG = 3.0       # 空白のあいだ、最初の向きからこれ以上ずれていたら戻す


def _schedule_tool():
    p = os.path.join(run_v2.REPO, "G", "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)
    import smith_yu_schedule
    return smith_yu_schedule


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def sign_test_p(k: int, n: int) -> float:
    """片側の符号検定：n人中k人以上が偶然（半々）で出る確率。"""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def run_one(seed: int, mode: str, *, ear_n: int, gap_s: float, log_dir: str, ticks: bool) -> dict:
    from ..brain import Brain
    sy = _schedule_tool().make(seed, gap_s=gap_s, test_iti_s=gap_s)
    run_v2.seed_all(seed)
    brain = Brain()
    brain.boxes["eyein"].setup()
    sem, ear = brain.boxes["semantic"], brain.boxes["listen"]
    ear.learning = True
    for s in make_corpus(ear_n, seed=seed):
        ear.step({"earin": s})
    ear.reset()
    sem.learning, sem.learn_mode = mode != "none", (mode if mode != "none" else "B")   # none＝覚えない太郎（比べる用）

    slides = sy["slides"]
    test_t0 = min(s["t0"] for s in slides if s["phase"] == "test")
    looks = {s["i"]: {"target": 0, "distractor": 0, "other": 0} for s in slides if s["phase"] == "test"}
    n_learn = {"train": 0, "skip": 0}
    rows = []

    def slide_at(t):
        return next((s for s in slides if s["t0"] <= t < s["t1"]), None)

    gaze = {"x": 0.0, "y": 0.0, "n_recenter": 0}

    def manual_eye(i, cmd):
        """空白のあいだ（画面が出ていないとき）は、3回に1回、目を最初の向きへ戻す指令に差し替える。"""
        t = float(env.unwrapped.data.time)
        if slide_at(t) is not None or slide_at(t + 0.3) is not None:
            return cmd
        if i % 3 or math.hypot(gaze["x"], gaze["y"]) <= RECENTER_DEG:
            return None
        gaze["n_recenter"] += 1
        return (max(-1.0, min(1.0, -gaze["x"] / HALF_VIEW_DEG)), max(-1.0, min(1.0, -gaze["y"] / HALF_VIEW_DEG)))

    def on_tick(row):
        t = row["t"]
        gaze["x"], gaze["y"] = row["x_gaze"], row["y_gaze"]
        if t >= test_t0 - 1.0 or mode == "none":   # テストの前から覚えるのを止める
            sem.learning = False
        if t >= test_t0 - 1.0:
            ear.learning = False
        ev = sem.last_event
        if ev == "learn":
            n_learn["train"] += 1
        elif ev == "learn_skip":
            n_learn["skip"] += 1
        s = slide_at(t)
        look = ""
        if s is not None:
            u = env.unwrapped
            angs = [(u._gaze_angle_to(f"sy_obj{k}"), k) for k in (s["left"], s["right"])]
            angs = [(a, k) for a, k in angs if a is not None and a <= LOOK_DEG]
            if angs:
                look = str(min(angs)[1])
            if s["phase"] == "test" and t >= s["t0"] + sy_word0:
                key = ("target" if look == str(s["target"]) else "distractor" if look == str(s["distractor"])
                       else "other")
                looks[s["i"]][key] += 1
        if ticks:
            rec = run_v2.path_record(brain)
            rec.update({"t": round(t, 2), "heard": row["heard"], "phase": s["phase"] if s else "",
                        "slide": s["i"] if s else "", "left": s["left"] if s else "",
                        "right": s["right"] if s else "", "look": look, "x_gaze": round(row["x_gaze"], 1), "y_gaze": round(row["y_gaze"], 1),
                        "sem.learn_mode": mode, "sem.n_learn": int(sem.n_learn)})
            rows.append(rec)

    sy_word0 = min(w["t"] for w in slides[-1]["words"]) - slides[-1]["t0"]
    world_extra = {"pairs": {"schedule": sy["pairs_schedule"]}, "word_test": {"schedule": sy["word_schedule"]}}
    t_wall = time.time()
    env = run_v2.build_env(SCENE, None, seed=seed, world_extra=world_extra)
    try:
        run_v2.run_trial(brain, env, seed=seed, duration_s=sy["t_end"] + 0.5, on_tick=on_tick,
                           manual_eye=manual_eye)
    finally:
        run_v2.close_env(env)

    T = sum(v["target"] for v in looks.values())
    D = sum(v["distractor"] for v in looks.values())
    names = [o["name"] for o in sy["objects"]]
    res = {
        "seed": seed, "mode": mode, "gap_s": gap_s, "ear_n": ear_n,
        "target_ticks": T, "distractor_ticks": D,
        "target_share": round(T / (T + D), 3) if T + D else None,
        "trials_target_longer": sum(v["target"] > v["distractor"] for v in looks.values()),
        "trials_distractor_longer": sum(v["target"] < v["distractor"] for v in looks.values()),
        "n_learn": n_learn, "n_recenter": gaze["n_recenter"],
        "known_new_words": sorted(set(sy["word_of"].values()) & ear.known()),
        "recall_norm": {w: round(float((sem.W @ word_tag(w)).norm()), 2) for w in sy["word_of"].values()},
        "word_of": {w: names[k] for k, w in sy["word_of"].items()},
        "trials": [{"i": s["i"], "word": sy["word_of"][s["target"]], "target": names[s["target"]],
                    "distractor": names[s["distractor"]],
                    "recalled": float((sem.W @ word_tag(sy["word_of"][s["target"]])).norm()) >= RECALL_MIN,
                    **looks[s["i"]]} for s in slides if s["phase"] == "test"],
        "sim_s": sy["t_end"], "wall_s": round(time.time() - t_wall, 1),
    }
    try:
        from ..save import code_version
        res["code"] = code_version()
    except Exception:  # noqa: BLE001
        pass
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{mode}_seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    if ticks and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(log_dir, f"{mode}_seed{seed}_回ごと.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return res


def summarize(results: list[dict]) -> dict:
    out = {}
    for mode in sorted({r["mode"] for r in results}):
        rs = [r for r in results if r["mode"] == mode and r["target_share"] is not None]
        k = sum(r["target_share"] > 0.5 for r in rs)
        n = sum(r["target_share"] != 0.5 for r in rs)
        out[mode] = {"taros": len(rs), "target_longer": k, "p_sign": round(sign_test_p(k, n), 4),
                     "mean_share": round(sum(r["target_share"] for r in rs) / len(rs), 3) if rs else None,
                     "pass": sign_test_p(k, n) < 0.05,
                     "human": "55人中46人"}
        for flag, key in ((True, "trials_recalled"), (False, "trials_not_recalled")):
            ts = [t for r in rs for t in r["trials"] if t["recalled"] == flag]
            out[mode][key] = {"n": len(ts), "target_longer": sum(t["target"] > t["distractor"] for t in ts),
                              "distractor_longer": sum(t["target"] < t["distractor"] for t in ts)}
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = {"--seeds": "0-19", "--modes": "B,A", "--ear-n": "1000", "--gap": str(GAP_S),
            "--log-dir": "段階6_スミスとユー"}
    ticks = "--ticks" in argv
    i = 0
    while i < len(argv):
        if argv[i] in opts:
            opts[argv[i]] = argv[i + 1]
            i += 2
        else:
            i += 1
    log_dir = os.path.join(run_v2.REPO, "G", "logs", opts["--log-dir"])
    results = []
    for mode in opts["--modes"].split(","):
        for seed in parse_seeds(opts["--seeds"]):
            r = run_one(seed, mode, ear_n=int(opts["--ear-n"]), gap_s=float(opts["--gap"]),
                        log_dir=log_dir, ticks=ticks)
            results.append(r)
            print(f"[{mode} 種{seed}] 正解を見た割合 {r['target_share']}（試行 正解>別 {r['trials_target_longer']}・"
                  f"別>正解 {r['trials_distractor_longer']}） 覚えた {r['n_learn']} 語になった新語 "
                  f"{len(r['known_new_words'])}/6 （{r['wall_s']}秒）", flush=True)
    summary = summarize(results)
    with open(os.path.join(log_dir, "まとめ.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "runs": [{k: r[k] for k in ("seed", "mode", "target_share",
                   "trials_target_longer", "trials_distractor_longer", "n_learn")} for r in results]},
                  f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


if __name__ == "__main__":
    main()
