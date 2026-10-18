"""段階8a：体ありの太郎で、驚きの係が「消えた」ときだけ立つか（合格の線 A の体ありの確かめ。仕様書 §3）。

体なしの試験場では A は作りから通る。本当の試しは体ありで、目の切り出しの取りこぼし（物を拾い損ねる回）や、
太郎が自分で目をそらすことが、物が消えていない場面で偽の驚きを出さないか。

段階2bの3つの物の場面で、親は黙ったまま太郎に自由に見させる（say_names と同じ）。
* 消える試行：VANISH_AFTER_S 秒たったあと、太郎が物を LOOK_DEG 以内で HOLD_S 秒続けて見ていたら、その物を急に
  視野の外の遠くへ移して見えなくする（親が持っていったのと違い、動くところは見えない＝消える）。
  消えてから WINDOW_S 秒以内に驚き（surprise→semantic の強さ）が 0 を超えたら「立った」
* 対照の試行：何もしない。試行の中で驚きが立った回数（0→正になった回数）を数える
意味の辞書・声は使わない（驚きの係と物の名簿だけを見る）。脳の他の係は段階2〜7のまま。

使い方：
  python -m core_v2.tools.nai_body [--taros 0,1,2] [--trials 10] [--trial-s 30] [--log-dir 段階8_体あり_驚き]
出力：G/logs/<log-dir>/まとめ.json と、試行ごとの驚きの記録 trials.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .. import run_v2
from .learn_names import DEFAULT_SCENES

LOOK_DEG = 10.0
VANISH_AFTER_S = 5.0
HOLD_S = 2.0
WINDOW_S = 3.0


def hide(env, key: str) -> None:
    """物を見えなくする。場面の物は親が手に持っている（toy_env の hold：毎歩、定位置へ置き直す）ので、
    場所を動かしても次の歩で戻る（依頼3の1回目はこれで1度も消えていなかった）。そこで前の太郎の親の
    消し方（run/world/parent.py の _hide_all）と同じく、物の面の色の透明度を0にし、材質の参照を外す
    （材質があると色の書き換えが効かない、toy_env の注記）。物はそこにあるが、目には映らない。"""
    u = env.unwrapped
    bid = int(u.model.body(run_v2.OBJECTS[key]).id)
    adr, num = int(u.model.body_geomadr[bid]), int(u.model.body_geomnum[bid])
    for g in range(adr, adr + num):
        u.model.geom_matid[g] = -1
        u.model.geom_rgba[g, 3] = 0.0


def seat_states(brain) -> str:
    sp = brain.boxes["surprise"]
    return " ".join(f"{i}:{s.state[:2]}:{s.enc:.2f}:{s.vanish:.0f}:{s.w:.2f}{'*' if s.fov else ''}"
                    for i, s in enumerate(sp.seats) if s.state != "empty")


def run_one(brain, scene: str, seed: int, trial_s: float, vanish: bool, trace: list | None = None) -> dict:
    env = run_v2.build_env(scene, None, seed=seed, silent=True)
    tick_s = run_v2.K * run_v2.DT
    st = {"look": "", "run": 0, "hid": None, "hid_obj": "", "level": []}

    def on_tick(row):
        cands = [(row[f"ang_{k}"], k) for k in run_v2.OBJECTS if row.get(f"ang_{k}") is not None
                 and row[f"ang_{k}"] <= LOOK_DEG]
        lk = min(cands)[1] if cands else ""
        st["run"] = st["run"] + 1 if lk and lk == st["look"] else (1 if lk else 0)
        st["look"] = lk
        st["level"].append(round(float(brain.buffer[("surprise", "semantic")]), 4))
        if trace is not None:
            trace.append({"tick": row["tick"], "look": lk, "hid": st["hid"] is not None,
                          "n_eye": len(brain.buffer[("eyein", "objfile")]) - 1,
                          "objfile": brain.boxes["objfile"].summary(), "seats": seat_states(brain),
                          "u": st["level"][-1], **{f"ang_{k}": row.get(f"ang_{k}") for k in run_v2.OBJECTS}})

    def manual_eye(i, cmd):
        if (vanish and st["hid"] is None and i * tick_s >= VANISH_AFTER_S and st["look"]
                and st["run"] * tick_s >= HOLD_S):
            hide(env, st["look"])
            st["hid"], st["hid_obj"] = i, st["look"]
        return cmd

    try:
        run_v2.run_trial(brain, env, seed=seed, duration_s=trial_s, on_tick=on_tick, manual_eye=manual_eye)
    finally:
        run_v2.close_env(env)
    lv = st["level"]
    onsets = sum(1 for a, b in zip([0.0] + lv, lv) if a == 0.0 and b > 0.0)
    out = {"scene": scene, "seed": seed, "vanish": vanish, "hid_tick": st["hid"], "hid_obj": st["hid_obj"],
           "onsets": onsets, "level": lv}
    if st["hid"] is not None:
        w = lv[st["hid"]:st["hid"] + int(WINDOW_S / tick_s) + 1]
        out["surprised"] = max(w, default=0.0) > 0.0
        out["u_max"] = max(w, default=0.0)
        out["onsets_before"] = sum(1 for a, b in zip([0.0] + lv[:st["hid"]], lv[:st["hid"]]) if a == 0.0 and b > 0.0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--taros", default="0,1,2")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--trial-s", type=float, default=30.0)
    ap.add_argument("--scenes", default=DEFAULT_SCENES)
    ap.add_argument("--log-dir", default="段階8_体あり_驚き")
    ap.add_argument("--trace", type=int, default=0, help="最初のこの数の太郎×試行について、1回ごとの記録を trace_*.jsonl に書く")
    a = ap.parse_args(argv)
    log_dir = os.path.join(run_v2.REPO, "G", "logs", a.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    from ..brain import Brain
    brain = Brain()
    brain.boxes["eyein"].setup()
    scenes = [x for x in a.scenes.split(",") if x]
    trials, t0 = [], time.time()
    for taro in [int(x) for x in a.taros.split(",")]:
        for k in range(a.trials):
            for vanish in (True, False):
                seed = taro * 1000 + k
                tr = [] if k < a.trace else None
                r = run_one(brain, scenes[k % len(scenes)], seed, a.trial_s, vanish, tr)
                if tr is not None:
                    with open(os.path.join(log_dir, f"trace_taro{taro}_trial{k}_{'vanish' if vanish else 'control'}.jsonl"),
                              "w", encoding="utf-8") as f:
                        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in tr)
                r["taro"] = taro
                trials.append(r)
                print(f"[{time.time() - t0:6.0f}s] 太郎{taro} 試行{k} {'消える' if vanish else '対照'}："
                      f"消した {r['hid_obj'] or '-'} 驚き {r.get('surprised', '-')} 立った回数 {r['onsets']}", flush=True)
    van = [r for r in trials if r["vanish"] and r["hid_tick"] is not None]
    ctl = [r for r in trials if not r["vanish"]]
    summary = {
        "vanish_done": len(van), "vanish_not_done": sum(1 for r in trials if r["vanish"] and r["hid_tick"] is None),
        "vanish_surprised": sum(r["surprised"] for r in van),
        "vanish_u_max_mean": round(sum(r["u_max"] for r in van) / max(1, len(van)), 3),
        "vanish_false_onsets_before": sum(r["onsets_before"] for r in van),
        "control_trials": len(ctl), "control_with_onset": sum(r["onsets"] > 0 for r in ctl),
        "control_onsets_total": sum(r["onsets"] for r in ctl), "trial_s": a.trial_s,
    }
    with open(os.path.join(log_dir, "trials.json"), "w", encoding="utf-8") as f:
        json.dump(trials, f, ensure_ascii=False)
    with open(os.path.join(log_dir, "まとめ.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
