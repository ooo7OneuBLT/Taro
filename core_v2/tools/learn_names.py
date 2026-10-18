"""意味の辞書に、親の名づけから名前を覚えさせる道具（段階3・仕様書 付録B-3）。

親（環境の「親が名づける仕組み」run/world/parent.py:32 ParentLabeling）が物を振って見せ、
太郎がそれを gaze_hold_sec（3秒）見続けたら名前を言う。意味の辞書（semantic、learning=True）は
名前を聞いた回に、視野の真ん中にある物の荷札と語の荷札を結び付ける。
各語 N 回覚えたら止めて、学んだ表を保存する。手で登録する道具（register_appearance.py）は使わない。

使い方：
  python -m core_v2.tools.learn_names [N] [--swap] [--out パス] [--scenes 場面,場面] [--max-s 秒] [--log-dir 名前]
    N       … 各語を何回覚えたら止めるか（既定 5）
    --swap  … 名前を入れ替える（ボールに「こっぷ」と言う）＝合格の線⑤の確かめ用
    --ear   … 段階5：親の話しかけで育てた耳（hear_parent.py の出力）を聞き分ける係に読む。名づけの間も耳は覚え続ける
既定の出力：G/models/semantic_learned_N{N}.pt（--swap なら _swap が付く）。記録は G/logs/段階3_名づけ/（--log-dir で G/logs/ の下の名前を変える）

覚えた回ごとに「そのとき実際に見ていた物」（視線と物の角度、10度以内で角の小さい方）を記録し、
名前と食い違った回（取り違え）を数える。段階4（2026-10-16）：意味の辞書は、物の名簿の「最後に1秒
（MIN_LOOK_S）以上続けて見た物」で覚えるので、取り違えも同じ決まり（環境の正解の角度で、最後に1秒以上
続けて見ていた物＝learned_object）で数える。
覚えた回にたまたま見ていた物（looking）も並べて残す。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

from .. import run_v2
from ..boxes.objfile import MIN_LOOK_S
from ..boxes.semantic import TRACE_TICKS

DEFAULT_SCENES = "段階2b_3物体_ボール左_2026-10-15,段階2b_3物体_ボール右_2026-10-15"
WORD_OF = {"toy1": "ぼおる", "toy2": "こっぷ"}       # toy1＝ボール（test_object1）・toy2＝コップ
OBJ_OF_WORD = {"ぼおる": "ball", "こっぷ": "cup"}
LOOK_DEG = 10.0
TRIAL_S = 60.0          # 1つの場面を続けて回す長さ。足りなければ種を変えてもう1回
PIPE_TICKS = 3          # 語は earin→listen→semantic で2回遅れて届く（幅を1回ぶん持たせる）


def learn_lag(sem) -> int:
    """聞いた回から覚える回までの遅れ。覚え方B（1cfe361 から）は名残りが消える TRACE_TICKS 回あとに覚える。"""
    return TRACE_TICKS if sem.learn_mode == "B" else 0


def word_of_learn(heard_hist, tick: int, lag: int):
    """覚えた回 tick を、その lag 回前（＋届くまでの遅れ）に聞いた語と、そのとき1秒以上見ていた物に結び付ける。"""
    lo, hi = tick - lag - PIPE_TICKS, tick - lag
    return next(((w, obj) for t0, w, obj in reversed(heard_hist) if lo <= t0 <= hi), ("", ""))


def labeling_world(swap: bool) -> dict:
    """段階2bの場面に重ねる world。名前の言い方は「〇〇だね」1通り、繰り返し・消える物・
    視線を追う・1つずつ見せる、は切る（まず一番素直な形で試す。仕様書 付録D）。"""
    words = dict(WORD_OF)
    if swap:
        words = {"toy1": WORD_OF["toy2"], "toy2": WORD_OF["toy1"]}
    return {
        "word_test": {"schedule": []},
        "parent_labeling": {
            "enabled": True,
            "utterances": {k: f"{w}だね" for k, w in words.items()},
            "gaze_hold_sec": 3.0,
            "attract_timeout_sec": 12.0,
            "respond_prob": 1.0,
            "repeat_labels": 0,
            "solo_presentation": False,
            "follow_gaze": False,
            "vanish_utterance": False,
            "vanish_wait_notice": False,
            "cycle_pick": True,
            "present_angles": {},
        },
    }


def looking_at(env) -> str:
    ang = run_v2.gaze_angles(env)
    cands = [(a, k) for k, a in ang.items() if a is not None and a <= LOOK_DEG]
    return min(cands)[1] if cands else ""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from ..world_eye.switch import pop_world_flag, select
    select(pop_world_flag(argv))          # --world eye|mimo（既定 mimo）
    swap = "--swap" in argv
    opts = {"--out": None, "--scenes": DEFAULT_SCENES, "--max-s": "900", "--log-dir": "段階3_名づけ", "--ear": None}
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in opts:
            opts[a] = argv[i + 1]
            i += 2
            continue
        if not a.startswith("--"):
            pos.append(a)
        i += 1
    n_target = int(pos[0]) if pos else 5
    tag = f"N{n_target}" + ("_swap" if swap else "")
    out = opts["--out"] or os.path.join(run_v2.REPO, "G", "models", f"semantic_learned_{tag}.pt")
    log_dir = os.path.join(run_v2.REPO, "G", "logs", opts["--log-dir"])
    os.makedirs(log_dir, exist_ok=True)
    max_s = float(opts["--max-s"])

    from ..brain import Brain
    brain = Brain()
    brain.boxes["eyein"].setup()
    sem = brain.boxes["semantic"]
    sem.learning = True
    ear = brain.boxes["listen"]
    if opts["--ear"]:
        ear.load_ear(os.path.join(run_v2.REPO, opts["--ear"]))
        ear.learning = True

    counts = {w: 0 for w in OBJ_OF_WORD}
    events = []          # 名前を聞いた回・覚えた回の記録
    heard_hist = []      # (tick, 語, そのとき1秒以上見ていた物)：覚えた回を、聞いた回に結び付けるため
    scenes = [x for x in opts["--scenes"].split(",") if x]
    sim_s, seed = 0.0, 0
    t_start = time.time()
    while min(counts.values()) < n_target and sim_s < max_s:
        scene = scenes[seed % len(scenes)]
        env = run_v2.build_env(scene, None, seed=seed, world_extra=labeling_world(swap))
        try:
            run_v2.seed_all(seed)
            obs, _ = env.reset(seed=seed)
            u = env.unwrapped
            u._orienting._map_target = None    # 前の試行の目の指令を持ち越さない（run_v2.run_trial の説明）
            brain.reset()                      # 意味の辞書の表 W は消えない
            act = run_v2.zero_action(env)
            n_ticks = int(round(TRIAL_S / (run_v2.K * run_v2.DT)))
            run_obj, run_len, long_obj = "", 0, ""   # 続けて見ている物とその回数、最後に1秒以上見た物
            heard_hist = []                    # tick は場面ごとに 0 から。前の場面の語を拾わない（brain.reset で名残りも消える）
            for tick in range(n_ticks):
                heard = []
                for _ in range(run_v2.K):
                    obs, _r, term, trunc, info = env.step(act)
                    utt = info.get("parent_utterance")
                    if utt and utt.get("text"):
                        heard.append(str(utt["text"]))
                    if term or trunc:
                        raise RuntimeError(f"途中で環境が終わった（t={u.data.time:.2f}）")
                text = " ".join(heard)
                to_env = brain.tick({"eyein": obs["eye_left"], "earin": text})
                cmd = to_env.get("eyeout")
                if cmd is not None:
                    u._orienting.set_map_target(float(cmd[0]), float(cmd[1]))
                look = looking_at(env)
                run_len = run_len + 1 if look and look == run_obj else (1 if look else 0)
                run_obj = look
                if run_obj and run_len * run_v2.K * run_v2.DT >= MIN_LOOK_S - 1e-9:
                    long_obj = run_obj
                if text:
                    word = next((w for w in OBJ_OF_WORD if w in text), "")
                    heard_hist.append((tick, word, long_obj))
                    events.append({"scene": scene, "seed": seed, "t": round(float(u.data.time), 2),
                                   "event": "heard", "text": text, "word": word, "looking": look})
                ev = sem.last_event
                if ev in ("learn", "learn_skip"):
                    # 覚え方Bは名残りが消えてから（聞いて TRACE_TICKS 回あと）覚える。名づけの取り違えは、聞いた回に
                    # 1秒以上見ていた物で数える（意味の辞書が重み1で足すのは聞く前3秒に見ていた物）
                    word, heard_obj = word_of_learn(heard_hist, tick, learn_lag(sem))
                    named = OBJ_OF_WORD.get(word, "")
                    if swap and named:
                        named = {"ball": "cup", "cup": "ball"}[named]   # 入れ替えた名前が指す物
                    learned_obj = heard_obj
                    events.append({"scene": scene, "seed": seed, "t": round(float(u.data.time), 2),
                                   "event": ev, "text": "", "word": word, "looking": look,
                                   "learned_object": learned_obj, "named_object": named,
                                   "mixup": ev == "learn" and learned_obj != named,
                                   "rows": brain.boxes["objfile"].summary(),
                                   "n_merge": brain.boxes["objfile"].n_merge})
                    if ev == "learn" and word in counts:
                        counts[word] += 1
                        print(f"[{scene} seed{seed} t={u.data.time:.1f}] 覚えた「{word}」"
                              f"（最後に1秒以上見ていた物: {learned_obj or 'なし'}、その回に見ていた物: {look or 'なし'}）"
                              f" 回数 {counts}", flush=True)
                sim_s += run_v2.K * run_v2.DT
                if min(counts.values()) >= n_target or sim_s >= max_s:
                    break
        finally:
            run_v2.close_env(env)
        seed += 1

    n_heard = sum(1 for e in events if e["event"] == "heard")
    n_learn = sum(1 for e in events if e["event"] == "learn")
    n_mix = sum(1 for e in events if e.get("mixup"))
    meta = {"ear": opts["--ear"], "n_target": n_target, "swap": swap, "counts": counts, "sim_s": round(sim_s, 1),
            "n_heard": n_heard, "n_learn": n_learn, "n_skip": sum(1 for e in events if e["event"] == "learn_skip"),
            "n_mixup": n_mix, "scenes": scenes, "seeds_used": seed}
    try:
        from ..save import code_version
        meta["code"] = code_version()
    except Exception:  # noqa: BLE001
        pass
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sem.save_learned(out, meta)
    with open(os.path.join(log_dir, f"名づけ_{tag}.csv"), "w", newline="", encoding="utf-8") as f:
        keys = ["scene", "seed", "t", "event", "text", "word", "looking", "learned_object", "named_object", "mixup", "rows", "n_merge"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(events)
    with open(os.path.join(log_dir, f"名づけ_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    done = min(counts.values()) >= n_target
    print(f"{'できた' if done else '時間切れ'}：{meta}  （{time.time() - t_start:.0f}秒）  保存: {out}")
    return meta


if __name__ == "__main__":
    main()
