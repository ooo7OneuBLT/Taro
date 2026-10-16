"""意味の辞書に、親の名づけから名前を覚えさせる道具（段階3・仕様書 付録B-3）。

親（環境の「親が名づける仕組み」run/world/parent.py:32 ParentLabeling）が物を振って見せ、
太郎がそれを gaze_hold_sec（3秒）見続けたら名前を言う。意味の辞書（semantic、learning=True）は
名前を聞いた回に、視野の真ん中にある物の荷札と語の荷札を結び付ける。
各語 N 回覚えたら止めて、学んだ表を保存する。手で登録する道具（register_appearance.py）は使わない。

使い方：
  python -m core_v2.tools.learn_names [N] [--swap] [--out パス] [--scenes 場面,場面] [--max-s 秒]
    N       … 各語を何回覚えたら止めるか（既定 5）
    --swap  … 名前を入れ替える（ボールに「こっぷ」と言う）＝合格の線⑤の確かめ用
既定の出力：G/models/semantic_learned_N{N}.pt（--swap なら _swap が付く）。記録は G/logs/段階3_名づけ/

覚えた回ごとに「そのとき実際に見ていた物」（視線と物の角度、10度以内で角の小さい方）を記録し、
名前と食い違った回（取り違え）を数える。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

from .. import run_v2

DEFAULT_SCENES = "段階2b_3物体_ボール左_2026-10-15,段階2b_3物体_ボール右_2026-10-15"
WORD_OF = {"toy1": "ぼおる", "toy2": "こっぷ"}       # toy1＝ボール（test_object1）・toy2＝コップ
OBJ_OF_WORD = {"ぼおる": "ball", "こっぷ": "cup"}
LOOK_DEG = 10.0
TRIAL_S = 60.0          # 1つの場面を続けて回す長さ。足りなければ種を変えてもう1回


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
    swap = "--swap" in argv
    opts = {"--out": None, "--scenes": DEFAULT_SCENES, "--max-s": "900"}
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
    log_dir = os.path.join(run_v2.REPO, "G", "logs", "段階3_名づけ")
    os.makedirs(log_dir, exist_ok=True)
    max_s = float(opts["--max-s"])

    from ..brain import Brain
    brain = Brain()
    brain.boxes["eyein"].setup()
    sem = brain.boxes["semantic"]
    sem.learning = True

    counts = {w: 0 for w in OBJ_OF_WORD}
    events = []          # 名前を聞いた回・覚えた回の記録
    heard_hist = []      # (tick, 語)：覚えた回を、2回前に聞いた語に結び付けるため
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
                if text:
                    word = next((w for w in OBJ_OF_WORD if w in text), "")
                    heard_hist.append((tick, word))
                    events.append({"scene": scene, "seed": seed, "t": round(float(u.data.time), 2),
                                   "event": "heard", "text": text, "word": word, "looking": look})
                ev = sem.last_event
                if ev in ("learn", "learn_skip"):
                    # 語は earin→listen→semantic で2回遅れて届く
                    word = next((w for t0, w in reversed(heard_hist) if tick - 3 <= t0 <= tick), "")
                    named = OBJ_OF_WORD.get(word, "")
                    if swap and named:
                        named = {"ball": "cup", "cup": "ball"}[named]   # 入れ替えた名前が指す物
                    events.append({"scene": scene, "seed": seed, "t": round(float(u.data.time), 2),
                                   "event": ev, "text": "", "word": word, "looking": look,
                                   "named_object": named, "mixup": ev == "learn" and look != named})
                    if ev == "learn" and word in counts:
                        counts[word] += 1
                        print(f"[{scene} seed{seed} t={u.data.time:.1f}] 覚えた「{word}」"
                              f"（見ていた物: {look or 'なし'}） 回数 {counts}", flush=True)
                sim_s += run_v2.K * run_v2.DT
                if min(counts.values()) >= n_target or sim_s >= max_s:
                    break
        finally:
            run_v2.close_env(env)
        seed += 1

    n_heard = sum(1 for e in events if e["event"] == "heard")
    n_learn = sum(1 for e in events if e["event"] == "learn")
    n_mix = sum(1 for e in events if e.get("mixup"))
    meta = {"n_target": n_target, "swap": swap, "counts": counts, "sim_s": round(sim_s, 1),
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
        keys = ["scene", "seed", "t", "event", "text", "word", "looking", "named_object", "mixup"]
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
