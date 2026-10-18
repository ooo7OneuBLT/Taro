"""段階7：体ありの太郎が、見ている物の名前を言うか（合格の線①の体ありの確かめ。仕様書 §3）。

体なしの試験場（voice_bed）で育てた声（口の帳面・M・聞いた語・言う語の一覧、export_voice の JSON）を
41係の脳に読み、段階4〜6で名前を覚えた意味の辞書（learn_names の出力）と一緒に、段階2bの3つの物の場面で
親は黙ったまま、太郎に自由に見させる。声の係は覚えず、喃語も出さない（言う語の係から語が来たときだけ言う）。

太郎が声を出したら、声を出し始めた回に見ていた物（視線と物の角度が LOOK_DEG 以内で近い方）と一緒に記録する。
当たり＝その声に一番近い名前（親の語の中で）が見ていた物の名前で、似かたが SIM_MIN（0.5）を超える
（体なしの試験場の試し①と同じ数え方）。物ごとに、声を出した回数・当たり・見に行った回数（1秒以上続けて見た回）を数える。

使い方：
  python -m core_v2.tools.say_names --voice core_v2/data/段階7_声_seed0.json --semantic G/models/semantic_learned_N5.pt
                                    [--taros 0,1,2] [--trials 6] [--trial-s 30] [--age 月齢] [--log-dir 段階7_体あり]
    --taros … 脳の種（試行の乱数）。同じ声の状態で、種ごとに別の太郎として数える
    --age   … 声の出口へ渡す月齢。省略すると JSON の月齢
  育ちの段（段4、2026-10-18）：
  python -m core_v2.tools.say_names --from auto [--growth-seed 0] [--taros 0,1,2] …
  段階9の写し（20語・部屋3つ）は、覚えた物そのもの（その語の部屋に、ほかの2語と並べる）で確かめる：
  python -m core_v2.tools.say_names --from auto --objects real [--eye dinov2|table] [--growth-seed 0] [--taros 0,1,2] …
    段階2bの丸と箱は、20語の太郎が名づけで一度も見ていない形なので、--objects なしでは「名前を言えるか」を測れない
    --from auto … 育ち帳の段3（無ければ段2…）のいちばん新しい写し（目だけの場面）を読む（--voice・--semantic は要らない）。写しの名前も書ける
    終わりに段4の写しを残す（段4_目_種<種>）。注意：段4を「育てる」道具はまだ無い。この道具は確かめだけで、声も意味の辞書も
    覚えないので、写しの中身は元の写しと同じ（育ち帳の note に書く）。--from のときの世界の既定は eye
出力：G/logs/<log-dir>/taro<種>.json（声ごとの記録）と まとめ.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .. import run_v2
from ..boxes.objfile import MIN_LOOK_S
from ..boxes.voice import SIM_MIN
from .learn_names import DEFAULT_SCENES
from .voice_bed import closest_word, import_voice

LOOK_DEG = 10.0
NAME_OF = {"ball": "ぼおる", "cup": "こっぷ"}
END_GAP_TICKS = 6


def looking_at(row) -> str:
    cands = [(row[f"ang_{k}"], k) for k in run_v2.OBJECTS if row.get(f"ang_{k}") is not None
             and row[f"ang_{k}"] <= LOOK_DEG]
    return min(cands)[1] if cands else ""


def score(rows, look_fn=None, objects=None) -> dict:
    """1試行の記録から、声と、そのとき見ていた物を拾う。look_fn(row)→見ていた物（既定は段階2bの3つの物の角度）。"""
    look_fn = look_fn or looking_at
    voices, cur, start_look, last = [], "", "", -99
    looks = {k: 0 for k in (objects or run_v2.OBJECTS)}
    run_obj, run_len = "", 0
    need = int(round(MIN_LOOK_S / (run_v2.K * run_v2.DT)))
    for r in rows:
        lk = look_fn(r)
        run_len = run_len + 1 if lk and lk == run_obj else (1 if lk else 0)
        run_obj = lk
        if lk and run_len == need:
            looks[lk] = looks.get(lk, 0) + 1
        if r["voice"]:
            if not cur:
                start_look = lk
            cur += r["voice"]
            last = r["tick"]
        elif cur and r["tick"] - last > END_GAP_TICKS:
            voices.append((start_look, cur))
            cur = ""
    if cur:
        voices.append((start_look, cur))
    return {"voices": voices, "looks": looks}


# ---------------------------------------------------------------- 段階9：部屋と20語の物で（--objects made|real）
STAGE9_WORDS = ("ぼおる", "こっぷ")    # 段階7と同じ2語を、その語の部屋で、ほかの2つの物と並べて見せる


def closest_name(voice: str, names) -> tuple[str, float]:
    from .. import mouth
    return max(((w, mouth.similarity(w, voice)) for w in names), key=lambda x: x[1])


def stage9(a, log_dir: str) -> dict:
    """段階9の写し（20語・部屋3つ・本物か自作の物）を、覚えた物そのもので確かめる。段階2bの丸と箱は、20語の太郎が
    名づけで一度も見ていない形なので、そこで数えても「名前を言えるか」にならない（2026-10-18）。
    1試行：STAGE9_WORDS の語を順に1つ、その語の部屋で覚えた個体（個体0〜2から乱数）を、同じ部屋のほかの語2つと並べ、
    親は黙ったまま a.trial_s 秒自由に見させる。声と、声を出し始めた回に見ていた物（noun_scene.looking_at）を数える。
    当たり＝その声に一番近い名前（20語の中で）が見ていた物の語で、似かたが SIM_MIN を超える。"""
    import random

    from ..world_eye import instances as ins
    from ..world_eye import noun_scene as ns
    from . import growth
    from .grow_names import EyeTaro

    row = growth.start_point(4, seed=a.growth_seed, world="eye") if a.start == "auto" else growth.find(a.start)
    if row is None:
        raise LookupError("段4の元にする写し（段3、目だけの場面）が育ち帳に無い")
    t = EyeTaro(a.growth_seed, a.objects, a.eye, a.table, log_root=os.path.join(log_dir, "画像"), start=row["name"],
                all_insts=True)
    b = t.brain.boxes
    for k in ("semantic", "listen", "kind"):
        b[k].learning = False
    month = growth.month_of(row)
    age = a.age if a.age is not None else float(month if month is not None else 24.0)
    print(f"[段4 段階9] 写し {row['name']}（月齢 {age}）　目：{t.eye_info}　物：{a.objects}", flush=True)
    summary = {"snapshot": row["name"], "objects": a.objects, "eye": t.eye_info, "age": age, "taros": {}}
    t0 = time.time()
    try:
        for taro in [int(x) for x in a.taros.split(",")]:
            rng = random.Random(taro)
            res = {"voices": [], "looks": {}}
            for k in range(a.trials):
                t.save_images(k < 1)
                word = STAGE9_WORDS[k % len(STAGE9_WORDS)]
                env = t.use_room(ins.ROOM_OF[word])
                others = rng.sample([w for w in ins.WORDS if ins.ROOM_OF[w] == ins.ROOM_OF[word] and w != word], 2)
                items = [(w, rng.choice(ins.TRAIN_INST), rng.randrange(ns.N_VIEWS)) for w in [word] + others]
                placed = ns.place(env, items, cause="say_names")
                torch.manual_seed(taro * 1000 + k)
                t.brain.reset()
                rows, echo = [], ""
                for i in range(int(round(a.trial_s / ns.TICK_S))):
                    heard = []
                    for _ in range(run_v2.K):
                        obs, _r, _t, _tr, info = env.step()
                        utt = info.get("parent_utterance")
                        if utt and utt.get("text"):
                            heard.append(str(utt["text"]))
                    ext = {"eyein": obs["eye_left"], "earin": " ".join(x for x in (echo, " ".join(heard)) if x),
                           "voiceout": age}
                    to_env = t.brain.tick(ext)
                    cmd = to_env.get("eyeout")
                    if cmd is not None:
                        env._orienting.set_map_target(float(cmd[0]), float(cmd[1]))
                    echo = to_env.get("voiceout", "") or ""
                    p = ns.looking_at(env, placed)
                    rows.append({"tick": i, "voice": echo, "look": p.word if p is not None else ""})
                ns.clear(env, placed, cause="say_names_end")
                sc = score(rows, look_fn=lambda r: r["look"], objects=[it[0] for it in items])
                res["voices"] += [{"trial": k, "shown": [it[0] for it in items], "looking": lk, "voice": v}
                                  for lk, v in sc["voices"]]
                for o, n in sc["looks"].items():
                    res["looks"][o] = res["looks"].get(o, 0) + n
                print(f"[{time.time() - t0:6.0f}s] 太郎{taro} 試行{k}（{'・'.join(it[0] for it in items)}）：声 {sc['voices']} "
                      f"見に行った {sc['looks']}", flush=True)
            per = {}
            for w in sorted(set(STAGE9_WORDS) | {e["looking"] for e in res["voices"] if e["looking"]}):
                vs = [e["voice"] for e in res["voices"] if e["looking"] == w]
                hit = 0
                for v in vs:
                    cw, sim = closest_name(v, ins.WORDS)
                    hit += cw == w and sim > SIM_MIN
                per[w] = {"looks": res["looks"].get(w, 0), "said": len(vs), "hit": hit, "examples": sorted(set(vs))[:5]}
            res["per_object"] = per
            summary["taros"][taro] = per
            with open(os.path.join(log_dir, f"taro{taro}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
            print(f"太郎{taro}：{per}", flush=True)
    finally:
        t.close()
    snap = growth.save(b, 4, growth.free_name(f"段4_目_種{a.growth_seed}"), end=True, parent=row["name"], month=age,
                       conditions={"world": "eye", "eye": a.eye, "seed": a.growth_seed, "tool": "say_names",
                                   "objects": a.objects, "taros": a.taros, "trials": a.trials},
                       note="段4を育てる道具はまだ無い。say_names は確かめだけ（声も意味の辞書も覚えない）ので、中身は元の写しと同じ")
    summary["snapshot_out"] = snap["name"]
    print(f"[段4] 写し：{snap['name']}", flush=True)
    with open(os.path.join(log_dir, "まとめ.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summary["taros"], ensure_ascii=False))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--voice", default=None)
    ap.add_argument("--semantic", default=None)
    ap.add_argument("--from", dest="start", default=None, help="育ちの段の写しから始める（auto か写しの名前）")
    ap.add_argument("--growth-seed", type=int, default=0, help="--from auto で引く写しの種")
    ap.add_argument("--taros", default="0,1,2")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--trial-s", type=float, default=30.0)
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--scenes", default=DEFAULT_SCENES)
    ap.add_argument("--log-dir", default="段階7_体あり")
    ap.add_argument("--world", default=None, choices=["mimo", "eye"], help="目だけの場面で回すなら eye（--from のときの既定）")
    ap.add_argument("--objects", default=None, choices=["made", "real"],
                    help="段階9の部屋と物で確かめる（--from が要る。段階9の20語の写しはこちら）")
    ap.add_argument("--eye", default="dinov2", choices=["dinov2", "table"], help="--objects のときの目")
    ap.add_argument("--table", default=None, help="--eye table の見た目の素の表")
    a = ap.parse_args(argv)
    if a.objects:
        if not a.start:
            ap.error("--objects には --from が要る")
        log_dir = os.path.join(run_v2.REPO, "G", "logs", f"{a.log_dir}_{a.objects}_{a.eye}_種{a.growth_seed}")
        os.makedirs(log_dir, exist_ok=True)
        return stage9(a, log_dir)
    a.world = a.world or ("eye" if a.start else "mimo")
    if not a.start and not (a.voice and a.semantic):
        ap.error("--voice と --semantic、または --from が要る")
    from ..world_eye.switch import select, suffix
    select(a.world)
    path = lambda p: p if os.path.isabs(p) else os.path.join(run_v2.REPO, p)   # noqa: E731
    log_dir = os.path.join(run_v2.REPO, "G", "logs", suffix(a.log_dir))
    os.makedirs(log_dir, exist_ok=True)

    from ..brain import Brain
    brain = Brain()
    brain.boxes["eyein"].setup()
    from . import growth
    row = growth.open_start(brain.boxes, a.start, 4, seed=a.growth_seed)
    if row is not None:
        meta = {"snapshot": row["name"], "month": growth.month_of(row)}
        print(f"[段4] 写し {row['name']} を読んだ（段{row['phase']}、月齢 {meta['month']}）", flush=True)
    else:
        brain.boxes["semantic"].load_learned(path(a.semantic))
        meta = import_voice(brain.boxes, path(a.voice))
    age = a.age if a.age is not None else float(meta["month"] if meta.get("month") is not None else 24.0)
    scenes = [x for x in a.scenes.split(",") if x]
    summary = {"voice": a.voice, "voice_meta": meta, "semantic": a.semantic, "age": age, "taros": {}}
    t0 = time.time()
    for taro in [int(x) for x in a.taros.split(",")]:
        res = {"voices": [], "looks": {k: 0 for k in run_v2.OBJECTS}}
        for k in range(a.trials):
            seed = taro * 1000 + k
            env = run_v2.build_env(scenes[k % len(scenes)], None, seed=seed, silent=True)
            try:
                rows = run_v2.run_trial(brain, env, seed=seed, duration_s=a.trial_s, age_months=age)
            finally:
                run_v2.close_env(env)
            s = score(rows)
            res["voices"] += [{"trial": k, "looking": lk, "voice": v} for lk, v in s["voices"]]
            for o, n in s["looks"].items():
                res["looks"][o] += n
            print(f"[{time.time() - t0:6.0f}s] 太郎{taro} 試行{k}：声 {s['voices']} 見に行った {s['looks']}", flush=True)
        per = {}
        for obj, name in NAME_OF.items():
            vs = [e["voice"] for e in res["voices"] if e["looking"] == obj]
            hit = 0
            for v in vs:
                w, sim = closest_word(v)
                hit += w == name and sim > SIM_MIN
            per[obj] = {"looks": res["looks"][obj], "said": len(vs), "hit": hit, "examples": sorted(set(vs))[:5]}
        res["per_object"] = per
        summary["taros"][taro] = per
        with open(os.path.join(log_dir, f"taro{taro}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"太郎{taro}：{per}", flush=True)
    if row is not None:
        snap = growth.save(brain.boxes, 4, growth.free_name(f"段4_{'目' if a.world == 'eye' else a.world}_種{a.growth_seed}"),
                           end=True, parent=row["name"], month=age,
                           conditions={"world": a.world, "eye": "dinov2", "seed": a.growth_seed, "tool": "say_names",
                                       "taros": a.taros, "trials": a.trials},
                           note="段4を育てる道具はまだ無い。say_names は確かめだけ（声も意味の辞書も覚えない）ので、中身は元の写しと同じ")
        summary["snapshot"] = snap["name"]
        print(f"[段4] 写し：{snap['name']}", flush=True)
    with open(os.path.join(log_dir, "まとめ.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary["taros"], ensure_ascii=False))


if __name__ == "__main__":
    main()
