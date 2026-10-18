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


def score(rows) -> dict:
    """1試行の記録から、声と、そのとき見ていた物を拾う。"""
    voices, cur, start_look, last = [], "", "", -99
    looks = {k: 0 for k in run_v2.OBJECTS}
    run_obj, run_len = "", 0
    need = int(round(MIN_LOOK_S / (run_v2.K * run_v2.DT)))
    for r in rows:
        lk = looking_at(r)
        run_len = run_len + 1 if lk and lk == run_obj else (1 if lk else 0)
        run_obj = lk
        if lk and run_len == need:
            looks[lk] += 1
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
    a = ap.parse_args(argv)
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
