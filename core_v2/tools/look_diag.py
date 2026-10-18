"""段階9の診断：目だけの場面で、太郎が1つの物を見る時間が短いわけを調べる（主のコードは変えない。記録するだけ）。

grow_names の段3の育ち（Grower）をそのまま回し、0.1秒ごとに次を書く：
* 親から見て太郎が見ている物（noun_scene.looking_at）、親が回した・鳴らした・名前を言った回
* 目を向ける反射（orienteye）の決め方：撃ったか・撃たない理由（refractory／holding／centered…）、
  撃った先の山の高さ peak、見続ける力の線 hold、今見ている行の慣れ hab、目的の抑え inhibit、探し物の地図が出ているか
* 物の名簿（objfile）の脱馴化の回数の増え、耳が意味の辞書へ送った語、意味の辞書の覚えた回
終わりに、見ていた時間の分布と「何が見るのを終わらせたか」の表、名づけが覚えるにつながらなかったわけを出す。
写しは作らない（育ち帳に書かない）。

使い方：MUJOCO_GL=osmesa python -m core_v2.tools.look_diag [--scenes 30] [--seed 0] [--from 段2_喃語_種0] [--out 置き場] [--ego efference|image]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import time

import torch

from ..boxes import orienteye as oe
from ..world_eye import noun_scene as ns
from . import grow_names as gn

CELL_PX = 224.0 / 28


def _peak_info(inbox):
    """orienteye の決め方を、係の中と同じ計算でもう一度出す（記録用）。"""
    m = inbox.get("prio")
    inhibit = float(inbox.get("gate3", 0.0))
    relief = 1.0 - oe.GOAL_RELIEF * min(max(inhibit, 0.0), 1.0)
    ior, att = oe.read_objfile(inbox.get("objfile"))
    d = {"inhibit": round(inhibit, 3), "relief": round(relief, 3), "hab": None, "m_held": None, "peak": None,
         "hold": None, "peak_rc": None, "ior_at_peak": None}
    if m is None:
        return d
    raw = m
    m = (m - oe.IOR_GAIN * relief * ior).clamp(min=0.0)
    if float(m.max()) <= 0.0:
        return d
    r, c = divmod(int(torch.argmax(m)), m.shape[1])
    d.update(peak=round(float(m[r, c]), 3), peak_rc=(r, c), raw_at_peak=round(float(raw[r, c]), 3),
             ior_at_peak=round(float(ior[r, c]), 3))
    if att is not None:
        hr, hc, hab = att
        d.update(hab=round(hab, 3), m_held=round(float(m[hr, hc]), 3), raw_held=round(float(raw[hr, hc]), 3),
                 ior_held=round(float(ior[hr, hc]), 3), held_rc=(hr, hc),
                 hold=round(float(m[hr, hc]) + oe.FIX_HOLD * (1.0 - hab) * relief, 3))
    return d


class Diag(gn.Grower):
    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.rows: list[dict] = []
        self.in_scene = False
        self.scene_t = 0
        self.flags = collections.Counter()
        self._oe_info = {}
        box = self.brain.boxes["orienteye"]
        orig = box.step

        def step(inbox, _orig=orig):
            self._oe_info = _peak_info(inbox)
            return _orig(inbox)
        box.step = step
        gz = self.brain.boxes["gaze"]
        gorig = gz.step

        def gstep(inbox, _orig=gorig):
            out = _orig(inbox)
            p = out.get("prio")
            self._seek_on = p is not None and float(p.max()) > 0.0
            return out
        gz.step = gstep
        self._seek_on = False
        ls = self.brain.boxes["listen"]
        lorig = ls.step

        def lstep(inbox, _orig=lorig):
            v = inbox.get("voice")
            self._own_voice = v is not None and float(v) > 0.5
            return _orig(inbox)
        ls.step = lstep
        self._own_voice = False
        # 親の「回す」と場面の出し入れを記録する
        orig_turn, orig_place, orig_clear = ns.turn, ns.place, ns.clear

        def turn(env, p, step, cause="parent_turn"):
            self.flags["turn"] += 1
            return orig_turn(env, p, step, cause)

        def place(env, items, cause="scene"):
            self.in_scene = True
            self.scene_t = 0
            return orig_place(env, items, cause)

        def clear(env, placed, cause="scene_end"):
            self.in_scene = False
            return orig_clear(env, placed, cause)
        ns.turn, ns.place, ns.clear = turn, place, clear

    def tick(self) -> dict:
        ob = self.brain.boxes["objfile"]
        dishab0 = ob.n_dishab
        turned = self.flags.pop("turn", 0)
        got = super().tick()
        o = self.brain.boxes["orienteye"]
        eyein = self.brain.boxes["eyein"]
        cur = self.parent.cur.word if (self.in_scene and self.parent.cur) else ""
        fired_to = ""
        info = dict(self._oe_info)
        if o.last_reason == "fire" and info.get("peak_rc"):
            r, c = info["peak_rc"]
            best = None
            for d in getattr(eyein, "last_dets", []) or []:
                dd = math.hypot(d["pos"][0] / CELL_PX - c - 0.5, d["pos"][1] / CELL_PX - r - 0.5)
                if best is None or dd < best[0]:
                    best = (dd, d["name"])
            if best is not None and best[0] <= 3.0:
                nm = best[1]
                fired_to = self.env.n9[nm]["word"] if nm in self.env.n9 else nm
            else:
                fired_to = "(物でない所)"
        heard_word = self.brain.boxes["listen"].last_word
        # 目の動きの見積もり（画像の位相相関）と、物の本当の動き（環境の正解の中心）を比べる
        dets = {d["name"]: (d["pos"][0] / CELL_PX, d["pos"][1] / CELL_PX) for d in getattr(eyein, "last_dets", []) or []}
        prev = getattr(self, "_prev_dets", {})
        common = [n for n in dets if n in prev]
        true_mv = est_mv = None
        if common and self.in_scene:
            n0 = common[0]
            true_mv = (dets[n0][0] - prev[n0][0], dets[n0][1] - prev[n0][1])
            sx, sy = getattr(eyein, "last_shift_px", (0.0, 0.0))
            est_mv = (sx / CELL_PX, sy / CELL_PX)
        self._prev_dets = dets
        row = {"scene": self.n_scenes, "in_scene": int(self.in_scene), "t": self.scene_t if self.in_scene else -1,
               "looking": cur, "turn": turned, "heard": got["heard"], "ear_word": heard_word,
               "sound": int(got["sound"]), "reason": o.last_reason, "fired_to": fired_to,
               "seek_on": int(self._seek_on), "dishab": ob.n_dishab - dishab0,
               "n_dets": len(getattr(eyein, "last_dets", []) or []),
               "sem": self.brain.boxes["semantic"].last_event, "rows": ob.summary(), "own_voice": int(self._own_voice),
               "true_mv": None if true_mv is None else tuple(round(v, 1) for v in true_mv),
               "est_mv": None if est_mv is None else tuple(round(v, 1) for v in est_mv),
               "shift_px": tuple(round(v, 1) for v in getattr(eyein, "last_shift_px", (0, 0))),
               "image_shift_px": tuple(round(v, 1) for v in getattr(eyein, "last_image_shift_px", (0, 0))),
               "pred_px": tuple(round(v, 1) for v in getattr(eyein, "last_pred_px", (0, 0))),
               "ego_mode": getattr(eyein, "last_ego_mode", ""), "ego_peak": round(getattr(eyein, "last_ego_peak", 0.0), 3)}
        info.pop("peak_rc", None)
        row.update(info)
        self.rows.append(row)
        if self.in_scene:
            self.scene_t += 1
        return got


def looks(rows):
    """親から見た「見ている物」の続き（場面の中だけ）。[(場面, 語, 始め, 長さ, 終わりの行の番号 or None)]"""
    out = []
    i, n = 0, len(rows)
    while i < n:
        r = rows[i]
        if not r["in_scene"] or not r["looking"]:
            i += 1
            continue
        j = i
        while j + 1 < n and rows[j + 1]["in_scene"] and rows[j + 1]["scene"] == r["scene"] \
                and rows[j + 1]["looking"] == r["looking"]:
            j += 1
        end = j + 1 if j + 1 < n and rows[j + 1]["in_scene"] and rows[j + 1]["scene"] == r["scene"] else None
        out.append((r["scene"], r["looking"], i, j - i + 1, end))
        i = j + 1
    return out


def end_cause(rows, start, length, end):
    """見るのを終わらせたもの。終わりの直前（撃った指令は1〜2回遅れて目に届く）の撃った回を探し、その回の様子で分ける。"""
    if end is None:
        return "場面の終わり", None
    fire = None
    for k in range(end, max(start, end - 4) - 1, -1):
        if rows[k]["reason"] == "fire":
            fire = k
            break
    if fire is None:
        return "撃たずに外れた（物が回って中心が動いた等）", None
    f = rows[fire]
    if f.get("hold") is None:
        return "見ている行が無い（見続ける力が働かない）", fire
    if f.get("raw_held") is not None and f["raw_held"] < 0.3:
        return "名簿の「今見ている行」が物からずれていた（目の動きの見積もりちがい）", fire
    if f["seek_on"]:
        return "探し物の地図（名前を聞いて探す）", fire
    if any(rows[k]["dishab"] for k in range(max(0, fire - 5), fire + 1)):
        return "脱馴化（目立ちの急な上がり）", fire
    return "慣れが育って、ほかの物が上回った", fire


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--objects", default="made", choices=["made", "real"])
    ap.add_argument("--eye", default="table", choices=["dinov2", "table"])
    ap.add_argument("--from", dest="start", default="段2_喃語_種0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ego", default=None, choices=["efference", "image"],
                    help="目が動いた分の求め方（既定は eyein の既定＝efference。image は段階9の直し前）")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(gn.growth.REPO, "G", "logs", f"段階9_見る時間_{a.objects}_{a.eye}_種{a.seed}")
    os.makedirs(out, exist_ok=True)
    _find = gn.growth.find

    def find(name):     # 育ち帳の置き場は Windows の区切り（\\）で書いてある
        row = dict(_find(name))
        row["file"] = row["file"].replace("\\", "/")
        return row
    gn.growth.find = find
    g = Diag(a.seed, a.objects, a.eye, None, log_root=None, start=a.start, kind="off")
    g.save_images(False)
    if a.ego is not None:
        g.brain.boxes["eyein"].ego = a.ego
    print(f"[診断] 始める太郎：{g.start_name()}　場面 {a.scenes}", flush=True)
    t0 = time.time()
    try:
        for s in range(a.scenes):
            g.scene()
            if (s + 1) % 10 == 0:
                print(f"[診断] 場面 {s + 1}　{time.time() - t0:.0f}秒　名づけ {g.parent.stats['names']}", flush=True)
    finally:
        g.close()
    rows = g.rows
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(os.path.join(out, "毎回.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    L = looks(rows)
    bins = [(0, 0.6), (0.6, 1.1), (1.1, 1.6), (1.6, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 13.0)]
    hist = {f"{lo}〜{hi}秒": sum(1 for x in L if lo <= x[3] * 0.1 + 1e-9 < hi) for lo, hi in bins}
    causes = collections.Counter()
    fire_detail = collections.defaultdict(list)
    for sc, wd, st, ln, end in L:
        c, fi = end_cause(rows, st, ln, end)
        causes[c] += 1
        if fi is not None:
            fire_detail[c].append({k: rows[fi].get(k) for k in ("hab", "m_held", "hold", "peak", "inhibit", "fired_to")}
                                  | {"look_s": round(ln * 0.1, 1)})
    # 撃たなかった理由の割合（場面の中）
    reasons = collections.Counter(r["reason"] for r in rows if r["in_scene"])
    # 撃つ前の hold と peak（場面の中で撃った回）
    fires = [r for r in rows if r["in_scene"] and r["reason"] == "fire"]
    # 名づけ → 覚えた？
    names = []
    for i, r in enumerate(rows):
        if r["heard"] and r["in_scene"]:
            ear = [rows[k]["ear_word"] for k in range(i, min(len(rows), i + 4)) if rows[k]["ear_word"]]
            learned = [rows[k]["sem"] for k in range(i, min(len(rows), i + 70)) if rows[k]["sem"] in ("learn", "learn_skip")]
            names.append({"text": r["heard"], "looking": r["looking"], "ear": ear, "sem": learned[:1],
                          "own_voice": r["own_voice"]})
    no_ear = [n for n in names if not n["ear"]]
    # 目が跳んだ回（物の本当の動きが2マス以上）の、見積もりの誤差
    errs = []
    for r in rows:
        if r.get("true_mv") and math.hypot(*r["true_mv"]) >= 2.0:
            t, e = r["true_mv"], r["est_mv"]
            errs.append((math.hypot(*t), math.hypot(t[0] - e[0], t[1] - e[1]), t[0] * e[0] + t[1] * e[1] > 0))
    errs.sort()
    ego = {"jumps": len(errs),
           "true_median_cells": round(errs[len(errs) // 2][0], 1) if errs else None,
           "err_median_cells": round(sorted(x[1] for x in errs)[len(errs) // 2], 1) if errs else None,
           "err_over_3cells": sum(x[1] > 3 for x in errs), "same_direction": sum(x[2] for x in errs),
           # 跳びの大きさ（マス）ごとに、見積もりが3マス以内で合った割合
           "by_jump_cells": {f"{lo}〜{hi}": {"n": len(b), "ok_within_3": round(sum(x[1] <= 3 for x in b) / len(b), 2),
                                                 "err_median": round(sorted(x[1] for x in b)[len(b) // 2], 1)}
                             for lo, hi in ((2, 4), (4, 8), (8, 12), (12, 16), (16, 99))
                             for b in [[x for x in errs if lo <= x[0] < hi]] if b}}
    # 着いてから3回目（撃った指令が名簿に届いたあと）に、「今見ている行」が物の上にあったか
    landed = []
    for sc, wd, st, ln, end in L:
        k = st + 3
        if k < len(rows) and rows[k]["looking"] == wd and rows[k].get("raw_held") is not None:
            landed.append((rows[k]["raw_held"] >= 0.3, ln * 0.1))
    on = [x[1] for x in landed if x[0]]
    off = [x[1] for x in landed if not x[0]]
    held_ok = {"on_object": len(on), "off_object": len(off),
               "look_median_on": round(sorted(on)[len(on) // 2], 1) if on else None,
               "look_median_off": round(sorted(off)[len(off) // 2], 1) if off else None}
    res = {"start": g.start_name(), "scenes": a.scenes, "parent": g.parent.stats, "looks": len(L),
           "look_median_s": sorted(x[3] for x in L)[len(L) // 2] * 0.1 if L else None,
           "hist": hist, "end_causes": dict(causes), "orienteye_reasons_in_scene": dict(reasons),
           "fires_in_scene": len(fires), "ego_estimate": ego, "held_row": held_ok,
           "fire_examples": {k: v[:8] for k, v in fire_detail.items()},
           "namings": len(names), "namings_ear_silent": len(no_ear),
           "ear_silent_texts": collections.Counter(n["text"] for n in no_ear).most_common(20),
           "ear_silent_while_own_voice": sum(n["own_voice"] for n in no_ear),
           "namings_learned": sum(1 for n in names if n["sem"] == ["learn"]),
           "ear_known": sorted(g.brain.boxes["listen"].known()),
           "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(out, "まとめ.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
