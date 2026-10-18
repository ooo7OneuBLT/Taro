"""物の名簿だけのテスト（段階4・仕様書 §7 の A）。脳は回さない。

目だけのテストの問題集（G/data/目のテスト、make_eye_dataset.py。場面×種ごとに20回続けて撮ったコマ、
4回ごとに目を乱数の向きへ跳ばす）を、目の入口 → 物の名簿 の順に流し、本当の物（masks）ごとに
* 名簿に載っている割合（物の画素の8画素以内に、行のかけらの中心がある）
* 前のコマでも載っていたとき、同じ番号のままの割合（合格 95%以上）
* 目の入口が拾い損ねたコマで、名簿には残っていた回数
* 1つの物に2行以上が付いた（割れた）コマの数
を出す。

使い方：
  python G/scripts/objfile_test.py [--data G/data/目のテスト] [--out G/logs/段階4_A_名簿だけ] [--diag]
  --diag … 行が作られた・追い出された理由を数えて diag.json に書き、短くまとめて表示する
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)
import core_v2  # noqa: E402,F401  負荷の決まり（core_v2/perf.py：スレッド1本など）を numpy・torch より先に効かせる
import numpy as np
os.chdir(REPO)

import torch  # noqa: E402

from core_v2.boxes.base import ENV  # noqa: E402
from core_v2.boxes.eyein import Eyein  # noqa: E402
from core_v2.boxes.objfile import Objfile, object_center  # noqa: E402

NAMES = ["ball", "cup", "distractor"]
NEAR_PX = 8.0


def near_mask(mask: np.ndarray, r: float, c: float) -> bool:
    """マスの (行, 列) の中心の画素が、物の画素から NEAR_PX 以内か。"""
    x, y = c * 8.0 + 4.0, r * 8.0 + 4.0
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return False
    return bool(np.min((xs - x) ** 2 + (ys - y) ** 2) <= NEAR_PX ** 2)


def main() -> None:
    args = sys.argv[1:]
    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default
    data = os.path.join(REPO, opt("--data", "G/data/目のテスト"))
    out = os.path.join(REPO, opt("--out", "G/logs/段階4_A_名簿だけ"))
    os.makedirs(out, exist_ok=True)

    seqs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(data, "*.npz"))):
        m = re.match(r"(.+)_t(\d+)\.npz$", os.path.basename(f))
        seqs[m.group(1)].append((int(m.group(2)), f))

    eye = Eyein()
    eye.setup()
    of = Objfile()
    diag = "--diag" in args
    cur = {"masks": None, "seq": ""}
    D = {"dets_by_class": defaultdict(int), "new_by_class": defaultdict(int),
         "victim_by_class": defaultdict(int), "refused": 0, "unmatched_dist": defaultdict(int),
         "frames_with_ndets": defaultdict(int), "examples": []}

    def classify(r, c):
        if cur["masks"] is None:
            return "?"
        hits = [NAMES[k] for k in range(len(NAMES)) if near_mask(cur["masks"][k], r, c)]
        return hits[0] if hits else "junk"

    if diag:
        orig_new, orig_match = Objfile._new_row, Objfile._match

        def new_row(self, part, **kw):
            before = list(self.rows)
            row = orig_new(self, part, **kw)
            cls = classify(part.r, part.c)
            D["new_by_class"][cls] += 1
            if row is None:
                D["refused"] += 1
            gone = [rw for rw in before if rw not in self.rows]
            for v in gone:
                vc = classify(*v.pos)
                D["victim_by_class"][vc] += 1
                if len(D["examples"]) < 15:
                    D["examples"].append(f"{cur['seq'][-12:]} t{self.t}: 新 {cls}@{part.r:.0f},{part.c:.0f} が "
                                         f"{vc}@{v.pos[0]:.0f},{v.pos[1]:.0f}(n_seen={sum(p.n_seen for p in v.parts)},"
                                         f"最後に見えた{self.t - v.last_seen}回前) を追い出した")
            return row

        def match(self, things, ego_moving):
            parts = [p for _, p in self._parts()]
            for tag, m in things:
                if float(torch.linalg.vector_norm(tag)) == 0.0 or float(m.max()) <= 0.0:
                    continue
                r, c = object_center(m)
                D["dets_by_class"][classify(r, c)] += 1
                dmin = min((((p.r - r) ** 2 + (p.c - c) ** 2) ** 0.5 for p in parts), default=99.0)
                b = "<1.5" if dmin <= 1.5 else "1.5-3" if dmin <= 3 else "3-6" if dmin <= 6 else ">6"
                D["unmatched_dist"][f"{classify(r, c)} {b}{' 跳び' if ego_moving else ''}"] += 1
            return orig_match(self, things, ego_moving)

        Objfile._new_row, Objfile._match = new_row, match
    zero2 = torch.zeros(2)
    stats = {k: {"present": 0, "listed": 0, "pairs": 0, "same": 0, "kept_when_missed": 0,
                 "missed_by_eye": 0, "split": 0} for k in NAMES}
    rows_per_frame = []
    for name, frames in sorted(seqs.items()):
        eye.reset()
        of.reset()
        prev = {}
        cur["seq"] = name
        for _, f in sorted(frames):
            d = np.load(f)
            cur["masks"] = d["masks"]
            eo = eye.step({ENV: d["img_l"]})
            of.step({"eyein": eo["objfile"], "orienteye": zero2})
            rows_per_frame.append(len(of.rows))
            dets = [object_center(m) for tag, m in eo["objfile"][:-1]]
            now = {}
            for k, key in enumerate(NAMES):
                mask = d["masks"][k]
                if mask.sum() == 0:
                    continue
                st = stats[key]
                st["present"] += 1
                rows = [rw for rw in of.rows if any(near_mask(mask, p.r, p.c) for p in rw.parts)]
                eye_has = any(near_mask(mask, r, c) for r, c in dets)
                if not eye_has:
                    st["missed_by_eye"] += 1
                    if rows:
                        st["kept_when_missed"] += 1
                if not rows:
                    continue
                st["listed"] += 1
                if len(rows) > 1:
                    st["split"] += 1
                nums = {rw.number for rw in rows}
                now[key] = nums
                if key in prev:
                    st["pairs"] += 1
                    st["same"] += bool(prev[key] & nums)
            prev = now
    summary = {k: dict(v, listed_rate=round(v["listed"] / max(v["present"], 1), 3),
                       same_rate=round(v["same"] / max(v["pairs"], 1), 3)) for k, v in stats.items()}
    summary["all"] = {
        "same_rate": round(sum(v["same"] for v in stats.values()) / max(sum(v["pairs"] for v in stats.values()), 1), 3),
        "kept_when_missed": sum(v["kept_when_missed"] for v in stats.values()),
        "missed_by_eye": sum(v["missed_by_eye"] for v in stats.values()),
        "rows_per_frame_mean": round(float(np.mean(rows_per_frame)), 2),
        "n_merge_last_seq": of.n_merge,
        "n_frames": len(rows_per_frame),
        "pass": None,
    }
    summary["all"]["pass"] = summary["all"]["same_rate"] >= 0.95
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=1)
    for k, v in summary.items():
        print(k, v)
    if diag:
        with open(os.path.join(out, "diag.json"), "w", encoding="utf-8") as fp:
            json.dump(D, fp, ensure_ascii=False, indent=1)
        print("目の入口が出した物（本当の物／床などの余分）:", dict(D["dets_by_class"]))
        print("新しい行を作った物:", dict(D["new_by_class"]), " 入れなかった:", D["refused"])
        print("追い出された行:", dict(D["victim_by_class"]))
        print("物ごとの、一番近いかけらまでの距離（マス）:", dict(sorted(D["unmatched_dist"].items())))
        for e in D["examples"]:
            print(" ", e)


if __name__ == "__main__":
    main()
