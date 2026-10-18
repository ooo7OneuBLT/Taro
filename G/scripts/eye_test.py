"""目だけのテストの採点（段階3 作り直し・2026-10-16）。

問題集（G/scripts/make_eye_dataset.py が作る）の左目の画像を1枚ずつ目の部品に見せ、
正解（物を1つずつ消して撮り直した差）と比べて採点する。太郎の脳も試行も回さない。

採点の項目（物ごと・正解が50画素以上見えているコマだけ）：
  （物ごとに「画像の端に触れない＝全身が見えている」と「端で切れている」を分けても数える）
  拾えた   … 拾った物の中心が、正解の重心から3度（≒10.2画素、視野60度・224画素）以内
  重なり   … 拾えた物のうち、正解との重なり（IoU、共通部分÷合併）が一番大きい値
  物でない所 … どの正解の重心からも3度より遠く、どの正解の範囲にも入らない拾い物の数
  時間     … 1枚あたりのミリ秒（GPU の待ちも含む）
拾えなかったコマは、正解の輪郭（緑・ボール 赤・コップ 青・妨害物）と拾った中心（黄）を描いて並べる。

使い方：
  python G/scripts/eye_test.py [--det sam] [--data G/data/目のテスト] [--max N]
出力：G/logs/目のテスト/<det>/summary.json・per_frame.csv・拾えなかったコマ_*.png
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)
import core_v2  # noqa: E402,F401  負荷の決まり（core_v2/perf.py：スレッド1本など）を numpy・torch より先に効かせる
import numpy as np
os.chdir(REPO)

NAMES = ["ball", "cup", "distractor"]
NAMES_JA = {"ball": "ボール", "cup": "コップ", "distractor": "妨害物"}
FOCAL_PX = 112.0 / math.tan(math.radians(30.0))     # 視野60度・224画素
HIT_PX = FOCAL_PX * math.tan(math.radians(3.0))     # 3度 ≒ 10.2画素
MIN_VISIBLE_PX = 50


# ---- 目の部品（拾い方）ごとの入口：画像 → [{"pos": (x, y), "mask": bool[224,224] か None}] ----

def make_sam():
    """今の目（core_v2 の eyein、毎回全体を切り出す detect_every=1）。切り抜きは
    object_detector._mask_appearance の呼び出しを横取りして控える（taro_core は変えない）。"""
    import torch
    from core_v2.boxes.eyein import Eyein
    eye = Eyein(detect_every=1, detector="sam")
    eye.setup()
    od = sys.modules["object_detector"]   # visual_attention.py:273 がこの名前で読み込む
    captured = []
    orig = od._mask_appearance

    def spy(mask, *a, **k):
        captured.append(np.asarray(mask, dtype=bool).copy())
        return orig(mask, *a, **k)
    od._mask_appearance = spy

    def run(img):
        captured.clear()
        eye.see(img)
        out = []
        for d in eye.last_dets:
            x, y = d["pos"]
            best = None
            for m in captured:
                ys, xs = np.nonzero(m)
                if len(xs) and abs(xs.mean() - x) < 1.0 and abs(ys.mean() - y) < 1.0:
                    best = m
                    break
            out.append({"pos": (float(x), float(y)), "mask": best})
        return out

    def reset():
        eye.reset()
    run.reset = reset
    run.sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    return run


def make_dino_floor_variant(**kw):
    def factory():
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from eye_dino_floor import make_detector
        return make_detector(**kw)
    return factory


def make_eyein_floor():
    """core_v2 に組み込んだ目の入口（detector="floor"）そのもの。試作 dino_floor_a と同じ成績になるかの確かめ。"""
    import torch
    from core_v2.boxes.eyein import Eyein
    from core_v2.floor_objects import cells_to_mask
    eye = Eyein()
    eye.setup()

    def run(img):
        eye.see(img)
        return [{"pos": d["pos"], "mask": cells_to_mask(d["cells"])} for d in eye.last_dets]
    run.reset = eye.reset
    run.sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    return run


DETECTORS = {
    "sam": make_sam,
    "eyein_floor": make_eyein_floor,
    # 試作①：DINOv2 の床らしさ＋床の典型色からの差（eye_dino_floor.py。値は掃引 2026-10-16 の上位）
    "dino_floor_nosplit": make_dino_floor_variant(floor_cos=0.1, min_patches=4, scale=2, k=2, color_w=1.0),
    "dino_floor": make_dino_floor_variant(floor_cos=0.1, min_patches=4, scale=2, k=2, color_w=1.0, split_cos=0.7),
    # 試作①a：床の典型色を画面全体の多数派から取る（掃引2 の上位。奥行きは視差が荒れて悪化したので入れない）
    "dino_floor_a": make_dino_floor_variant(floor_cos=0.0, min_patches=4, scale=2, k=2, color_w=1.5,
                                            split_cos=0.7, color_src="all"),
    "dino_floor_224": make_dino_floor_variant(floor_cos=0.3, min_patches=2, scale=1, k=2, color_w=0.5),
}


def iou(a, b) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(u) if u else 0.0


def score_frame(dets, masks):
    res = {}
    cents = []
    for k, name in enumerate(NAMES):
        m = masks[k]
        if m.sum() < MIN_VISIBLE_PX:
            cents.append(None)
            res[name] = None
            continue
        ys, xs = np.nonzero(m)
        c = (xs.mean(), ys.mean())
        cents.append(c)
        hits = [d for d in dets if math.dist(d["pos"], c) <= HIT_PX]
        best_iou = max((iou(d["mask"], m) for d in hits if d["mask"] is not None), default=None)
        edge = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        res[name] = {"found": bool(hits), "iou": best_iou, "edge": edge}
    false = 0
    for d in dets:
        x, y = d["pos"]
        near = any(c is not None and math.dist((x, y), c) <= HIT_PX for c in cents)
        inside = any(masks[k][min(max(int(y), 0), 223), min(max(int(x), 0), 223)]
                     for k in range(len(NAMES)))
        if not (near or inside):
            false += 1
    return res, false


def draw_misses(misses, out_dir, det_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "sans-serif"]
    cols = {"ball": "lime", "cup": "red", "distractor": "deepskyblue"}
    for page in range(0, len(misses), 12):
        chunk = misses[page:page + 12]
        fig, axes = plt.subplots(3, 4, figsize=(12, 9.5))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (name, img, masks, dets, missed) in zip(axes.flat, chunk):
            ax.imshow(img)
            for k, n in enumerate(NAMES):
                if masks[k].any():
                    ax.contour(masks[k], levels=[0.5], colors=cols[n], linewidths=1)
            for d in dets:
                ax.plot(*d["pos"], "x", color="yellow", ms=7, mew=2)
            ax.set_title(f"{name}\n拾えず: {'・'.join(NAMES_JA[m] for m in missed)}", fontsize=7)
        fig.suptitle(f"{det_name}：拾えなかったコマ（輪郭＝正解 緑ボール・赤コップ・青妨害物、×＝拾った中心）")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"拾えなかったコマ_{page // 12 + 1}.png"), dpi=90)
        plt.close(fig)


def main() -> None:
    args = sys.argv[1:]

    def opt(name, default):
        return type(default)(args[args.index(name) + 1]) if name in args else default
    det_name = opt("--det", "sam")
    data = os.path.join(REPO, opt("--data", "G/data/目のテスト"))
    max_n = opt("--max", 0)
    out_dir = os.path.join(REPO, "G/logs/目のテスト", det_name + ("" if "--data" not in args
                                                                else "_" + os.path.basename(data)))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(data, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    if max_n:
        index = index[:max_n]
    run = DETECTORS[det_name]()
    rows, misses = [], []
    last_key = None
    # 最初の1枚は読み込み・準備の時間が乗るので、時間の集計から外す
    for i, e in enumerate(index):
        key = (e["scene"], e["seed"])
        if key != last_key and hasattr(run, "reset"):
            run.reset()
            last_key = key
        d = np.load(os.path.join(data, e["file"]))
        img, masks = d["img_l"], d["masks"]
        run.sync()
        t0 = time.perf_counter()
        dets = run(img, d["img_r"]) if getattr(run, "stereo", False) else run(img)
        run.sync()
        ms = (time.perf_counter() - t0) * 1000.0
        res, false = score_frame(dets, masks)
        row = {"file": e["file"], "scene": e["scene"], "jump_age": e["jump_age"], "ms": round(ms, 1),
               "n_dets": len(dets), "false": false}
        missed = []
        for n in NAMES:
            r = res[n]
            row[f"{n}_found"] = "" if r is None else int(r["found"])
            row[f"{n}_iou"] = "" if r is None or r["iou"] is None else round(r["iou"], 3)
            row[f"{n}_edge"] = "" if r is None else int(r["edge"])
            if r is not None and not r["found"]:
                missed.append(n)
        rows.append(row)
        if missed:
            misses.append((e["file"].replace(".npz", ""), img, masks, dets, missed))
        if i < 1:
            row["warmup"] = 1
    with open(os.path.join(out_dir, "per_frame.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()) + ["warmup"])
        w.writeheader()
        w.writerows(rows)

    def summarize(sel):
        s = {"n_frames": len(sel)}
        for n in NAMES:
            v = [r[f"{n}_found"] for r in sel if r[f"{n}_found"] != ""]
            ious = [r[f"{n}_iou"] for r in sel if r[f"{n}_iou"] != ""]
            s[n] = {"visible": len(v), "found_rate": round(sum(v) / len(v), 3) if v else None,
                    "iou_median": round(float(np.median(ious)), 3) if ious else None}
            for tag, e in (("inside", 0), ("edge", 1)):
                vv = [r[f"{n}_found"] for r in sel if r[f"{n}_found"] != "" and r[f"{n}_edge"] == e]
                s[n][f"found_{tag}"] = f"{sum(vv)}/{len(vv)}"
        all3 = [r for r in sel if all(r[f"{n}_found"] != "" for n in NAMES)]
        s["all3_found_rate"] = (round(sum(all(r[f"{n}_found"] for n in NAMES) for r in all3) / len(all3), 3)
                                if all3 else None)
        s["false_per_frame"] = round(float(np.mean([r["false"] for r in sel])), 2) if sel else None
        timed = [r["ms"] for r in sel if not r.get("warmup")]
        s["ms_median"] = round(float(np.median(timed)), 1) if timed else None
        return s
    summary = {"det": det_name, "hit_deg": 3.0, "all": summarize(rows),
               "just_jumped": summarize([r for r in rows if r["jump_age"] == 1]),
               "per_scene": {sc: summarize([r for r in rows if r["scene"] == sc])
                             for sc in sorted({r["scene"] for r in rows})}}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    draw_misses(misses, out_dir, det_name)
    print(json.dumps(summary["all"], ensure_ascii=False))
    print("跳んだ直後:", json.dumps(summary["just_jumped"], ensure_ascii=False))
    for sc, s in summary["per_scene"].items():
        print(sc, json.dumps(s, ensure_ascii=False))
    print(f"拾えなかったコマ {len(misses)} 枚 → {out_dir}")


if __name__ == "__main__":
    main()
