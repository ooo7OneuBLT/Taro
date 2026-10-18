"""新しい目の試作①：DINOv2 のマス特徴で「床らしくない所」をかたまりにする（段階3 作り直し・2026-10-16）。

仕様書 2026-10-16_段階3_仕様書_作り直し 技術付録 A の 3〜5 の試作。core_v2 本体には入れない。
1. DINOv2 のマス特徴（224入力で16×16、448入力で32×32）を長さ1にそろえる
2. 画面の外周 BORDER マスの特徴を k 個に分け（球面 k平均）、床の見た目 g1..gk とする
3. 床らしさ s = max_j f·g_j。s < floor_cos のマスを物のマスとする
4. 隣り合う物のマスをつないでかたまりにし、min_patches 未満は捨てる
5. 位置はマスの重心（床らしくなさ 1−s で重み）。範囲はマスをそのまま画素へ広げたもの
color_w > 0 なら、床の典型色（外周マスの色の中央値 k 個）からの色の差も「床らしくなさ」に足す（比べる案）。

使い方（採点は eye_test.py の DETECTORS から呼ぶ。ここは引数の掃引用）：
  python G/scripts/eye_dino_floor.py --sweep   … 特徴を1回だけ計算して、閾値を振って表を出す
"""

from __future__ import annotations

import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)
import core_v2  # noqa: E402,F401  負荷の決まり（core_v2/perf.py：スレッド1本など）を numpy・torch より先に効かせる
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.join(REPO, "taro_core", "src", "senses"))

BORDER = 2
_MODEL = {}


def load_model():
    if "m" not in _MODEL:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL["m"] = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                     verbose=False).eval().to(dev)
        _MODEL["dev"] = dev
    return _MODEL["m"], _MODEL["dev"]


def features(img: np.ndarray, scale: int = 1) -> tuple[np.ndarray, int]:
    """(n, n, 384) の長さ1の特徴と n。scale=2 なら画像を448へ広げて32×32マス。"""
    from object_detector import patch_features
    model, dev = load_model()
    if scale != 1:
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((224 * scale, 224 * scale), Image.BILINEAR))
    p, n = patch_features(model, img, dev)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    return p.reshape(n, n, -1), n


def border_mask(n: int, b: int = BORDER) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[:b], m[-b:], m[:, :b], m[:, -b:] = True, True, True, True
    return m


def spherical_kmeans(x: np.ndarray, k: int, iters: int = 10) -> np.ndarray:
    """x: (m, d) 長さ1。初期値は「平均に一番似たもの」から順に、既に選んだものと一番似ていないもの。"""
    cents = [x[int(np.argmax(x @ x.mean(0)))]]
    while len(cents) < k:
        sim = np.max(np.stack([x @ c for c in cents]), axis=0)
        cents.append(x[int(np.argmin(sim))])
    c = np.stack(cents)
    for _ in range(iters):
        lab = np.argmax(x @ c.T, axis=1)
        for j in range(k):
            if (lab == j).any():
                v = x[lab == j].mean(0)
                c[j] = v / (np.linalg.norm(v) + 1e-8)
    return c


def floor_score(f: np.ndarray, k: int = 2) -> np.ndarray:
    n = f.shape[0]
    g = spherical_kmeans(f[border_mask(n)], k)
    return np.max(np.einsum("ijd,kd->ijk", f, g), axis=-1)


def color_offness(img: np.ndarray, n: int, k: int = 2, src: str = "border") -> np.ndarray:
    """マスごとの平均色と、床の典型色（k 平均）との最小距離（0〜1 に縮める）。
    src="border"：外周マスの色から取る。src="all"：画面全体のマスを k+2 個に分け、
    マスの15%以上を占める群（多数派）だけを床の色とする（端で切れた物が外周に混ざっても汚れない）。"""
    from PIL import Image
    small = np.array(Image.fromarray(img).resize((n, n), Image.BOX), dtype=np.float32)
    if src == "all":
        cols = small.reshape(-1, 3)
        kk = k + 2
        cents = cols[np.random.default_rng(0).choice(len(cols), size=kk, replace=False)]
        for _ in range(15):
            lab = np.argmin(((cols[:, None] - cents[None]) ** 2).sum(-1), axis=1)
            for j in range(kk):
                if (lab == j).any():
                    cents[j] = np.median(cols[lab == j], axis=0)
        share = np.bincount(lab, minlength=kk) / len(lab)
        cents = cents[share >= 0.15]
        d = np.sqrt(((small[:, :, None] - cents[None, None]) ** 2).sum(-1)).min(-1)
        return np.clip(d / 100.0, 0.0, 1.0)
    cols = small[border_mask(n)]
    cents = cols[np.random.default_rng(0).choice(len(cols), size=k, replace=False)]
    for _ in range(10):
        lab = np.argmin(((cols[:, None] - cents[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if (lab == j).any():
                cents[j] = np.median(cols[lab == j], axis=0)
    d = np.sqrt(((small[:, :, None] - cents[None, None]) ** 2).sum(-1)).min(-1)
    return np.clip(d / 100.0, 0.0, 1.0)


def stereo_offness(img_l: np.ndarray, img_r: np.ndarray, n: int, scale_px: float = 6.0) -> np.ndarray:
    """両目のずれ（視差）から「床の面から浮いている度合い」をマスごとに出す（0〜1）。
    視差は半大域ブロック照合（OpenCV StereoSGBM、左目基準）。床は平らなので、マスごとの視差に
    平面 d = a·x + b·y + c を外れ値を除きながら当てはめ、平面より手前（視差が大きい）ぶんを
    scale_px 画素で割って 0〜1 にする。人間の両眼視差（V1 から）にあたる。"""
    import cv2
    gl = cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7,
                                 P1=8 * 49, P2=32 * 49, uniquenessRatio=5,
                                 speckleWindowSize=50, speckleRange=2)
    disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    c = 224 // n if 224 % n == 0 else 224 / n
    cells = np.full((n, n), np.nan, dtype=np.float32)
    step = 224.0 / n
    for i in range(n):
        for j in range(n):
            blk = disp[int(i * step):int((i + 1) * step), int(j * step):int((j + 1) * step)]
            if np.isfinite(blk).sum() >= 3:
                cells[i, j] = np.nanmedian(blk)
    ys, xs = np.mgrid[0:n, 0:n]
    ok = np.isfinite(cells)
    A = np.stack([xs[ok], ys[ok], np.ones(ok.sum())], 1).astype(np.float64)
    y = cells[ok].astype(np.float64)
    keep = np.ones(len(y), dtype=bool)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        r = y - A @ coef
        mad = np.median(np.abs(r[keep])) + 1e-3
        keep = np.abs(r) < 3 * mad
    plane = coef[0] * xs + coef[1] * ys + coef[2]
    res = np.nan_to_num(cells - plane, nan=0.0)
    return np.clip(res / scale_px, 0.0, 1.0)


def split_by_look(lab: np.ndarray, k: int, f: np.ndarray, split_cos: float) -> tuple[np.ndarray, int]:
    """くっついた2つの物を見た目で分ける：かたまりの中の特徴を2つに分け、2つの中心の
    似かたが split_cos より低ければ（見た目が違う物どうし）、別々のかたまりにする。"""
    out = np.zeros_like(lab)
    nxt = 0
    for i in range(1, k + 1):
        m = lab == i
        parts = [m]
        if m.sum() >= 6:
            x = f[m]
            c = spherical_kmeans(x, 2)
            if float(c[0] @ c[1]) < split_cos:
                side = np.argmax(x @ c.T, axis=1)
                parts = []
                for j in range(2):
                    mj = np.zeros_like(m)
                    mj[m] = side == j
                    sub, ns = ndimage.label(mj, structure=np.ones((3, 3)))
                    parts += [sub == q for q in range(1, ns + 1)]
        for p in parts:
            nxt += 1
            out[p] = nxt
    return out, nxt


def blobs(s: np.ndarray, floor_cos: float, min_patches: int, img_px: int = 224,
          f: np.ndarray | None = None, split_cos: float = -1.0) -> list[dict]:
    n = s.shape[0]
    obj = s < floor_cos
    lab, k = ndimage.label(obj, structure=np.ones((3, 3)))
    if f is not None and split_cos > -1.0:
        lab, k = split_by_look(lab, k, f, split_cos)
    out = []
    cell = img_px / n
    for i in range(1, k + 1):
        m = lab == i
        if m.sum() < min_patches:
            continue
        w = (1.0 - s) * m
        ys, xs = np.mgrid[0:n, 0:n]
        cx = float((w * (xs + 0.5)).sum() / w.sum()) * cell
        cy = float((w * (ys + 0.5)).sum() / w.sum()) * cell
        mask = np.kron(m, np.ones((int(round(cell)), int(round(cell))), dtype=bool))[:img_px, :img_px]
        if mask.shape != (img_px, img_px):
            full = np.zeros((img_px, img_px), dtype=bool)
            full[:mask.shape[0], :mask.shape[1]] = mask
            mask = full
        out.append({"pos": (cx, cy), "mask": mask, "n_patches": int(m.sum())})
    return out


def make_detector(floor_cos: float = 0.5, min_patches: int = 2, scale: int = 1, k: int = 2,
                  color_w: float = 0.0, split_cos: float = -1.0, color_src: str = "border",
                  depth_w: float = 0.0):
    import torch
    load_model()

    def run(img, img_r=None):
        f, n = features(img, scale)
        s = floor_score(f, k)
        if color_w > 0:
            s = s - color_w * color_offness(img, n, k, color_src)
        if depth_w > 0 and img_r is not None:
            s = s - depth_w * stereo_offness(img, img_r, n)
        return blobs(s, floor_cos, min_patches, f=f, split_cos=split_cos)
    run.sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    return run


def sweep() -> None:
    import eye_test as et
    data = os.path.join(REPO, "G/data/目のテスト")
    index = json.load(open(os.path.join(data, "index.json"), encoding="utf-8"))
    frames = [np.load(os.path.join(data, e["file"])) for e in index]
    rows = []
    scale, k = 2, 2
    cache = []
    for d in frames:
        f, n = features(d["img_l"], scale)
        cache.append((floor_score(f, k), color_offness(d["img_l"], n, k, "border"),
                      color_offness(d["img_l"], n, k, "all"), stereo_offness(d["img_l"], d["img_r"], n), f))
    for color_src in ("border", "all"):
        for color_w in (0.5, 1.0, 1.5):
            for depth_w in (0.0, 0.5, 1.0, 2.0):
                for floor_cos in (0.0, 0.1, 0.2):
                    for min_patches in (4, 8):
                        tot = {nm: [0, 0, 0, 0] for nm in et.NAMES}
                        false = 0
                        for (s, cb, ca, so, f), d in zip(cache, frames):
                            co = cb if color_src == "border" else ca
                            dets = blobs(s - color_w * co - depth_w * so, floor_cos, min_patches,
                                         f=f, split_cos=0.7)
                            res, fa = et.score_frame(dets, d["masks"])
                            false += fa
                            for nm in et.NAMES:
                                r = res[nm]
                                if r is None:
                                    continue
                                jj = 2 if r["edge"] else 0
                                tot[nm][jj] += 1
                                tot[nm][jj + 1] += int(r["found"])
                        row = {"color_src": color_src, "color_w": color_w, "depth_w": depth_w,
                               "floor_cos": floor_cos, "min_patches": min_patches,
                               "false": round(false / len(frames), 2)}
                        for nm, (a, b, c, e) in tot.items():
                            row[f"{nm}_in"] = round(b / a, 3) if a else None
                            row[f"{nm}_edge"] = round(e / c, 3) if c else None
                        rows.append(row)
    out = os.path.join(REPO, "G/logs/目のテスト/dino_floor_掃引2.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = [r for r in rows if r["false"] <= 1.0 and min(r[f"{nm}_in"] for nm in et.NAMES) >= 0.95]
    key = lambda r: (min(r[f"{nm}_edge"] for nm in et.NAMES), -r["false"])
    for r in sorted(ok, key=key, reverse=True)[:10]:
        print(r)


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sys.path.insert(0, os.path.dirname(__file__))
        sweep()
