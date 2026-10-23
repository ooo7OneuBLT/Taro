"""床から浮いて見えるまとまりを物として拾う（目の入口の切り出し、段階3 作り直し・2026-10-16）。

人間の視覚野との対応（どれも粗い対応。詳しくは eyein.py の冒頭）：
* V1〜V4・下側頭葉 … DINOv2 のマスごとの特徴（見た目）。DINO のような自己学習の画像AIの中〜後ろの
  層は、サルの V4・下側頭葉の反応をよく予測する（Yamins & DiCarlo 2014 ほか）
* V2（図と地・境界の持ち主）… 「床（地）らしくないマス」を図とし、隣り合う図のマスをつなぐ。
  つながった2つの物でも見た目がはっきり違えば分ける
* 色の差 … V1〜V4 の色の働き。床の典型色を画面全体の多数派から決め、それからの差を足す

手順（目だけのテストの試作 dino_floor_a、G/scripts/eye_dino_floor.py と同じ。採点は
G/scripts/eye_test.py、同じ200枚と別の100枚で確かめた）：
1. 画像を448画素へ広げて DINOv2（vits14）にかけ、32×32 マスの特徴を長さ1にそろえる
2. 画面の外周2マスの特徴を2つに分けて（球面 k平均）床の見た目 g1・g2 とし、床らしさ s=max(f·g)
3. マスの平均色と床の典型色（画面全体のマスの色を4つに分け、15%以上を占める群）との距離を
   100 で割って 0〜1 にし、COLOR_W 倍して s から引く
4. s < FLOOR_COS のマスを物のマスとし、8近傍でつなぐ。かたまりの中の特徴を2つに分け、2つの中心の
   似かたが SPLIT_COS 未満なら別々の物にする。MIN_PATCHES マス未満は捨てる
5. 位置はマスの重心（床らしくなさ 1−s で重み、224画素の座標）。見た目は、かたまりのマスの DINOv2
   特徴（長さをそろえる前）の平均（384）
値は目だけのテストで決めた［仮置き］。端で切れた物も「ふちが画面の外へ続く物」としてそのまま拾う。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

SCALE = 2              # 224 → 448 画素（32×32 マス）。16×16 ではマスが粗くコップを半分しか拾えなかった
BORDER = 2             # 床の見た目を取る外周のマス
FLOOR_K = 2            # 床の見た目の数（市松模様の2色）
FLOOR_COS = 0.0
FLOOR_SRC = "all"      # 床の見た目を取る所（"border"＝外周／"all"＝画面全体の多数派。目だけのテストで all が丈夫）
COLOR_W = 1.5
COLOR_SHARE = 0.15     # 床の典型色とみなす、画面全体のマスに占める割合の下限
COLOR_ITERS = 5        # 色の k平均の繰り返し（15→5 に減らして速さを詰める）
SPLIT_COS = 0.7
MIN_PATCHES = 4
IMG = 224


def dino_features(model, device, img224: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(n*n, 384) の生の特徴・長さ1の特徴・n を返す。"""
    import torch
    from PIL import Image
    big = np.array(Image.fromarray(img224).resize((IMG * SCALE, IMG * SCALE), Image.BILINEAR))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = torch.from_numpy(((big.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1))[None]
    # GPU では半分の精度で計算（約2倍速い。特徴の似かた 0.99 以上・目だけのテストの成績は同じ、2026-10-17）
    half = torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda"))
    with torch.no_grad(), half:
        p = model.forward_features(x.to(device))["x_norm_patchtokens"][0].float().cpu().numpy()
    n = int(round(np.sqrt(p.shape[0])))
    f = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    return p, f, n


def border_mask(n: int, b: int = BORDER) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[:b], m[-b:], m[:, :b], m[:, -b:] = True, True, True, True
    return m


def spherical_kmeans(x: np.ndarray, k: int, iters: int = 10) -> np.ndarray:
    """x: (m, d) 長さ1。初期値は平均に一番似たものから、既に選んだものと一番似ていないものを順に足す。"""
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


def floor_score(f: np.ndarray, n: int) -> np.ndarray:
    """床らしさ。FLOOR_SRC="border"：外周マスの見た目を FLOOR_K 個に分けたものを床とする。
    "all"：画面全体のマスを FLOOR_K+2 個に分け、COLOR_SHARE 以上を占める群（多数派）だけを床とする。
    外周だけだと、画面の端で切れた灰色のコップが床の見た目の1つに数えられ、コップの胴が床になった
    （段階2の合格テスト、ボール右・seed0・t42、2026-10-16）。"""
    fg = f.reshape(n, n, -1)
    if FLOOR_SRC == "all":
        x = f.reshape(-1, f.shape[-1])
        g = spherical_kmeans(x, FLOOR_K + 2, iters=COLOR_ITERS)
        share = np.bincount(np.argmax(x @ g.T, axis=1), minlength=len(g)) / len(x)
        g = g[share >= COLOR_SHARE]
    else:
        g = spherical_kmeans(fg[border_mask(n)], FLOOR_K)
    return np.max(fg @ g.T, axis=-1)


def color_offness(img224: np.ndarray, n: int) -> np.ndarray:
    from PIL import Image
    small = np.array(Image.fromarray(img224).resize((n, n), Image.BOX), dtype=np.float32)
    cols = small.reshape(-1, 3)
    kk = FLOOR_K + 2
    cents = cols[np.random.default_rng(0).choice(len(cols), size=kk, replace=False)]
    for _ in range(COLOR_ITERS):
        lab = np.argmin(((cols[:, None] - cents[None]) ** 2).sum(-1), axis=1)
        for j in range(kk):
            if (lab == j).any():
                cents[j] = np.median(cols[lab == j], axis=0)
    share = np.bincount(lab, minlength=kk) / len(lab)
    cents = cents[share >= COLOR_SHARE]
    d = np.sqrt(((small[:, :, None] - cents[None, None]) ** 2).sum(-1)).min(-1)
    return np.clip(d / 100.0, 0.0, 1.0)


def _split_by_look(lab: np.ndarray, k: int, f: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.zeros_like(lab)
    nxt = 0
    for i in range(1, k + 1):
        m = lab == i
        parts = [m]
        if m.sum() >= 6:
            x = f[m]
            c = spherical_kmeans(x, 2)
            if float(c[0] @ c[1]) < SPLIT_COS:
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


def detect(model, device, img224: np.ndarray) -> list[dict]:
    """画像（224×224×3、uint8）→ [{"pos": (x, y) 224画素, "area": 画面に占める割合,
    "appearance": 384, "cells": bool[n,n]}]。"""
    p, f, n = dino_features(model, device, img224)
    s = floor_score(f, n) - COLOR_W * color_offness(img224, n)
    lab, k = ndimage.label(s < FLOOR_COS, structure=np.ones((3, 3)))
    fg = f.reshape(n, n, -1)
    lab, k = _split_by_look(lab, k, fg)
    pg = p.reshape(n, n, -1)
    cell = IMG / n
    ys, xs = np.mgrid[0:n, 0:n]
    out = []
    for i in range(1, k + 1):
        m = lab == i
        if m.sum() < MIN_PATCHES:
            continue
        w = (1.0 - s) * m
        tot = float(w.sum())
        if tot <= 0:
            w, tot = m.astype(np.float64), float(m.sum())
        cx = float((w * (xs + 0.5)).sum() / tot) * cell
        cy = float((w * (ys + 0.5)).sum() / tot) * cell
        out.append({"pos": (cx, cy), "area": float(m.sum()) / (n * n),
                    "appearance": pg[m].mean(axis=0), "cells": m})
    return out


def cells_to_mask(cells: np.ndarray) -> np.ndarray:
    n = cells.shape[0]
    rep = IMG // n
    return np.kron(cells, np.ones((rep, rep), dtype=bool))
