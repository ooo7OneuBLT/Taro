# -*- coding: utf-8 -*-
"""物の切り出し ── 「ここに何かがある」だけを返す。種類は出さない。

【なぜ、2026-09-03】太郎はこれまで画像1枚を384個の数字（DINOv2のCLSトークン）に
要約するだけで、「どこに」「いくつ」物があるかを持っていなかった。人間の物体ファイル
（Kahneman & Treisman 1992 [Tier1]）は名前より先に、位置に紐づく入れ物として物を
保持する。切り出しは太郎の経験では育たない知覚の機構と位置づけ、DINOv2に肩代わり
させる（視覚をDINOv2で代用しているのと同じ層の判断。研究の原則・原則2）。

やり方（初版・2026-09-03）：視野の中心のマスを種にして、それに似たマスを
「図（figure）」とする（`figure_mask()`）。成人でも注視している対象が図になるので、
中心を種にするのは不自然ではない[Tier3・工学的近似]。ただし実測（F2-63）で
「視野に何も無くても必ず何か検出する」「離れた2物体を安定して見分けられない」
という弱点が確認された（`F/docs/物体ファイルと注意/文献調査/2026-09-04_複数物体検出_診断.md`）。

【2026-09-04・切り出しをMobileSAMに交換】感覚の入出力は既製部品を先に検討する方針
（[[project-policy-emergence-scope-2026-08-19]]、視覚をDINOv2で代用しているのと同じ
既製部品ファーストの判断）を、切り出しにも素直に適用し直した。訓練済みの軽量
セグメンテーションモデルMobileSAM（Zhang et al. 2023, arXiv:2306.14289, Apache 2.0）
を導入し、実測で両方の弱点が改善することを確認した
（`F/docs/物体ファイルと注意/2026-09-04c_統合まとめ.md`、`F/logs/F2-64_視野に物が無い場合/`）。
`detect()` に `mask_generator`（`load_mobilesam()`で取得）を渡すとMobileSAM経由になる。
**既定はOFF**（`mask_generator=None`なら従来のfigure_mask方式のまま。新機構は明示的に
有効化するまで既存の呼び出し元に影響しない）。DINOv2は「見た目ベクトル」の抽出役として
引き続き使う（MobileSAMは「どこからどこまでが1つの物か」の境界だけを返し、見た目の
特徴は返さない2段構成）。[[feedback-run-system-self-contained-ui]]
"""
import os
import numpy as np
from scipy import ndimage

PATCH_GRID = 16          # dinov2_vits14・224px入力でのマス目の一辺
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
MODELS_CACHE_DIR = os.path.join(_REPO_ROOT, ".models_cache")


def patch_features(model, img224, device="cpu"):
    """DINOv2 のマスごとの特徴（パッチトークン）を取り出す。

    Args:
        model: torch.hub.load("facebookresearch/dinov2", "dinov2_vits14") で得たモデル。
        img224: (224, 224, 3) の uint8 配列。
    Returns:
        (マス数, 384) の特徴と、マス目の一辺 n（224px入力なら16）。
    """
    import torch
    x = torch.from_numpy(np.asarray(img224, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    x = (x - torch.from_numpy(IMAGENET_MEAN).float()) / torch.from_numpy(IMAGENET_STD).float()
    with torch.no_grad():
        out = model.forward_features(x.to(device))
    p = out["x_norm_patchtokens"][0].cpu().numpy()
    n = int(round(np.sqrt(p.shape[0])))
    return p, n


def figure_mask(patch_feats, n, thresh=0.55):
    """視野の中心のマスを種にして、図（0=地, 1=図）のマスクを返す。thresh は0〜1、
    大きいほど図の範囲が広がる（外周との似方の差の中でどこを境にするか）。"""
    v = patch_feats / (np.linalg.norm(patch_feats, axis=1, keepdims=True) + 1e-9)
    g = v.reshape(n, n, -1)
    c = n // 2
    seed = g[max(c - 1, 0):c + 1, max(c - 1, 0):c + 1].reshape(-1, g.shape[-1]).mean(axis=0)
    seed = seed / (np.linalg.norm(seed) + 1e-9)
    sim = (v @ seed).reshape(n, n)
    edge = np.concatenate([sim[0], sim[-1], sim[:, 0], sim[:, -1]])
    lo, hi = float(edge.mean()), float(sim.max())
    t = lo + (hi - lo) * (1.0 - thresh)
    return (sim >= t).astype(np.uint8)


def detect(patch_feats, n, img_size=224, thresh=0.55, min_cells=1, img=None, mask_generator=None):
    """検出の一覧を返す。各検出＝ (中心xy[画素・img_size基準], 面積比[0-1], 見た目ベクトル[384])。

    `mask_generator` を渡すとMobileSAM経由（`img`も必須）。渡さなければ既定どおり
    旧来のfigure_mask方式（視野中心を種にする）のまま、完全に後方互換。
    """
    if mask_generator is not None:
        if img is None:
            raise ValueError("mask_generator を使うときは img（生のRGB画像）も渡す必要がある")
        return _detect_mobilesam(patch_feats, n, img, mask_generator, img_size)

    mask = figure_mask(patch_feats, n, thresh)
    # 均一な面でも1マスぶんのノイズ（グリッド線などの陰影差）で分断されるので、
    # 隣接3x3の範囲でつなげる（一般的な画像処理の後処理。意味的な判断は入れない）
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=1).astype(np.uint8)
    lab, k = ndimage.label(mask)
    scale = img_size / n
    out = []
    for i in range(1, k + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < min_cells:
            continue
        cx = float(xs.mean() + 0.5) * scale
        cy = float(ys.mean() + 0.5) * scale
        area = len(ys) / (n * n)
        v = patch_feats.reshape(n, n, -1)[ys, xs].mean(axis=0)
        out.append({"pos": (cx, cy), "area": area, "appearance": v})
    return out


def load_mobilesam(weights_path=None, device=None):
    """MobileSAM（訓練済み・軽量な物体切り出しモデル）を読み込む。1回だけ呼んで
    使い回すこと（読み込みが重い）。返り値を `detect(..., mask_generator=...)` に渡す。

    【出典】Zhang et al. 2023, arXiv:2306.14289「Faster Segment Anything」。
    Apache License 2.0（`F/docs/物体ファイルと注意/文献調査/2026-09-04c_mobilesam_ライセンス.md`
    で確認済み・商用利用可・AI学習禁止条項なし）。
    パラメータ（points_per_side等）はF2-64検証時と同じ設定
    （`F/docs/物体ファイルと注意/2026-09-04c_統合まとめ.md`）。
    """
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
    import torch
    if weights_path is None:
        weights_path = os.path.join(MODELS_CACHE_DIR, "checkpoints", "mobile_sam.pt")
    if device is None:
        want = os.environ.get("TARO_VISION_DEVICE")
        device = want if want else ("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(weights_path):
        raise RuntimeError(
            "MobileSAMの重み(%s)が無い。取得元: "
            "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"
            % weights_path)
    sam = sam_model_registry["vit_t"](checkpoint=weights_path)
    sam.to(device=device)
    sam.eval()
    return SamAutomaticMaskGenerator(
        sam, points_per_side=32, pred_iou_thresh=0.86,
        stability_score_thresh=0.92, min_mask_region_area=50)


def _mask_touches_edge(bbox, w, h, margin=1):
    """背景（壁・床など）は画像の端まで届くことが多く、物体は中央寄りになる
    ことを使った足切り。F2-64実測：物なし画面のマスク3枚は全て端に接触、
    物あり画面ではコップ本体だけが端に触れなかった（詳細は上記統合まとめ）。"""
    x, y, bw, bh = bbox
    return x <= margin or y <= margin or (x + bw) >= (w - margin) or (y + bh) >= (h - margin)


def _detect_mobilesam(patch_feats, n, img, mask_generator, img_size=224, min_area_frac=0.01):
    img = np.asarray(img)
    h, w = img.shape[0], img.shape[1]
    raw_masks = mask_generator.generate(img)
    feats_grid = patch_feats.reshape(n, n, -1)
    scale = img_size / n
    out = []
    for m in raw_masks:
        if _mask_touches_edge(m["bbox"], w, h):
            continue
        area = m["area"] / (h * w)
        if area < min_area_frac:
            continue
        seg = m["segmentation"]
        bx, by, bw, bh = m["bbox"]
        cx = float(bx) + float(bw) / 2.0
        cy = float(by) + float(bh) / 2.0
        # マスクの範囲に半分以上重なるDINOv2のマスだけを、この物体の見た目とする
        patch_ids = []
        for i in range(n):
            y0, y1 = int(i * scale), int((i + 1) * scale)
            for j in range(n):
                x0, x1 = int(j * scale), int((j + 1) * scale)
                cell = seg[y0:y1, x0:x1]
                if cell.size and cell.mean() >= 0.5:
                    patch_ids.append((i, j))
        if patch_ids:
            v = np.mean([feats_grid[i, j] for i, j in patch_ids], axis=0)
        else:
            # マスクが1マスより小さいときは中心座標に一番近いマスで代用
            i = min(max(int(cy / scale), 0), n - 1)
            j = min(max(int(cx / scale), 0), n - 1)
            v = feats_grid[i, j]
        out.append({"pos": (cx, cy), "area": area, "appearance": v})
    return out
