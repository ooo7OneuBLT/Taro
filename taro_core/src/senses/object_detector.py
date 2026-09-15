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
という弱点が確認された（`doc/文献調査/物体ファイルと注意/2026-09-04_複数物体検出_診断.md`）。

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


def detect(patch_feats, n, img_size=224, thresh=0.55, min_cells=1, img=None, mask_generator=None,
           cohesion_gap_px=None, point_predictor=None, blobs=None):
    """検出の一覧を返す。各検出＝ (中心xy[画素・img_size基準], 面積比[0-1], 見た目ベクトル[384])。

    `mask_generator` を渡すとMobileSAM経由（`img`も必須）。渡さなければ既定どおり
    旧来のfigure_mask方式（視野中心を種にする）のまま、完全に後方互換。

    `cohesion_gap_px`（【2026-09-09・凝集性】仕様_物体ファイルの人間寄せ_凝集性・
    連続性・上限）：Noneなら従来どおり（既定不変）。整数を渡すと、MobileSAM経由の
    検出だけに効く。詳細は `_detect_mobilesam` 参照。

    `point_predictor`（【2026-09-09・予測して確かめる検出「追記3」1節】）：
    Noneなら従来どおり（既定不変・`emb`キーは付かない）。`make_point_predictor()`
    で得た `SamPredictor` を渡すと、見回り（scan）の各検出にも確認(confirm)と
    同じ埋め込みから計算した `emb`（256次元）を足す。詳細は `_detect_mobilesam` 参照。

    `blobs`（【2026-09-09・仕様_見る側の段構成】段1の動いた塊の一覧）：Noneなら
    従来どおり（既定不変）。渡すと`_cohesion_groups`のblobベースの凝集も併用する。
    """
    if mask_generator is not None:
        if img is None:
            raise ValueError("mask_generator を使うときは img（生のRGB画像）も渡す必要がある")
        return _detect_mobilesam(patch_feats, n, img, mask_generator, img_size,
                                  cohesion_gap_px=cohesion_gap_px, point_predictor=point_predictor,
                                  blobs=blobs)

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


def load_mobilesam(weights_path=None, device=None, points_per_batch=None):
    """MobileSAM（訓練済み・軽量な物体切り出しモデル）を読み込む。1回だけ呼んで
    使い回すこと（読み込みが重い）。返り値を `detect(..., mask_generator=...)` に渡す。

    【出典】Zhang et al. 2023, arXiv:2306.14289「Faster Segment Anything」。
    Apache License 2.0（`doc/文献調査/物体ファイルと注意/2026-09-04c_mobilesam_ライセンス.md`
    で確認済み・商用利用可・AI学習禁止条項なし）。
    パラメータ（points_per_side等）はF2-64検証時と同じ設定
    （`F/docs/物体ファイルと注意/2026-09-04c_統合まとめ.md`）。

    【なぜ points_per_batch を引数化したか、2026-09-07】メモリ削減候補の検証
    （実装・仕事1）。`SamAutomaticMaskGenerator` の既定は64（1バッチで同時に
    処理する候補点の数。大きいほど中間テンソルが太る）。既定 None のときは
    SamAutomaticMaskGenerator 自身の既定値（64）がそのまま使われる＝
    従来と1ビットも変わらない。値を渡したときだけ明示的に上書きする。
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
    kwargs = dict(pred_iou_thresh=0.86, stability_score_thresh=0.92,
                  min_mask_region_area=50, points_per_side=32)
    if points_per_batch is not None:
        kwargs["points_per_batch"] = int(points_per_batch)
    return SamAutomaticMaskGenerator(sam, **kwargs)


def make_point_predictor(mask_generator):
    """点プロンプトによる問い合わせ（`segment_at_points`）用の `SamPredictor` を返す。

    【なぜ、2026-09-09】仕様_予測して確かめる検出_2026-09-09「後半」1節。
    斥候報告（2026-09-09）で確認済み：点プロンプト `SamPredictor` は自動生成
    （`SamAutomaticMaskGenerator`）と同じ重み（`mask_generator.predictor.model`）で
    使える別インスタンス。1回だけ作って呼び出し元で使い回すこと（読み込みは軽いが、
    毎回作り直す必要は無い）。"""
    from mobile_sam import SamPredictor
    return SamPredictor(mask_generator.predictor.model)


def _mask_touches_edge(bbox, w, h, margin=1):
    """背景（壁・床など）は画像の端まで届くことが多く、物体は中央寄りになる
    ことを使った足切り。F2-64実測：物なし画面のマスク3枚は全て端に接触、
    物あり画面ではコップ本体だけが端に触れなかった（詳細は上記統合まとめ）。"""
    x, y, bw, bh = bbox
    return x <= margin or y <= margin or (x + bw) >= (w - margin) or (y + bh) >= (h - margin)


def _cohesion_groups(masks, gap, blobs=None):
    """`masks`（各要素に "bbox" を持つ辞書のリスト）の bbox を `gap` 画素だけ
    広げた箱同士が重なるものを union-find で連結成分にまとめ、インデックスの
    グループのリストを返す（1個だけの物は要素1個のグループのまま）。

    【なぜ、2026-09-09・凝集性】Spelke の凝集性の原理（乳児は連結した面の
    かたまりを1つの物として扱う[Tier1]）を、検出器が出したマスク同士の
    近さ（隙間が`gap`画素以内）で近似する[Tier3・工学的近似：本当の面の
    連結ではなく矩形の近さで代用]。

    `blobs`（【2026-09-09・仕様_見る側の段構成】段1が出す動いた塊の一覧、
    各要素に"bbox"を持つ）：Noneなら従来どおり（既定不変）。渡すと、2つの
    マスクの重心が同じblobのbboxに入るものも追加でunionする（接触判定は
    従来どおり残す＝人間の分節の最初の手がかりは共通運動、Kellman & Spelke
    1983[Tier1]）。"""
    m = len(masks)
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    boxes = []
    for mk in masks:
        x, y, bw, bh = mk["bbox"]
        boxes.append((x - gap, y - gap, x + bw + gap, y + bh + gap))
    for i in range(m):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, m):
            bx0, by0, bx1, by1 = boxes[j]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                union(i, j)

    if blobs:
        centroids = []
        for mk in masks:
            x, y, bw, bh = mk["bbox"]
            centroids.append((x + bw / 2.0, y + bh / 2.0))
        for b in blobs:
            bx, by, bw, bh = b["bbox"]
            members = [i for i, (cx, cy) in enumerate(centroids)
                       if bx <= cx <= bx + bw and by <= cy <= by + bh]
            for k in range(1, len(members)):
                union(members[0], members[k])

    groups = {}
    for i in range(m):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def _mask_appearance(mask, feats_grid, n, scale, cx=None, cy=None):
    """マスク範囲に半分以上重なるDINOv2のマスだけを見た目ベクトルとする
    （detect()内の元の計算を切り出しただけ。segment_at_pointsと共用するため
    ＝仕様_見る側の段構成_実装_2026-09-09.md 3節「新しい計算を作らない」）。
    `cx`/`cy`（省略可）：フォールバック（マスクが1マスより小さいとき）で
    使う中心座標。省略時はマスク自身の重心から計算する（_detect_mobilesamは
    従来どおりbbox中心のcx/cyを明示的に渡し、挙動を1ビットも変えない）。"""
    patch_ids = []
    for i in range(n):
        y0, y1 = int(i * scale), int((i + 1) * scale)
        for j in range(n):
            x0, x1 = int(j * scale), int((j + 1) * scale)
            cell = mask[y0:y1, x0:x1]
            if cell.size and cell.mean() >= 0.5:
                patch_ids.append((i, j))
    if patch_ids:
        return np.mean([feats_grid[i, j] for i, j in patch_ids], axis=0)
    if cx is None or cy is None:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return feats_grid[n // 2, n // 2]
        cy = float(ys.mean())
        cx = float(xs.mean())
    i = min(max(int(cy / scale), 0), n - 1)
    j = min(max(int(cx / scale), 0), n - 1)
    return feats_grid[i, j]


def _dedup_masks(raw):
    """同じ物を指す複数の点から出たマスクを1枚にまとめる（重複除け）。
    マスクのIoUが0.5以上、または重心の距離が10px以内なら重複とみなし、
    先に見つかった方だけを残す（[[仕様_見る側_道を1本にする_2026-09-09]]
    後半1節「同じマスクの重複除け」。旧・確認専用の重複除けヘルパーと同じ基準）。"""
    import math
    kept = []
    for r in raw:
        ys, xs = np.where(r["mask"])
        if len(ys) == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        dup = False
        for k in kept:
            inter = float(np.logical_and(k["mask"], r["mask"]).sum())
            union = float(np.logical_or(k["mask"], r["mask"]).sum())
            if union > 0 and (inter / union) >= 0.5:
                dup = True
                break
            if math.hypot(cx - k["_cx"], cy - k["_cy"]) <= 10.0:
                dup = True
                break
        if not dup:
            r2 = dict(r)
            r2["_cx"], r2["_cy"] = cx, cy
            kept.append(r2)
    for k in kept:
        k.pop("_cx", None)
        k.pop("_cy", None)
    return kept


def segment_at_points(predictor, img224, points, patch_feats, n, img_size=224,
                       min_area_frac=0.002, max_area_frac=0.5, blobs=None):
    """段2 分節：点プロンプトで指定した場所だけを切り出す（`predictor.set_image`
    1回＋点ごとに`predict`）。

    【なぜ、2026-09-09】仕様_見る側の段構成_実装_2026-09-09.md 3節。段1
    （前注意の地図）の「説明できない変化」や段6（優先度地図）の「カードの
    無い高優先度の場所」だけを頼まれた点として切り出す横取りの置き換え。
    ここは既存カードの確認ではなく**新しい候補**を作るための切り出しなので、
    対応づけ（どのfile_idか）は持たない。

    【2026-09-09・仕様_見る側_道を1本にする 後半1節】背景（端に届くマスク）は
    `_mask_touches_edge` で足切りする（中3。`_detect_mobilesam`と同じ基準）。
    同じマスクを指す複数の点（例：カードの予測位置と探索点が同じ物に当たった）
    は`_dedup_masks`で1枚にまとめる。

    【2026-09-09・追記1「直し」】SAMは1点につき部品/物/場面の3段階マスクを
    返し、これまでは常にSAM自身の自信（iou_pred）が最大のものを選んでいた。
    小さい物の上の点でも自信最大は場面ぜんたいのマスクであることが多く、
    それは端に届くので背景の足切りで捨てられ、結果を何も返さない事故が
    起きた（F2-107pre実測：n_dets平均0.52）。直しは「採るべき大きさの当て」
    を点ごとに渡すこと。`points` の各要素は `{"pos": (x, y),
    "expect_area": float|None}`。
      - `expect_area` があるとき：3段階のマスクのうち、端に届かないものの中で
        面積比が `expect_area` に最も近いものを選ぶ。ただし
        `0.5 <= 面積/expect_area <= 2.0` を満たすものが1枚も無ければ、
        その点は捨てる（`n_reject_scale` として数える）。
      - `expect_area` が None のとき：`min_area_frac`〜`max_area_frac` に
        入るもののうち面積が最大のものを選ぶ（従来の探索点の扱いに近い）。

    Args:
        predictor: `make_point_predictor()` で得た `SamPredictor`。
        img224: (224, 224, 3) の uint8 RGB画像。
        points: `[{"pos": (x, y), "expect_area": float|None}, ...]`
            切り出したい座標（img224の画素基準）と、そこにあるはずの
            大きさの当て（画面全体に対する面積比。無ければNone）。
        patch_feats, n: `patch_features()` の戻り値（見た目ベクトル計算用）。
        min_area_frac, max_area_frac: `expect_area` が None の点に使う
            採用面積比の範囲（範囲外は捨てる）。`expect_area` がある点には
            使わない（上記の比 0.5〜2.0 判定を使う）。
        blobs: 段1の`blobs`（各要素に"bbox"を持つ）。Noneなら凝集しない。
            渡すと`_cohesion_groups`で「同じblobに入る点から出たマスクは
            1つの物」としてまとめる（仕様3節「共通運動の凝集」）。
    Returns:
        `detect()`と同じ形の辞書のリスト（`mask, bbox, pos, area, appearance,
        emb`）。`emb`は常にNone（点プロンプトのMobileSAM埋め込みは計算しない
        ＝仕様に無かった判断。`ObjectFile.emb`は次の見回りで埋まる）。
        `cohesion_reason`は"motion"（同じblobでまとめられた）または""。
        `n_points_expect`・`n_reject_scale`（呼び出し側のCSV記録用）は検出
        リストにはぶら下げず、`(dets, stats)` のタプルの2つめとして返す
        （`stats = {"n_points_expect": int, "n_reject_scale": int}`）。
    """
    img224 = np.asarray(img224)
    h, w = img224.shape[0], img224.shape[1]
    feats_grid = patch_feats.reshape(n, n, -1)
    scale = img_size / n
    total_px = float(h * w)

    n_points_expect = 0
    n_reject_scale = 0
    # 【2026-09-11・計測】数えていない足切りが3つあった。どれで落ちているか
    #   分からないと直せない（F2-126 で「点は出したが検出0」が 228 コマ）。
    n_reject_edge = 0      # 3枚の候補が全部「画像の端に接している」で落ちた点の数
    n_reject_area = 0      # 当てが無い側で、面積が min/max の外だった点の数
    n_reject_dedup = 0     # 重複除けで消えた数

    predictor.set_image(img224)
    raw = []
    for p in points:
        x, y = p["pos"]
        expect_area = p.get("expect_area")
        masks, iou_preds, _ = predictor.predict(
            point_coords=np.array([[float(x), float(y)]], dtype=np.float64),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        # 【2026-09-09・追記1「直し」2】候補ごとに端接触を先に足切りし、
        # そのうえで当ての有無に応じて選び方を変える（規則は1つ：
        # 「当てがあれば面積が最も近いもの、無ければ許容範囲内で最大」）。
        candidates = []
        for i in range(masks.shape[0]):
            mask_i = masks[i].astype(bool)
            ys, xs = np.where(mask_i)
            if len(ys) == 0:
                continue
            bx0, bx1 = int(xs.min()), int(xs.max())
            by0, by1 = int(ys.min()), int(ys.max())
            bbox_i = (bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1)
            if _mask_touches_edge(bbox_i, w, h):
                continue
            area_i = float(mask_i.sum()) / total_px
            candidates.append({"mask": mask_i, "bbox": bbox_i, "area": area_i})
        if not candidates:
            n_reject_edge += 1
            continue

        if expect_area is not None:
            n_points_expect += 1
            best = None
            best_diff = None
            for c in candidates:
                if expect_area <= 0:
                    continue
                ratio = c["area"] / expect_area
                if ratio < 0.5 or ratio > 2.0:
                    continue
                diff = abs(c["area"] - expect_area)
                if best_diff is None or diff < best_diff:
                    best_diff, best = diff, c
            if best is None:
                n_reject_scale += 1
                continue
        else:
            best = None
            best_area = None
            for c in candidates:
                if c["area"] < min_area_frac or c["area"] > max_area_frac:
                    continue
                if best_area is None or c["area"] > best_area:
                    best_area, best = c["area"], c
            if best is None:
                n_reject_area += 1
                continue

        # 【2026-09-10・見る側3段目】この点から新しい記録を作ってよいか。
        #   注意が向いた点だけ True。カードの維持のために打つ点は False。
        #   キーが無ければ True＝従来どおり（既定不変）。
        raw.append({"mask": best["mask"], "bbox": best["bbox"],
                    "can_create": bool(p.get("can_create", True))})
    predictor.reset_image()

    stats = {"n_points_expect": n_points_expect, "n_reject_scale": n_reject_scale,
             "n_reject_edge": n_reject_edge, "n_reject_area": n_reject_area,
             "n_reject_dedup": 0}

    if not raw:
        return [], stats

    _n_before_dedup = len(raw)
    raw = _dedup_masks(raw)
    stats["n_reject_dedup"] = _n_before_dedup - len(raw)
    if not raw:
        return [], stats

    if blobs:
        groups = _cohesion_groups(raw, gap=0, blobs=blobs)
    else:
        groups = [[i] for i in range(len(raw))]

    out = []
    for group in groups:
        if len(group) == 1:
            mask = raw[group[0]]["mask"]
            bx, by, bw, bh = raw[group[0]]["bbox"]
            reason = ""
        else:
            mask = raw[group[0]]["mask"].copy()
            for gi in group[1:]:
                mask = mask | raw[gi]["mask"]
            ys, xs = np.where(mask)
            bx, bx1 = int(xs.min()), int(xs.max())
            by, by1 = int(ys.min()), int(ys.max())
            bw, bh = bx1 - bx + 1, by1 - by + 1
            reason = "motion"
        can_create = any(bool(raw[gi].get("can_create", True)) for gi in group)
        area = float(mask.sum()) / (h * w)
        if area < min_area_frac or area > max_area_frac:
            continue
        cx = float(bx) + float(bw) / 2.0
        cy = float(by) + float(bh) / 2.0
        appearance = _mask_appearance(mask, feats_grid, n, scale)
        out.append({"mask": mask, "bbox": (bx, by, bw, bh), "pos": (cx, cy),
                     "area": area, "appearance": appearance, "emb": None,
                     "cohesion_reason": reason, "can_create": can_create})
    return out, stats


def _detect_mobilesam(patch_feats, n, img, mask_generator, img_size=224, min_area_frac=0.01,
                       cohesion_gap_px=None, point_predictor=None, blobs=None):
    """`cohesion_gap_px`（【2026-09-09・凝集性】仕様_物体ファイルの人間寄せ_
    凝集性・連続性・上限）：Noneなら従来どおり（マスク1枚＝検出1件、既定不変）。
    整数を渡すと、端接触・面積の足切りを終えたマスクの bbox を `cohesion_gap_px`
    画素だけ広げた箱同士が重なるものを1つの物としてまとめる（`_cohesion_groups`）。
    連結成分が1つだけのマスクは従来の計算式（bbox中心・m["area"]）のまま。
    まとめる場合だけ、segmentationの論理和を取り、重心・画素数から
    pos/area を計算し直す（仕様「後半」1節）。

    `point_predictor`（【2026-09-09・予測して確かめる検出「追記3」1節】）：
    Noneなら従来どおり（`emb`キーを一切付けない＝既定不変・追加コストも無し）。
    `SamPredictor` を渡すと、`mask_generator.generate()`（自動生成器が内部で
    別インスタンスの `set_image`→`generate`→`reset_image` を行う）とは別に、
    ここで自前の `point_predictor.set_image(img)` を1回呼んで埋め込みを取る。
    【なぜこの形にしたか】斥候報告（2026-09-09）どおり、自動生成器の内部
    `predictor` は `generate()` の最後に `reset_image()` されるため、外から
    その埋め込みを横取りすることはできない。自前の `SamPredictor` でもう1回
    `set_image` する＝エンコーダのforwardが1回余分にかかる
    （実測を報告に書く）。confirm_emb_cos を使わない走行（既定）では
    `point_predictor=None` のまま呼ばれるので、この追加コストは一切発生しない。"""
    img = np.asarray(img)
    h, w = img.shape[0], img.shape[1]
    emb_np = None
    if point_predictor is not None:
        point_predictor.set_image(img)
        emb_np = point_predictor.get_image_embedding().detach().cpu().numpy()[0]   # (256, eh, ew)
        point_predictor.reset_image()
    raw_masks = mask_generator.generate(img)
    feats_grid = patch_feats.reshape(n, n, -1)
    scale = img_size / n

    # 端に接するマスク・面積が小さすぎるマスクの足切りは従来どおり先に済ませる
    # （凝集性でまとめる対象は、そもそも検出として有効なマスクだけにする）。
    kept = []
    for m in raw_masks:
        if _mask_touches_edge(m["bbox"], w, h):
            continue
        if m["area"] / (h * w) < min_area_frac:
            continue
        kept.append(m)

    if cohesion_gap_px is not None and len(kept) > 1:
        groups = _cohesion_groups(kept, int(cohesion_gap_px), blobs=blobs)
    else:
        groups = [[i] for i in range(len(kept))]

    out = []
    for group in groups:
        if len(group) == 1:
            # 従来どおり（マスク1枚だけなら挙動を一切変えない）
            m = kept[group[0]]
            area = m["area"] / (h * w)
            seg = m["segmentation"]
            bx, by, bw, bh = m["bbox"]
            cx = float(bx) + float(bw) / 2.0
            cy = float(by) + float(bh) / 2.0
        else:
            # 凝集性：複数マスクを論理和でまとめ、pos＝union重心、
            # area＝unionの画素数/(h*w)（従来のareaと同じ尺度）で計算し直す。
            seg = kept[group[0]]["segmentation"].copy()
            for gi in group[1:]:
                seg = seg | kept[gi]["segmentation"]
            ys, xs = np.where(seg)
            cx = float(xs.mean())
            cy = float(ys.mean())
            area = float(seg.sum()) / (h * w)
        # マスクの範囲に半分以上重なるDINOv2のマスだけを、この物体の見た目とする（従来どおり）。
        # 【2026-09-09・仕様_見る側の段構成】segment_at_pointsと共用するため
        #   _mask_appearance に切り出した（新しい計算は作らない・計算式は不変）。
        v = _mask_appearance(seg, feats_grid, n, scale, cx=cx, cy=cy)
        det = {"pos": (cx, cy), "area": area, "appearance": v}
        if emb_np is not None:
            # 【追記3「直し」1節】旧・確認専用の埋め込み計算と同じやり方（マスクを
            # 埋め込み格子(eh,ew)へ最近傍で縮め、その領域の平均）。point_predictorが
            # Noneのときはこのブロックごと実行されない＝既定不変。
            from PIL import Image
            eh, ew = emb_np.shape[1], emb_np.shape[2]
            small = np.array(
                Image.fromarray(seg.astype(np.uint8) * 255).resize((ew, eh), Image.NEAREST))
            region = small > 0
            if region.any():
                det["emb"] = emb_np[:, region].mean(axis=1)
            else:
                gy = min(max(int(cy / h * eh), 0), eh - 1)
                gx = min(max(int(cx / w * ew), 0), ew - 1)
                det["emb"] = emb_np[:, gy, gx]
        out.append(det)
    return out
