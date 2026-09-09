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


def detect(patch_feats, n, img_size=224, thresh=0.55, min_cells=1, img=None, mask_generator=None,
           cohesion_gap_px=None, point_predictor=None):
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
    """
    if mask_generator is not None:
        if img is None:
            raise ValueError("mask_generator を使うときは img（生のRGB画像）も渡す必要がある")
        return _detect_mobilesam(patch_feats, n, img, mask_generator, img_size,
                                  cohesion_gap_px=cohesion_gap_px, point_predictor=point_predictor)

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
    Apache License 2.0（`F/docs/物体ファイルと注意/文献調査/2026-09-04c_mobilesam_ライセンス.md`
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
    """点プロンプトによる確認（confirm）用の `SamPredictor` を返す。

    【なぜ、2026-09-09】仕様_予測して確かめる検出_2026-09-09「後半」1節。
    斥候報告（2026-09-09）で確認済み：点プロンプト `SamPredictor` は自動生成
    （`SamAutomaticMaskGenerator`）と同じ重み（`mask_generator.predictor.model`）で
    使える別インスタンス。1回だけ作って呼び出し元で使い回すこと（読み込みは軽いが、
    毎回作り直す必要は無い）。"""
    from mobile_sam import SamPredictor
    return SamPredictor(mask_generator.predictor.model)


def confirm_points(predictor, img224, points, expected_areas, file_ids=None,
                    iou_thresh=0.7, area_tol=0.5, card_embs=None, emb_cos_thresh=None):
    """点プロンプトで「さっきの場所に、さっきの大きさの物がまだあるか」を確認する。

    【なぜ、2026-09-09】仕様_予測して確かめる検出_2026-09-09「後半」1節。
    画面全体を256か所走査する重い `SamAutomaticMaskGenerator.generate()` の
    代わりに、いま追跡中のカードの数（4枚以内）だけ点で問い合わせる軽い経路。
    画像の埋め込み計算（`set_image`）は1回だけ、`predict()` は点の数だけ呼ぶ。

    【なぜ file_id・mask を返すか、2026-09-09・追記2】F2-101pre2/F2-102pre の
    実測（`確認・見回り・横取り` の絵）で、確認コマの結果が呼び出し元で
    どのカードの問い合わせだったか分からなくなり、`ofs.step()` に渡すと
    ハンガリアン法が結果を別のカードに結び直してしまい、対応しなかった
    カードが新規カードとして再作成される事故が起きた（追記2「絵」参照）。
    各結果に頼んだ側の `file_id` を付けて返し、呼び出し元はそのカードにしか
    結ばない（`ObjectFileSystem.confirm_update`）。`mask` も返すのは、
    同じ物を複数のカードが確認したときの重複判定（マスクのIoU）に使うため。

    【なぜ card_embs・emb_cos_thresh・emb を返すか、2026-09-09・追記3】
    F2-103pre の実測（追記3「何が起きたか」）：点プロンプトは何も無い場所でも
    机の影・縁のマスクを返し、面積さえ近ければ「確認できた」になってしまう
    （偽の確認）。人間側は同一性の判断に位置だけでなく見た目も使う
    （Xu & Carey 1996、12か月で特徴も使う[Tier1]）。`get_image_embedding()`は
    `set_image()`の時点で計算済みのテンソルを返すだけで追加のforwardは無い
    （斥候報告2026-09-09で確認済み＝追加コスト零）。採用候補マスクの領域だけ
    埋め込み格子（64x64）上で平均した256次元ベクトルを`emb`として返し、
    `card_embs`（呼び出し元がカードごとに持つ`ObjectFile.emb`）が渡されていれば
    コサイン類似度`emb_cos`を計算し、`emb_cos_thresh`未満なら不採用にする。
    `card_embs`がNone、または個々のカードのembがNone（emb_cos_threshが
    Noneのとき・古いカード）のときは従来どおり照合しない（既定不変）。

    Args:
        predictor: `make_point_predictor()` で得た `SamPredictor`。
        img224: (224, 224, 3) の uint8 RGB画像。
        points: [(x, y), ...] 確認したい座標（img224の画素基準）。
        expected_areas: 各pointに対応する期待面積比[0-1]のリスト（pointsと同じ長さ）。
        file_ids: 各pointに対応する物体ファイルid（pointsと同じ長さ）。Noneなら
            結果の "file_id" は全てNone（後方互換：呼ばなくても動く）。
        iou_thresh: `predict()` が返す `iou_predictions` の下限。これ未満の候補マスクは
            採用対象から外す。
        area_tol: 採用する面積の許容比率（|area-expected|/expected <= area_tol）。
        card_embs: 各pointに対応するカードの見た目埋め込み（256次元 or None）の
            リスト（pointsと同じ長さ）。Noneなら全点で照合しない（既定不変）。
        emb_cos_thresh: 採用条件に足す `cos(emb, card_emb)` の下限。Noneなら
            照合しない（既定不変）。
    Returns:
        pointsと同じ長さ・同じ順のリスト。iou_thresh・area_tolの両方を満たす
        候補が見つからなければ {"ok": False, "file_id": 対応するfile_id}。
        見つかれば（emb_cos_threshで最終的に落ちても）
        {"ok": bool, "file_id": fid, "iou": float, "area_ratio": float
         （採用候補の面積÷期待面積）, "emb_cos": float（card_embが無ければ無し）}
        に加え、ok=Trueのときだけ
        {"pos": マスク重心(x,y), "area": 面積比, "mask": bool配列(H,W)}
        を足す（失敗の場合もiou/area_ratio/emb_cosは書く＝「何で落ちたか」が
        分かるように。追記3「直し」2節）。appearanceは返さない（呼び出し元が
        カードのものをそのまま使う）。
    """
    import numpy as np
    from PIL import Image
    img224 = np.asarray(img224)
    h, w = img224.shape[0], img224.shape[1]
    if file_ids is None:
        file_ids = [None] * len(points)
    if card_embs is None:
        card_embs = [None] * len(points)
    predictor.set_image(img224)
    emb_np = None
    if emb_cos_thresh is not None:
        # 【追記3「直し」1節】emb_cos_threshが指定されているときだけ埋め込みを
        # 読む（get_image_embedding自体は追加コスト零だが、Noneのとき・
        # 従来呼び出しでは一切触らない＝既定不変）。
        image_embedding = predictor.get_image_embedding()
        emb_np = image_embedding.detach().cpu().numpy()[0]   # (256, eh, ew)
    eh = int(emb_np.shape[1]) if emb_np is not None else None
    ew = int(emb_np.shape[2]) if emb_np is not None else None

    out = []
    for (x, y), expected_area, fid, card_emb in zip(points, expected_areas, file_ids, card_embs):
        masks, iou_preds, _ = predictor.predict(
            point_coords=np.array([[float(x), float(y)]], dtype=np.float64),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best = None   # (面積差, マスクindex, 面積)
        for i in range(masks.shape[0]):
            if float(iou_preds[i]) < iou_thresh:
                continue
            area = float(masks[i].sum()) / (h * w)
            if expected_area is None or expected_area <= 0:
                continue
            if abs(area - expected_area) / expected_area > area_tol:
                continue
            diff = abs(area - expected_area)
            if best is None or diff < best[0]:
                best = (diff, i, area)
        if best is None:
            out.append({"ok": False, "file_id": fid})
            continue
        _diff, i, area = best
        mask = masks[i]
        ys, xs = np.where(mask)
        if len(ys) == 0:
            out.append({"ok": False, "file_id": fid})
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        area_ratio = area / expected_area

        emb = None
        emb_cos = None
        ok = True
        if emb_np is not None:
            small = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize((ew, eh), Image.NEAREST))
            region = small > 0
            if region.any():
                emb = emb_np[:, region].mean(axis=1)
            else:
                # マスクが埋め込み格子で1マスにも満たない極小マスク→中心1マスで代用。
                gy = min(max(int(cy / h * eh), 0), eh - 1)
                gx = min(max(int(cx / w * ew), 0), ew - 1)
                emb = emb_np[:, gy, gx]
            if card_emb is not None:
                a = emb / (np.linalg.norm(emb) + 1e-9)
                b = np.asarray(card_emb, dtype=np.float64)
                b = b / (np.linalg.norm(b) + 1e-9)
                emb_cos = float(a @ b)
                ok = emb_cos >= emb_cos_thresh

        result = {"ok": ok, "file_id": fid, "iou": float(iou_preds[i]), "area_ratio": area_ratio}
        if emb_cos is not None:
            result["emb_cos"] = emb_cos
        if ok:
            result["pos"] = (cx, cy)
            result["area"] = area
            result["mask"] = mask.astype(bool)
            if emb is not None:
                result["emb"] = emb
        out.append(result)
    predictor.reset_image()
    return out


def _mask_touches_edge(bbox, w, h, margin=1):
    """背景（壁・床など）は画像の端まで届くことが多く、物体は中央寄りになる
    ことを使った足切り。F2-64実測：物なし画面のマスク3枚は全て端に接触、
    物あり画面ではコップ本体だけが端に触れなかった（詳細は上記統合まとめ）。"""
    x, y, bw, bh = bbox
    return x <= margin or y <= margin or (x + bw) >= (w - margin) or (y + bh) >= (h - margin)


def _cohesion_groups(masks, gap):
    """`masks`（各要素に "bbox" を持つ辞書のリスト）の bbox を `gap` 画素だけ
    広げた箱同士が重なるものを union-find で連結成分にまとめ、インデックスの
    グループのリストを返す（1個だけの物は要素1個のグループのまま）。

    【なぜ、2026-09-09・凝集性】Spelke の凝集性の原理（乳児は連結した面の
    かたまりを1つの物として扱う[Tier1]）を、検出器が出したマスク同士の
    近さ（隙間が`gap`画素以内）で近似する[Tier3・工学的近似：本当の面の
    連結ではなく矩形の近さで代用]。"""
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
    groups = {}
    for i in range(m):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def _detect_mobilesam(patch_feats, n, img, mask_generator, img_size=224, min_area_frac=0.01,
                       cohesion_gap_px=None, point_predictor=None):
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
        groups = _cohesion_groups(kept, int(cohesion_gap_px))
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
        # マスクの範囲に半分以上重なるDINOv2のマスだけを、この物体の見た目とする（従来どおり）
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
        det = {"pos": (cx, cy), "area": area, "appearance": v}
        if emb_np is not None:
            # 【追記3「直し」1節】confirm_points と同じやり方（マスクを埋め込み
            # 格子(eh,ew)へ最近傍で縮め、その領域の平均）。point_predictorが
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
