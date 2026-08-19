"""F1-3b「視覚バックエンドの差し替え機構」の単体テスト
（F/docs/仕様_F1-3b_視覚バックエンドの差し替え機構.md 技術付録「新設2」の3項目）。

1. 既定（lexicon_vision=None→customバックエンド）で従来経路とbit-identical
   （同一obsに対して同一state）
2. dinov2バックエンド：F/logs/F0_視覚弁別プローブ/の10枚で箱vs球の分離比
   （条件間距離÷条件内距離）>10倍を自動判定
3. フォビア切り出しの寸法・中央位置の検証

taro_core側の新設ファイル（vision_backends.py）と、既存のVisionEncoder（読むだけ・
比較用に呼ぶだけで変更しない）を使う。
"""
import glob
import os
import sys

import numpy as np
import torch

_BRIDGE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_BRIDGE, "src", "senses"))

from vision_backends import (  # noqa: E402
    CustomVisionBackend, DINOv2VisionBackend, fovea_crop, get_backend,
)
from vision_encoder import VisionEncoder  # noqa: E402

_ROOT = os.path.abspath(os.path.join(_BRIDGE, ".."))
_PROBE_DIR = os.path.join(_ROOT, "F", "logs", "F0_視覚弁別プローブ")


def _random_image(rng, size=128):
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


# ============================================================================
# 1. 既定（custom）＝従来経路とbit-identical
# ============================================================================
def test_1_custom_backend_bit_identical():
    print("=" * 60)
    print("1. 既定（custom）バックエンドがfusion.vision直呼びとbit-identicalか")
    print("=" * 60)
    torch.manual_seed(0)
    enc = VisionEncoder(embedding_dim=64, image_size=128)

    rng = np.random.default_rng(0)
    img_left = _random_image(rng)
    img_right = _random_image(rng)

    # 従来経路（F1-3実装当初のrun/trainer.pyのコード：直接呼んでdetach().cpu().tolist()）
    old_state = enc(img_left, img_right).detach().cpu().tolist()

    # 新経路：get_backend(None, ...) → customバックエンド → encode().tolist()
    backend = get_backend(None, vision_encoder=enc)
    assert backend.name == "custom"
    assert backend.dim == 64
    new_state = backend.encode(img_left, img_right).tolist()

    print(f"従来経路の先頭5要素: {old_state[:5]}")
    print(f"新経路の先頭5要素  : {new_state[:5]}")
    assert old_state == new_state, "customバックエンド経由の出力が従来経路と一致しない"
    print(f"全{len(old_state)}要素が完全一致（bit-identical） PASS")


def test_1b_custom_backend_requires_vision_encoder():
    print("=" * 60)
    print("1b. customバックエンドはvision_encoderが必須（無いとValueError）")
    print("=" * 60)
    try:
        CustomVisionBackend(vision_encoder=None)
        raised = False
    except ValueError:
        raised = True
    assert raised, "vision_encoder無しでもエラーにならなかった"
    print("PASS")


# ============================================================================
# 2. dinov2バックエンド：箱vs球の分離比 > 10倍
# ============================================================================
def _load_probe_images():
    boxes = sorted(glob.glob(os.path.join(_PROBE_DIR, "赤い箱_*.png")))
    balls = sorted(glob.glob(os.path.join(_PROBE_DIR, "白い球_*.png")))
    assert len(boxes) == 5 and len(balls) == 5, (
        f"F0プローブ画像が想定と違う（箱{len(boxes)}枚・球{len(balls)}枚、各5枚のはず）"
        f": dir={_PROBE_DIR}")
    from PIL import Image

    def _load(paths):
        return [np.asarray(Image.open(p).convert("RGB")) for p in paths]

    return _load(boxes), _load(balls)


def _cosine_distance(a, b):
    """1 - コサイン類似度。プロジェクト全体の連合分離指標
    （run/plugins/common/word_learning.py の assoc_sep）と同じ、L2正規化済みベクトル
    向けの距離。DINOv2backendの出力は仕様どおりL2正規化済みなので、これに揃える。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def test_2_dinov2_box_vs_ball_separation():
    print("=" * 60)
    print("2. dinov2バックエンド：箱vs球の分離比（条件間÷条件内）> 10倍")
    print("=" * 60)
    box_imgs, ball_imgs = _load_probe_images()

    # fovea_px既定（None＝画像辺の1/2＝128px画像なら64）で構築。
    backend = DINOv2VisionBackend()
    assert backend.dim == 384

    # 単眼キャプチャなので両目に同じ画像を渡す。
    box_feats = [backend.encode(im, im) for im in box_imgs]
    ball_feats = [backend.encode(im, im) for im in ball_imgs]

    def _pairwise_mean_dist(feats_a, feats_b, exclude_self=False):
        dists = []
        for i, a in enumerate(feats_a):
            for j, b in enumerate(feats_b):
                if exclude_self and i == j:
                    continue
                dists.append(_cosine_distance(a, b))
        return sum(dists) / len(dists)

    intra_box = _pairwise_mean_dist(box_feats, box_feats, exclude_self=True)
    intra_ball = _pairwise_mean_dist(ball_feats, ball_feats, exclude_self=True)
    intra = (intra_box + intra_ball) / 2.0
    inter = _pairwise_mean_dist(box_feats, ball_feats)

    ratio = inter / intra if intra > 1e-12 else float("inf")
    print(f"条件内平均距離（箱内={intra_box:.6f} 球内={intra_ball:.6f} 平均={intra:.6f}）")
    print(f"条件間平均距離（箱↔球={inter:.6f}）")
    print(f"分離比 inter/intra = {ratio:.3f}")
    assert ratio > 10.0, f"分離比が不十分: {ratio}"
    print("PASS")


# ============================================================================
# 3. フォビア切り出しの寸法・中央位置の検証
# ============================================================================
def test_3_fovea_crop_shape_and_center():
    print("=" * 60)
    print("3. フォビア切り出しの寸法・中央位置の検証")
    print("=" * 60)
    # 既定（fovea_px=None）＝画像の辺の1/2。128px画像なら中央64×64。
    img = np.arange(128 * 128 * 3, dtype=np.int64).reshape(128, 128, 3)
    cropped = fovea_crop(img, fovea_px=None)
    print(f"128px画像・既定fovea_px: 出力shape={cropped.shape}")
    assert cropped.shape == (64, 64, 3), f"既定フォビアの寸法が違う: {cropped.shape}"
    expected = img[32:96, 32:96]
    assert np.array_equal(cropped, expected), "既定フォビアの中央位置がずれている"
    print("既定（辺の1/2）: 寸法(64,64,3)・中央位置一致 PASS")

    # 明示的にfovea_pxを指定した場合
    cropped32 = fovea_crop(img, fovea_px=32)
    assert cropped32.shape == (32, 32, 3), f"fovea_px=32の寸法が違う: {cropped32.shape}"
    expected32 = img[48:80, 48:80]
    assert np.array_equal(cropped32, expected32), "fovea_px=32の中央位置がずれている"
    print("明示指定（32）: 寸法(32,32,3)・中央位置一致 PASS")

    # 画像よりフォビアが大きい場合は画像そのものにクランプされる
    small_img = np.zeros((20, 20, 3), dtype=np.uint8)
    cropped_clamped = fovea_crop(small_img, fovea_px=64)
    assert cropped_clamped.shape == (20, 20, 3), (
        f"画像よりフォビアが大きいときのクランプが効いていない: {cropped_clamped.shape}")
    print("画像よりフォビアが大きい場合のクランプ PASS")


def main():
    test_1_custom_backend_bit_identical()
    test_1b_custom_backend_requires_vision_encoder()
    test_3_fovea_crop_shape_and_center()
    test_2_dinov2_box_vs_ball_separation()
    print("=" * 60)
    print("全項目 PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
