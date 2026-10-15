"""荷札を作る道具（段階2の仮部品が共有する）。

* 見た目→荷札：目の入口（eyein）と、意味の辞書の登録道具（tools/register_appearance.py）が
  **同じ行列**を使わないと、登録した荷札と見えている荷札が比べられなくなる。
  だから行列はここ1か所で、種を固定して作る（学習しない。段階3でNNに替える対象）
* 語→荷札：聞き分ける係（listen）が使う。語ごとに種を決めた乱数ベクトル
* 雑音：仮部品の荷札はきれいすぎて、NNに替えたときに初めて問題が出る（設計書 リスク1）。
  semantic と eyein が出す荷札に N(0, NOISE_SIGMA²) を足して正規化し直す
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from .formats import TAG_DIM, make_tag

APPEARANCE_DIM = 384          # DINOv2 ViT-S/14 の特徴の長さ（旧太郎の物体ファイルと同じ）
PROJECTION_SEED = 20261015    # 変えると登録済みの意味の辞書が使えなくなる
NOISE_SIGMA = 0.05

_proj = None


def projection() -> np.ndarray:
    """384→64 の固定の乱数行列。"""
    global _proj
    if _proj is None:
        g = np.random.default_rng(PROJECTION_SEED)
        _proj = g.standard_normal((APPEARANCE_DIM, TAG_DIM)) / np.sqrt(TAG_DIM)
    return _proj


def appearance_tag(appearance) -> torch.Tensor:
    """物の見た目（384）を荷札（64、L2正規化）へ縮める。雑音は足さない。"""
    a = np.asarray(appearance, dtype=np.float64).reshape(APPEARANCE_DIM)
    return make_tag((a @ projection()).astype(np.float32))


def word_tag(word: str) -> torch.Tensor:
    """語ごとに決まった乱数の荷札（同じ語ならいつ作っても同じ）。"""
    seed = int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")
    g = np.random.default_rng(seed)
    return make_tag(g.standard_normal(TAG_DIM).astype(np.float32))


def noisy(tag: torch.Tensor, sigma: float = NOISE_SIGMA) -> torch.Tensor:
    """荷札に雑音を足して正規化し直す。ゼロの荷札（何も無い）はゼロのまま。
    乱数は torch の大域の乱数（run_v2 が試行ごとに種を撒く）。"""
    if float(torch.linalg.vector_norm(tag)) == 0.0 or sigma <= 0:
        return tag
    return make_tag(tag + sigma * torch.randn(TAG_DIM))
