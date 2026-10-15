"""目の入口（入口（感覚））。

段階2・工程2：旧太郎の物の切り出し（MobileSAM）と見た目（DINOv2）、目立ちの地図
（SalienceMap）を借りて、書式に包む。
* eyein→gaze  : 物ごとの (荷札, 地図)。荷札＝見た目384を固定の乱数行列で64へ縮めて
                L2正規化し雑音を足したもの（tagging.py）。地図＝物の中心にガウスの山
* eyein→prio  : 目立ちの地図（28×28、0〜1）
* それ以外    : 素通し（ゼロ）

**仕様書 §2-2 からの変更（2026-10-15、実測による）**：仕様書は旧太郎の「物体ファイル」
（result["files"]、毎コマ追跡して名簿を持つ仕組み）を読むことにしていたが、
① 旧太郎の設定（cohesion_gap_px=4）では3つの物と机のマスが1つにまとめられ、物が
  1つも取れなかった（3つの物の真ん中に1枚だけ）
② まとめを切っても、追跡は動く妨害物に2枚目3枚目の名簿を作り、見失った名簿が
  予測位置を漂い続けた（misses 3〜7）
ので、追跡は使わず**毎回その場で切り出す**（旧太郎の最初のコマと同じ全体切り出し）。
こうすると「今見えている物」だけが出る。20コマで3つの物は毎コマ取れた（points_per_side=12）。
机のマスなども物として出るので、物が MAX_OBJECTS（8）を超えたら目立ちの高い順に残す。

環境からの入力（inbox[ENV]）：左目の画像（224×224×3、uint8）。
setup() を呼ぶまでは素通し（骨組みのテストは重い視覚を作らずに回る）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from ..formats import GRID, MAX_OBJECTS, DTYPE
from ..tagging import appearance_tag, noisy
from .base import ENV, Box

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

# 旧太郎の _門2やり直し_2026-09-16 の設定から、物の切り出しと目立ちに要る所だけを残した。
# cohesion_gap_px は切る（上の①）。points_per_side=12：8だと 224ms・12だと 402ms・16だと
# 650ms（1コマ、RTX 3060 Ti、20コマの中央値）。3つとも全部のコマで取れた
DEFAULT_CONFIG = {
    "points_per_side": 12,
    "points_per_batch": 32,
    "empty_cache_after_detect": False,
    "interval_s": 0.09,
    "cohesion_gap_px": None,
    "attention": {"cell": GRID, "ior_tau_s": 0.7, "ior_gain": 1.0, "acc_tau_s": 0.11,
                  "dog_iters": 10},
    "gaze_from_attention": False,   # 目を動かすのは orienteye 箱だけ（仕様書 §2-2）
}
BUMP_SIGMA_CELLS = 1.0


def bump(cx_px: float, cy_px: float, sigma: float = BUMP_SIGMA_CELLS) -> torch.Tensor:
    """224画素の座標 (cx, cy) に、高さ1のガウスの山を置いた 28×28 の地図。"""
    step = 224.0 / GRID
    gx, gy = cx_px / step - 0.5, cy_px / step - 0.5     # マス c の中心は 8c+4 画素
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    m = np.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2.0 * sigma ** 2))
    return torch.as_tensor(m, dtype=DTYPE)


class Eyein(Box):
    BOX_ID = "eyein"

    def __init__(self) -> None:
        super().__init__()
        self.va = None
        self.last_dets: list[dict] = []   # 記録用（その回の切り出し結果）

    def setup(self, config: dict | None = None) -> None:
        """旧太郎の VisualAttention を、Taro を作らずに直接作る（DINOv2・MobileSAM を読む）。"""
        core_src = os.path.join(_REPO, "taro_core", "src")
        if core_src not in sys.path:
            sys.path.insert(0, core_src)
        from brain.cerebral_cortex.parietal_lobe.visual_attention import VisualAttention
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        self.va = VisualAttention(config=cfg)
        self.cohesion_gap_px = cfg.get("cohesion_gap_px")

    def reset(self) -> None:
        if self.va is not None and self.va._sal is not None:
            # 目立ちの地図は前のコマとの差（動き）を持つので、試行の頭で前のコマを捨てる
            self.va._sal._prev_gray = None
        self.last_dets = []

    def see(self, img) -> tuple[list, torch.Tensor]:
        va = self.va
        img224 = np.asarray(img)
        if img224.shape[:2] != (224, 224):
            from PIL import Image
            img224 = np.array(Image.fromarray(np.clip(img224, 0, 255).astype(np.uint8))
                              .resize((224, 224)))
        img224 = np.clip(img224, 0, 255).astype(np.uint8)
        sal = va._sal.update(img224, (0.0, 0.0), False)["salience"]
        sal = np.clip(np.nan_to_num(np.asarray(sal, dtype=np.float32)), 0.0, 1.0)
        p, n = va._patch_features(va._model, img224, va.device)
        dets = va._detect(p, n, img=img224, mask_generator=va._mgen,
                          cohesion_gap_px=self.cohesion_gap_px)
        step = 224.0 / GRID
        for d in dets:
            c = int(min(max(d["pos"][0] // step, 0), GRID - 1))
            r = int(min(max(d["pos"][1] // step, 0), GRID - 1))
            d["sal"] = float(sal[r, c])
        dets = sorted(dets, key=lambda d: -d["sal"])[:MAX_OBJECTS]
        self.last_dets = dets
        objs = [(noisy(appearance_tag(d["appearance"])), bump(*d["pos"])) for d in dets]
        return objs, torch.as_tensor(sal, dtype=DTYPE)

    def step(self, inbox):
        out = self.passthrough()
        if self.va is None or inbox.get(ENV) is None:
            return out
        objs, sal = self.see(inbox[ENV])
        out["gaze"] = objs
        out["prio"] = sal
        return out
