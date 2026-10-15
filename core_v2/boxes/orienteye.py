"""見えた方へ目を向ける反射（1段目 反射、上丘の役）。

段階2・工程3の仮部品（仕様書 §3）：合わせた地図（prio）の最大のマスが、今の注視点
（地図は網膜の座標なので真ん中 13.5,13.5）から MIN_DIST_CELLS マス以上離れていて、前回の
発射から REFRACTORY_TICKS 回（1.2秒）以上たっていれば、そのマスの方向
((x−13.5)/13.5, (13.5−y)/13.5)（[-1,1]、右・上が正）を目の出口へ渡す。撃たない回はゼロ。

抑える強さ（門番から）が 0.5 を超えるときは、目立ちだけの山には撃たない。
**仕様書からの変更（2026-10-15）**：仕様書は「探し物らしさ<0.2 の山へは撃たない」だが、
この係に届くのは合わせた地図だけで、探し物らしさは別には届かない。合わせた地図の値 m と
w=0.9×抑える強さ から、探し物らしさの下限 (m−(1−w))/w（目立ちが最大の1だった場合）を出し、
それが 0.2 未満なら撃たない。目立ちだけの山は m ≤ 1−w なので必ず止まる。

**段階3（2026-10-15）：復帰抑制（一度見た所はしばらく見ない）を足した。**生まれつきの部品
（新生児からある。Valenza 1994 ほか、旧太郎 `doc/文献調査/2026-08-19_乳児のIORと馴化.md`）
なので手で決め、学習しない。旧太郎の上丘の版（`taro_core/src/brain/midbrain/orienting.py:657-705`、
既定OFFだった）と同じ形：
* 「もう見た地図」self.ior（28×28）を持つ。毎回 exp(−0.1/IOR_DECAY_S) 倍に薄れる
* どこを見るか決める前に、合わせた地図から IOR_GAIN×self.ior を引く（負は0）
* 撃ったら、地図を目が動く分だけ逆向きにずらし（地図は網膜の座標。目を動かす指令の写しで
  ずらす）、新しい注視点＝真ん中に高さ1・σ=IOR_SIGMA_CELLS の山を足す（1で頭打ち）
use_ior=False で段階2と1ビットも変わらない。
"""

import math

import torch

from ..formats import GRID, DTYPE
from .base import Box

MIN_DIST_CELLS = 2.0
REFRACTORY_TICKS = 12          # 1.2秒（脳の1回＝0.1秒）
INHIBIT_ON = 0.5
MIN_SEEK = 0.2
W_GOAL = 0.9
CENTER = (GRID - 1) / 2.0      # 13.5
TICK_S = 0.1

# 復帰抑制（段階3・仕様書 付録A）
IOR_DECAY_S = 3.0        # 旧太郎 orienting.py:703（成人の約3秒を出発点）[Tier2〜3]
IOR_SIGMA_CELLS = 2.0    # 旧太郎 spatial_priority_map.py の「注意の半径の半分」≒2.3マス［仮置き］
IOR_GAIN = 1.0           # 旧太郎と同じ［仮置き］


def _center_bump(sigma: float = IOR_SIGMA_CELLS) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(GRID, dtype=DTYPE), torch.arange(GRID, dtype=DTYPE),
                            indexing="ij")
    return torch.exp(-((xs - CENTER) ** 2 + (ys - CENTER) ** 2) / (2.0 * sigma ** 2))


def shift_map(m: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    """地図を下へ dr・右へ dc マスずらす。はみ出た所は捨て、空いた所はゼロ。"""
    out = torch.zeros_like(m)
    r0, r1 = max(dr, 0), GRID + min(dr, 0)
    c0, c1 = max(dc, 0), GRID + min(dc, 0)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = m[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


class Orienteye(Box):
    BOX_ID = "orienteye"

    def __init__(self, use_ior: bool = True) -> None:
        super().__init__()
        self.use_ior = bool(use_ior)
        self._bump = _center_bump()
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.last_fire = -10 ** 9
        self.last_reason = ""     # 記録用
        self.ior = torch.zeros(GRID, GRID, dtype=DTYPE)
        self.last_ior_at_peak = 0.0   # 記録用：選んだマス（引く前の最大）に乗っていた印

    def _mark_fire(self, r: int, c: int) -> None:
        """撃った回：目が (r, c) へ動くので、地図を逆向きにずらして、真ん中に印を足す。"""
        dr, dc = int(round(CENTER - r)), int(round(CENTER - c))
        self.ior = (shift_map(self.ior, dr, dc) + self._bump).clamp(max=1.0)

    def step(self, inbox):
        out = self.passthrough()
        self.t += 1
        m = inbox.get("prio")
        inhibit = float(inbox.get("gate3", 0.0))
        if self.use_ior:
            self.ior = self.ior * math.exp(-TICK_S / IOR_DECAY_S)
            if m is not None:
                raw_peak = int(torch.argmax(m))
                self.last_ior_at_peak = float(self.ior.reshape(-1)[raw_peak])
                m = (m - IOR_GAIN * self.ior).clamp(min=0.0)
        if m is None or float(m.max()) <= 0.0:
            self.last_reason = "empty"
            return out
        r, c = divmod(int(torch.argmax(m)), GRID)
        peak = float(m[r, c])
        if math.hypot(c - CENTER, r - CENTER) < MIN_DIST_CELLS:
            self.last_reason = "centered"
            return out
        if self.t - self.last_fire < REFRACTORY_TICKS:
            self.last_reason = "refractory"
            return out
        if inhibit > INHIBIT_ON:
            w = W_GOAL * inhibit
            if (peak - (1.0 - w)) / w < MIN_SEEK:
                self.last_reason = "inhibited"
                return out
        self.last_fire = self.t
        self.last_reason = "fire"
        if self.use_ior:
            self._mark_fire(r, c)
        out["eyeout"] = torch.tensor([(c - CENTER) / CENTER, (CENTER - r) / CENTER], dtype=DTYPE)
        return out
