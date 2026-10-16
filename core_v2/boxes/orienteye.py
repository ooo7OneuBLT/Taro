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

**段階3（2026-10-15〜16）：復帰抑制（一度見た所・物はしばらく見ない）を足した。**生まれつきの部品
（新生児からある。Valenza 1994 ほか、旧太郎 `doc/文献調査/2026-08-19_乳児のIORと馴化.md`）
なので手で決め、学習しない。
* 「もう見た」の印を、場所の点（網膜の座標、行・列）と強さの組で持つ（self.marks）。強さは毎回
  exp(−0.1/IOR_DECAY_S) 倍に薄れる。撃ったら全部の印を目が動く分だけ逆向きにずらし（目を動かす
  指令の写しを使う）、新しい注視点＝真ん中に強さ1の印を置く
* **印は物に付いて動く（2026-10-16、ユーザー決定）**：毎回、目の入口から届いた物ごとの山の
  中心を見て、印から TRACK_CELLS マス以内に物があれば、印をその物の中心へ移す。物が無ければ
  その場に残る（＝場所の印）。最初の版は網膜の場所にだけ印を付けたため、横に動き続ける妨害物が
  自分の印の外へ逃げ、太郎は妨害物を追いかけ直し続けた（段階3①のボール右の場面 1/10）
  - 物を見分けるのは見た目ではなく「つながって動いていること」（場所と時間の連続）。乳児は
    物を個体として追うとき、見た目より場所と時間の連続を使う（Xu & Carey 1996 系）。妨害物は
    点滅して見た目が変わるので、見た目で照合すると取り違える
  - 乳児で「動く物に印が付いて動く」ことを直接確かめた文献は見つけていない（成人は Tipper 1991。
    乳児は場所の復帰抑制が3〜6か月で整い、物の新しさを好む傾向は6か月で出る、Harman ほか 1994）。
    逸脱リスト行き［Tier3・モデル上の選択］
* **目の動いた分は、画像の中の物の動きから求める（2026-10-16、①の本番の診断から）**：最初の版は
  撃った回に「目を動かす指令の分」だけ印をずらしたが、実際の目の跳びは指令の 0.55〜0.95 倍で、
  印が数マスずれた所に残った。それで印が机のマスなどの近くの別の物へ移り、2回跳ぶと前の物の印は
  ほぼ消えていた。そこで毎回、前の回と今の回の物の中心の組を見比べ、いちばん多くの物が重なる
  ずれ（＝目が実際に動いた分。机の上の止まった物の多くは一緒にずれる）を全部の印に足す。
  重なる物が2つ未満で決められないときは、撃った直後の1回だけ指令の分を使う。物が届かない
  ときは前と同じく指令の分でずらす。人も目を動かしたあと、見えている物の位置で照合し直すと
  考えられる［Tier3・モデル上の選択］
* 印を新しく置くのは「撃った先のマス」（撃った回の座標）。その後、上のずれで目と一緒に動く
* 印が物に付いていける1回の距離 TRACK_CELLS は 1.5 マス（目の動きを別に引いたので、物そのものの
  動きだけ追えばよい）
* どこを見るか決める前に、合わせた地図から「印の地図」（印ごとに高さ＝強さ・σ=IOR_SIGMA_CELLS の
  山、重なりは大きい方）× IOR_GAIN を引く（負は0）
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
IOR_DECAY_S = 8.0        # 旧太郎 orienting.py:703 は3秒（成人）。①の実測（2026-10-16）で3秒では
                         # 1.2秒後の妨害物（目立ち1.0）に残る印0.67が、コップ（0.19）より上に残すので、
                         # 6.4秒以上が要る。8秒にした［Tier3・仮置き］
IOR_SIGMA_CELLS = 2.0    # 旧太郎 spatial_priority_map.py の「注意の半径の半分」≒2.3マス［仮置き］
IOR_GAIN = 1.0           # 旧太郎と同じ［仮置き］
TRACK_CELLS = 1.5        # 印が物に付いていける1回の距離。妨害物は横に約12度/秒＝約0.6マス/回［仮置き］
MATCH_CELLS = 1.0        # 目の動いた分を求めるとき、これより近ければ「同じ物が重なった」とみる
MIN_MATCH = 2            # 重なる物がこれ未満なら、画像からは目の動きを決めない
MERGE_CELLS = 1.0        # これより近い印は1つにまとめる（強い方を残す）
MAX_MARKS = 8
MARK_MIN = 0.02          # これより弱い印は捨てる

_YS, _XS = torch.meshgrid(torch.arange(GRID, dtype=DTYPE), torch.arange(GRID, dtype=DTYPE),
                          indexing="ij")


def bump_at(r: float, c: float, sigma: float = IOR_SIGMA_CELLS) -> torch.Tensor:
    return torch.exp(-((_XS - c) ** 2 + (_YS - r) ** 2) / (2.0 * sigma ** 2))


def shift_map(m: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    """地図を下へ dr・右へ dc マスずらす。はみ出た所は捨て、空いた所はゼロ。"""
    out = torch.zeros_like(m)
    r0, r1 = max(dr, 0), GRID + min(dr, 0)
    c0, c1 = max(dc, 0), GRID + min(dc, 0)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = m[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def object_centers(objs) -> list[tuple[float, float]]:
    """目の入口の物ごとの (荷札, 地図) から、山の中心（行・列、小数）を出す。山の 0.5 以上の重心。"""
    out = []
    for _tag, m in objs or []:
        if float(m.max()) <= 0.0:
            continue
        w = (m >= 0.5 * float(m.max())).to(DTYPE) * m
        tot = float(w.sum())
        out.append((float((w * _YS).sum()) / tot, float((w * _XS).sum()) / tot))
    return out


def estimate_shift(prev, cur):
    """前の回と今の回の物の中心の組から、いちばん多くの物が重なるずれ (行, 列) と重なった数を返す。
    同じ数なら、重なりの距離の合計が小さい方。"""
    if not prev or not cur:
        return (0.0, 0.0), 0
    cands = {(0.0, 0.0)} | {(c[0] - p[0], c[1] - p[1]) for p in prev for c in cur}
    best, best_key = (0.0, 0.0), (0, 0.0)
    for dr, dc in cands:
        n, err = 0, 0.0
        for p in prev:
            d = min(math.hypot(p[0] + dr - c[0], p[1] + dc - c[1]) for c in cur)
            if d <= MATCH_CELLS:
                n += 1
                err += d
        key = (n, -err)
        if key > best_key:
            best, best_key = (dr, dc), key
    if best_key[0] >= MIN_MATCH:
        # 重なった組の平均でならす
        pairs = []
        for p in prev:
            d, c = min((math.hypot(p[0] + best[0] - c[0], p[1] + best[1] - c[1]), c) for c in cur)
            if d <= MATCH_CELLS:
                pairs.append((c[0] - p[0], c[1] - p[1]))
        best = (sum(x for x, _ in pairs) / len(pairs), sum(y for _, y in pairs) / len(pairs))
    return best, best_key[0]


class Orienteye(Box):
    BOX_ID = "orienteye"

    def __init__(self, use_ior: bool = True) -> None:
        super().__init__()
        self.use_ior = bool(use_ior)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.last_fire = -10 ** 9
        self.last_reason = ""     # 記録用
        self.marks: list[list[float]] = []    # [行, 列, 強さ]（網膜の座標）
        self.ior = torch.zeros(GRID, GRID, dtype=DTYPE)
        self.last_ior_at_peak = 0.0   # 記録用：選んだマス（引く前の最大）に乗っていた印
        self._prev_centers = None     # 前の回の物の中心（目の動いた分を求める）
        self._pending = None          # 撃った直後：まだ画像で確かめていない指令の分 (行, 列, 残り回数)
        self.last_shift = (0.0, 0.0)  # 記録用：その回に印をずらした分

    def _render(self) -> None:
        m = torch.zeros(GRID, GRID, dtype=DTYPE)
        for r, c, a in self.marks:
            m = torch.maximum(m, a * bump_at(r, c))
        self.ior = m

    def _merge(self) -> None:
        kept: list[list[float]] = []
        for mk in sorted(self.marks, key=lambda x: -x[2]):
            if mk[2] < MARK_MIN:
                continue
            if any(math.hypot(mk[0] - k[0], mk[1] - k[1]) < MERGE_CELLS for k in kept):
                continue
            kept.append(mk)
        self.marks = kept[:MAX_MARKS]

    def _shift_marks(self, dr: float, dc: float) -> None:
        lim = 3 * IOR_SIGMA_CELLS
        for mk in self.marks:
            mk[0] += dr
            mk[1] += dc
        self.marks = [mk for mk in self.marks
                      if -lim <= mk[0] <= GRID - 1 + lim and -lim <= mk[1] <= GRID - 1 + lim]
        self.last_shift = (dr, dc)

    def _track(self, objs) -> None:
        """目の動いた分だけ印をずらし、そのあと印から TRACK_CELLS マス以内の物の中心へ移す。"""
        self.last_shift = (0.0, 0.0)
        if objs is None:
            return
        centers = object_centers(objs)
        prev, self._prev_centers = self._prev_centers, centers
        (dr, dc), n = estimate_shift(prev, centers)
        if n >= MIN_MATCH:
            if abs(dr) + abs(dc) > 0:
                self._shift_marks(dr, dc)
            if self._pending is not None and math.hypot(dr, dc) >= 1.0:
                self._pending = None          # 目が動いたのを画像で確かめた
        elif self._pending is not None:
            self._shift_marks(*self._pending[:2])
            self._pending = None
        if self._pending is not None:
            # 目が動くのは撃った次の回の画像から（脳は1回遅れで受け取る）。3回待って動かなければ忘れる
            self._pending = (self._pending[0], self._pending[1], self._pending[2] - 1)
            if self._pending[2] <= 0:
                self._pending = None
        for mk in self.marks:
            if not centers:
                break
            d, (r, c) = min((math.hypot(r - mk[0], c - mk[1]), (r, c)) for r, c in centers)
            if d <= TRACK_CELLS:
                mk[0], mk[1] = r, c

    def _mark_fire(self, r: int, c: int) -> None:
        """撃った回：撃った先 (r, c) に強さ1の印を置く。目が動いた分は、物が届いていれば
        次の回から画像で求めて足す（_track）。物が届いていなければ指令の分でずらす。"""
        self.marks.insert(0, [float(r), float(c), 1.0])
        dr, dc = CENTER - r, CENTER - c
        if self._prev_centers is None:
            self._shift_marks(dr, dc)
        else:
            self._pending = (dr, dc, 3)
        self._merge()
        self._render()

    def step(self, inbox):
        out = self.passthrough()
        self.t += 1
        m = inbox.get("prio")
        inhibit = float(inbox.get("gate3", 0.0))
        if self.use_ior:
            k = math.exp(-TICK_S / IOR_DECAY_S)
            for mk in self.marks:
                mk[2] *= k
            self._track(inbox.get("eyein"))
            self._merge()
            self._render()
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
