# -*- coding: utf-8 -*-
"""目立ちの地図 ── 視野ぜんたいの各場所に「どれくらい注意を向けるべきか」の値を1つ持つ。

【置き場所について】人間で下からの目立ちを計算するのは上丘の表層と視覚野各所
（V1〜V4）で、それを場所ごとの値にまとめるのが頭頂間溝の優先度地図（LIP）。
本ファイルはそのうち**下からの目立ちを1枚にまとめるところまで**を担う。
上からの目的・復帰抑制・1位の選択は、この上の段（優先度地図）の仕事で、ここには入れない。

【なぜ作るか、2026-09-10】太郎の「前注意の地図」は**動いた場所にしか値が無かった**。
実測：机の上に皿がはっきり見えているのに物体ファイルが0枚（F2-109pre、frame_00053）。
値が無い場所は順位争いに参加できないので、静止した物には注意が向きようがない。
人間の優先度地図は「網膜座標の、空間に限定された地図」であり、物があるかどうかに
関係なく各場所に値がある（`F/docs/二語文/文献調査/2026-09-10_注意は場所か物か_人間側.md`
で原文確認）。

【手順の出典】Itti, Koch & Niebur (1998)（`…/2026-09-10_目立ちの地図の作り方_人間とAI側.md`
で原文確認）。
  1. 画像を特徴ごとに分ける（明るさ・色の反対色・向き）
  2. 粗さの違う段（ピラミッド）を作り、細かい段から粗い段を引く（中心-周辺差）
     ［原文：c∈{2,3,4}、s=c+{3,4} の6通り］
  3. 各枚を正規化する（少数の場所だけ突出していれば強め、団子なら弱め）
  4. 特徴ごとに足し、最後に全部を足して1枚にする ［原文：S = 1/3(N(Ī)+N(C̄)+N(Ō))］
**動きは特別扱いしない。** 明るさ・色と横並びの1つの特徴として同じ手順を通す
［原文：Itti & Koch 2001 の設計思想。1998年版に動きチャネルは無く、後から
1特徴として追加された］。

【既製品を使わなかった理由】OpenCV の saliency モジュール（Apache 2.0）は
この環境に入っていない（cv2 4.11.0、`hasattr(cv2,"saliency")` は False）。
加えて、既製の静止画用では**動きを同じ枠で混ぜられない**。太郎の方針
（既製部品ファースト）からの逸脱にあたるので、逸脱リストに登録すること。

【この段でやらないこと】1位の選択（勝者総取り）、復帰抑制、上からの目的の加算、
物体ファイルの生成。すべて上の段の仕事。
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def _pyramid(a, levels):
    """粗さの違う段を作る。段 k は 1/2^k の大きさ。

    半分ずつ実際に縮める。ぼかしの半径を 2^k と大きくしていく作りは、224 画素の
    画像に半径 256 のぼかしまで掛かって 1 コマ 400 ms かかった（2026-09-10 実測）。
    太郎は 0.1 秒刻みなので、それでは使えない。
    """
    out = [a]
    cur = a
    for _ in range(1, levels):
        if min(cur.shape) <= 2:
            out.append(cur)
            continue
        cur = gaussian_filter(cur, sigma=1.0)[::2, ::2]
        out.append(cur)
    return out


def _resize_to(a, shape):
    """最近傍で shape に合わせる（段どうしの大きさを揃える・升目へ落とす）。"""
    h, w = shape
    yi = (np.arange(h) * a.shape[0] // h).clip(0, a.shape[0] - 1)
    xi = (np.arange(w) * a.shape[1] // w).clip(0, a.shape[1] - 1)
    return a[yi][:, xi]


def _normalize(m):
    """N(・)：1枚を0〜1に伸ばしたうえで、(最大値 − 局所最大の平均)^2 を掛ける。

    少数の場所だけ突出していれば大きく、どこも同じくらいなら小さくなる
    ［原文：Itti et al. 1998 の N(・)］。
    """
    mx = float(m.max())
    if mx <= 1e-12:
        return np.zeros_like(m)
    m = m / mx
    # 局所最大の平均：最大値の 1/2 を超える点を「局所最大」の代わりに使う近似
    thr = 0.5
    peaks = m[m >= thr]
    m_bar = float(peaks.mean()) if peaks.size > 1 else 0.0
    return m * ((1.0 - m_bar) ** 2)


class SalienceMap:
    """下からの目立ちを1枚にまとめる。

    Args:
        cell: 出力する升目の一辺の数（視野を何分割するか）。
        levels: ピラミッドの段数。
        cs_pairs: 中心-周辺差の組み合わせ。既定は原文の c∈{2,3,4}, s=c+{3,4}。
        weights: 特徴ごとの重み。順に 明るさ・色・向き・動き。
            動きと静止特徴の重みの配分は人間側のデータが見つからなかった［仮置き］。
        blur_cell: 升目に落としたあとのぼかし（升単位）。
    """

    def __init__(self, cell=28, levels=7, cs_pairs=None,
                 weights=(1.0, 1.0, 1.0, 1.0), blur_cell=0.8):
        self.cell = int(cell)
        self.levels = int(levels)
        self.cs_pairs = tuple(cs_pairs) if cs_pairs else (
            (2, 5), (2, 6), (3, 6), (3, 7), (4, 7), (4, 8))
        self.weights = tuple(float(w) for w in weights)
        self.blur_cell = float(blur_cell)
        self._prev_gray = None
        # ピラミッドは cs_pairs が指す最大の段まで必要
        self._need = max(max(c, s) for c, s in self.cs_pairs) + 1

    # ---- 特徴の取り出し -------------------------------------------------
    @staticmethod
    def _features(img_rgb):
        """明るさと反対色（赤緑・青黄）を取り出す［原文：Itti et al. 1998 2節］。"""
        a = np.asarray(img_rgb, dtype=np.float32)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        inten = (r + g + b) / 3.0
        # 明るいところだけで色を評価する（暗部の色は当てにならない）
        mx = float(inten.max())
        mask = inten >= 0.1 * mx if mx > 0 else np.zeros_like(inten, dtype=bool)
        denom = np.where(inten > 1e-6, inten, 1.0)
        R = np.where(mask, r - (g + b) / 2.0, 0.0) / denom
        G = np.where(mask, g - (r + b) / 2.0, 0.0) / denom
        B = np.where(mask, b - (r + g) / 2.0, 0.0) / denom
        Y = np.where(mask, (r + g) / 2.0 - np.abs(r - g) / 2.0 - b, 0.0) / denom
        return inten, np.abs(R - G), np.abs(B - Y)

    @staticmethod
    def _orientation(inten):
        """向きの手がかり。4方向の1次微分の絶対値の和で代用する。

        ［簡略化］原文は Gabor フィルタ。ここでは計算を軽くするため微分で代用した。
        逸脱リストに登録すること。
        """
        gx = np.zeros_like(inten); gy = np.zeros_like(inten)
        gx[:, 1:-1] = inten[:, 2:] - inten[:, :-2]
        gy[1:-1, :] = inten[2:, :] - inten[:-2, :]
        d1 = np.zeros_like(inten); d2 = np.zeros_like(inten)
        d1[1:-1, 1:-1] = inten[2:, 2:] - inten[:-2, :-2]
        d2[1:-1, 1:-1] = inten[2:, :-2] - inten[:-2, 2:]
        return np.abs(gx) + np.abs(gy) + np.abs(d1) + np.abs(d2)

    def _channel(self, a):
        """1つの特徴 → ピラミッド → 中心-周辺差 → 正規化 → 升目に落として足す。

        中心-周辺差は「細かい段」の大きさで取り、粗い段はそこへ伸ばして引く
        ［原文：Itti et al. 1998 の across-scale difference］。段ごとに大きさが
        違うので、正規化したあと升目へ落としてから足す。
        """
        pyr = _pyramid(a, self._need)
        acc = np.zeros((self.cell, self.cell), dtype=np.float32)
        for c, s in self.cs_pairs:
            fine = pyr[c]
            coarse = _resize_to(pyr[s], fine.shape)
            acc += _resize_to(_normalize(np.abs(fine - coarse)),
                              (self.cell, self.cell))
        return _normalize(acc)

    def _to_cells(self, a):
        """画素の地図を升目に落とす（升の中の平均）。"""
        h, w = a.shape
        ch, cw = h // self.cell, w // self.cell
        a = a[:ch * self.cell, :cw * self.cell]
        return a.reshape(self.cell, ch, self.cell, cw).mean(axis=(1, 3))

    # ---- 本体 -----------------------------------------------------------
    def update(self, img_rgb, shift_actual=(0.0, 0.0), moving=False):
        """1コマぶん。

        Returns:
            {"salience": (cell,cell) 0〜1, "peak": (x_px, y_px), "peak_cell": (col,row),
             "channels": {"明るさ","色","向き","動き"} それぞれ (cell,cell),
             "valid": bool}
        `moving`（目が動いている最中）のコマは、動きの特徴だけ 0 にする。静止特徴は
        そのまま出す（人間は目を動かしている最中も場面の目立ちを失わない）。
        """
        img = np.asarray(img_rgb, dtype=np.float32)
        if img.ndim == 2:
            img = np.dstack([img, img, img])
        inten, rg, by = self._features(img)
        ori = self._orientation(inten)

        prev = self._prev_gray
        self._prev_gray = inten.copy()
        if prev is None or moving:
            motion = np.zeros_like(inten)
        else:
            dx, dy = shift_actual
            prev_shifted = np.roll(prev, shift=(int(round(dy)), int(round(dx))),
                                    axis=(0, 1))
            motion = np.abs(inten - prev_shifted)

        chans = {
            "明るさ": self._channel(inten),
            "色": self._channel(rg) + self._channel(by),
            "向き": self._channel(ori),
            "動き": self._channel(motion),
        }
        cells = chans          # _channel が升目まで落として返す
        w = dict(zip(("明るさ", "色", "向き", "動き"), self.weights))
        total = sum(w[k] * _normalize(v) for k, v in cells.items())
        if self.blur_cell > 0:
            total = gaussian_filter(total, sigma=self.blur_cell)
        mx = float(total.max())
        total = total / mx if mx > 1e-12 else total

        r, c = np.unravel_index(int(np.argmax(total)), total.shape)
        px = (c + 0.5) * img.shape[1] / self.cell
        py = (r + 0.5) * img.shape[0] / self.cell
        return {"salience": total, "peak": (float(px), float(py)),
                "peak_cell": (int(c), int(r)), "channels": cells,
                "valid": prev is not None and not moving}
