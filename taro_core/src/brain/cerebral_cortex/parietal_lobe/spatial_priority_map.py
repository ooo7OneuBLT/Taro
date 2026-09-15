# -*- coding: utf-8 -*-
"""場所の優先度地図 ── 「次にどこへ注意を向けるか」を、視野の升目ごとの値で持つ。

【置き場所について】人間で下からの目立ちと上からの目的を1枚にまとめるのは
頭頂間溝（LIP）。「網膜座標の、空間に限定された地図」であり、物の一覧ではない
（`doc/文献調査/二語文/2026-09-10_注意は場所か物か_人間側.md` で原文確認）。
復帰抑制は LIP・FEF・上丘の3か所に痕跡があり1か所に限定されないが、太郎では
この地図の中に1つだけ持つ［簡略化・逸脱リストへ］。

【なぜ作るか、2026-09-10】太郎の従来の「優先度地図」は**物の一覧から1枚選ぶ**もので、
カードが無い場所には値が無かった。値が無い場所は順位争いに参加できないため、
机の上に皿がはっきり見えていても注意が向かない（F2-109pre 実測）。

【この段でやること】
  1. 下からの目立ち（`SalienceMap` の出力）に、上からの目的を足す（今は 0）
  2. 直前に注意した場所を下げる（復帰抑制）
  3. 優先度を時間をかけて溜め、先に閾値へ達した升が勝つ（毎コマ最大値を取り直さない）
  4. 目が動いた分だけ、持ち越している値（抑制と溜め）をずらす

【この段でやらないこと】物体ファイルの生成（次の段）、目を動かすこと（その次）。

【数値の根拠】
  - 復帰抑制の持続 0.5〜0.9 秒 ［原文：Itti & Koch 2000。既定は中央の 0.7 秒］
  - 抑制の広がり＝注意の半径の半分 ［原文：同上］
  - 注意の半径＝画像幅の 1/6 ［仮置き：今回の調査では一次資料に届かなかった］
  - 抑制の強さ ［仮置き］
  - 溜めの時定数 0.11 秒 ［人間の注視 200〜330ms に合わせて実測で決めた］
"""
import numpy as np
from scipy.ndimage import gaussian_filter


class SpatialPriorityMap:
    """升目ごとの優先度を持ち、いちばん高い升を返す。

    Args:
        cell: 升目の一辺の数（`SalienceMap` と揃えること）。
        img_size: 元画像の一辺の画素数（升→画素の換算と、注意の半径の計算に使う）。
        ior_tau_s: 復帰抑制が薄れる時定数［秒］。
        ior_gain: 復帰抑制の強さ（0 で無効）。
        foa_radius_frac: 注意の半径を画像幅の何倍にするか。
    """

    def __init__(self, cell=28, img_size=224.0, ior_tau_s=0.7, ior_gain=1.0,
                 foa_radius_frac=1.0 / 6.0, acc_tau_s=0.11, acc_leak_s=1.0):
        self.cell = int(cell)
        self.img_size = float(img_size)
        self.ior_tau_s = float(ior_tau_s)
        self.ior_gain = float(ior_gain)
        self.foa_radius_px = float(foa_radius_frac) * self.img_size
        # 抑制の広がり＝注意の半径の半分（升単位に直す）
        self.ior_sigma_cell = max(
            (self.foa_radius_px / 2.0) / (self.img_size / self.cell), 0.5)
        self.ior = np.zeros((self.cell, self.cell), dtype=np.float32)
        # 勝者を決める溜めの層。acc_tau_s は「優先度 1.0 の升が閾値 1 に達するまでの
        #   秒数」＝注意が次へ移る間隔。人間の注視は 200〜330ms［原文：Itti & Koch
        #   2000 が引く実測サッケード間隔］。録画した視界で振って測り、平均の注視が
        #   その範囲に入る 0.11 秒を既定にした（0.10→241ms、0.12→284ms、2026-09-10 実測）。
        #   ＝この数は勘で置いたのではなく、人間の実測値に合わせて決めた。
        self.acc = np.zeros((self.cell, self.cell), dtype=np.float32)
        self.acc_tau_s = float(acc_tau_s)
        self.acc_leak_s = float(acc_leak_s)
        self.last_winner = None      # (col, row)

    # ---- 目が動いた分だけ持ち越しをずらす -------------------------------
    def _remap(self, shift_px):
        """遠心性コピーの shift（画素）だけ、抑制の地図をずらす。

        【なぜ必要か】地図は網膜に貼り付いているので、目が動くと地図ぜんたいが
        ずれる。持ち越している値（抑制）をずらさないと、前に見た場所の抑制が
        違う場所に残る。人間は目を動かす命令の写しで先回りしてずらす
        （Duhamel 1992・Wurtz 2008、要旨のみ）。
        """
        dx, dy = shift_px
        cx = int(round(dx / (self.img_size / self.cell)))
        cy = int(round(dy / (self.img_size / self.cell)))
        if cx == 0 and cy == 0:
            return
        self.ior = np.roll(self.ior, shift=(cy, cx), axis=(0, 1))
        # はみ出した帯は「知らない場所」なので抑制を 0 に戻す
        if cx > 0:
            self.ior[:, :cx] = 0.0
        elif cx < 0:
            self.ior[:, cx:] = 0.0
        if cy > 0:
            self.ior[:cy, :] = 0.0
        elif cy < 0:
            self.ior[cy:, :] = 0.0
        if self.last_winner is not None:
            c, r = self.last_winner[0] + cx, self.last_winner[1] + cy
            # 【2026-09-11・直し】ここだけ範囲を見ていなかった（地図は上で帯を0に戻し、
            #   `_bump` は clip している）。覚えている勝者が 0〜cell-1 の外へ出ると、
            #   `winner_px` が画像の外を指し、`gaze_from_attention` がその外へ目を
            #   向ける命令を出す。さらに `_v[_wr, _wc]` は +側で落ち、−側では numpy が
            #   末尾に回り込んで黙って別の升を読む（F2-126 が3分で IndexError）。
            #   丸めると注意が端に貼り付くので、**上の帯と同じ扱い＝「知らない場所」**
            #   にして覚えるのをやめ、次のコマで選び直させる。
            #   選び直す側に入ると溜めが大域リセットされ抑制がかかり、`switched` も
            #   True になる（視野外への脱落が切り替えに数えられる。解析時は注意）。
            self.last_winner = ((c, r) if 0 <= c < self.cell and 0 <= r < self.cell
                                else None)

    def _remap_acc(self, shift_px):
        """溜めの層も、目が動いた分だけずらす（抑制と同じ理由）。"""
        dx, dy = shift_px
        step = self.img_size / self.cell
        cx, cy = int(round(dx / step)), int(round(dy / step))
        if cx == 0 and cy == 0:
            return
        self.acc = np.roll(self.acc, shift=(cy, cx), axis=(0, 1))
        if cx > 0: self.acc[:, :cx] = 0.0
        elif cx < 0: self.acc[:, cx:] = 0.0
        if cy > 0: self.acc[:cy, :] = 0.0
        elif cy < 0: self.acc[cy:, :] = 0.0

    def _bump(self, col, row):
        """勝った升を中心に、抑制の山を1つ足す。"""
        g = np.zeros((self.cell, self.cell), dtype=np.float32)
        r = int(np.clip(row, 0, self.cell - 1))
        c = int(np.clip(col, 0, self.cell - 1))
        g[r, c] = 1.0
        g = gaussian_filter(g, sigma=self.ior_sigma_cell)
        mx = float(g.max())
        if mx > 1e-12:
            g /= mx
        self.ior = np.clip(self.ior + g, 0.0, 3.0)

    # ---- 本体 -----------------------------------------------------------
    def update(self, salience, shift_px=(0.0, 0.0), dt=0.1, goal=None):
        """1コマぶん。

        Args:
            salience: `SalienceMap` の出力（cell×cell、0〜1）。
            shift_px: 遠心性コピーの「画像の中身がどれだけ動いたか」（画素）。
            dt: 前回からの経過［秒］。
            goal: 上からの目的の地図（cell×cell）。None なら 0。
        Returns:
            {"priority","winner_cell","winner_px","ior","acc_max","switched"}
        """
        sal = np.asarray(salience, dtype=np.float32)
        self._remap(shift_px)
        if self.ior_tau_s > 0:
            self.ior *= float(np.exp(-max(dt, 0.0) / self.ior_tau_s))

        pri = sal.copy()
        if goal is not None:
            pri = pri + np.asarray(goal, dtype=np.float32)
        pri = pri - self.ior_gain * self.ior

        # 【原文：Itti et al. 1998】勝者を決めるのは、優先度を**時間をかけて溜める**
        #   層（leaky integrate-and-fire ニューロンの2次元層）。毎コマ最大値を
        #   取り直すのではない。溜めが先に閾値へ達した升が勝ち、勝った瞬間に
        #   層ぜんたいがリセットされ、その場所に復帰抑制がかかる。
        #   毎コマ取り直す作りだと、僅差の山の間で毎コマ飛んだ（2026-09-10 実測：
        #   注意が同じ升に留まった割合 0%、抑制を切っても 5%）。
        self._remap_acc(shift_px)
        if self.acc_leak_s > 0:
            self.acc *= float(np.exp(-max(dt, 0.0) / self.acc_leak_s))
        self.acc += np.clip(pri, 0.0, None) * (max(dt, 0.0) / self.acc_tau_s)

        fired = float(self.acc.max()) >= 1.0
        if fired or self.last_winner is None:
            r, c = np.unravel_index(int(np.argmax(self.acc)), self.acc.shape)
            switched = (self.last_winner is None) or ((c, r) != self.last_winner)
            self.acc[:] = 0.0                      # 大域リセット［原文］
            self._bump(c, r)
            self.last_winner = (int(c), int(r))
        else:
            c, r = self.last_winner
            switched = False
        step = self.img_size / self.cell
        return {"priority": pri, "winner_cell": (int(c), int(r)),
                "winner_px": ((c + 0.5) * step, (r + 0.5) * step),
                "ior": self.ior.copy(), "acc_max": float(self.acc.max()),
                "switched": bool(switched)}
