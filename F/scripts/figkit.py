# -*- coding: utf-8 -*-
"""回路図・ブロック図の共通部品（2026-09-10）。

【なぜ】ユーザー指摘（2026-09-10）「矢印をボックスに触れるまで伸ばさないで。見にくい」
「回路図の作り方を調べてから作って」。調べた手引き（Vexlio / serialized.net /
Formation / GitHub Docs / artifact-diagramming スキル）から、毎回守る規則を
コードに落とした。図ごとに書き直さない。

守る規則：
  1. 矢印は箱に触れない。箱の縁から gap 分だけ手前で始まり、手前で終わる。
  2. 矢印には「何が流れるか」を書く（label）。無地の矢印は「なんか関係ある」しか言わない。
  3. 箱は格子に置く（col/row 指定）。目分量で置かない。
  4. 余白は箱と同じくらい取る。
  5. 文字の大きさは3種類だけ（題・箱・注記）。
  6. 確認レベルの弱い線は破線にする（dashed=True）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]

FS_TITLE, FS_BOX, FS_NOTE = 15.0, 10.5, 8.8
GAP = 0.18          # 箱の縁と矢印の先端のすきま（データ座標）


class Panel:
    """1枚の図（または左右に並べる1面）。col/row の格子で箱を置く。"""

    def __init__(self, ax, ncol, nrow, box_w=2.9, box_h=1.05, gap_x=0.75, gap_y=0.95,
                 x0=0.35, y0=0.35):
        self.ax = ax
        self.bw, self.bh = box_w, box_h
        self.gx, self.gy = gap_x, gap_y
        self.x0, self.y0 = x0, y0
        self.ncol, self.nrow = ncol, nrow
        self.boxes = {}
        ax.set_xlim(0, x0 * 2 + ncol * box_w + (ncol - 1) * gap_x)
        ax.set_ylim(0, y0 * 2 + nrow * box_h + (nrow - 1) * gap_y)
        ax.axis("off")

    def rect(self, col, row, colspan=1):
        """格子の (col,row) の箱の (x, y, w, h)。row は上が 0。"""
        w = self.bw * colspan + self.gx * (colspan - 1)
        x = self.x0 + col * (self.bw + self.gx)
        y = self.y0 + (self.nrow - 1 - row) * (self.bh + self.gy)
        return x, y, w, self.bh

    def box(self, name, col, row, text, colspan=1, fc="white", ec="#333", lw=1.6,
            note=None, note_color="#c1121f"):
        x, y, w, h = self.rect(col, row, colspan)
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                          fc=fc, ec=ec, lw=lw, zorder=2))
        self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                     fontsize=FS_BOX, weight="bold", zorder=3)
        if note:
            # 注記は箱の真下だと出ていく矢印と重なるので、白地を敷いて上に出す
            self.ax.text(x + w / 2, y - 0.16, note, ha="center", va="top",
                         fontsize=FS_NOTE, color=note_color, zorder=5,
                         bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                    ec="none", alpha=0.92))
        self.boxes[name] = (x, y, w, h)
        return self.boxes[name]

    @staticmethod
    def _edge(rect, toward):
        """rect の中心から toward へ向かう線が、rect の縁と交わる点。"""
        x, y, w, h = rect
        cx, cy = x + w / 2, y + h / 2
        dx, dy = toward[0] - cx, toward[1] - cy
        if dx == 0 and dy == 0:
            return cx, cy
        sx = (w / 2) / abs(dx) if dx else np.inf
        sy = (h / 2) / abs(dy) if dy else np.inf
        s = min(sx, sy)
        return cx + dx * s, cy + dy * s

    def arrow(self, a, b, label=None, color="#2a4d8f", lw=2.0, dashed=False,
              both=False, gap=GAP, label_side=0.0, label_color=None):
        """箱 a から箱 b へ。両端とも箱の縁から gap 手前で止める。"""
        ra, rb = self.boxes[a], self.boxes[b]
        ca = (ra[0] + ra[2] / 2, ra[1] + ra[3] / 2)
        cb = (rb[0] + rb[2] / 2, rb[1] + rb[3] / 2)
        p0 = np.array(self._edge(ra, cb))
        p1 = np.array(self._edge(rb, ca))
        v = p1 - p0
        L = np.hypot(*v)
        if L <= 2 * gap:
            return
        u = v / L
        p0 = p0 + u * gap
        p1 = p1 - u * gap
        self.ax.annotate("", xy=tuple(p1), xytext=tuple(p0),
                         arrowprops=dict(arrowstyle="<->" if both else "->",
                                          lw=lw, color=color,
                                          linestyle="--" if dashed else "-",
                                          shrinkA=0, shrinkB=0), zorder=1)
        if label:
            mid = (p0 + p1) / 2
            perp = np.array([-u[1], u[0]]) * (label_side if label_side else 0.22)
            self.ax.text(mid[0] + perp[0], mid[1] + perp[1], label, ha="center",
                         va="center", fontsize=FS_NOTE,
                         color=label_color or color, zorder=4,
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def figure(ncol, nrow, panels=1, **kw):
    """panels 面ぶんの Panel を作って返す。"""
    fig, axes = plt.subplots(1, panels, figsize=(8.2 * panels, 1.55 * nrow + 1.4))
    axes = [axes] if panels == 1 else list(axes)
    return fig, [Panel(ax, ncol, nrow, **kw) for ax in axes]
