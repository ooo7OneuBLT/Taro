# -*- coding: utf-8 -*-
"""図：今の太郎で「眼球の筋への指令」を書いている経路を全部並べる（2026-09-10）。

事実の出典（コードを読んで確認）：
  E/scripts/e_toy_env.py:2171-2180  step() で action を書き換える順番
    ①_vor.override(action)  ②_orienting.apply(action)  ③_vergence.apply(action)
  taro_core/src/brain/brainstem/vor.py:195,258  方策の眼球出力を上書きする
  taro_core/src/brain/midbrain/orienting.py     自前の動き検出＋set_voluntary_target
  run/plugins/common/object_files.py:675-681    優先度地図の勝者→set_voluntary_target
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figkit import figure
import matplotlib.pyplot as plt

fig, (p,) = figure(4, 4, panels=1, box_w=3.0, box_h=1.0, gap_x=0.8, gap_y=1.05)
p.ax.set_title("太郎で眼球を動かしている経路（F2-111pre 時点）", fontsize=15, weight="bold")

p.box("pol", 0, 0, "方策\n（強化学習）", fc="#f3f3f3")
p.box("head", 1, 0, "頭の回転\n（半規管）", fc="#eaf3ff")
p.box("img", 2, 0, "左目の画像", colspan=2, fc="#eaf3ff")

p.box("vor", 1, 1, "前庭動眼反射", fc="#fff6e0")
p.box("mot", 2, 1, "自前の動き検出\n（視線誘導反射）", fc="#fff6e0")
p.box("pri", 3, 1, "目立ちの地図\n→場所の優先度地図", fc="#e8f7ea")

p.box("ver", 0, 2, "輻輳反射", fc="#fff6e0")
p.box("sc", 2, 2, "サッケードの測定（上丘）", colspan=2, fc="#fff6e0")

p.box("mus", 1, 3, "眼球の筋への指令", colspan=2, fc="#ffe9e9")

p.arrow("pol", "vor", "眼球ぶんは捨てられる", color="#999", dashed=True, label_color="#c1121f")
p.arrow("head", "vor", "回った速さ")
p.arrow("vor", "mus", "頭と逆向き")
p.arrow("img", "mot", "コマの差")
p.arrow("img", "pri", "明るさ・色・向き・動き", label_side=0.45)
p.arrow("mot", "sc", "動きの方向")
p.arrow("pri", "sc", "注意する場所")
p.arrow("sc", "mus", "跳ぶ向き")
p.arrow("ver", "mus", "左右の差だけ足す")

p.ax.text(0.35, 0.08, "黄＝反射　緑＝今回作った地図　灰＝学習した方策（眼球には届かない）",
          fontsize=8.8, color="#444")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "_図",
                   "眼球を動かす経路_2026-09-10.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
print(os.path.abspath(out))
