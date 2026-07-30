"""月齢を上げたときに何が起きるかを1枚の図にする（2026-07-28の実測から作図）。

数値は `e_age_body_audit.py` / `e_age_muscle_check.py` / `e_age_weight_check.py` の
実測結果をそのまま書き写したもの。**この図を描き直すときは必ず測り直すこと**
（値を手で持つのは図の再現のためで、測定の代わりではない）。

使い方:
    .venv/Scripts/python.exe E/scripts/e_age_body_figure.py
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

for _f in ("Meiryo", "Yu Gothic", "MS Gothic", "IPAexGothic"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.pardir, "figures", "月齢を4ヶ月にすると何が起きるか_2026-07-28.png")

# ---- 実測値（2026-07-28） ----
ages = np.array([0, 2, 3, 4, 6])
# 首の筋力 [N·m]（太郎の補正込み。4ヶ月以降は補正が解除されMIMoの素の値）
neck_strength = np.array([0.066, 15.778, 23.590, 31.390, 31.390])
# 持ち上げ能力比＝首の筋力 ÷ 頭の重力モーメント
lift_ratio = np.array([0.17, 15.64, 24.38, 47.51, 44.17])
# 首のバネの強さ [N·m/rad]
neck_spring = np.array([0.40, 0.40, 0.0, 0.0, 0.0])
# 体重 [kg]
w_shape = np.array([3.367, 6.899, 7.674, 8.303, 9.327])    # 新生児用の体型補正あり
w_bare = np.array([2.888, 5.851, 6.504, 7.033, 7.895])     # 素のMIMo
w_human = np.array([3.3, 5.6, 6.4, 7.0, 7.9])              # WHO 2006 男児中央値

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
fig.suptitle("月齢を4ヶ月に上げると太郎の体はどうなるか（実測・2026-07-28）",
             fontsize=15, fontweight="bold")

# --- (1) 首の筋力 ---
ax = axes[0, 0]
ax.plot(ages, neck_strength, "o-", color="#c0392b", lw=2.5, ms=9)
ax.axvline(4, color="#7f8c8d", ls="--", lw=1.5)
ax.text(4.06, 16, "4ヶ月で補正が解除", color="#7f8c8d", fontsize=10, rotation=90,
        va="center")
ax.annotate("0.066", (0, 0.066), textcoords="offset points", xytext=(6, 12),
            fontsize=10, color="#c0392b")
ax.annotate("31.39\n(475倍)", (4, 31.39), textcoords="offset points", xytext=(-38, -6),
            fontsize=10, color="#c0392b", fontweight="bold")
ax.set_title("① 首の筋力が4ヶ月で475倍に跳ぶ", fontsize=12, fontweight="bold")
ax.set_xlabel("月齢"); ax.set_ylabel("首の筋力 [N·m]")
ax.grid(alpha=0.3)

# --- (2) 持ち上げ能力比 ---
ax = axes[0, 1]
bars = ax.bar([str(a) for a in ages], lift_ratio,
              color=["#27ae60", "#f39c12", "#f39c12", "#c0392b", "#c0392b"])
ax.axhline(1.0, color="#2c3e50", ls="-", lw=2)
ax.text(4.3, 2.5, "1.0 = 頭の重さとちょうど釣り合う", fontsize=9.5, color="#2c3e50",
        ha="right")
for b, v in zip(bars, lift_ratio):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center",
            fontsize=10)
ax.set_title("② 頭の重さの何倍を持ち上げられるか", fontsize=12, fontweight="bold")
ax.set_xlabel("月齢"); ax.set_ylabel("持ち上げ能力比")
ax.set_ylim(0, 56)
ax.grid(alpha=0.3, axis="y")

# --- (3) 首のバネ ---
ax = axes[1, 0]
ax.step(ages, neck_spring, where="post", color="#2980b9", lw=3)
ax.fill_between(ages, 0, neck_spring, step="post", color="#2980b9", alpha=0.2)
ax.axvline(3, color="#7f8c8d", ls="--", lw=1.5)
ax.text(3.08, 0.2, "3ヶ月でバネが消える", color="#7f8c8d", fontsize=10, rotation=90,
        va="center")
ax.text(4.3, 0.05, "支えるものが何も無い状態", fontsize=10, color="#c0392b",
        fontweight="bold")
ax.set_title("③ 首を支えるバネ（筋緊張）は3ヶ月で消える", fontsize=12,
             fontweight="bold")
ax.set_xlabel("月齢"); ax.set_ylabel("バネの強さ [N·m/rad]")
ax.set_ylim(-0.03, 0.5)
ax.grid(alpha=0.3)

# --- (4) 体重 ---
ax = axes[1, 1]
w = 0.27
x = np.arange(len(ages))
ax.bar(x - w, w_human, w, label="人間（WHO中央値）", color="#2c3e50")
ax.bar(x, w_bare, w, label="素のMIMo", color="#16a085")
ax.bar(x + w, w_shape, w, label="新生児用の体型補正あり", color="#e67e22")
for i, (b, s, h) in enumerate(zip(w_bare, w_shape, w_human)):
    ax.text(i + w, s + 0.15, f"{s/h*100:.0f}%", ha="center", fontsize=9,
            color="#e67e22", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([str(a) for a in ages])
ax.set_title("④ 新生児用の体型補正は4ヶ月では重すぎる", fontsize=12, fontweight="bold")
ax.set_xlabel("月齢"); ax.set_ylabel("体重 [kg]")
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(alpha=0.3, axis="y")

fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.text(0.5, 0.005,
         "★MIMoは筋肉モデルの筋力を月齢で変えていない（FMAX 180個すべてが0/4/18ヶ月で完全一致）。"
         "①の変化はすべて太郎側の補正が解除された結果。",
         ha="center", fontsize=10, color="#c0392b")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved:", os.path.abspath(OUT))
