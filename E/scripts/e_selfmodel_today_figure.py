"""2026-07-29 に確定したことを1枚にまとめる。

【この日の流れ】
  夕方  「新生児期を経ると自己モデルが良くなる（+7.6%、3シード一致）」と結論
  夜    学習量をそろえたら差が消えた ⇒ 夕方の結論を撤回
  深夜  四肢の筋力補正のバグ発覚（0〜3ヶ月の体が仕様と違っていた）

注意：①②のデータは**バグを直す前**に取ったもの。0ヶ月の太郎は
  自分の体に対して2倍強い腕を持っていた（落とし穴チェックリスト 項76）。
  ＝**0ヶ月を含む条件の絶対値は仕様と違う体での値**。図にもそう書く。

使い方:
    .venv/Scripts/python.exe E/scripts/e_selfmodel_today_figure.py
出力:
    E/logs/selfmodel/今日わかったこと.png
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import re
import sys
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

import matplotlib                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402

for _f in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(_ROOT, "E", "logs", "selfmodel")
C_A, C_B, C_S, C_G = "#2b6cb0", "#c05621", "#2f855a", "#718096"


def margins(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return []
    r = []
    for l in open(p, encoding="utf-8", errors="replace"):
        m = re.search(r"margin=([+-][\d.]+)%", l)
        if m:
            r.append(float(m.group(1)))
    return r


def tail(r, k=8):
    return st.mean(r[-k:]) if r else float("nan")


def main():
    seeds = [0, 1, 2]
    # ① 学習量をそろえた対照
    A = [tail(margins(f"train_A_seed{s}.log")) for s in seeds]
    B = [tail(margins(f"train_B18k_seed{s}.log")) for s in seeds]
    # ② 体を変えたときだけ落ちる（保存・読み込みは同じ手順）
    def growth(name):
        r = margins(name)
        return (st.mean(r[-6:]) - st.mean(r[:2])) if len(r) > 10 else float("nan")
    S = [growth(f"train_S_seed{s}.log") for s in seeds]
    J = [growth(f"train_A_seed{s}.log") for s in seeds]

    fig = plt.figure(figsize=(14.0, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=0.30)

    # ---- ① ----
    ax = fig.add_subplot(gs[0])
    x = np.arange(3)
    ax.bar(x - 0.19, A, 0.36, color=C_A, label="新生児期を経る")
    ax.bar(x + 0.19, B, 0.36, color=C_B, label="最初から4ヶ月")
    for i, (a, b) in enumerate(zip(A, B)):
        ax.text(i - 0.19, a + 0.5, f"{a:+.1f}", ha="center", fontsize=9, color=C_A)
        ax.text(i + 0.19, b + 0.5, f"{b:+.1f}", ha="center", fontsize=9, color=C_B)
    d = [a - b for a, b in zip(A, B)]
    ax.set_xticks(x); ax.set_xticklabels([f"シード{s}" for s in seeds])
    ax.set_ylabel("自己モデルの質（margin, %）")
    ax.set_ylim(0, max(A + B) * 1.32)
    ax.set_title("① 学習量をそろえたら差が消えた", loc="left", fontsize=11.5)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    _sd = st.stdev(d)
    _t = st.mean(d) / (_sd / len(d) ** 0.5)
    ax.text(0.5, 0.955,
            f"差 平均{st.mean(d):+.1f} ± {_sd:.1f}（3シードとも A>B）\n"
            f"t={_t:.2f}（自由度2・5%の臨界値4.30）＝偶然を否定できない",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.2,
            color="#c53030",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f5", ec="#feb2b2"))

    # ---- ② ----
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(x - 0.19, S, 0.36, color=C_S, label="体はそのまま")
    ax2.bar(x + 0.19, J, 0.36, color=C_A, label="4ヶ月へ載せ替え")
    for i, (s_, j_) in enumerate(zip(S, J)):
        ax2.text(i - 0.19, s_ + (0.4 if s_ >= 0 else -1.1), f"{s_:+.1f}",
                 ha="center", fontsize=9, color=C_S)
        ax2.text(i + 0.19, j_ + (0.4 if j_ >= 0 else -1.1), f"{j_:+.1f}",
                 ha="center", fontsize=9, color=C_A)
    ax2.axhline(0, color="#444", lw=1.0)
    ax2.set_xticks(x); ax2.set_xticklabels([f"シード{s}" for s in seeds])
    ax2.set_ylabel("6000回での伸び（%ポイント）")
    ax2.set_title("② 落ちるのは「体を変えたとき」だけ", loc="left", fontsize=11.5)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(0.5, 0.04,
             "保存・読み込みの手順はどちらも同じ\n"
             "⇒「経験のメモ帳が消える」は無罪",
             transform=ax2.transAxes, ha="center", va="bottom", fontsize=9.5,
             color="#276749",
             bbox=dict(boxstyle="round,pad=0.35", fc="#f0fff4", ec="#9ae6b4"))

    # ---- ③ ----
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    ax3.text(0.0, 1.00, "③ そして前提が崩れた", fontsize=11.5, va="top",
             transform=ax3.transAxes)
    body = (
        "四肢の筋力補正にバグ（落とし穴 項76）\n\n"
        "補正の「基準」を、太郎の体を作る途中で\n"
        "裏に18ヶ月の体を1体作って測っていた。\n"
        "新生児の体型補正がそこに漏れ、\n"
        "基準が2.1倍ずれていた。\n\n"
        "            補正のかかり方    相対的な強さ\n"
        "  0ヶ月     79個中31個だけ     137.88\n"
        "  4ヶ月     79個すべて            64.73\n"
        "  本来      どちらも              64 前後\n\n"
        "＝新生児が自分の体に対して2倍強い腕を\n"
        "  持っていた。直すはずの補正が\n"
        "  新生児でだけ効いていなかった。\n\n"
        "⇒ ①②はバグを直す前のデータ。\n"
        "  0ヶ月を含む条件は仕様と違う体での値。\n"
        "  直した体で3条件を回し直し中。"
    )
    ax3.text(0.0, 0.90, body, fontsize=9.3, va="top",
             transform=ax3.transAxes, linespacing=1.5)

    fig.suptitle("2026-07-29 に確定したこと ─ 夕方の結論は撤回し、前提のバグを見つけた",
                 fontsize=13, y=0.99)
    fig.text(0.008, 0.012,
             "margin＝自分の行動と他人の行動で予測させたときの予測誤差の差。"
             "大きいほど「自分の体を分かっている」。すべて4ヶ月の体で評価・終盤8点の平均",
             fontsize=8, color="#555")
    path = os.path.join(OUT, "今日わかったこと.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"→ {os.path.relpath(path, _ROOT)}")
    print(f"① 差 平均{st.mean(d):+.1f} ばらつき{st.pstdev(d):.1f}")
    print(f"② 体そのまま 平均{st.mean(S):+.1f} / 載せ替え 平均{st.mean(J):+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
