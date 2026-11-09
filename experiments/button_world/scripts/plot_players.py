"""手書きのプレイヤーの結果を、表（md 形式の txt）と図にする。python plot_players.py <results.json> <出力の置き場>"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf")
plt.rcParams["font.family"] = "IPAPGothic"

data = json.load(open(sys.argv[1], encoding="utf-8"))
res = data["結果"]
out = sys.argv[2]
PLAYERS = ["総当たりの人", "覚える人", "答えを知っている人", "でたらめ"]
RULE_JA = {"none": "何も起きない", "reset_to_start": "始めに戻される", "end_episode": "押したら終わり"}


def key(rule, nb, view):
    return f"{rule}・ボタン{nb}個・{view}"


lines = ["# 手書きのプレイヤーの比べ（学習なし）", ""]
lines.append(f"条件: {data['条件']}。計測にかかった時間は合計 {data['計測にかかった秒数（各行は jit の初回を含む）']['全部']} 秒。")
lines.append("")
for view in ("上から", "横から"):
    lines.append(f"## {view}")
    lines.append("")
    lines.append("| ルール | ボタン | 人 | ゲーム数 | 4回の合計点 | 回ごとの点（1〜4回目） | 成功率（1回目/4回目） | 成功した回の歩数の中央値（1〜4回目） | 最初のボタンが当たり（1回目/4回目） | 時間切れ（1回目/4回目） | 外れで終わり（1回目/4回目） |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for rule in ("none", "reset_to_start", "end_episode"):
        for nb in (3, 6):
            for pl in PLAYERS:
                r = res[key(rule, nb, view)][pl]
                e = r["回ごと"]
                lines.append(
                    f"| {RULE_JA[rule]} | {nb} | {pl} | {r['ゲーム数']} | {r['4回の合計点_平均']} | "
                    + "／".join(f"{x['点の平均']:.3f}" for x in e)
                    + f" | {e[0]['成功率']:.2f}／{e[3]['成功率']:.2f} | "
                    + "／".join(str(x["成功した回の歩数_中央値"]) for x in e)
                    + f" | {e[0]['最初に押したボタンが当たりの割合']:.2f}／{e[3]['最初に押したボタンが当たりの割合']:.2f}"
                    + f" | {e[0]['時間切れの割合']:.2f}／{e[3]['時間切れの割合']:.2f} | {e[0]['外れで終わった割合']:.2f}／{e[3]['外れで終わった割合']:.2f} |"
                )
    lines.append("")
lines.append("## 覚える人 − 総当たりの人 の点の差（回ごと）")
lines.append("")
lines.append("| 視点 | ルール | ボタン | 1回目 | 2回目 | 3回目 | 4回目 | 2〜4回目の平均 |")
lines.append("|---|---|---|---|---|---|---|---|")
diffs = {}
for view in ("上から", "横から"):
    for rule in ("none", "reset_to_start", "end_episode"):
        for nb in (3, 6):
            a = [x["点の平均"] for x in res[key(rule, nb, view)]["覚える人"]["回ごと"]]
            b = [x["点の平均"] for x in res[key(rule, nb, view)]["総当たりの人"]["回ごと"]]
            d = [round(x - y, 3) for x, y in zip(a, b)]
            diffs[(view, rule, nb)] = d
            lines.append(f"| {view} | {RULE_JA[rule]} | {nb} | " + " | ".join(f"{x:+.3f}" for x in d) + f" | {sum(d[1:]) / 3:+.3f} |")
open(os.path.join(out, "table.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

colors = {"none": "#888888", "reset_to_start": "#2a78d6", "end_episode": "#eb6834"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=120, sharey=True)
for ax, view in zip(axes, ("上から", "横から")):
    for rule in ("none", "reset_to_start", "end_episode"):
        for nb, ls in ((3, "-"), (6, "--")):
            ax.plot([1, 2, 3, 4], diffs[(view, rule, nb)], color=colors[rule], ls=ls, marker="o", lw=2, label=f"{RULE_JA[rule]}・ボタン{nb}個")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("ゲームの中の回")
    ax.set_title(view)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("点の差（覚える人 − 総当たりの人）")
axes[1].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("ゲームの中で覚えると、どれだけ点が上がるか（手書きのプレイヤー、学習なし）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(out, "score_diff.png"))
print("saved")
