# -*- coding: utf-8 -*-
import sys; sys.path.insert(0, "F/scripts")
import matplotlib.pyplot as plt
from figkit import Panel, FS_TITLE
out = sys.argv[1]
NROW = 7
fig, axes = plt.subplots(1, 2, figsize=(28.0, 2.15 * NROW + 2.2))
BLUE="#a8dadc"; ORG="#f4a261"; YEL="#f6d55c"; GRAY="#ededed"; RED="#f6bcb2"

L = Panel(axes[0], 4, NROW, box_w=3.15, box_h=1.3, gap_x=2.15, gap_y=1.7)
R = Panel(axes[1], 3, NROW, box_w=3.15, box_h=1.3, gap_x=2.15, gap_y=1.7)

# ================= 左：人間（実行の順に上から下） =================
stages = ["① 入力", "② 面のまとまり", "③ 注意の前の候補", "④ 優先度の地図",
          "⑤ 注意が向く", "⑥ 中身の入った記録", "⑦ 目を動かす"]
for i, s in enumerate(stages):
    x, y, w, h = L.rect(0, i)
    L.ax.text(x + w, y + h / 2, s, ha="right", va="center",
              fontsize=13, weight="bold", color="#555")

L.box("ret", 1, 0, "網膜")
L.box("goal", 3, 0, "上からの目的", fc=GRAY, note="出どころ未確認", note_color="#888")
L.box("vis", 1, 1, "視覚野各所 V2・V4・MT・IT\n面のまとまりと見た目", fc=BLUE, colspan=2)
L.box("proto", 1, 2, "原物体（中身の無い粗い塊）\n視野ぜんたいで同時にできる\n次の入力ですぐ上書きされる", fc=BLUE, colspan=2)
L.box("lip", 1, 3, "頭頂間溝 LIP　優先度の地図\n座標の地図・場所ごとに連続値\n負けた候補の値も残る", fc=ORG, colspan=2)
L.box("ior", 3, 3, "復帰抑制\nLIP・FEF・SC の3か所", fc=GRAY)
L.box("att", 1, 4, "注意が1か所へ向く", fc=ORG, colspan=2)
L.box("of", 1, 5, "中身の入った物体ファイル\nそこで初めて性質が結び付く・3〜5個", fc=ORG, colspan=2)
L.box("fef", 3, 5, "前頭眼野 FEF\n意思のサッケード", fc=YEL)
L.box("sc", 1, 6, "上丘 SC → 眼球を動かす筋", fc=BLUE, colspan=2)

L.arrow("ret", "vis", label="光")
L.arrow("vis", "proto", label="面")
L.arrow("proto", "lip", label="下からの目立ち")
L.arrow("goal", "lip", label="上からの目的", dashed=True, color="#888")
L.arrow("ior", "lip", label="直前の場所を下げる", color="#888", label_side=-0.95)
L.arrow("lip", "att", label="いちばん高い場所")
L.arrow("att", "of", label="そこの塊に中身が入る")
L.arrow("lip", "fef", label="地図を見て決める", label_side=1.15)
L.arrow("fef", "sc", label="意思の命令")

# ================= 右：太郎（実行の順に上から下。人間の段番号を添える） =================
R.box("img", 0, 0, "左目の画像 224×224", colspan=2)
R.box("pre", 0, 1, "前注意の地図　［人間の③］\n動いた塊だけ\n目立ちの地図は作るが使われない", fc=RED, colspan=2)
R.box("ref", 2, 1, "視線誘導反射　［人間の⑦］", fc=BLUE)
R.box("seg", 0, 2, "MobileSAM＋DINOv2　［人間の②］\n頼んだ点だけ切り出す", fc=BLUE, colspan=2)
R.box("eye", 2, 2, "眼球を動かす筋", fc=BLUE)
R.box("of", 0, 3, "物体ファイル　［人間の⑥］\n上限4枚\n注意より先に作られる", fc=RED, colspan=2)
R.box("pri", 0, 4, "優先度地図　［人間の④］\n物の一覧から選ぶ・座標の地図でない\n負けた候補は消える", fc=RED, colspan=2)
R.box("goal", 2, 4, "目的にあたるもの", fc=GRAY, note="無い")
R.box("att", 0, 5, "注意＝一覧から1枚　［人間の⑤］", fc=ORG, colspan=2)
R.box("out", 2, 5, "発話・単語学習へ", fc=GRAY)

R.arrow("img", "pre", label="前のコマとの差")
R.arrow("img", "ref", label="動きを自分で見つける", color="#888")
R.arrow("ref", "eye", label="サッケード", color="#888")
R.arrow("pre", "seg", label="動いた場所を見に行く")
R.arrow("seg", "of", label="切り出せたら記録を作る", color="#c1121f")
R.arrow("of", "pri", label="できた記録の一覧", color="#c1121f")
R.arrow("pri", "att", label="1位")
R.arrow("att", "out", label="注意した物", label_side=0.42)
R.arrow("goal", "pri", label="繋がっていない", dashed=True, color="#888", label_side=-0.95)

for ax, t in ((axes[0], "人間"), (axes[1], "太郎（今）")):
    ax.set_title(t, fontsize=20, weight="bold", pad=18)
axes[1].text(axes[1].get_xlim()[1] / 2, 0.10,
             "赤い箱＝人間と作りが違うところ。上から順に ③→②→⑥→④→⑤ で、"
             "記録（⑥）が優先度地図（④）と注意（⑤）より先にできている。目（⑦）は画像から直接で、注意とつながっていない。",
             ha="center", va="bottom", fontsize=13.5, color="#c1121f")
fig.suptitle("注意まわりの回路：人間と太郎（今）", fontsize=23, weight="bold")
fig.tight_layout(rect=[0, 0.02, 1, 0.955])
fig.savefig(out, dpi=110, bbox_inches="tight"); print("saved")
