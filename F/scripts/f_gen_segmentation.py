# -*- coding: utf-8 -*-
"""F2-23：語の切れ目を自力で見つける（統計的分節）実験一式を作る。

【何を確かめるか】親が自然な混合のセリフ（「わんわんいた」「わんわんだよ」
「わんわん」）で話しかけたとき、太郎が**教わっていない語の切れ目**を、
予測の自信の谷から自力で見つけられるか（Saffran 1996 の統計的分節の再現）。

  合否：学習後の語彙（lexicon.counts）に「わんわん」と「いた」「だよ」が
        **別々の単位**として立っていること。
  対照：real_confidence=false（従来の1.0決め打ち）なら「わんわんいた」が
        丸ごと1語として立つはず。

【セリフの混合比の根拠】F/docs/二語文/文献調査/2026-08-31_日本語の親の語りかけ.md
  「名詞+動詞（助詞省略）」型と「名詞+だよ」型が典型・裸の名詞単独は18.4%
  （小椋・浜辺2021）。→ 「Xいた」0.42／「Xだよ」0.40／「X」0.18 とする。

【前提】⓪（脳の名簿を耳側に開く・白紙バグ修正）が2026-08-31に完了していること。

    .venv/Scripts/python.exe F/scripts/f_gen_segmentation.py
"""
import os
import sys
import io
import json

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")

DATE = "2026-08-31"
TAG = "F2-24_分節単独形"
ROUNDS = 2          # 【時短】3周→2周（ユーザー判断。リスクは設計に記載）
STEPS = 600         # 【時短】1000→600（1本14回聞く。プローブは18回で単位が立った）
START = "F/models/F2-6_P1_喃語900s_12ヶ月_seed10_2026-08-24.pt"
WORDS = ["わんわん", "にゃんにゃん", "ぶーぶー", "でんしゃ", "りんご",
         "ボール", "くつ", "ばなな", "コップ", "ぼうし"]


# 【2026-08-31・最終版＝ユーザー（母語話者）が作成】
#   経緯：私の作文が3回外れた（①全語「いた」＝有生性の誤り ②「あった」＝
#   発見の言葉で見せる場面に不適 ③「はいX」＝Fernaldの実例の引き写しだが
#   場面に合わないと母語話者判定）。最終的にユーザーが場面を見て自作：
#     生き物（犬・猫）: Xだね / Xだよ / Xいるね（有生性の手がかりが自然に入る）
#     物（8語）      : Xだね / Xだよ
#   単独形はユーザーのリストに無いため外した。
ANIMATE = {"わんわん", "にゃんにゃん"}


# 【F2-24・2026-08-31】単独形18%を追加。前回（F2-23）で物の語が頭2モーラに
#   砕けた（わん100回等）。入力に語の単独提示が無く、正しい右端を一度も
#   聞いていないのが原因と仮説。単独18.4%は実測値（小椋・浜辺2021）。
#   固まれば「孤立提示が語の型を固める」（乳児研究の知見）の再現になる。
SOLO = 0.18


def mixture(w):
    """親のセリフの混合（ユーザー作＋実測の単独形18%・2026-08-31）。"""
    if w in ANIMATE:
        base = [[w + "だね", 1.0], [w + "だよ", 1.0], [w + "いるね", 1.0]]
    else:
        base = [[w + "だね", 1.0], [w + "だよ", 1.0]]
    n = len(base)
    return [[t, (1.0 - SOLO) / n] for t, _ in base] + [[w, SOLO]]


def main():
    # 台：F2-21と同じ学習実験・シーン（変えるのはセリフと分節の2点だけ）
    chain_src = [l for l in io.open("F/logs/_f221_chain.txt", encoding="utf-8")
                 .read().splitlines() if l.strip()]
    made, prev = [], START
    for p in chain_src:
        if "3周目" in p and ROUNDS < 3:
            continue
        ex = json.load(io.open(p, encoding="utf-8"))
        ex["run"]["steps"] = STEPS
        ex["run"]["checkpoint"] = STEPS
        word = ex["name"].split("_")[-1]           # 例: F2-21_聞く学習_1周目_わんわん
        ex["name"] = ex["name"]                    # （TAG置換は下の行でまとめて行う）
        # シーンを複製してセリフを混合に差し替える
        sc = json.load(io.open("run/scenes/%s.json" % ex["scene"], encoding="utf-8"))
        sname = ex["scene"].replace("F2-20_10語", TAG)
        sc["name"] = sname
        sc["note"] = ("2026-08-31 F2-23。親のセリフを実測の混合（Xいた/Xだよ/X）に。"
                      "分節の自信度は本物（real_confidence）。")
        sc["world"]["parent_labeling"]["utterances"] = {
            "toy1": mixture(word), "toy2": mixture(word)}
        io.open("run/scenes/%s.json" % sname, "w", encoding="utf-8").write(
            json.dumps(sc, ensure_ascii=False, indent=1))

        ex["name"] = ex["name"].replace("F2-21_聞く学習", TAG)
        ex["note"] = "2026-08-31 F2-23。統計的分節の再現。" + ex.get("note", "")
        ex["scene"] = sname
        ex["taro"]["produce"]["real_confidence"] = True
        ex["taro"]["model"] = prev
        ex["taro"]["save"] = ex["taro"]["save"].replace("F2-21_聞く学習", TAG)
        ex["run"]["csv"] = ex["run"]["csv"].replace("F2-21_聞く学習", TAG)
        out = p.replace("F2-21_聞く学習", TAG)
        io.open(out, "w", encoding="utf-8").write(json.dumps(ex, ensure_ascii=False, indent=2))
        made.append(out)
        prev = ex["taro"]["save"]
    print("学習 %d本（最後: %s）" % (len(made), os.path.basename(prev)))
    io.open("F/logs/_f224_chain.txt", "w", encoding="utf-8",
            newline="\n").write("\n".join(made) + "\n")


if __name__ == "__main__":
    main()
