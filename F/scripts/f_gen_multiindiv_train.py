# -*- coding: utf-8 -*-
"""複数個体で学習する実験一式を作る（F2-19）。

【何を確かめるか】いまの太郎は「1枚の絵」だけで語を学んでいるため、
初めて見る個体への判断が紙一重になっている（実測：学習した絵では2位を
+0.42〜0.70 引き離すのに、初見の絵では +0.12〜0.16 しかない）。
**複数の個体で学習すれば、その余裕は広がるのか**を測る。

  学習   12本の鎖。本ごとに違う個体を見せる（1周目=個体1／2周目=個体2／3周目=個体3）
  テスト 学習に使っていない個体（4・5）で言わせる
  判定   2位との差が +0.3 以上に広がれば「複数の見えが般化を作る」

【視覚の設定】2026-08-30に実測で決めたもの
  視力フィルタ  なし（あり/なしで分離が 0.542 対 0.536＝差は誤差）
  中心窩の解像度 336px と 112px で分離が変わらないので、視野5度なら112pxで足りる
                 ただし今回は中心窩15度のまま（物体が中心窩に収まる大きさにしてあるため）
  認識に使う目  片目（測定の結果を見て決める。既定は both のまま）

    .venv/Scripts/python.exe F/scripts/f_gen_multiindiv_train.py
"""
import os
import sys
import io
import json

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")

DATE = "2026-08-30"
TAG = "F2-19_複数個体"
EYE = "left"                      # 認識に使う目（片目比較の結果で決める）
ACUITY_FILTER = False             # 視力フィルタ（2026-08-30に外す方針）

# 語 → (学習に使う個体, テストに使う個体)
PLAN = {
    "わんわん": (["犬1", "犬2", "犬3"], ["犬4", "犬5"]),
    "ぶーぶー": (["車1", "車2", "車3"], ["車4", "車5"]),
    "りんご": (["りんご1", "りんご2", "りんご3"], ["りんご4", "りんご5"]),
    "くつ": (["くつ1", "くつ2", "くつ3"], ["くつ4", "くつ5"]),
}
# ABAB の A/B がどの2語を見せるか（既存の構成と同じ）
PAIR = {"A": ("わんわん", "ぶーぶー"), "B": ("りんご", "くつ")}
START = "F/models/F2-6_P1_喃語900s_12ヶ月_seed10_2026-08-24.pt"


def scene_name(ind):
    return "座位_12ヶ月_F2-17_%s_%s" % (ind, "2026-08-30")


def make_pair_scene(word1, ind1, word2, ind2, name, note):
    """2個体を同時に見せるシーン。個体1のシーンを台にして、2つ目を足す。"""
    sc = json.load(io.open("run/scenes/%s.json" % scene_name(ind1), encoding="utf-8"))
    sc["name"] = name
    sc["note"] = note
    sc["body"]["acuity_filter"] = ACUITY_FILTER
    # 2個目は同じXMLに入っていないので、1個体ずつ見せる方式にする
    sc["world"]["toy2"]["enabled"] = False
    # 【重要】台にした F2-17 の物体シーンは産出テスト用で respond_prob=0（親が
    #   名前を言わない）。学習用は 1.0 にしないと語彙イベントがゼロになる
    #   （2026-08-30 の起動前チェックで実際に0件を検出して発覚）。
    sc["world"]["parent_labeling"]["respond_prob"] = 1.0
    sc["world"]["parent_labeling"]["utterances"] = {"toy1": word1, "toy2": word1}
    io.open("run/scenes/%s.json" % name, "w", encoding="utf-8").write(
        json.dumps(sc, ensure_ascii=False, indent=1))
    return name


def main():
    made, prev = [], START
    # 1周目→3周目で個体を入れ替える。1本 = 1語 × 1個体（1イラスト方式）
    for rd in (1, 2, 3):
        for word in ("わんわん", "ぶーぶー", "りんご", "くつ"):
            ind = PLAN[word][0][rd - 1]
            sname = "座位_12ヶ月_%s_%s_%s" % (TAG, ind, DATE)
            make_pair_scene(word, ind, word, ind, sname,
                            "2026-08-30 F2-19。%s（%d周目）を見せて「%s」を学ぶ。"
                            "視力フィルタなし。" % (ind, rd, word))
            ex = json.load(io.open(
                "F/experiments/F2-15_1枚ずつ_%d周目_%s_2026-08-28.json" % (rd, word),
                encoding="utf-8"))
            ex["name"] = "%s_%d周目_%s" % (TAG, rd, word)
            ex["note"] = ("2026-08-30 F2-19。複数個体で学習して般化が強まるかを見る。"
                          "%d周目は %s を見せる。" % (rd, ind))
            ex["scene"] = sname
            ex["taro"]["lexicon_vision"] = dict(ex["taro"]["lexicon_vision"], eye=EYE)
            ex["taro"]["model"] = prev
            ex["taro"]["save"] = "F/models/%s_%d周目_%s_seed10_%s.pt" % (TAG, rd, word, DATE)
            ex["run"]["csv"] = "F/logs/%s_本学習/%d周目_%s/run.csv" % (TAG, rd, word)
            p = "F/experiments/%s_%d周目_%s_%s.json" % (TAG, rd, word, DATE)
            io.open(p, "w", encoding="utf-8").write(json.dumps(ex, ensure_ascii=False, indent=2))
            made.append(p)
            prev = ex["taro"]["save"]
    print("学習 %d本（最後: %s）" % (len(made), os.path.basename(prev)))

    # テスト：学習に使っていない個体で言わせる
    tests = []
    for word, (_tr, te) in PLAN.items():
        for ind in te:
            sname = "座位_12ヶ月_%s_テスト_%s_%s" % (TAG, ind, DATE)
            sc = json.load(io.open("run/scenes/%s.json" % scene_name(ind), encoding="utf-8"))
            sc["name"] = sname
            sc["note"] = "2026-08-30 F2-19 テスト。%s は学習に使っていない個体。" % ind
            sc["body"]["acuity_filter"] = ACUITY_FILTER
            sc["world"]["parent_labeling"]["respond_prob"] = 0.0
            io.open("run/scenes/%s.json" % sname, "w", encoding="utf-8").write(
                json.dumps(sc, ensure_ascii=False, indent=1))
            ex = json.load(io.open(
                "F/experiments/F2-15_30cm_言わせる_%s_2026-08-28.json" % word,
                encoding="utf-8"))
            ex["name"] = "%s_テスト_%s" % (TAG, ind)
            ex["note"] = "2026-08-30 F2-19 テスト。学習に使っていない %s で言わせる。" % ind
            ex["scene"] = sname
            ex["taro"]["lexicon_vision"] = dict(ex["taro"]["lexicon_vision"], eye=EYE)
            ex["taro"]["model"] = prev
            ex["taro"]["save"] = "F/models/_%s_test_%s.pt" % (TAG, ind)
            ex["run"]["steps"] = 1000
            ex["run"]["checkpoint"] = 1000
            d = "F/logs/%s_テスト/%s" % (TAG, ind)
            ex["run"]["csv"] = d + "/run.csv"
            ex["plugins"] = {
                "word_production": {"events_out": d + "/発話イベント.csv"},
                "produce_snapshot": {"out_dir": d + "/中心窩", "full_dir": d + "/両目",
                                     "max_images": 20},
            }
            os.makedirs(d, exist_ok=True)
            p = "F/experiments/%s_テスト_%s_%s.json" % (TAG, ind, DATE)
            io.open(p, "w", encoding="utf-8").write(json.dumps(ex, ensure_ascii=False, indent=2))
            tests.append(p)
    print("テスト %d本" % len(tests))
    io.open("F/logs/_f219_chain.txt", "w", encoding="utf-8",
            newline="\n").write("\n".join(made) + "\n")
    io.open("F/logs/_f219_tests.txt", "w", encoding="utf-8",
            newline="\n").write("\n".join(tests) + "\n")


if __name__ == "__main__":
    main()
