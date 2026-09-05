# -*- coding: utf-8 -*-
"""10語×5個体の実験一式を作る（F2-20）。XML→シーンJSON→実験ファイルまで一気通貫。

【何を確かめるか】F2-19（4語）で「複数個体で学習すると般化が2倍」まで確認した。
次は語数を10に増やし、**表引き（語ごとの見えの平均とのコサイン）がどこで壊れるか**を測る。
そのために、わざと似せた「そっくりペア」を3組仕込んである（f_make_objects.py 参照）：

  犬-猫         見えの近さ 0.864 ≒ カテゴリ内   → ほぼ区別不能のはず
  車-電車       0.619                            → 中間
  りんご-ボール  0.491                            → 区別できるはず

人間の12ヶ月児は猫を見て「わんわん」と言う（過大般化）。太郎が同じ間違いをするかが見どころ。

【学習量：F2-19 の 3000→1000ステップに削減（2026-08-30・ユーザー決定）】
F2-19 実測で1語204回聞いていたが、lexiconの学習は累積平均なので20〜30回で収束する。
1000ステップ＝太郎の体感50秒・1本あたり約22回聞く＝3個体で1語約66回。十分。
テストは1000ステップのまま（産出は±8ポイント揺れるので試行数は減らさない）。

【F2-19のXMLは上書きしない】りんごの倍率を 2.60→1.30 に直した（四辺見切れの修正）ため、
同じ名前で吐くとF2-19の再現が壊れる。F2-20用は f20_obj_*.xml / F2-20_* シーン名で分ける。

    .venv/Scripts/python.exe F/scripts/f_gen_10words.py
"""
import os
import sys
import io
import re
import json

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.path.abspath("F/scripts"))
from f_make_objects import CATS   # noqa: E402

DATE = "2026-08-30"
TAG = "F2-20_10語"
STEPS_TRAIN = 1000            # F2-19 は 3000。削減の根拠は冒頭コメント
STEPS_TEST = 1000
EYE = "left"                  # 認識に使う目（F2-18で片目化）
ACUITY_FILTER = False         # 視力フィルタなし（F2-18で実測の上で除去）
START = "F/models/F2-6_P1_喃語900s_12ヶ月_seed10_2026-08-24.pt"

BASE_XML = "MIMo/mimoEnv/assets/benchmarkv2_scene_fillust.xml"
OUT_DIR = "MIMo/mimoEnv/assets"
# ファイル名は英数字のみ（MuJoCoのXMLパスは日本語を開けない・実測）
# 「電車」を「車」より先に置換する（辞書の順序どおりに replace するため、
#   短い語が先だと「電車1」→「電car1」になる。実測で踏んだ）
ASCII = {"電車": "train", "犬": "dog", "猫": "neko", "車": "car",
         "りんご": "apple", "ボール": "ball", "くつ": "shoe",
         "ばなな": "banana", "コップ": "cup", "ぼうし": "hat"}
# 学習・テストのシーンJSONの台（F2-17と同じ。板→立体の差し替え済みの構成）
SRC_SCENE = "run/scenes/座位_12ヶ月_F2-17_犬1_2026-08-30.json"
# 実験ファイルの台（F2-15。run/taroの構成だけ借り、名前・シーン・モデルは差し替える）
SRC_TRAIN = "F/experiments/F2-15_1枚ずつ_1周目_わんわん_2026-08-28.json"
SRC_TEST = "F/experiments/F2-15_30cm_言わせる_わんわん_2026-08-28.json"


def find_object_body(src):
    """test_object1 の body 要素（開始タグ〜閉じタグ）の範囲を返す。"""
    i = src.index('<body name="test_object1"')
    depth = 0
    j = i
    while True:
        nb = src.find("<body", j + 1)
        ne = src.find("</body>", j + 1)
        if ne == -1:
            raise RuntimeError("test_object1 の閉じタグが見つからない")
        if nb != -1 and nb < ne:
            depth += 1
            j = nb
        else:
            if depth == 0:
                return i, ne + len("</body>")
            depth -= 1
            j = ne


def ascii_stem(name):
    for ja, en in ASCII.items():
        name = name.replace(ja, en)
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise RuntimeError("ASCII化できていない: " + name)
    return name


def main():
    # ---- ① XML 50本 --------------------------------------------------------
    src = io.open(BASE_XML, encoding="utf-8").read()
    i, j = find_object_body(src)
    body = src[i:j]
    keep = [m.group(0) for m in
            re.finditer(r"<(freejoint|joint|site|inertial)\b[^>]*/?>", body)]
    head = re.match(r"<body[^>]*>", body).group(0)
    xml_of = {}
    for cat, items, fn in CATS:
        for name, prm in items:
            geoms, _lift = fn(prm)
            new_body = head + "".join(keep) + geoms + "</body>"
            path = "%s/f20_obj_%s.xml" % (OUT_DIR, ascii_stem(name))
            io.open(path, "w", encoding="utf-8", newline="\n").write(
                src[:i] + new_body + src[j:])
            xml_of[name] = path
    print("XML %d本" % len(xml_of))

    # ---- ② シーンJSON 50本（学習=個体1-3・親が名づける／テスト=個体4-5・黙る）----
    base = json.load(io.open(SRC_SCENE, encoding="utf-8"))
    scene_of = {}
    for cat, items, fn in CATS:
        for k, (name, _prm) in enumerate(items):
            is_train = k < 3
            sc = json.loads(json.dumps(base))
            sname = "座位_12ヶ月_%s_%s_%s" % (TAG, name, DATE)
            sc["name"] = sname
            sc["note"] = ("2026-08-30 F2-20。10語×5個体。%s は%s用。"
                          "そっくりペア（犬-猫/車-電車/りんご-ボール）で表引きの限界を測る。"
                          % (name, "学習" if is_train else "テスト"))
            sc["world"]["xml"] = xml_of[name]
            sc["world"]["toy"]["shape"] = "asis"
            sc["world"]["toy"].pop("rgba", None)
            sc["world"]["toy2"]["enabled"] = False
            sc["body"]["acuity_filter"] = ACUITY_FILTER
            pl = sc["world"]["parent_labeling"]
            pl["respond_prob"] = 1.0 if is_train else 0.0
            pl["utterances"] = {"toy1": cat, "toy2": cat}
            io.open("run/scenes/%s.json" % sname, "w", encoding="utf-8").write(
                json.dumps(sc, ensure_ascii=False, indent=1))
            scene_of[name] = sname
    print("シーンJSON %d本" % len(scene_of))

    # ---- ③ 学習30本（鎖：1周目=個体1×10語 → 2周目=個体2 → 3周目=個体3） -------
    tmpl = json.load(io.open(SRC_TRAIN, encoding="utf-8"))
    made, prev = [], START
    for rd in (1, 2, 3):
        for cat, items, _fn in CATS:
            name = items[rd - 1][0]           # この周に見せる個体
            ex = json.loads(json.dumps(tmpl))
            ex["name"] = "%s_%d周目_%s" % (TAG, rd, cat)
            ex["note"] = ("2026-08-30 F2-20。10語学習の%d周目、%s を見せて「%s」を学ぶ。"
                          "1000ステップ（F2-19の3分の1）に削減、根拠は f_gen_10words.py。"
                          % (rd, name, cat))
            ex["scene"] = scene_of[name]
            ex["taro"]["lexicon_vision"] = dict(ex["taro"]["lexicon_vision"], eye=EYE)
            ex["taro"]["model"] = prev
            ex["taro"]["save"] = "F/models/%s_%d周目_%s_seed10_%s.pt" % (TAG, rd, cat, DATE)
            ex["run"]["steps"] = STEPS_TRAIN
            ex["run"]["checkpoint"] = STEPS_TRAIN
            ex["run"]["csv"] = "F/logs/%s_本学習/%d周目_%s/run.csv" % (TAG, rd, cat)
            p = "F/experiments/%s_%d周目_%s_%s.json" % (TAG, rd, cat, DATE)
            io.open(p, "w", encoding="utf-8").write(
                json.dumps(ex, ensure_ascii=False, indent=2))
            made.append(p)
            prev = ex["taro"]["save"]
    print("学習 %d本（最後: %s）" % (len(made), os.path.basename(prev)))

    # ---- ④ テスト20本（学習に使っていない個体4・5で言わせる） ------------------
    tmpl = json.load(io.open(SRC_TEST, encoding="utf-8"))
    tests = []
    for cat, items, _fn in CATS:
        for name, _prm in items[3:]:
            ex = json.loads(json.dumps(tmpl))
            ex["name"] = "%s_テスト_%s" % (TAG, name)
            ex["note"] = "2026-08-30 F2-20 テスト。学習に使っていない %s で言わせる。" % name
            ex["scene"] = scene_of[name]
            ex["taro"]["lexicon_vision"] = dict(ex["taro"]["lexicon_vision"], eye=EYE)
            ex["taro"]["model"] = prev
            ex["taro"]["save"] = "F/models/_%s_test_%s.pt" % (TAG, ascii_stem(name))
            ex["run"]["steps"] = STEPS_TEST
            ex["run"]["checkpoint"] = STEPS_TEST
            d = "F/logs/%s_テスト/%s" % (TAG, name)
            ex["run"]["csv"] = d + "/run.csv"
            ex["plugins"] = {
                "word_production": {"events_out": d + "/発話イベント.csv"},
                "produce_snapshot": {"out_dir": d + "/中心窩", "full_dir": d + "/両目",
                                     "max_images": 20},
            }
            os.makedirs(d, exist_ok=True)
            p = "F/experiments/%s_テスト_%s_%s.json" % (TAG, name, DATE)
            io.open(p, "w", encoding="utf-8").write(
                json.dumps(ex, ensure_ascii=False, indent=2))
            tests.append(p)
    print("テスト %d本" % len(tests))
    io.open("F/logs/_f220_chain.txt", "w", encoding="utf-8",
            newline="\n").write("\n".join(made) + "\n")
    io.open("F/logs/_f220_tests.txt", "w", encoding="utf-8",
            newline="\n").write("\n".join(tests) + "\n")


if __name__ == "__main__":
    main()
