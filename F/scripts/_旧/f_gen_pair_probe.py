# -*- coding: utf-8 -*-
"""V1b の前提プローブ一式を作る：1つのシーンに3D物体を2つ（犬1＋くつ1）。

【何を確かめるか】接地の圧力を作るには語が入れ替わる世界が要る（F2-26の診断：
1本1語のブロック学習では視覚に予測上の価値が無く、変換層が定数に崩壊した）。
板の時代のABAB＝「2枚置くが提示は常に1枚ずつ（solo_presentation・親が
もう片方を隠す）」を3D物体で復活させる。その前提——
  ・2物体入りXMLが物理的に成立するか（発散しないか）
  ・親が交互に差し出し、太郎の注視が片方ずつに正しく乗るか
  ・中心窩に各物体がちゃんと映るか（数値でなく絵で確認する）
を、学習を回す前に1本のプローブで確かめる。

    .venv/Scripts/python.exe F/scripts/f_gen_pair_probe.py
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
from f_gen_10words import find_object_body   # noqa: E402

BASE_XML = "MIMo/mimoEnv/assets/benchmarkv2_scene_fillust.xml"
DATE = "2026-08-31"


def build_pair_xml(nameA, nameB, out_path):
    """test_object1=A・test_object2=B の2物体XMLを作る。"""
    src = io.open(BASE_XML, encoding="utf-8").read()

    def geoms_of(name):
        for cat, items, fn in CATS:
            for nm, prm in items:
                if nm == name:
                    g, _lift = fn(prm)
                    return g
        raise KeyError(name)

    def replace_body(src, body_name, geoms):
        i = src.index('<body name="%s"' % body_name)
        # find_object_body は test_object1 固定なので、同じ走査をここで行う
        depth = 0
        j = i
        while True:
            nb = src.find("<body", j + 1)
            ne = src.find("</body>", j + 1)
            if nb != -1 and nb < ne:
                depth += 1
                j = nb
            else:
                if depth == 0:
                    end = ne + len("</body>")
                    break
                depth -= 1
                j = ne
        body = src[i:end]
        head = re.match(r"<body[^>]*>", body).group(0)
        keep = [m.group(0) for m in
                re.finditer(r"<(freejoint|joint|site|inertial)\b[^>]*/?>", body)]
        return src[:i] + head + "".join(keep) + geoms + "</body>" + src[end:]

    src = replace_body(src, "test_object1", geoms_of(nameA))
    src = replace_body(src, "test_object2", geoms_of(nameB))
    io.open(out_path, "w", encoding="utf-8", newline="\n").write(src)
    return out_path


def main():
    xml = build_pair_xml("犬1", "くつ1", "MIMo/mimoEnv/assets/f20_pair_dog1_shoe1.xml")
    print("XML:", xml)

    # シーン：F2-23の犬1を台に、toy2を有効化してABAB（提示は1個ずつ）
    sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-23_分節_犬1_2026-08-30.json",
                           encoding="utf-8"))
    sname = "座位_12ヶ月_V1bプローブ_犬くつ_%s" % DATE
    sc["name"] = sname
    sc["note"] = ("2026-08-31 V1bプローブ。3D物体2つ（犬1・くつ1）のABAB提示。"
                  "親は太郎の注視を確認してから、見せている方の名前を言う。")
    sc["world"]["xml"] = xml
    sc["world"]["toy2"] = {"enabled": True, "shape": "asis", "radius": 0.04,
                           "dist": 0.2, "angle_deg": 0.0}
    pl = sc["world"]["parent_labeling"]
    pl["solo_presentation"] = True
    pl["utterances"] = {
        "toy1": [["わんわんだね", 1.0], ["わんわんだよ", 1.0], ["わんわんいるね", 1.0]],
        "toy2": [["くつだね", 1.0], ["くつだよ", 1.0]]}
    io.open("run/scenes/%s.json" % sname, "w", encoding="utf-8").write(
        json.dumps(sc, ensure_ascii=False, indent=1))
    print("シーン:", sname)

    # プローブ実験：V1b構成（視覚トークンON）・800ステップ・スナップショット多め
    ex = json.load(io.open("F/experiments/F2-26_視覚統合V1_1周目_わんわん_2026-08-31.json",
                           encoding="utf-8"))
    ex["name"] = "V1bプローブ_犬くつ"
    ex["note"] = "2026-08-31 V1bの前提プローブ。学習の本走行ではない。"
    ex["scene"] = sname
    ex["taro"]["model"] = "F/models/F2-6_P1_喃語900s_12ヶ月_seed10_2026-08-24.pt"
    ex["taro"]["save"] = "F/logs/V1bプローブ/model.pt"
    ex["run"]["steps"] = 800
    ex["run"]["checkpoint"] = 800
    ex["run"]["csv"] = "F/logs/V1bプローブ/run.csv"
    ex["plugins"] = {
        "word_production": {"events_out": "F/logs/V1bプローブ/発話イベント.csv"},
        "produce_snapshot": {"out_dir": "F/logs/V1bプローブ/中心窩",
                             "full_dir": "F/logs/V1bプローブ/両目",
                             "max_images": 30},
    }
    os.makedirs("F/logs/V1bプローブ", exist_ok=True)
    p = "F/experiments/V1bプローブ_犬くつ_%s.json" % DATE
    io.open(p, "w", encoding="utf-8").write(json.dumps(ex, ensure_ascii=False, indent=2))
    print("実験:", p)


if __name__ == "__main__":
    main()
