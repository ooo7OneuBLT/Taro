# -*- coding: utf-8 -*-
"""10択の世界を作る（V1c）：1シーンに10物体、親がランダムに1個ずつ見せて名づける。

【なぜ10択か】F2-26の診断：1本1語のブロック学習では文脈だけで次の語が当たり、
視覚に予測上の価値が無く、視覚トークンが定数に崩壊した（近道学習）。
選択肢が多いほど文脈で当てられず、視覚が唯一の手がかりになる＝接地の圧力が最大。
2択でなく10択にするのはユーザー判断（4択案は工学的保身のみで科学的理由なし）。

構成：
  XML         test_object1/2（既存body）＋test_object5〜12（新設・freejoint）に
              10語の個体を1つずつ。周ごとに個体セットを替える（1周目=個体1、2周目=個体2）
  シーン      toy1/toy2=asis有効・present_slots=[test_object5..12]・solo提示・
              follow差し出し・セリフはユーザー作の混合（f_gen_segmentation.mixture）
  実験        視覚トークンON（vision_context）・聞く学習ON・λ0・600ステップ

    .venv/Scripts/python.exe F/scripts/f_gen_10way.py
"""
import os
import sys
import io
import re
import json

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.path.abspath("F/scripts"))
from f_make_objects import CATS               # noqa: E402
from f_gen_segmentation import mixture        # noqa: E402（ユーザー作セリフ＋単独形18%）

BASE_XML = "MIMo/mimoEnv/assets/benchmarkv2_scene_fillust.xml"
DATE = "2026-08-31"
WORDS = [c for c, _, _ in CATS]               # 10語（定義順）
SLOT_BODIES = (["test_object1", "test_object2"]
               + ["test_object%d" % k for k in range(5, 13)])
SLOT_KEYS = ["toy1", "toy2"] + ["toy%d" % k for k in range(5, 13)]


def geoms_of(name):
    for cat, items, fn in CATS:
        for nm, prm in items:
            if nm == name:
                g, _ = fn(prm)
                return g
    raise KeyError(name)


def replace_body(src, body_name, geoms):
    i = src.index('<body name="%s"' % body_name)
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


def build_xml(round_idx, out_path):
    """round_idx周目の個体（items[round_idx-1]）で10物体XMLを作る。"""
    src = io.open(BASE_XML, encoding="utf-8").read()
    inds = []
    for cat, items, fn in CATS:
        inds.append(items[round_idx - 1][0])
    # 既存body 2つを置換
    src = replace_body(src, "test_object1", geoms_of(inds[0]))
    src = replace_body(src, "test_object2", geoms_of(inds[1]))
    # 新設body 8つを </worldbody> の直前に足す（freejoint・初期位置は遠方地下）
    add = []
    for k, ind in enumerate(inds[2:]):
        add.append('<body name="test_object%d" pos="%.1f 3.0 -2.0"><freejoint/>%s</body>'
                   % (5 + k, 5.0 + 0.5 * k, geoms_of(ind)))
    src = src.replace("</worldbody>", "".join(add) + "</worldbody>")
    io.open(out_path, "w", encoding="utf-8", newline="\n").write(src)
    return inds


def main():
    made_ex, prev = [], "F/models/F2-6_P1_喃語900s_12ヶ月_seed10_2026-08-24.pt"
    base_sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-23_分節_犬1_2026-08-30.json",
                                encoding="utf-8"))
    tmpl_ex = json.load(io.open("F/experiments/F2-26_視覚統合V1_1周目_わんわん_2026-08-31.json",
                                encoding="utf-8"))
    for rd in (1, 2):
        xml = "MIMo/mimoEnv/assets/f20_10way_r%d.xml" % rd
        inds = build_xml(rd, xml)
        sc = json.loads(json.dumps(base_sc))
        sname = "座位_12ヶ月_F2-27_10択_%d周目_%s" % (rd, DATE)
        sc["name"] = sname
        sc["note"] = ("2026-08-31 F2-27。10択の世界（%d周目の個体）。親がランダムに"
                      "1個ずつ見せて名づける。" % rd)
        sc["world"]["xml"] = xml
        sc["world"]["toy"]["shape"] = "asis"
        sc["world"]["toy"].pop("rgba", None)
        sc["world"]["toy2"] = {"enabled": True, "shape": "asis", "radius": 0.04,
                               "dist": 0.2, "angle_deg": 0.0}
        sc["world"]["present_slots"] = SLOT_BODIES[2:]
        pl = sc["world"]["parent_labeling"]
        pl["solo_presentation"] = True
        pl["utterances"] = {k: mixture(w) for k, w in zip(SLOT_KEYS, WORDS)}
        io.open("run/scenes/%s.json" % sname, "w", encoding="utf-8").write(
            json.dumps(sc, ensure_ascii=False, indent=1))

        ex = json.loads(json.dumps(tmpl_ex))
        ex["name"] = "F2-27_10択_%d周目" % rd
        ex["note"] = "2026-08-31 F2-27。10択×視覚トークン。接地の圧力の本命条件。"
        ex["scene"] = sname
        ex["taro"]["model"] = prev
        ex["taro"]["save"] = "F/models/F2-27_10択_%d周目_seed10_%s.pt" % (rd, DATE)
        # 10語が1本に混ざるので、1本の長さを10語ぶんに（600×10=6000ステップ）
        ex["run"]["steps"] = 6000
        ex["run"]["checkpoint"] = 6000
        ex["run"]["csv"] = "F/logs/F2-27_10択/%d周目/run.csv" % rd
        p = "F/experiments/F2-27_10択_%d周目_%s.json" % (rd, DATE)
        io.open(p, "w", encoding="utf-8").write(json.dumps(ex, ensure_ascii=False, indent=2))
        made_ex.append(p)
        prev = ex["taro"]["save"]
        print("%d周目: XML=%s 個体=%s" % (rd, os.path.basename(xml), "・".join(inds)))
    io.open("F/logs/_f227_chain.txt", "w", encoding="utf-8",
            newline="\n").write("\n".join(made_ex) + "\n")
    print("実験:", len(made_ex), "本（各6000ステップ＝従来の10本ぶんを1本に）")


if __name__ == "__main__":
    main()
