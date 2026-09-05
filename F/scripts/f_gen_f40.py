# -*- coding: utf-8 -*-
"""F2-40：経験の多様性増強（訓練個体8体/語・毛の模様・親の連呼）の鎖を生成する。

  ラウンド r（8つ）＝各語の訓練個体 [1,2,3,6,7,8,9,10] の r 番目で10択XMLを作る。
  個体4・5（般化テスト用）は使わない。模様つき個体（犬6/8/10・猫7/10）は
  f_make_objects.pattern_assets_xml() のテクスチャを<asset>へ注入する。
  シーン＝F2-31の10択シーンに parent_labeling.repeat_labels=2（連呼）を足したもの。
  実験＝F2-38c（短命海馬）の設定で、モデル＋海馬を鎖で引き継ぐ。各3000歩・500歩ごと睡眠。

    .venv/Scripts/python.exe F/scripts/f_gen_f40.py
"""
import os, sys, io, re, json
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.path.abspath("F/scripts"))
import f_make_objects as M
from f_gen_10way import replace_body, BASE_XML, SLOT_BODIES, SLOT_KEYS, WORDS
from f_gen_segmentation import mixture

DATE = "2026-09-02"
ROUNDS = [0, 1, 2, 5, 6, 7, 8, 9]       # 個体1,2,3,6,7,8,9,10（index）
STEPS_PER_ROUND = 3000


def geoms_of_idx(cat_i, idx):
    cat, items, fn = M.CATS[cat_i]
    name, prm = items[idx]
    g, _ = fn(prm)
    return M.apply_pattern(g, name, prm), name


def build_xml(idx, out_path):
    src = io.open(BASE_XML, encoding="utf-8").read()
    names = []
    geoms = []
    for cat_i in range(len(M.CATS)):
        g, nm = geoms_of_idx(cat_i, idx)
        geoms.append(g); names.append(nm)
    src = replace_body(src, "test_object1", geoms[0])
    src = replace_body(src, "test_object2", geoms[1])
    add = []
    for k, g in enumerate(geoms[2:]):
        add.append('<body name="test_object%d" pos="%.1f 3.0 -2.0"><freejoint/>%s</body>'
                   % (5 + k, 5.0 + 0.5 * k, g))
    src = src.replace("</worldbody>", "".join(add) + "</worldbody>")
    src = src.replace("</asset>", M.pattern_assets_xml() + "</asset>", 1)   # 模様のテクスチャ
    io.open(out_path, "w", encoding="utf-8", newline="\n").write(src)
    return names


def main(probe_only=False):
    base_sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-31_10択_3周目_2026-08-31.json",
                                encoding="utf-8"))
    tmpl_ex = json.load(io.open("F/experiments/F2-38c_短命海馬_2026-09-01.json",
                                encoding="utf-8"))
    prev = "F/models/F2-38c_短命海馬_seed10_2026-09-01.pt"
    made = []
    for r, idx in enumerate(ROUNDS, start=1):
        xml = "MIMo/mimoEnv/assets/f40_10way_r%d.xml" % r
        names = build_xml(idx, xml)
        sc = json.loads(json.dumps(base_sc))
        sname = "座位_12ヶ月_F2-40_10択_r%d_個体%d_%s" % (r, idx + 1, DATE)
        sc["name"] = sname
        sc["note"] = ("%s F2-40 ラウンド%d（各語の個体%d）。訓練個体8体/語・毛の模様・"
                      "親の連呼(repeat_labels=2, 1秒おき・呼び戻し方式)。" % (DATE, r, idx + 1))
        sc["world"]["xml"] = xml
        sc["world"]["parent_labeling"]["repeat_labels"] = 2
        sc["world"]["parent_labeling"]["repeat_gap_sec"] = 1.0
        sc["world"]["parent_labeling"]["utterances"] = {k: mixture(w) for k, w in zip(SLOT_KEYS, WORDS)}
        io.open("run/scenes/%s.json" % sname, "w", encoding="utf-8").write(
            json.dumps(sc, ensure_ascii=False, indent=1))
        ex = json.loads(json.dumps(tmpl_ex))
        ex["name"] = "F2-40_r%d" % r
        ex["note"] = "%s F2-40 ラウンド%d/8。F2-38cからモデル＋海馬を鎖で引き継ぐ。" % (DATE, r)
        ex["scene"] = sname
        ex["taro"]["model"] = prev
        ex["taro"]["save"] = "F/models/F2-40_r%d_seed%d_%s.pt" % (r, 10 + r, DATE)
        # 【2026-09-02・中間測定で発覚】全ラウンド同一シードだと親の提示順（乱数列）が
        #   ラウンドごとに丸ごと同じになり、1ラウンドの偏り（犬3回vs猫6回）が8回繰り返された。
        #   ラウンドごとにシードを変えて提示の偏りを平均化する（別の日＝別の乱数）。
        ex["run"]["seed"] = 10 + r
        ex["run"]["steps"] = STEPS_PER_ROUND
        ex["run"]["checkpoint"] = 500
        ex["run"]["csv"] = "F/logs/F2-40_多様性増強/r%d/run.csv" % r
        ex["plugins"]["word_learning"]["events_out"] = "F/logs/F2-40_多様性増強/r%d/発話イベント.csv" % r
        ex["plugins"]["model_snapshots"] = {"out_dir": "F/logs/F2-40_多様性増強/r%d/snapshots" % r}
        p = "F/experiments/F2-40_r%d_%s.json" % (r, DATE)
        io.open(p, "w", encoding="utf-8").write(json.dumps(ex, ensure_ascii=False, indent=2))
        made.append(p)
        prev = ex["taro"]["save"]
        print("r%d: 個体%d %s" % (r, idx + 1, "・".join(names)))
    io.open("F/logs/_f240_chain.txt", "w", encoding="utf-8", newline="\n").write("\n".join(made) + "\n")
    # 下見（600歩・r1のシーン・保存は使い捨て）
    ex = json.load(io.open(made[0], encoding="utf-8"))
    ex["name"] = "F2-40probe"; ex["run"]["steps"] = 600; ex["run"]["checkpoint"] = 600
    ex["taro"]["save"] = "F/models/_F2-40probe_使い捨て.pt"
    ex["run"]["csv"] = "F/logs/F2-40_多様性増強/probe/run.csv"
    ex["plugins"] = {"word_learning": {"events_out": "F/logs/F2-40_多様性増強/probe/発話イベント.csv"},
                     "trace": {"out": "F/logs/F2-40_多様性増強/probe/trace.csv",
                               "dump_dir": "F/logs/F2-40_多様性増強/probe/dump",
                               "dump_keys": ["obs_out.eye_left"], "dump_until": 600}}
    io.open("F/experiments/F2-40probe_%s.json" % DATE, "w", encoding="utf-8").write(
        json.dumps(ex, ensure_ascii=False, indent=2))
    print("鎖:", len(made), "本 ＋ 下見1本")


if __name__ == "__main__":
    main()
