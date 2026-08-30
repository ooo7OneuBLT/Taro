# -*- coding: utf-8 -*-
"""基本図形の物体を見せるシーンJSONを個体ごとに作る（F2-17）。

台にするのは F2-15 の 1イラスト・30cm・輻輳ON のシーン。
違いは2つだけ：
  world.xml    …… 個体ごとのXML（`f_gen_object_scenes.py` が作ったもの）
  toy.shape    …… "asis"（形を書き換えずXMLのまま使う。2026-08-30に追加した経路）

    .venv/Scripts/python.exe F/scripts/f_gen_object_scene_json.py
"""
import os
import sys
import io
import json

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")

DATE = "2026-08-30"
SRC = {"わんわん": "run/scenes/座位_12ヶ月_1イラスト_F2-15_30cm_わんわんだけ_2026-08-28.json",
       "ぶーぶー": "run/scenes/座位_12ヶ月_1イラスト_F2-15_30cm_ぶーぶーだけ_2026-08-28.json",
       "りんご": "run/scenes/座位_12ヶ月_1イラスト_F2-15_30cm_りんごだけ_2026-08-28.json",
       "くつ": "run/scenes/座位_12ヶ月_1イラスト_F2-15_30cm_くつだけ_2026-08-28.json"}
NOTE = ("2026-08-30 F2-17。板のイラストをやめ、基本図形で組んだ3Dの物体を見せる。"
        "同じカテゴリの別個体を用意して般化を測るため。素材の経緯は現在地.md参照。"
        "toy.shape='asis' で e_toy_env は形を書き換えずXMLのまま使う。")


def main():
    items = json.load(io.open("F/logs/_f217_scenes.json", encoding="utf-8"))
    made = []
    for it in items:
        cat, name, xml = it["cat"], it["name"], it["xml"]
        sc = json.load(io.open(SRC[cat], encoding="utf-8"))
        scene = "座位_12ヶ月_F2-17_%s_%s" % (name, DATE)
        sc["name"] = scene
        sc["note"] = NOTE
        sc["world"]["xml"] = xml
        sc["world"]["toy"]["shape"] = "asis"
        # 板ではなく立体なので、色の指定は使わない（XMLのrgbaが効く）
        sc["world"]["toy"].pop("rgba", None)
        p = "run/scenes/%s.json" % scene
        io.open(p, "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
        made.append((cat, name, scene))
    print("シーンJSON %d本" % len(made))
    for cat, name, scene in made:
        print("  %-8s %-8s %s" % (cat, name, scene))
    io.open("F/logs/_f217_scene_names.json", "w", encoding="utf-8").write(
        json.dumps([{"cat": c, "name": n, "scene": s} for c, n, s in made],
                   ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
