# -*- coding: utf-8 -*-
"""基本図形の物体（4語×5個体）を置いたシーンXMLを生成する（F2-17）。

`f_make_objects.py` が組み立てた図形の並びを、`benchmarkv2_scene_fillust.xml` の
`test_object1` の中身と差し替えたXMLを個体ごとに作る。
シーンJSONの `world.xml` でこれを指し、`toy.shape` を `"asis"` にすると、
e_toy_env が形を書き換えずXMLのまま使う（2026-08-30に追加した経路）。

    .venv/Scripts/python.exe F/scripts/f_gen_object_scenes.py
"""
import os
import sys
import io
import re
import json

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.path.abspath("F/scripts"))
from f_make_objects import CATS, dog, car, apple, shoe   # noqa: E402

BASE_XML = "MIMo/mimoEnv/assets/benchmarkv2_scene_fillust.xml"
# assets/ 直下に置く：サブフォルダだと <include file="mimo/..."> の相対パスが壊れる。
# ファイル名は英数字のみ：MuJoCoのXMLパスは日本語を開けない（実測）。
OUT_DIR = "MIMo/mimoEnv/assets"
ASCII = {"犬": "dog", "車": "car", "りんご": "apple", "くつ": "shoe"}
DATE = "2026-08-30"
# シーンJSONで使う語 → f_make_objects のカテゴリ名
WORD_OF = {"わんわん": "わんわん", "ぶーぶー": "ぶーぶー",
           "りんご": "りんご", "くつ": "くつ"}


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src = io.open(BASE_XML, encoding="utf-8").read()
    i, j = find_object_body(src)
    body = src[i:j]
    # 元の body から、freejoint と site（親が掴む点など）だけ残して geom を消す
    keep = []
    for m in re.finditer(r"<(freejoint|joint|site|inertial)\b[^>]*/?>", body):
        keep.append(m.group(0))
    head = re.match(r"<body[^>]*>", body).group(0)
    print("元の body:", head)
    print("残す要素:", keep if keep else "（なし）")

    made = []
    for cat, items, fn in CATS:
        for name, prm in items:
            geoms, lift = fn(prm)
            # 物体の中心が freejoint の原点に来るよう、生成した図形をそのまま入れる
            new_body = head + "".join(keep) + geoms + "</body>"
            out = src[:i] + new_body + src[j:]
            stem = name
            for ja, en in ASCII.items():
                stem = stem.replace(ja, en)
            path = os.path.join(OUT_DIR, "f_scene_%s.xml" % stem)
            io.open(path, "w", encoding="utf-8", newline="\n").write(out)
            made.append((cat, name, path, lift))
    print("\nXML %d本を %s に生成" % (len(made), OUT_DIR))
    for cat, name, path, lift in made:
        print("  %-8s %-8s lift=%.4f  %s" % (cat, name, lift, os.path.basename(path)))
    io.open("F/logs/_f217_scenes.json", "w", encoding="utf-8").write(
        json.dumps([{"cat": c, "name": n, "xml": p.replace("\\", "/"), "lift": l}
                    for c, n, p, l in made], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
