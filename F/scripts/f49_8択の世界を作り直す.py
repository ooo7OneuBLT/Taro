# -*- coding: utf-8 -*-
"""消えた8択の世界 `MIMo/mimoEnv/assets/f49_8way_r1〜r8.xml` を作り直す（2026-09-16）。

【なぜ要るか】2026-09-16、git worktree をジャンクションごと `--force` で消したとき、
`MIMo/mimoEnv/assets/` の自作XMLが全部消えた（落とし穴 項118）。目標Fの土台の場面
`run/scenes/土台_座位12ヶ月_壁あり_胡坐_8択_消失発話_2026-09-15.json` は
`f49_8way_r1.xml` を名指ししているので、**目標Fの実験が1本も走らない**。

【作り直せる根拠】消えたのはXMLだけで、素材と作り方は残っていた。
    メッシュ・テクスチャ  F/assets/gso/（47体）・F/assets/ycb/（6体）
    どの語にどれを使うか  F/assets/gso/_選定.json・F/assets/ycb/_選定.json
    寸法・提示角度・個体  F/scripts/_旧/f_gen_f49.py（2026-09-03。この本の元）

【元と違うところ・ここだけは戻らない】元の鎖は
    benchmarkv2_scene_fillust.xml →（f_gen_f40）→ f40_10way_r8.xml →（f_gen_f49）→ f49_8way_r1.xml
だったが、根の `benchmarkv2_scene_fillust.xml` は**手作りで生成コードが無く**、消えた。
そこで MIMo 標準の `benchmarkv2_scene.xml` を根に据え直した。中間の f40 が置いた
おもちゃの見た目（基本図形・毛の模様）は f49 が全部メッシュに差し替えるので、
最終のXMLに残るのは **`test_object2` の見た目だけ**。その枠は場面で
`toy2.enabled=false` にしてあり使わない（＝影響なし）。

壁・照明・床は失われていない：
  壁   run/world/toy_env.py:1375 の `walls` が走行時にコードから足す（場面の backdrop 設定）
  枠の位置  同 _hold_present_slots() が毎歩、親の手の位置へ動かす（XMLの初期位置は上書きされる）

【使い方】
    .venv/Scripts/python.exe F/scripts/f49_8択の世界を作り直す.py            # r1〜r8 を書く
    .venv/Scripts/python.exe F/scripts/f49_8択の世界を作り直す.py --確認だけ  # 書かずに素材の有無だけ見る
"""
import argparse
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
ROOT = os.getcwd().replace(os.sep, "/")

BASE_XML = "MIMo/mimoEnv/assets/benchmarkv2_scene.xml"   # MIMo 標準（GitHub から復旧済み）
OUT_FMT = "MIMo/mimoEnv/assets/f49_8way_r%d.xml"
TARGET_M = 0.15                      # 最長辺（乳児のおもちゃ15〜20cmの中央）。f_gen_f49 と同じ
ROUND_INDIV = [0, 1, 2, 0, 1, 2, 0, 1]   # ラウンドrで見せる訓練個体（_選定.json の index）
# 【般化テスト用・2026-09-16】_選定.json の先頭3体が訓練個体、4体目以降がテスト用
#   （消えた f_gen_f49.py の説明文どおり）。バスは3体しか無いので初見が作れない
#   ＝その語だけ訓練個体のまま置き、採点から外す（元の設計も「バスはテスト無し」）。
TEST_INDIV = [3, 4]                      # 個体4・個体5
TEST_FMT = "MIMo/mimoEnv/assets/f49_8way_test%d.xml"

# 語 →（素材の出どころ, _選定.json のキー）。並び順がそのまま枠に対応する
WORDS = [("くつ", "gso", "くつ"), ("コップ", "gso", "コップ"), ("おさら", "gso", "おさら"),
         ("おわん", "gso", "ボウル"), ("かばん", "gso", "かばん"),
         ("がおー", "gso", "きょうりゅう"), ("バス", "gso", "バス"), ("ボール", "ycb", "ボール")]
SLOT_BODIES = ["test_object1"] + ["test_object%d" % k for k in range(5, 5 + len(WORDS) - 1)]
ADD_SLOTS = ["test_object%d" % k for k in range(5, 13)]   # 根に足す枠（f_gen_10way と同じ8つ）

GSO = json.load(io.open("F/assets/gso/_選定.json", encoding="utf-8"))
YCB = json.load(io.open("F/assets/ycb/_選定.json", encoding="utf-8"))


def mesh_files(src, name):
    """OBJ とテクスチャの絶対パス。f_gen_f49.mesh_files と同じ規則。"""
    if src == "gso":
        d = "%s/F/assets/gso/%s" % (ROOT, name)
        tex = None
        tdir = d + "/materials/textures"
        if os.path.isdir(tdir):
            for f in sorted(os.listdir(tdir)):
                if f.lower().endswith(".png"):
                    tex = tdir + "/" + f
                    break
        return d + "/meshes/model.obj", tex
    d = "%s/F/assets/ycb/%s/google_16k" % (ROOT, name)
    return d + "/textured.obj", d + "/texture_map.png"


def prep(src, name):
    """最長辺を TARGET_M に、中心を原点に。(obj, tex, scale, center) を返す。"""
    import trimesh
    obj, tex = mesh_files(src, name)
    m = trimesh.load(obj, force="mesh", process=False)
    scale = TARGET_M / float(max(m.bounding_box.extents))
    return obj, tex, scale, m.bounding_box.centroid * scale


def asset_and_geom(tag, src, name):
    """<asset> に足す断片と、<body> に入れる geom 断片。

    contype/conaffinity=0 ＝ぶつからない（親が手で持って差し出す前提。f_gen_f49 と同じ）。
    """
    obj, tex, scale, center = prep(src, name)
    asset = '<mesh name="mesh_%s" file="%s" scale="%.6f %.6f %.6f"/>' % (tag, obj, scale, scale, scale)
    if tex:
        asset += ('<texture name="tex_%s" type="2d" file="%s"/>'
                  '<material name="mat_%s" texture="tex_%s"/>' % (tag, tex, tag, tag))
        mat = 'material="mat_%s"' % tag
    else:
        mat = 'rgba=".7 .7 .7 1"'
    cx, cy, cz = (-float(v) for v in center)
    geom = ('<geom type="mesh" mesh="mesh_%s" pos="%.5f %.5f %.5f" %s contype="0" conaffinity="0"/>'
            % (tag, cx, cy, cz, mat))
    return asset, geom


def replace_body(src, body_name, geoms):
    """<body name=...> の中の geom だけを差し替える（freejoint・site・inertial は残す）。

    f_gen_10way.replace_body をそのまま写した。入れ子の </body> を数えて端を見つける。
    """
    i = src.index('<body name="%s"' % body_name)
    depth, j = 0, i
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
    keep = [m.group(0) for m in re.finditer(r"<(freejoint|joint|site|inertial)\b[^>]*/?>", body)]
    return src[:i] + head + "".join(keep) + geoms + "</body>" + src[end:]


def remove_body(src, body_name):
    """<body name=...> … </body> を丸ごと消す（差し出し枠は入れ子を持たない）。"""
    i = src.find('<body name="%s"' % body_name)
    if i < 0:
        return src
    j = src.index("</body>", i) + len("</body>")
    return src[:i] + src[j:]


def 根を作る():
    """MIMo標準のXMLに、差し出し枠 test_object5〜12 を足したものを返す。

    位置 `5.0+0.5k 3.0 -2.0` は f_gen_10way / f_gen_f40 と同じ「遠方の地下」。
    走行中は _hold_present_slots() が毎歩この枠を親の手の位置へ動かすので、
    ここに書く初期位置は「太郎の視界に映り込まない場所」でありさえすればよい。
    中身の geom は仮（この後 f49 が全部メッシュへ差し替え、余った枠は消す）。
    """
    src = io.open(BASE_XML, encoding="utf-8").read()
    add = ['<body name="%s" pos="%.1f 3.0 -2.0"><freejoint/>'
           '<geom type="box" material="matgeom" size="0.05 0.05 0.05"/></body>'
           % (b, 5.0 + 0.5 * k) for k, b in enumerate(ADD_SLOTS)]
    return src.replace("</worldbody>", "".join(add) + "</worldbody>")


def 個体名(idx):
    """ラウンドの個体 index から、語ごとの素材名を並べて返す。"""
    return [(GSO if s == "gso" else YCB)[k][idx] for _w, s, k in WORDS]


def build_xml(idx, out_path):
    """訓練個体 idx を8枠に差し込んだ世界XMLを書き、使った素材名を返す。"""
    src = 根を作る()
    assets, names = [], []
    for k, (_word, s, key) in enumerate(WORDS):
        name = (GSO if s == "gso" else YCB)[key][idx]
        a, g = asset_and_geom("w%d" % k, s, name)
        assets.append(a)
        names.append(name)
        src = replace_body(src, SLOT_BODIES[k], g)
    src = src.replace("</asset>", "".join(assets) + "</asset>", 1)
    for b in ADD_SLOTS:                       # 使わない枠は世界から消す（関節数を合わせるため）
        if b not in SLOT_BODIES:
            src = remove_body(src, b)
    io.open(out_path, "w", encoding="utf-8", newline="\n").write(src)
    return names


def build_test_xml(idx, out_path):
    """初見の個体 idx で世界XMLを書く。素材が足りない語は訓練個体のまま置く。

    戻り値は (使った素材名の並び, 初見にできた語の並び)。
    **採点は「初見にできた語」だけを対象にすること**（残りは見たことのある個体）。
    """
    src = 根を作る()
    assets, names, 初見 = [], [], []
    for k, (word, s, key) in enumerate(WORDS):
        一覧 = (GSO if s == "gso" else YCB)[key]
        if idx < len(一覧):
            name = 一覧[idx]
            初見.append(word)
        else:
            name = 一覧[0]              # 素材が足りない（バス）＝訓練個体のまま
        a, g = asset_and_geom("w%d" % k, s, name)
        assets.append(a)
        names.append(name)
        src = replace_body(src, SLOT_BODIES[k], g)
    src = src.replace("</asset>", "".join(assets) + "</asset>", 1)
    for b in ADD_SLOTS:
        if b not in SLOT_BODIES:
            src = remove_body(src, b)
    io.open(out_path, "w", encoding="utf-8", newline="\n").write(src)
    return names, 初見


def 素材を確かめる():
    """8語×使う個体のファイルが全部あるか。無ければ名前を並べて False を返す。"""
    欠け = []
    for idx in sorted(set(ROUND_INDIV)):
        for (word, s, key), name in zip(WORDS, 個体名(idx)):
            obj, tex = mesh_files(s, name)
            if not os.path.isfile(obj):
                欠け.append("%s 個体%d %s の OBJ" % (word, idx + 1, name))
            if not tex or not os.path.isfile(tex):
                欠け.append("%s 個体%d %s のテクスチャ" % (word, idx + 1, name))
    for x in 欠け:
        print("  無い: %s" % x)
    return not 欠け


def main():
    ap = argparse.ArgumentParser(description="消えた f49_8way_r1〜r8.xml を作り直す")
    ap.add_argument("--確認だけ", action="store_true", help="書かずに素材の有無だけ見る")
    a = ap.parse_args()
    if not os.path.isfile(BASE_XML):
        raise SystemExit("根のXMLがありません: %s" % BASE_XML)
    print("根: %s" % BASE_XML)
    if not 素材を確かめる():
        raise SystemExit("素材が足りないので作れません")
    print("素材は8語×個体1〜3ぶん揃っている")
    if a.確認だけ:
        return 0
    for r, idx in enumerate(ROUND_INDIV, start=1):
        out = OUT_FMT % r
        names = build_xml(idx, out)
        print("r%d（個体%d）: %s" % (r, idx + 1, "・".join(n[:20] for n in names)))
    print("\n%d本 書いた: %s" % (len(ROUND_INDIV), OUT_FMT % 1))
    # --- 般化テスト用（初見の個体）-------------------------------------------
    表 = {}
    for idx in TEST_INDIV:
        out = TEST_FMT % (idx + 1)
        names, 初見 = build_test_xml(idx, out)
        表["個体%d" % (idx + 1)] = {
            "xml": out,
            "素材": dict(zip([w for w, _s, _k in WORDS], names)),
            "初見の語": 初見}
        訓 = [w for w, _s, _k in WORDS if w not in 初見]
        print("テスト個体%d: 初見 %d 語（%s）%s"
              % (idx + 1, len(初見), "・".join(初見),
                 "／※ %s は素材が足りず訓練個体のまま＝採点から外す" % "・".join(訓) if 訓 else ""))
    io.open("F/assets/_般化テストの素材.json", "w", encoding="utf-8").write(
        json.dumps(表, ensure_ascii=False, indent=1))
    print("対応表: F/assets/_般化テストの素材.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
