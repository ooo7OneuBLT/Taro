# -*- coding: utf-8 -*-
"""F2-49：実物スキャン素材（Google Scanned Objects / YCB）で8語を学ぶ鎖を生成する（2026-09-03）。

語（ユーザー承認 2026-09-03）：くつ・コップ・おさら・おわん(旧ボウル)・かばん・がおー(旧きょうりゅう)・バス・ボール の8語
  （つみきは素材が「散らばったブロック」にしか見えず外した・2026-09-03）
  ・各語の素材は _選定.json の順で、先頭3体＝訓練個体、4体目以降＝般化テスト用（バスは3体のみ→テスト無し）
  ・8ラウンド＝訓練個体 [1,2,3,1,2,3,1,2] を順に見せる（F2-40の「ラウンドごとに別個体」と同じ考え）
  ・シーン＝F2-44 r8（目224・回転なし・一巡・連呼）の9択配置（toy1＋toy5〜12）をそのまま使い、
    基本図形の物体を実物スキャンのメッシュ（テクスチャ付き・最長辺0.15m）に差し替える
  ・実験＝F2-48 r8 の設定（短命海馬・海馬即答・listen_self=false）で、起点はF2-6（語の学習前）
  ・メッシュの寸法合わせ・中心合わせは f_gso_gallery.prep と同じ（元OBJ直読み・<mesh scale>・geom pos）

    .venv/Scripts/python.exe F/scripts/f_gen_f49.py          # 鎖8本＋下見1本（視界動画付き）を生成
"""
import os, sys, io, json
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.path.abspath("F/scripts"))
import trimesh
from f_gen_10way import replace_body
from f_gen_segmentation import mixture, ANIMATE, SOLO

DATE = "2026-09-03"
ROOT = os.getcwd().replace(os.sep, "/")
TARGET_M = 0.15                       # 最長辺（乳児のおもちゃ15〜20cmの中央）
BASE_SCENE = "run/scenes/座位_12ヶ月_F2-44_9択目224_r8_個体10_2026-09-02.json"
BASE_EX = "F/experiments/F2-48_r8_2026-09-02.json"
START_MODEL = "F/models/F2-6_P1_喃語900s_12ヶ月_seed10_2026-08-24.pt"
STEPS_PER_ROUND = 3000
ROUND_INDIV = [0, 1, 2, 0, 1, 2, 0, 1]   # 各ラウンドで見せる訓練個体（index）
N_TRAIN = 3

# 語 → (素材の出どころ, _選定.json のキー)。順番が枠（toy1, toy5..toy12）に対応する
WORDS = [("くつ", "gso", "くつ"), ("コップ", "gso", "コップ"), ("おさら", "gso", "おさら"),
         ("おわん", "gso", "ボウル"), ("かばん", "gso", "かばん"),
         ("がおー", "gso", "きょうりゅう"), ("バス", "gso", "バス"), ("ボール", "ycb", "ボール")]
SLOT_BODIES = ["test_object1"] + ["test_object%d" % k for k in range(5, 5 + len(WORDS) - 1)]
SLOT_KEYS = ["toy1"] + ["toy%d" % k for k in range(5, 5 + len(WORDS) - 1)]
# 物ごとの見せ方（本番環境の静止測定 図_提示角度_案.png・2026-09-03）。傾きは負＝上面を太郎へ
PRESENT_ANGLES = {"くつ": {"yaw": 0, "tilt": -20}, "コップ": {"yaw": 60, "tilt": -25},
                  "おさら": {"yaw": 30, "tilt": -60}, "おわん": {"yaw": 30, "tilt": -55},
                  "かばん": {"yaw": 30, "tilt": -20}, "がおー": {"yaw": 0, "tilt": -10},
                  "バス": {"yaw": -30, "tilt": -15}, "ボール": {"yaw": 30, "tilt": -25}}
GSO = json.load(io.open("F/assets/gso/_選定.json", encoding="utf-8"))
YCB = json.load(io.open("F/assets/ycb/_選定.json", encoding="utf-8"))


def individuals(src, key):
    """語の素材名リスト（_選定.json の順）。"""
    return (GSO if src == "gso" else YCB)[key]


def mesh_files(src, name):
    """OBJ とテクスチャの絶対パス。"""
    if src == "gso":
        d = "%s/F/assets/gso/%s" % (ROOT, name)
        obj = d + "/meshes/model.obj"
        tex = None
        tdir = d + "/materials/textures"
        if os.path.isdir(tdir):
            for f in sorted(os.listdir(tdir)):
                if f.lower().endswith(".png"):
                    tex = tdir + "/" + f; break
        return obj, tex
    d = "%s/F/assets/ycb/%s/google_16k" % (ROOT, name)
    return d + "/textured.obj", d + "/texture_map.png"


def prep(src, name):
    """最長辺を TARGET_M に、中心を原点に。(obj, tex, scale, center) を返す。"""
    obj, tex = mesh_files(src, name)
    m = trimesh.load(obj, force="mesh", process=False)
    ext = m.bounding_box.extents
    scale = TARGET_M / float(max(ext))
    center = m.bounding_box.centroid * scale
    return obj, tex, scale, center


def asset_and_geom(tag, src, name):
    """<asset> に足す断片と、<body> に入れる geom 断片。"""
    obj, tex, scale, center = prep(src, name)
    asset = '<mesh name="mesh_%s" file="%s" scale="%.6f %.6f %.6f"/>' % (tag, obj, scale, scale, scale)
    if tex:
        asset += '<texture name="tex_%s" type="2d" file="%s"/><material name="mat_%s" texture="tex_%s"/>' % (tag, tex, tag, tag)
        mat = 'material="mat_%s"' % tag
    else:
        mat = 'rgba=".7 .7 .7 1"'
    cx, cy, cz = (-float(v) for v in center)
    geom = ('<geom type="mesh" mesh="mesh_%s" pos="%.5f %.5f %.5f" %s contype="0" conaffinity="0"/>'
            % (tag, cx, cy, cz, mat))
    return asset, geom


def build_xml(base_xml, idx, out_path):
    """各語の訓練個体 idx を8枠に差し込んだ世界XMLを書く。"""
    src = io.open(base_xml, encoding="utf-8").read()
    assets, names = [], []
    for k, (word, s, key) in enumerate(WORDS):
        name = individuals(s, key)[idx]
        a, g = asset_and_geom("w%d" % k, s, name)
        assets.append(a); names.append(name)
        src = replace_body(src, SLOT_BODIES[k], g)
    src = src.replace("</asset>", "".join(assets) + "</asset>", 1)
    # 使わない枠の本体は世界から消す（保存姿勢114値＋枠×7＝関節数、の照合を通すため・2026-09-03）
    for k in range(5, 13):
        b = "test_object%d" % k
        if b not in SLOT_BODIES:
            src = remove_body(src, b)
    io.open(out_path, "w", encoding="utf-8", newline="\n").write(src)
    return names


def remove_body(src, body_name):
    """<body name=...> … </body> を丸ごと消す（入れ子の body が無い前提＝差し出し枠はそう）。"""
    i = src.find('<body name="%s"' % body_name)
    if i < 0:
        return src
    j = src.index("</body>", i) + len("</body>")
    return src[:i] + src[j:]


def utter(word):
    """親のセリフの混合。がおー（生き物）は「いるね」枠も持つ。"""
    if word == "がおー":
        base = [[word + "だね", 1.0], [word + "だよ", 1.0], [word + "いるね", 1.0]]
        n = len(base)
        return [[t, (1.0 - SOLO) / n] for t, _ in base] + [[word, SOLO]]
    return mixture(word)


def main():
    base_sc = json.load(io.open(BASE_SCENE, encoding="utf-8"))
    tmpl_ex = json.load(io.open(BASE_EX, encoding="utf-8"))
    base_xml = base_sc["world"]["xml"]
    prev = START_MODEL
    made = []
    for r, idx in enumerate(ROUND_INDIV, start=1):
        xml = "MIMo/mimoEnv/assets/f49_8way_r%d.xml" % r
        names = build_xml(base_xml, idx, xml)
        sc = json.loads(json.dumps(base_sc))
        sname = "座位_12ヶ月_F2-49_実物8択_r%d_個体%d_%s" % (r, idx + 1, DATE)
        sc["name"] = sname
        sc["note"] = ("%s F2-49 ラウンド%d（各語の訓練個体%d/3）。実物スキャン素材（GSO/YCB）8語・物ごとの提示角度。"
                      "配置・親の振る舞いはF2-44 r8と同一。" % (DATE, r, idx + 1))
        sc["world"]["xml"] = xml
        sc["world"]["toy2"]["enabled"] = False
        sc["world"]["present_slots"] = SLOT_BODIES[1:]          # 使う枠だけ（余った枠は世界に置かない）
        sc["world"]["parent_labeling"]["utterances"] = {k: utter(w[0]) for k, w in zip(SLOT_KEYS, WORDS)}
        sc["world"]["parent_labeling"]["present_angles"] = {k: PRESENT_ANGLES[w[0]] for k, w in zip(SLOT_KEYS, WORDS)}
        io.open("run/scenes/%s.json" % sname, "w", encoding="utf-8").write(
            json.dumps(sc, ensure_ascii=False, indent=1))
        ex = json.loads(json.dumps(tmpl_ex))
        ex["name"] = "F2-49_r%d" % r
        ex["note"] = "%s F2-49 ラウンド%d/8。実物スキャン8語をF2-6（語の学習前）から鎖で。物ごとの提示角度。" % (DATE, r)
        ex["scene"] = sname
        ex["taro"]["model"] = prev
        ex["taro"]["save"] = "F/models/F2-49_r%d_seed%d_%s.pt" % (r, 90 + r, DATE)
        ex["run"]["seed"] = 90 + r
        ex["run"]["steps"] = STEPS_PER_ROUND
        ex["run"]["checkpoint"] = 500
        ex["run"]["csv"] = "F/logs/F2-49_実物スキャン/r%d/run.csv" % r
        ex["plugins"]["word_learning"]["events_out"] = "F/logs/F2-49_実物スキャン/r%d/発話イベント.csv" % r
        ex["plugins"]["model_snapshots"] = {"out_dir": "F/logs/F2-49_実物スキャン/r%d/snapshots" % r}
        ex["plugins"]["word_production"] = {"events_out": "F/logs/F2-49_実物スキャン/r%d/太郎の発話.csv" % r}
        p = "F/experiments/F2-49_r%d_%s.json" % (r, DATE)
        io.open(p, "w", encoding="utf-8").write(json.dumps(ex, ensure_ascii=False, indent=2))
        made.append(p); prev = ex["taro"]["save"]
        print("r%d: 個体%d %s" % (r, idx + 1, "・".join(n[:18] for n in names)))
    os.makedirs("F/logs/F2-49_実物スキャン", exist_ok=True)
    io.open("F/logs/_f249_chain.txt", "w", encoding="utf-8", newline="\n").write("\n".join(made) + "\n")
    # 下見（r1のシーン・視界ダンプ付き・保存は使い捨て）：ユーザーの目視承認用
    ex = json.load(io.open(made[0], encoding="utf-8"))
    ex["name"] = "F2-49probe"; ex["run"]["steps"] = 900; ex["run"]["checkpoint"] = 900
    ex["taro"]["save"] = "F/models/_F2-49probe_使い捨て.pt"
    ex["run"]["csv"] = "F/logs/F2-49_実物スキャン/probe/run.csv"
    ex["plugins"] = {"word_learning": {"events_out": "F/logs/F2-49_実物スキャン/probe/発話イベント.csv"},
                     "view_video": {"out": "F/logs/F2-49_実物スキャン/probe/動画_視界_8語.mp4"}}
    io.open("F/experiments/F2-49probe_%s.json" % DATE, "w", encoding="utf-8").write(
        json.dumps(ex, ensure_ascii=False, indent=2))
    print("鎖:", len(made), "本 ＋ 下見1本")


if __name__ == "__main__":
    main()
