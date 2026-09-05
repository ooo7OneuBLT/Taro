# -*- coding: utf-8 -*-
"""ユーザーの実写真で太郎を試す（2026-09-03）。ノイズあり（本物の背景・照明）の認識テスト。

2通りの見せ方：
  board  ＝ 写真を板に貼って親が持つ位置に立て、太郎の左目で見る（本番と同じ距離・環境）
  direct ＝ 写真をそのまま224pxに縮めて目の画像として渡す（環境を通さない）
採点はしない（正解語が太郎の語彙に無い写真もある）。太郎が何と言ったか・先頭音の自信を記録する。

    .venv/Scripts/python.exe F/scripts/f49_test_photos.py <model.pt> ...
入力: F/assets/user_photos/converted/*.jpg（ファイル名を写真の名前として記録）
出力: F/logs/F2-49_実物スキャン/test_photos/ 結果_<モデル名>.csv・図_写真テスト_<モデル名>.png
"""
import os, sys, io, json, csv, warnings, glob
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco, torch
from PIL import Image, ImageDraw, ImageFont
import f_gen_f49 as G
import f49_test as T
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod
from senses.vision_backends import get_backend

PHOTOS = sorted(glob.glob("F/assets/user_photos/converted/*.jpg"))
OUT = "F/logs/F2-49_実物スキャン/test_photos"
FONT = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 13)
BOARD_HALF = 0.075
BOARD_ROT = Image.ROTATE_90      # 板の貼り付けの癖を打ち消す回転（静止測定で決める）


def square_png(jpg):
    """板のテクスチャ用：中央を正方形に切って512pxのPNGに。"""
    png = jpg.replace(".jpg", "_sq.png")
    if not os.path.exists(png):
        im = Image.open(jpg).convert("RGB"); w, h = im.size; s = min(w, h)
        sq = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((512, 512))
        # 【2026-09-03】板（箱）の面への貼り付けで絵が90度回るので、先に逆向きに回しておく
        sq.transpose(BOARD_ROT).save(png)
    return os.path.abspath(png).replace(os.sep, "/")


def board_world(xml_out):
    base_sc = json.load(io.open(G.BASE_SCENE, encoding="utf-8"))
    src = io.open(base_sc["world"]["xml"], encoding="utf-8").read()
    assets = []
    for k, p in enumerate(PHOTOS):
        assets.append('<texture name="tex_u%d" type="cube" file="%s"/><material name="mat_u%d" texture="tex_u%d"/>' % (k, square_png(p), k, k))
        g = '<geom type="box" size="0.003 %.3f %.3f" material="mat_u%d" contype="0" conaffinity="0"/>' % (BOARD_HALF, BOARD_HALF, k)
        src = G.replace_body(src, G.SLOT_BODIES[k], g)
    src = src.replace("</asset>", "".join(assets) + "</asset>", 1)
    for k in range(5, 13):
        b = "test_object%d" % k
        if b not in G.SLOT_BODIES[:len(PHOTOS)]:
            src = G.remove_body(src, b)
    io.open(xml_out, "w", encoding="utf-8", newline="\n").write(src)


def render_boards():
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    r1 = "run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE
    sc = json.load(io.open(r1, encoding="utf-8"))
    xml = "MIMo/mimoEnv/assets/f49_user_photos.xml"; board_world(xml)
    sc["name"] = "座位_12ヶ月_F2-49_ユーザー写真_%s" % G.DATE
    sc["note"] = "ユーザーの実写真を板に貼って見せるテスト用。学習には使わない。"
    sc["world"]["xml"] = xml
    sc["world"]["present_slots"] = G.SLOT_BODIES[1:len(PHOTOS)]
    io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    dist = float(sc["world"]["parent_labeling"].get("follow_dist", 0.3))
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}; qadr["toy1"] = int(u._toy_qadr)
    cid = int(m.camera("eye_left").id); ren = mujoco.Renderer(m, height=224, width=224)
    out = []
    for k, p in enumerate(PHOTOS):
        slot = G.SLOT_KEYS[k]
        for a in qadr.values():
            d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
        mujoco.mj_forward(m, d)
        eye = np.array(d.cam_xpos[cid]); fwd = -np.array(d.cam_xmat[cid]).reshape(3, 3)[:, 2]
        goal = eye + fwd * dist; goal[2] = max(goal[2], 0.05)
        a = qadr[slot]; d.qpos[a:a + 3] = goal; d.qpos[a + 3:a + 7] = present_quat(u, 0, 0)
        mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
        out.append(("board", os.path.basename(p), ren.render().copy()))
    ren.close(); env.close()
    return out


def direct_imgs():
    out = []
    for p in PHOTOS:
        im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((224, 224))
        out.append(("direct", os.path.basename(p), np.asarray(im, dtype=np.uint8)))
    return out


def main(models):
    os.makedirs(OUT, exist_ok=True)
    stim = render_boards() + direct_imgs()
    be = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9})
    vecs = [np.asarray(be.encode(s[2], s[2]), dtype=np.float32) for s in stim]
    T_, L = 224, 34
    for mp in models:
        tag = os.path.splitext(os.path.basename(mp))[0]
        b, vp, pv = T.load_model(mp)
        rows = []
        for (mode, name, img), vec in zip(stim, vecs):
            said, conf = T.greedy(b, vp, pv, vec)
            rows.append({"見せ方": mode, "写真": name, "GRUの発話": said, "先頭音の自信": round(conf, 2)})
        with io.open("%s/結果_%s.csv" % (OUT, tag), "w", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        canvas = Image.new("RGB", (len(stim) * (T_ + 4), T_ + L + 20), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
        dr.text((4, 2), tag, fill=(0, 0, 0), font=FONT)
        for i, ((mode, name, img), r) in enumerate(zip(stim, rows)):
            x = i * (T_ + 4); canvas.paste(Image.fromarray(img), (x, 20 + L))
            dr.text((x + 2, 20), "%s %s" % (mode, name), fill=(0, 0, 0), font=FONT)
            dr.text((x + 2, 36), "→「%s」%.2f" % (r["GRUの発話"], r["先頭音の自信"]), fill=(0, 0, 160), font=FONT)
        canvas.save("%s/図_写真テスト_%s.png" % (OUT, tag))
        print(tag + ": " + "  ".join("%s/%s→「%s」%.2f" % (r["見せ方"], r["写真"], r["GRUの発話"], r["先頭音の自信"]) for r in rows))


if __name__ == "__main__":
    main(sys.argv[1:] or ["F/models/F2-49c_r3_seed93_%s.pt" % G.DATE])
