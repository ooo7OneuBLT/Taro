# -*- coding: utf-8 -*-
"""取得したGSOモデルを本番の提示条件（0.3m・60度・224px・傾き25度）で描いた一覧図を作る（承認用）。

    .venv/Scripts/python.exe F/scripts/f_gso_gallery.py
出力: F/logs/F2-49_実物スキャン/図_GSO一覧_<語>.png と 全体版
大きさは各モデルの最長辺を TARGET_M（既定0.15m）に揃える（乳児のおもちゃ15〜20cmの中央）。
"""
import os, sys, io, json, math
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
import numpy as np, mujoco, trimesh
from PIL import Image, ImageDraw, ImageFont

TARGET_M = 0.15
SELECT = json.load(io.open("F/assets/gso/_選定.json", encoding="utf-8"))
font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 14)


def prep(name):
    """OBJを読み、最長辺をTARGET_Mに正規化・原点を中心に。テクスチャのパスを返す。"""
    d = os.path.join("F/assets/gso", name)
    obj = os.path.join(d, "meshes", "model.obj")
    if not os.path.exists(obj):
        return None
    # 【2026-09-02】trimeshで再書き出しするとUV座標が落ちて真っ黒になった。
    #   元のOBJをMuJoCoに直接読ませ（texcoord保持）、拡大率と中心合わせはMJCF側で行う。
    m = trimesh.load(obj, force="mesh", process=False)
    ext = m.bounding_box.extents
    scale = TARGET_M / float(max(ext))
    center = m.bounding_box.centroid * scale
    tex = None
    tdir = os.path.join(d, "materials", "textures")
    if os.path.isdir(tdir):
        for f in os.listdir(tdir):
            if f.lower().endswith(".png"):
                tex = os.path.join(tdir, f); break
    return obj, tex, len(m.faces), scale, center


def render(name, degs=(30, 150), tilt=25):
    p = prep(name)
    if p is None:
        return None
    obj, tex, nf, scale, center = p
    root = os.getcwd().replace(os.sep, "/")
    tex_xml = ('<texture name="t" type="2d" file="%s/%s"/><material name="mt" texture="t"/>' % (root, tex.replace(os.sep, "/"))) if tex else ""
    mat = 'material="mt"' if tex else 'rgba=".7 .7 .7 1"'
    cx, cy, cz = (-float(v) for v in center)
    XML = f"""<mujoco><compiler meshdir="{root}"/>
<visual><global offwidth="640" offheight="640"/><headlight ambient=".45 .45 .45" diffuse=".35 .35 .35"/></visual>
<asset><mesh name="mm" file="{obj.replace(os.sep, '/')}" scale="{scale} {scale} {scale}"/>{tex_xml}
<texture type="skybox" builtin="gradient" rgb1=".62 .70 .82" rgb2=".80 .84 .90" width="16" height="16"/></asset>
<worldbody><light pos="0.3 -0.26 0.55" dir="-0.5 0.45 -0.72" castshadow="false"/>
<camera name="wide" fovy="60" pos="0.30 0 0.5" euler="0 90 0"/>
<body name="obj" pos="0 0 0.5"><freejoint/><geom type="mesh" mesh="mm" pos="{cx} {cy} {cz}" {mat} contype="0" conaffinity="0"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    imgs = []
    for deg in degs:
        a = math.radians(deg); t = math.radians(tilt)
        qz = [math.cos(a/2), 0, 0, math.sin(a/2)]; qy = [math.cos(t/2), 0, math.sin(t/2), 0]
        w1, x1, y1, z1 = qy; w2, x2, y2, z2 = qz
        d.qpos[3:7] = [w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2]
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, height=224, width=224); r.update_scene(d, camera="wide")
        imgs.append(r.render().copy()); r.close()
    return imgs, nf, scale


if __name__ == "__main__":
    os.makedirs("F/logs/F2-49_実物スキャン", exist_ok=True)
    T, L = 224, 18
    rows = []
    for jp, names in SELECT.items():
        tiles = []
        for n in names:
            try:
                res = render(n)
            except Exception as e:
                print("失敗", n, str(e)[:80]); res = None
            if res is None:
                continue
            imgs, nf, scale = res
            tiles.append((n, imgs, nf))
            print("%s %-50s 面数%6d 倍率%.3f" % (jp, n[:50], nf, scale), flush=True)
        if not tiles:
            continue
        W = len(tiles) * 2 * (T + 2)
        canvas = Image.new("RGB", (W, T + L + 2), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        for i, (n, imgs, nf) in enumerate(tiles):
            for k, im in enumerate(imgs):
                x = (i * 2 + k) * (T + 2)
                canvas.paste(Image.fromarray(im), (x, L))
            dr.text((i * 2 * (T + 2) + 2, 1), "%s %s (%d面)" % (jp, n[:34], nf), fill=(0, 0, 0), font=font)
        out = "F/logs/F2-49_実物スキャン/図_GSO一覧_%s.png" % jp
        canvas.save(out); rows.append(out)
    print("保存:", len(rows), "枚")
