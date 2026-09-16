# -*- coding: utf-8 -*-
"""手の形の確認（2026-09-03）：手だけを取り出して、3つの形を真横と正面から見る。
世界の中だと体やコップに隠れて確かめられないので、確認用の小さな世界を別に作る。

    .venv/Scripts/python.exe F/scripts/f53_hand_shapes.py
出力: F/logs/F2-53_親のIK/図_手の形.png
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
import f53_parent_ik_probe as P

OUT = "F/logs/F2-53_親のIK"
SHAPES = ["point", "hold", "open"]
LABEL = {"point": "指さし", "hold": "持つ", "open": "開く"}

XML = """<mujoco>
 <visual><global offwidth="900" offheight="400"/><headlight diffuse="0.7 0.7 0.7"/></visual>
 <asset><texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.9 0.9 0.92" width="8" height="8"/></asset>
 <worldbody>
  <light pos="0 -1 1.5" dir="0 0.5 -1"/>
%s
 </worldbody>
</mujoco>""" % "\n".join(
    "  " + line.replace('_r"', '_%s"' % s)
    for s in SHAPES
    for line in P.body_xml().split("\n")
    if '_r"' in line and ("p_palm" in line or any("p_%s" % f in line for f in P.FING + ["thumb"])))


def main():
    os.makedirs(OUT, exist_ok=True)
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    # 3つの形を横に並べる。手首は原点、+x（画面右）が指の向き、+z が手の甲
    R = np.eye(3)
    for i, sh in enumerate(SHAPES):
        parts = P.hand_parts(np.array([i * 0.30 - 0.30, 0.0, 0.0]), R, sh, sh)
        for name, (pos, quat) in parts.items():
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            mi = int(m.body_mocapid[bid]); d.mocap_pos[mi] = pos; d.mocap_quat[mi] = quat
    mujoco.mj_forward(m, d)
    views = [("真横から", 90.0, -8.0), ("斜め上から", 75.0, -50.0)]
    imgs = []
    for title, az, el in views:
        ren = mujoco.Renderer(m, height=300, width=760)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.03, 0.0, -0.01]; cam.distance = 0.72; cam.azimuth = az; cam.elevation = el
        ren.update_scene(d, camera=cam); imgs.append((title, ren.render().copy())); ren.close()
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 15)
    canvas = Image.new("RGB", (760, 2 * 322 + 6), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    for k, (title, im) in enumerate(imgs):
        y = k * 322
        dr.text((6, y + 2), "%s ／ 左から %s" % (title, "・".join(LABEL[s] for s in SHAPES)), fill=(0, 0, 120), font=font)
        canvas.paste(Image.fromarray(im), (0, y + 22))
    p = OUT + "/図_手の形.png"; canvas.save(p); print("図:", p)


if __name__ == "__main__":
    main()
