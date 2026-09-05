# -*- coding: utf-8 -*-
"""離れた場所に2つの物があるとき、両方を検出できるか（2026-09-04）。

F2-60は「重なっている」場合の弱点（1つの塊にされる）を確認した。今回は
「重なっていない・離れている」場合を確かめる。今の検出（中心を種にする
図地分離）は「画面の中心に必ず1つだけ物がある」前提で作られているため、
中心から離れた2つ目の物がそもそも検出されない可能性がある。

    .venv/Scripts/python.exe F/scripts/f63_multi_object_probe.py
出力: F/logs/F2-63_複数物体/図_離れた2物体.png
"""
import os, sys, io, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("taro_core/src/brain"))
sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, torch, mujoco
from PIL import Image, ImageDraw, ImageFont
import f_gen_f49 as G
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod
from senses.object_detector import patch_features, detect

OUT = "F/logs/F2-63_複数物体"


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE, encoding="utf-8"))
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    cid = int(m.camera("eye_left").id)
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
    qadr["toy1"] = int(u._toy_qadr)
    for a in qadr.values():
        d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
    mujoco.mj_forward(m, d)
    eye = np.array(d.cam_xpos[cid]); Rc = np.array(d.cam_xmat[cid]).reshape(3, 3)
    fwd = -Rc[:, 2]; right = Rc[:, 0]; up = Rc[:, 1]
    ren = mujoco.Renderer(m, height=224, width=224)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()

    ang5 = sc["world"]["parent_labeling"]["present_angles"]["toy5"]   # コップ
    ang6 = sc["world"]["parent_labeling"]["present_angles"]["toy6"]   # がおー

    def place(slot, pos, ang):
        a = qadr[slot]; d.qpos[a:a + 3] = pos; d.qpos[a + 3:a + 7] = present_quat(u, ang["yaw"], ang["tilt"])

    scenes = [
        ("1つだけ（中心）", [("toy5", eye + fwd * 0.30, ang5)]),
        ("2つ・離れている（左右）", [("toy5", eye + fwd * 0.30 - right * 0.12, ang5),
                                    ("toy6", eye + fwd * 0.30 + right * 0.12, ang6)]),
        ("2つ・さらに離す（画面端寄り）", [("toy5", eye + fwd * 0.28 - right * 0.18, ang5),
                                        ("toy6", eye + fwd * 0.28 + right * 0.18, ang6)]),
    ]
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 14)
    f2 = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 12)
    canvas = Image.new("RGB", (250 * len(scenes), 300), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    for i, (title, placements) in enumerate(scenes):
        for a in qadr.values():
            d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
        for slot, pos, ang in placements:
            place(slot, pos, ang)
        mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
        img = ren.render().copy()
        p, n = patch_features(model, img)
        dets = detect(p, n, thresh=0.55)
        note = "検出 %d 件（正解は%d件）" % (len(dets), len(placements))
        print("%s: %s" % (title, note))
        for k, dd in enumerate(dets):
            print("   検出%d: 位置(画素) %.0f,%.0f 面積比 %.2f" % (k, dd["pos"][0], dd["pos"][1], dd["area"]))
        x = i * 250
        dr.text((x + 4, 4), title, fill=(0, 0, 120), font=font)
        canvas.paste(Image.fromarray(img).resize((224, 224)), (x + 10, 26))
        for dd in dets:
            cx, cy = dd["pos"]; r = 8
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=2)
        dr.text((x + 4, 255), note, fill=(180, 0, 0) if len(dets) != len(placements) else (0, 120, 0), font=f2)
    ren.close(); env.close()
    p = OUT + "/図_離れた2物体.png"; canvas.save(p); print("図:", p)


if __name__ == "__main__":
    main()
