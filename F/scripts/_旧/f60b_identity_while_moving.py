# -*- coding: utf-8 -*-
"""物が動いている間、同一性が保てるか（2026-09-03）：F2-60の続き。

F2-60は物を止めたまま検証した（同一性は自明に保てる）。ここでは物を連続的に
動かし（落ちる・横に転がる）、遮蔽物なしで、同じ物体ファイルのIDが保たれるか、
新しいIDが誤って生まれないか、を確かめる。

    .venv/Scripts/python.exe F/scripts/f60b_identity_while_moving.py
出力: F/logs/F2-60_物体ファイル/図_動いている間の同一性.png
"""
import os, sys, io, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, torch, mujoco
from PIL import Image, ImageDraw, ImageFont
import f_gen_f49 as G
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod
from senses.object_detector import patch_features, detect
from brain.object_files import ObjectFileSystem

OUT = "F/logs/F2-60_物体ファイル"
N_FRAMES = 16


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    env, sc, _ = scene_mod.build("座位_12ヶ月_F2-49_実物8択_r1_個体1_%s" % G.DATE,
                                 taro=spec["taro"], seed=0, verbose=False)
    env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    cid = int(m.camera("eye_left").id)
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
    qadr["toy1"] = int(u._toy_qadr)
    for a in qadr.values():
        d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
    mujoco.mj_forward(m, d)
    eye = np.array(d.cam_xpos[cid]); Rc = np.array(d.cam_xmat[cid]).reshape(3, 3)
    fwd = -Rc[:, 2]; right = Rc[:, 0]; up = Rc[:, 1]
    slot = "toy5"; a = qadr[slot]
    ang = sc["world"]["parent_labeling"]["present_angles"][slot]
    ren = mujoco.Renderer(m, height=224, width=224)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()

    def shot(pos):
        d.qpos[a:a + 3] = pos; d.qpos[a + 3:a + 7] = present_quat(u, ang["yaw"], ang["tilt"])
        mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
        return ren.render().copy()

    base = eye + fwd * 0.30; base[2] = max(base[2], 0.10)
    paths = {
        "落ちる（連続）": [base - up * (0.010 * t) for t in range(N_FRAMES)],
        "横へ転がる（連続）": [base + right * (0.012 * t) for t in range(N_FRAMES)],
    }
    fonts = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 14)
    f2 = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 11)
    canvas = Image.new("RGB", (N_FRAMES * 90, 2 * 118 + 8), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    for row, (title, seq) in enumerate(paths.items()):
        ofs = ObjectFileSystem(appearance_weight=0.6, max_missed=20, reappear_gap=4, gate=0.75,
                       meas_noise=64.0)  # F2-57実測（1歩8画素）に合わせた観測ノイズ。既定6.0は未較正だった
        y0 = row * 118
        dr.text((4, y0 + 2), title, fill=(0, 0, 120), font=fonts)
        ids_seen, resids = [], []
        for t, pos in enumerate(seq):
            img = shot(pos)
            p, n = patch_features(model, img)
            dets = detect(p, n, thresh=0.55)
            r = ofs.step(dets)
            note = ""
            if r["created"]:
                note = "新規%s" % r["created"]; ids_seen += r["created"]
            if r["matched"]:
                fid, j, res = r["matched"][0]
                note = "id%d %.2f" % (fid, res); resids.append(res)
            x = t * 90
            canvas.paste(Image.fromarray(img).resize((84, 84)), (x, y0 + 22))
            dr.text((x + 2, y0 + 106), note, fill=(0, 0, 0), font=f2)
        n_ids = len(set(ids_seen))
        ok = "○ 同一IDを維持" if n_ids == 1 else "× ID分裂（%d個生成）" % n_ids
        print("%s：%s／マハラノビス距離の平均 %.2f・最大 %.2f（対応づけの境界=%.2f）" % (title, ok, np.mean(resids) if resids else -1, max(resids) if resids else -1, 2.4477468306808161))
    p = OUT + "/図_動いている間の同一性.png"; canvas.save(p); print("図:", p)
    ren.close(); env.close()


if __name__ == "__main__":
    main()
