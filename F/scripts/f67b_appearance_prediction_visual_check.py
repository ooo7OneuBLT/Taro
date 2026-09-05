# -*- coding: utf-8 -*-
"""F2-67cで使った画像を実際に並べて見る（2026-09-04）。

予測誤差の向きが逆転した原因（実物スキャンの方が角度変化で見た目が大きく動いた
のではないか、という考察）を、数値でなく目で確かめる。

    .venv/Scripts/python.exe F/scripts/f67b_appearance_prediction_visual_check.py
出力: F/logs/F2-67c_見た目予測プローブ/図_使った画像.png
"""
import os, sys, io, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import f_gen_f49 as G
import f49_test as T
import f58b_novelty_control as U

OUT = "F/logs/F2-67c_見た目予測プローブ"


def main():
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()

    # ① 既知：2個体ぶん、角度順(-40,0,+40)
    stim = T.render_stimuli()
    groups = {}
    for word, idx, unseen, yaw, img, name in stim:
        groups.setdefault((word, idx), []).append((yaw, img, name))
    known_show = list(groups.items())[:2]

    # ② 未知：2個体ぶん(積み木・車)、角度順(-30,0,+30)
    xml = "MIMo/mimoEnv/assets/f58_unknown.xml"
    U.build_world(xml)
    r1 = "run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE
    sc = json.load(io.open(r1, encoding="utf-8"))
    sc["name"] = "座位_12ヶ月_F2-67b_視覚確認_%s" % G.DATE
    sc["note"] = "見た目の予測誤差テストで使った画像の目視確認。学習には使わない。"
    sc["world"]["xml"] = xml
    sc["world"]["present_slots"] = G.SLOT_BODIES[1:len(U.UNKNOWN)]
    io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    from run.plugins.common import scene as scene_mod
    from f_present_angle_probe import present_quat
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    dist = float(sc["world"]["parent_labeling"].get("follow_dist", 0.3))
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
    qadr["toy1"] = int(u._toy_qadr)
    cid = int(m.camera("eye_left").id)
    ren = mujoco.Renderer(m, height=224, width=224)
    unknown_show = []
    for k, (word, name) in enumerate(U.UNKNOWN[:2]):
        slot = G.SLOT_KEYS[k]
        imgs = []
        for off in U.YAW:
            for a in qadr.values():
                d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
            mujoco.mj_forward(m, d)
            eye = np.array(d.cam_xpos[cid]); fwd = -np.array(d.cam_xmat[cid]).reshape(3, 3)[:, 2]
            goal = eye + fwd * dist; goal[2] = max(goal[2], 0.05)
            a = qadr[slot]; d.qpos[a:a + 3] = goal
            d.qpos[a + 3:a + 7] = present_quat(u, off, -20)
            mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
            imgs.append((off, ren.render().copy(), name))
        unknown_show.append(((word, k), imgs))
    ren.close(); env.close()

    fig, axes = plt.subplots(4, 3, figsize=(8, 10.5))
    for row, ((word, idx), items) in enumerate(known_show):
        imgs = sorted(items, key=lambda t: t[0])
        for col, (yaw, img, name) in enumerate(imgs):
            ax = axes[row, col]; ax.imshow(img); ax.axis("off")
            ax.set_title("既知:%s(%s) 角度%+d" % (word, name, yaw), fontsize=8)
    for row, ((word, idx), items) in enumerate(unknown_show):
        imgs = sorted(items, key=lambda t: t[0])
        for col, (yaw, img, name) in enumerate(imgs):
            ax = axes[row + 2, col]; ax.imshow(img); ax.axis("off")
            ax.set_title("未知:%s(%s) 角度%+d" % (word, name, yaw), fontsize=8)
    fig.suptitle("見た目の予測誤差テストで使った画像（太郎の左目視点・224×224）", fontsize=11)
    fig.tight_layout()
    p = OUT + "/図_使った画像.png"; fig.savefig(p, dpi=130); print("図:", p)


if __name__ == "__main__":
    main()
