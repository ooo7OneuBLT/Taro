# -*- coding: utf-8 -*-
"""親の体の下見（2026-09-03・段1）：頭・胴・腕・手を操り人形（mocap）として世界に置き、
手が物を持って太郎に見せる姿を、太郎の左目と第三者視点で描く。動作の部品はまだ無い（見た目の承認用）。

    .venv/Scripts/python.exe F/scripts/f52_parent_body_probe.py
出力: F/logs/F2-52_親の体/図_親の体_下見.png
"""
import os, sys, io, json
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
import f_gen_f49 as G
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod

OUT = "F/logs/F2-52_親の体"
SKIN, SHIRT = ".95 .80 .70 1", ".35 .55 .80 1"
PARENT_XML = """
<body name="parent_torso" mocap="true" pos="0 0 0"><geom type="capsule" size="0.11" fromto="0 0 -0.45 0 0 0.0" rgba="%s" contype="0" conaffinity="0"/></body>
<body name="parent_head" mocap="true" pos="0 0 0">
  <geom type="sphere" size="0.09" rgba="%s" contype="0" conaffinity="0"/>
  <geom type="sphere" size="0.013" pos="0.083 0.032 0.02" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
  <geom type="sphere" size="0.013" pos="0.083 -0.032 0.02" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
  <geom type="capsule" size="0.009" fromto="0.085 0 -0.005 0.098 0 -0.025" rgba=".92 .72 .62 1" contype="0" conaffinity="0"/>
  <geom type="ellipsoid" size="0.095 0.098 0.06" pos="-0.02 0 0.045" rgba="0.25 0.15 0.1 1" contype="0" conaffinity="0"/>
</body>
<body name="parent_upperarm" mocap="true" pos="0 0 0"><geom type="capsule" size="0.035" fromto="0 0 -0.13 0 0 0.13" rgba="%s" contype="0" conaffinity="0"/></body>
<body name="parent_forearm" mocap="true" pos="0 0 0"><geom type="capsule" size="0.03" fromto="0 0 -0.12 0 0 0.12" rgba="%s" contype="0" conaffinity="0"/></body>
<body name="parent_hand" mocap="true" pos="0 0 0">
  <geom type="box" size="0.045 0.035 0.012" rgba="%s" contype="0" conaffinity="0"/>
  <geom type="capsule" size="0.009" fromto="0.04 0.025 0.01 0.07 0.03 0.05" rgba="%s" contype="0" conaffinity="0"/>
  <geom type="capsule" size="0.009" fromto="0.04 0.008 0.01 0.075 0.01 0.055" rgba="%s" contype="0" conaffinity="0"/>
  <geom type="capsule" size="0.009" fromto="0.04 -0.01 0.01 0.072 -0.012 0.052" rgba="%s" contype="0" conaffinity="0"/>
  <geom type="capsule" size="0.009" fromto="0.04 -0.027 0.01 0.065 -0.03 0.045" rgba="%s" contype="0" conaffinity="0"/>
  <geom type="capsule" size="0.01" fromto="-0.01 0.035 0.005 0.02 0.06 0.035" rgba="%s" contype="0" conaffinity="0"/>
</body>
""" % (SHIRT, SKIN, SHIRT, SKIN, SKIN, SKIN, SKIN, SKIN, SKIN, SKIN)


def quat_from_x(x_dir, up=(0, 0, 1)):
    """ローカル+xを x_dir に向ける姿勢。"""
    x = np.asarray(x_dir, float); x /= np.linalg.norm(x)
    z = np.asarray(up, float); z = z - x * (z @ x); z /= np.linalg.norm(z); y = np.cross(z, x)
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).ravel()); return q


def quat_from_z(z_dir):
    z = np.asarray(z_dir, float); z /= np.linalg.norm(z)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = ref - z * (ref @ z); x /= np.linalg.norm(x); y = np.cross(z, x)
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).ravel()); return q


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    r1 = "run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE
    sc = json.load(io.open(r1, encoding="utf-8"))
    src = io.open(sc["world"]["xml"], encoding="utf-8").read()
    xml = "MIMo/mimoEnv/assets/f52_parent_probe.xml"
    io.open(xml, "w", encoding="utf-8", newline="\n").write(src.replace("</worldbody>", PARENT_XML + "</worldbody>"))
    sc["name"] = "座位_12ヶ月_F2-52_親の体_下見_%s" % G.DATE; sc["note"] = "親の体（操り人形）の見た目の下見。学習には使わない。"; sc["world"]["xml"] = xml
    io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 800); m.vis.global_.offheight = max(int(m.vis.global_.offheight), 600)
    cid = int(m.camera("eye_left").id)
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}; qadr["toy1"] = int(u._toy_qadr)
    for a in qadr.values():
        d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
    mujoco.mj_forward(m, d)
    eye = np.array(d.cam_xpos[cid]); R = np.array(d.cam_xmat[cid]).reshape(3, 3)
    fwd = -R[:, 2]; right = R[:, 0]; fwd_h = fwd.copy(); fwd_h[2] = 0; fwd_h /= np.linalg.norm(fwd_h)
    right_h = right.copy(); right_h[2] = 0; right_h /= np.linalg.norm(right_h)
    dist = float(sc["world"]["parent_labeling"].get("follow_dist", 0.3))
    obj = eye + fwd * dist; obj[2] = max(obj[2], 0.05)
    # 物（コップ）を親の手の位置に
    slot = "toy5"; a = qadr[slot]; d.qpos[a:a + 3] = obj
    angles = sc["world"]["parent_labeling"]["present_angles"][slot]; d.qpos[a + 3:a + 7] = present_quat(u, angles["yaw"], angles["tilt"])
    # 親：太郎の正面やや右、距離0.75m。頭は目の高さ+0.15、胴はその下。腕は肩から手へ。
    base = eye + fwd_h * 0.75 + right_h * 0.30
    head = base + np.array([0, 0, 0.15]); torso = base + np.array([0, 0, -0.05])
    shoulder = torso + np.array([0, 0, -0.02]) - right_h * 0.16
    hand = obj + np.array([0, 0, -0.075])
    elbow = (shoulder + hand) / 2 + np.array([0, 0, -0.12])
    def set_mocap(name, pos, quat):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name); mi = int(m.body_mocapid[bid])
        d.mocap_pos[mi] = pos; d.mocap_quat[mi] = quat
    set_mocap("parent_torso", torso, np.array([1, 0, 0, 0.0]))
    set_mocap("parent_head", head, quat_from_x(obj - head))
    set_mocap("parent_upperarm", (shoulder + elbow) / 2, quat_from_z(elbow - shoulder))
    set_mocap("parent_forearm", (elbow + hand) / 2, quat_from_z(hand - elbow))
    set_mocap("parent_hand", hand, quat_from_x(-fwd_h))
    mujoco.mj_forward(m, d)
    ren = mujoco.Renderer(m, height=224, width=224); ren.update_scene(d, camera="eye_left"); eye_img = ren.render().copy(); ren.close()
    ren = mujoco.Renderer(m, height=560, width=760); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (eye + base) / 2 + np.array([0, 0, -0.1]); cam.distance = 1.9; cam.azimuth = 120; cam.elevation = -15
    ren.update_scene(d, camera=cam); third = ren.render().copy(); ren.close()
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 14)
    canvas = Image.new("RGB", (760 + 460, 580), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(third), (0, 20)); canvas.paste(Image.fromarray(eye_img).resize((448, 448)), (770, 20))
    dr.text((4, 2), "第三者視点：親（操り人形）がコップを持って太郎に見せる", fill=(0, 0, 0), font=font)
    dr.text((774, 2), "太郎の左目（224px）", fill=(0, 0, 0), font=font)
    p = OUT + "/図_親の体_下見.png"; canvas.save(p); print("図:", p)
    env.close()


if __name__ == "__main__":
    main()
