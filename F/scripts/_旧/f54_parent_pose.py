# -*- coding: utf-8 -*-
"""親の姿勢を「条件」から解く（2026-09-03・段3）。

人が書くのは対象の名前だけ。手首や肘の座標は書かない。
  指さす  → 人差し指の先の向きが対象を通る（LookAtTask）
  顔を向ける → 顔の前方が対象を通る（LookAtTask）
  持つ    → 手のひらの把持点が物の表面にある（FrameTask）
  自然さ  → 基準の姿勢からできるだけ離れない（PostureTask）
  可動域  → 関節は人間の範囲内（ConfigurationLimit）
  ぶつからない → どの部位も相手と2cm以上離れる（CollisionAvoidanceLimit）

    .venv/Scripts/python.exe F/scripts/f54_parent_pose.py
出力: F/logs/F2-54_親を条件で解く/図_条件から解いた姿勢.png
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco, mink
from PIL import Image, ImageDraw, ImageFont
import f54_parent_model as M

OUT = "F/logs/F2-54_親を条件で解く"
ARM_GEOMS = {s: ["upper_%s" % s, "fore_%s" % s, "palm_%s" % s, "thumb_%s" % s]
             + ["%s%d_%s" % (f, g, s) for f in M.FING for g in (1, 2)] for s in ("r", "l")}


def geom_ids(model, names):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]


def solve(model, data, cmd, obstacles, q0=None, iters=600, dt=0.04, verbose=False):
    """cmd = {"point": (手, 目標点), "look": 目標点, "hold": (手, 物の中心, 物の半径)}
    obstacles = {名前: (位置, 半径)}  ← モデルに mocap で入れてある代理形状
    返り値: 解けた qpos と、条件をどれだけ満たしたか"""
    cfg = mink.Configuration(model)
    cfg.update(model.key_qpos[0].copy() if model.nkey else np.zeros(model.nq) if q0 is None else q0)

    def near_hand(pt):
        return min(("r", "l"), key=lambda sd: np.linalg.norm(
            np.array(cfg.data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "p_upper_%s" % sd)])
            - np.asarray(pt, float)))

    cmd = dict(cmd)
    if "hold" in cmd and cmd["hold"][0] is None:
        cmd["hold"] = (near_hand(cmd["hold"][1]),) + tuple(cmd["hold"][1:])
    if "point" in cmd and cmd["point"][0] is None:
        other = {"r": "l", "l": "r"}[cmd["hold"][0]] if "hold" in cmd else near_hand(cmd["point"][1])
        cmd["point"] = (other, cmd["point"][1])
    tasks, targets = [], []
    if "look" in cmd:
        t = mink.LookAtTask(frame_name="gaze", frame_type="site", axis=(1, 0, 0), cost=1.0)
        t.set_target(cmd["look"]); tasks.append(t); targets.append(("顔の向き", t))
    if "reach_only" in cmd:                       # 切り分け用：位置の条件だけ
        side, at = cmd["reach_only"]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "p_upper_%s" % side)
        sh = np.array(cfg.data.xpos[bid]); v = np.asarray(at, float) - sh
        far = float(np.linalg.norm(v)); v = v / far
        arm = M.D["upper"] + M.D["fore"] + M.D["palm"] + M.D["seg1"] + M.D["seg2"]
        t = mink.FrameTask(frame_name="tip_%s" % side, frame_type="site", position_cost=1.2, orientation_cost=0.0)
        t.set_target(mink.SE3.from_translation(sh + v * min(arm * 0.85, far - 0.16)))
        tasks.append(t); targets.append(("指先の位置", t))
    if "point" in cmd:
        side, at = cmd["point"]
        t = mink.LookAtTask(frame_name="tip_%s" % side, frame_type="site", axis=(1, 0, 0), cost=1.0)
        t.set_target(at); tasks.append(t); targets.append(("指の向き", t))
        # 人間は物の斜め上・手前から指す（下から回り込んで指さない）。
        # 目標は対象と自分の位置から決まる規則で、対象ごとに書く数字ではない。
        sh = np.array(cfg.data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "p_upper_%s" % side)])
        v = np.asarray(at, float) - sh; far = float(np.linalg.norm(v)); v = v / far
        arm = M.D["upper"] + M.D["fore"] + M.D["palm"] + M.D["seg1"] + M.D["seg2"]
        approach = sh + v * min(arm * 0.85, max(far - 0.18, 0.15)) + np.array([0, 0, 0.10])
        t2 = mink.FrameTask(frame_name="tip_%s" % side, frame_type="site",
                            position_cost=0.5, orientation_cost=0.0)
        t2.set_target(mink.SE3.from_translation(approach))
        tasks.append(t2)

    if "hold" in cmd:
        side, center, radius = cmd["hold"]
        # 把持点は物の「中心」ではなく、親の側の表面（中心を掴むと物にめり込む）
        sh = np.array(cfg.data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "p_upper_%s" % side)])
        v = np.asarray(center, float) - sh; v = v / np.linalg.norm(v)
        grip = np.asarray(center, float) - v * (float(radius) + 0.015)
        t = mink.FrameTask(frame_name="grip_%s" % side, frame_type="site",
                           position_cost=2.0, orientation_cost=0.0)
        t.set_target(mink.SE3.from_translation(np.asarray(grip, float)))
        tasks.append(t); targets.append(("持つ手", t))
    # 基準の姿勢＝その動作の「構え」。向きの条件だけでは腕が下がったまま向きを合わせて
    # しまうので、指さしなら腕を前に上げた形、持つなら胸の前の形を基準にする。
    # 対象がどこにあっても同じ構えを使う（対象ごとの座標は書かない）。
    qref = np.zeros(model.nq)

    def _set(joint, deg):
        qref[model.joint(joint).qposadr[0]] = np.radians(deg)

    _set("spine", 12)
    if "point" in cmd:
        sd = cmd["point"][0]
        _set("sh_%s_y" % sd, 62); _set("el_%s" % sd, 18)
    if "hold" in cmd:
        sd = cmd["hold"][0]
        _set("sh_%s_y" % sd, 48); _set("el_%s" % sd, 62)
    posture = mink.PostureTask(model, cost=1.2e-2)
    posture.set_target(qref)
    tasks.append(posture)
    # ぶつからない：持つ手だけは物に触れてよいので回避から外す
    hold_side = cmd["hold"][0] if "hold" in cmd else None
    if not tasks[:-1]:
        pass
    body_geoms = []
    for s in ("r", "l"):
        if s != hold_side:
            body_geoms += ARM_GEOMS[s]
        else:
            body_geoms += ["upper_%s" % s, "fore_%s" % s]   # 持つ手でも腕は物を避ける
    body_geoms += ["head", "torso", "hair"]
    obs_geoms = ["obs_%s" % k for k in obstacles]
    limits = [mink.ConfigurationLimit(model)]
    if body_geoms and obs_geoms:
        limits.append(mink.CollisionAvoidanceLimit(
            model, geom_pairs=[(geom_ids(model, body_geoms), geom_ids(model, obs_geoms))],
            minimum_distance_from_collisions=0.02, collision_detection_distance=0.10))
    for i in range(iters):
        vel = mink.solve_ik(cfg, tasks, dt, solver="daqp", damping=1e-3, limits=limits)
        cfg.integrate_inplace(vel, dt)
        if np.linalg.norm(vel) < 1e-5:
            break
    # ずれは実際の距離・角度で測る（タスクの内部誤差ではなく、目で確かめられる量で）
    mujoco.mj_kinematics(model, cfg.data) if False else None
    err = {}
    for name, t in targets:
        if isinstance(t, mink.LookAtTask):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, t.frame_name)
            pos = cfg.data.site_xpos[sid]; R = cfg.data.site_xmat[sid].reshape(3, 3)
            v = np.asarray(t.target_pos, float) - pos; v = v / np.linalg.norm(v)
            err[name] = float(np.degrees(np.arccos(np.clip(R[:, 0] @ v, -1, 1))))
        else:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, t.frame_name)
            err[name] = float(np.linalg.norm(cfg.data.site_xpos[sid] - t.transform_target_to_world.translation()) * 100)
    if verbose:
        print("   反復 %d 回／" % (i + 1) + "・".join(
            "%s %s" % (k, ("%.1f度" % v) if "向き" in k else ("%.1fcm" % v)) for k, v in err.items()))
    return cfg.q.copy(), err


def main():
    os.makedirs(OUT, exist_ok=True)
    # 太郎（頭・胴）とコップを、よけたい相手として置く
    obstacles = {"taro_head": (np.array([0.0, 0.0, 0.37]), 0.085),
                 "taro_body": (np.array([-0.02, 0.0, 0.18]), 0.10),
                 "cup": (np.array([0.30, 0.0, 0.37]), 0.065)}
    xml = M.build_xml(shape_r="point", shape_l="hold",
                      obstacles=[(k, "sphere", "%.3f" % v[1]) for k, v in obstacles.items()])
    # 親は太郎の正面 0.80m に座り、太郎の方（-x）を向く → 世界を x 反転して置くのではなく
    # 親のモデルの原点を親の腰にし、太郎たちを親から見た座標で置く
    P = np.array([0.80, 0.20, 0.0])          # 親から見た太郎の位置（親は +x を向く）
    obst_world = {k: (P + np.array([-v[0][0], -v[0][1], 0]) * 0 + np.array([P[0] * 0, 0, 0]) + v[0] * 0, v[1])
                  for k, v in obstacles.items()}
    # 親は原点、太郎は親の前方 x=+0.80、やや左 y=+0.20
    obst_world = {"taro_head": (np.array([0.72, 0.20, 0.37]), 0.085),
                  "taro_body": (np.array([0.74, 0.20, 0.18]), 0.10),
                  "cup": (np.array([0.42, 0.20, 0.37]), 0.065)}
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for k, (pos, _) in obst_world.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obs_%s" % k)
        data.mocap_pos[int(model.body_mocapid[bid])] = pos
    mujoco.mj_forward(model, data)

    taro_eye = obst_world["taro_head"][0] + np.array([-0.08, 0, 0.01])
    cup_c, cup_r = obst_world["cup"]
    scenes = [
        ("【切り分け】コップを指さすだけ", {"point": (None, cup_c)}),
        ("【切り分け】腕を伸ばすだけ（向きの条件なし）", {"reach_only": ("r", cup_c)}),
        ("コップを指さして、太郎を見て", {"look": taro_eye, "point": (None, cup_c), "hold": (None, cup_c, cup_r)}),
        ("コップを見せて", {"look": cup_c, "hold": (None, cup_c, cup_r)}),
        ("コップを持って、太郎を指さして", {"look": taro_eye, "point": (None, taro_eye), "hold": (None, cup_c, cup_r)}),
    ]
    imgs = []
    for title, cmd in scenes:
        print("「%s」" % title)
        q, err = solve(model, data, cmd, obst_world, verbose=True)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        shots = []
        for az, el, dist in ((92.0, -8.0, 1.55), (152.0, -55.0, 1.60)):
            ren = mujoco.Renderer(model, height=380, width=470)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.34, 0.10, 0.34]; cam.distance = dist; cam.azimuth = az; cam.elevation = el
            ren.update_scene(data, camera=cam); shots.append(ren.render().copy()); ren.close()
        imgs.append((title, err, shots))
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 15)
    f2 = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 13)
    W = 470 * 2 + 8
    canvas = Image.new("RGB", (W * 3, 430), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    for i, (title, err, shots) in enumerate(imgs):
        x = i * W
        dr.text((x + 6, 4), "「%s」" % title, fill=(0, 0, 120), font=font)
        canvas.paste(Image.fromarray(shots[0]), (x, 26))
        canvas.paste(Image.fromarray(shots[1]), (x + 474, 26))
        dr.text((x + 6, 410), "   ".join("%s のずれ %s" % (k, ("%.1f度" % v) if "向き" in k else ("%.1fcm" % v))
                                          for k, v in err.items()), fill=(0, 0, 0), font=f2)
        dr.text((x + 380, 8), "横から", fill=(90, 90, 90), font=f2)
        dr.text((x + 860, 8), "上から", fill=(90, 90, 90), font=f2)
    p = OUT + "/図_条件から解いた姿勢.png"; canvas.save(p); print("図:", p)


if __name__ == "__main__":
    main()
