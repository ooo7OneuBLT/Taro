# -*- coding: utf-8 -*-
"""条件から解いた親の姿勢を、太郎の世界に持ち込む（2026-09-03・段3の続き）。

親は太郎の世界の物理には参加しない。別モデルとして解き、見た目だけを写す
（太郎の姿勢データの長さが変わらないので、保存済みモデル・過去の実験は無傷）。
めり込みは解く側の制約で防ぎ、写した後にも実測して確かめる。

    .venv/Scripts/python.exe F/scripts/f55_parent_in_world.py
出力: F/logs/F2-55_親を太郎の世界へ/図_太郎から見た親.png
"""
import os, sys, io, json
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
import f_gen_f49 as G
import f54_parent_model as M
import f54_parent_pose as P
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod

OUT = "F/logs/F2-55_親を太郎の世界へ"
PRE = "pa"
HIP_DIST, HIP_SIDE = 0.72, 0.20      # 親の腰＝太郎の正面 0.72m、やや横 0.20m


def quat_of(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q


def penetration(m, d, prefix, margin=0.0):
    ids = [g for g in range(m.ngeom)
           if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "").startswith(prefix + "_")]
    ps = set(ids); bad = []
    for a in ids:
        if d.geom_xpos[a][2] < -1.0:
            continue
        na = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a) or "?"
        for b in range(m.ngeom):
            if b in ps or d.geom_xpos[b][2] < -1.0:
                continue
            dist = mujoco.mj_geomDistance(m, d, a, b, 0.05, None)
            if dist < margin:
                nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b) or "?"
                bad.append((na, nb, float(dist)))
    return bad


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE, encoding="utf-8"))
    src = io.open(sc["world"]["xml"], encoding="utf-8").read()
    xml = "MIMo/mimoEnv/assets/f55_parent.xml"
    io.open(xml, "w", encoding="utf-8", newline="\n").write(
        src.replace("</worldbody>", M.mocap_xml(prefix=PRE) + "\n</worldbody>"))
    sc["name"] = "座位_12ヶ月_F2-55_親を条件で解く_%s" % G.DATE
    sc["note"] = "条件から解いた親の姿勢を写す下見。学習には使わない。"
    sc["world"]["xml"] = xml
    io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 900)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 700)
    cid = int(m.camera("eye_left").id)
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
    qadr["toy1"] = int(u._toy_qadr)
    for a in qadr.values():
        d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
    mujoco.mj_forward(m, d)
    eye = np.array(d.cam_xpos[cid]); Rc = np.array(d.cam_xmat[cid]).reshape(3, 3)
    fwd = -Rc[:, 2]
    fwd_h = fwd.copy(); fwd_h[2] = 0; fwd_h /= np.linalg.norm(fwd_h)
    right_h = Rc[:, 0].copy(); right_h[2] = 0; right_h /= np.linalg.norm(right_h)
    # 親の座標系：腰が原点、+x が太郎の方向
    hip_w = eye + fwd_h * HIP_DIST + right_h * HIP_SIDE
    hip_w[2] = M.D["hip_z"]
    px = -fwd_h; pz = np.array([0, 0, 1.0]); py = np.cross(pz, px)
    Rp = np.stack([px, py, pz], axis=1)

    def to_parent(w):
        return Rp.T @ (np.asarray(w, float) - hip_w)

    # 物は従来どおり太郎の目の前 0.30m に置く（学習と同じ提示位置）
    obj_w = eye + fwd * 0.30; obj_w[2] = max(obj_w[2], 0.06)
    slot = "toy5"; ang = sc["world"]["parent_labeling"]["present_angles"][slot]
    a = qadr[slot]; d.qpos[a:a + 3] = obj_w; d.qpos[a + 3:a + 7] = present_quat(u, ang["yaw"], ang["tilt"])
    mujoco.mj_forward(m, d)
    # 太郎の頭・胴とコップを、親のモデル側で「よけたい相手」として置く
    head_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")
    # 物の代理の大きさは実測（外接球の半径）。決め打ちだと腕が実物に食い込む
    slot_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, G.SLOT_BODIES[G.SLOT_KEYS.index(slot)])
    cup_r = max([float(m.geom_rbound[g]) for g in range(m.ngeom) if int(m.geom_bodyid[g]) == slot_b] or [0.07])
    print("コップの外接半径 %.3f m" % cup_r)
    obst = {"taro_head": (to_parent(d.xpos[head_b]), 0.085),
            "taro_body": (to_parent(eye - fwd_h * 0.02 - np.array([0, 0, 0.20])), 0.10),
            "cup": (to_parent(obj_w), cup_r)}
    pmodel = mujoco.MjModel.from_xml_string(
        M.build_xml(shape_r="point", shape_l="hold",
                    obstacles=[(k, "sphere", "%.3f" % v[1]) for k, v in obst.items()]))
    pdata = mujoco.MjData(pmodel)
    for k, (pos, _) in obst.items():
        bid = mujoco.mj_name2id(pmodel, mujoco.mjtObj.mjOBJ_BODY, "obs_%s" % k)
        pdata.mocap_pos[int(pmodel.body_mocapid[bid])] = pos
    mujoco.mj_forward(pmodel, pdata)

    eye_p = to_parent(eye)
    scenes = [("コップを見せて（目の前0.30m・今の提示）", 0.30, lambda c: {"look": c, "hold": (None, c, cup_r)}),
              ("コップを指さして、太郎を見て（0.30m）", 0.30, lambda c: {"look": eye_p, "point": (None, c), "hold": (None, c, cup_r)}),
              ("コップを指さして、太郎を見て（0.55m・物を離す）", 0.55, lambda c: {"look": eye_p, "point": (None, c), "hold": (None, c, cup_r)})]
    tiles = []
    for title, dist, mk in scenes:
        obj_w = eye + fwd * dist; obj_w[2] = max(obj_w[2], 0.06)
        d.qpos[a:a + 3] = obj_w; d.qpos[a + 3:a + 7] = present_quat(u, ang["yaw"], ang["tilt"])
        mujoco.mj_forward(m, d)
        cup_p = to_parent(obj_w)
        bid_obs = mujoco.mj_name2id(pmodel, mujoco.mjtObj.mjOBJ_BODY, "obs_cup")
        pdata.mocap_pos[int(pmodel.body_mocapid[bid_obs])] = cup_p
        mujoco.mj_forward(pmodel, pdata)
        cmd = mk(cup_p)
        q, err = P.solve(pmodel, pdata, cmd, obst, verbose=False)
        pdata.qpos[:] = q; mujoco.mj_forward(pmodel, pdata)
        # 解いた姿勢を太郎の世界へ写す
        for body in M.part_geoms():
            bid = mujoco.mj_name2id(pmodel, mujoco.mjtObj.mjOBJ_BODY, body)
            lp = np.array(pdata.xpos[bid]); lR = np.array(pdata.xmat[bid]).reshape(3, 3)
            wb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "%s_%s" % (PRE, body))
            mi = int(m.body_mocapid[wb])
            d.mocap_pos[mi] = hip_w + Rp @ lp
            d.mocap_quat[mi] = quat_of(Rp @ lR)
        mujoco.mj_forward(m, d)
        bad = [t for t in penetration(m, d, PRE) if "floor" not in t[1]]
        hold_g = ("palm_%s" % cmd["hold"][0]) if "hold" in cmd else "@"
        bad = [t for t in bad if not (t[2] > -0.02 and hold_g[-1] in t[0][-2:])]
        note = "めり込みなし" if not bad else "めり込み %d 箇所（最悪 %.1fcm）" % (len(bad), min(t[2] for t in bad) * 100)
        for t in sorted(bad, key=lambda t: t[2])[:4]:
            print("     %s ↔ %s  %+.1fcm" % t[:2] + "" if False else "     %s ↔ %s  %+.1fcm" % (t[0], t[1], t[2] * 100))
        print("%s | %s | %s" % (title, note, "・".join("%s %.1f" % (k, v) for k, v in err.items())))
        ren = mujoco.Renderer(m, height=224, width=224); ren.update_scene(d, camera="eye_left")
        im_eye = ren.render().copy(); ren.close()
        ren = mujoco.Renderer(m, height=430, width=470)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (eye + hip_w) / 2 + np.array([0, 0, 0.05]); cam.distance = 1.65
        cam.azimuth = np.degrees(np.arctan2(fwd_h[1], fwd_h[0])) + 70.0; cam.elevation = -8
        ren.update_scene(d, camera=cam); im3 = ren.render().copy(); ren.close()
        tiles.append((title, note, im3, im_eye))
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 15)
    f2 = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 13)
    W = 470 + 240
    canvas = Image.new("RGB", (W * 3, 500), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    for i, (title, note, im3, im_eye) in enumerate(tiles):
        x = i * W
        dr.text((x + 6, 4), "「%s」" % title, fill=(0, 0, 120), font=font)
        canvas.paste(Image.fromarray(im3), (x + 2, 26))
        canvas.paste(Image.fromarray(im_eye).resize((224, 224)), (x + 478, 26))
        dr.text((x + 478, 254), "太郎の左目（学習に使う224画素）", fill=(0, 0, 0), font=f2)
        canvas.paste(Image.fromarray(im_eye).resize((224 * 2, 224 * 2), Image.NEAREST).crop((0, 0, 448, 190)), (x + 478 - 460, 280))
        dr.text((x + 20, 458), "↑ 左目の絵を2倍に拡大（上半分）", fill=(0, 0, 0), font=f2)
        dr.text((x + 6, 476), note, fill=(0, 120, 0) if "なし" in note else (180, 0, 0), font=f2)
    p = OUT + "/図_太郎から見た親.png"; canvas.save(p); print("図:", p)
    env.close()


if __name__ == "__main__":
    main()
