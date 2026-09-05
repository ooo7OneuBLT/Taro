# -*- coding: utf-8 -*-
"""提示角度の静止測定（2026-09-03・F2-49）。本番と同じ環境を組み、親が持つ位置に物を置いて、
太郎の左目（学習に使う画像そのもの）から見た「傾き」「横回転」の掃引を一覧にする。
向きの計算は parent_labeling._present_quat と同じ式（yaw/tilt を引数にしただけ）。

    .venv/Scripts/python.exe F/scripts/f_present_angle_probe.py [実験JSON]
出力: F/logs/F2-49_実物スキャン/図_提示角度_傾き掃引.png / 図_提示角度_横回転掃引.png
"""
import os, sys, io, json
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
from run.plugins.common import scene as scene_mod

FONT = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 15)
SPEC = sys.argv[1] if len(sys.argv) > 1 else "F/experiments/F2-49probe_2026-09-03.json"
# 枠 → 語（f_gen_f49.py の WORDS と同じ順）
import f_gen_f49 as _G                                   # 枠→語は生成器と同じ定義を使う（8語版・2026-09-03）
SLOT_WORD = {k: w for k, (w, _, _) in zip(_G.SLOT_KEYS, _G.WORDS)}
FAR = np.array([5.0, 3.0, -2.0])


def present_quat(env, yaw_deg, tilt_deg):
    cid = int(env.model.camera("eye_left").id)
    fwd = -np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]
    ex = -fwd; ex[2] = 0.0
    n = np.linalg.norm(ex); ex = ex / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0]); ey = np.cross(ez, ex)
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    ex2 = c * ex + s * ey; ey2 = -s * ex + c * ey
    ct, st = np.cos(np.radians(tilt_deg)), np.sin(np.radians(tilt_deg))
    ex3 = ct * ex2 + st * ez; ez3 = -st * ex2 + ct * ez
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.stack([ex3, ey2, ez3], axis=1).ravel())
    return q


def main():
    spec = json.load(io.open(SPEC, encoding="utf-8"))
    env, sc, _ = scene_mod.build(spec["scene"], taro=spec["taro"], seed=int(spec["run"].get("seed", 0)), verbose=False)
    env.reset(seed=int(spec["run"].get("seed", 0)))
    u = env.unwrapped; m, d = u.model, u.data
    dist = float(sc["world"]["parent_labeling"].get("follow_dist", 0.3))
    slots = dict(getattr(u, "_present_slots", {}))
    qadr = {k: v["qadr"] for k, v in slots.items()}
    qadr["toy1"] = int(u._toy_qadr)
    print("枠:", sorted(qadr))
    cid = int(m.camera("eye_left").id)
    ren = mujoco.Renderer(m, height=224, width=224)

    def shot(slot, yaw, tilt):
        for k, a in qadr.items():              # 全部を遠くへ
            d.qpos[a:a + 3] = FAR
        mujoco.mj_forward(m, d)
        eye = np.array(d.cam_xpos[cid]); fwd = -np.array(d.cam_xmat[cid]).reshape(3, 3)[:, 2]
        goal = eye + fwd * dist; goal[2] = max(goal[2], 0.05)
        a = qadr[slot]
        d.qpos[a:a + 3] = goal
        d.qpos[a + 3:a + 7] = present_quat(u, yaw, tilt)
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera="eye_left")
        return ren.render().copy()

    T, L = 224, 20
    order = [k for k in _G.SLOT_KEYS if k in qadr]
    for title, sweep, fixed, out in (
            ("傾き掃引（横30固定）", [("傾%d" % t, 30, t) for t in (-60, -25, 0, 25, 60)], None, "図_提示角度_傾き掃引.png"),
            ("横回転掃引（傾25固定）", [("横%d" % y, y, 25) for y in (-60, -30, 0, 30, 60, 90)], None, "図_提示角度_横回転掃引.png")):
        canvas = Image.new("RGB", (len(order) * (T + 4) + 70, len(sweep) * (T + 4) + L), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        dr.text((4, 2), "太郎の左目・親が持つ位置 " + title, fill=(0, 0, 0), font=FONT)
        for j, slot in enumerate(order):
            dr.text((70 + j * (T + 4) + 60, 2 + 0), "", fill=(0, 0, 0), font=FONT)
        for i, (lab, yaw, tilt) in enumerate(sweep):
            dr.text((4, L + i * (T + 4) + 100), lab, fill=(0, 0, 0), font=FONT)
            for j, slot in enumerate(order):
                img = shot(slot, yaw, tilt)
                x, y = 70 + j * (T + 4), L + i * (T + 4)
                canvas.paste(Image.fromarray(img), (x, y))
                dr.text((x + 2, y + 2), SLOT_WORD[slot], fill=(255, 255, 0), font=FONT)
        p = "F/logs/F2-49_実物スキャン/" + out
        canvas.save(p); print("保存:", p)
    # 案：物ごとの「それらしく見える」角度（静止測定 2026-09-03 から選定。傾きは負が「上面を太郎へ」）
    canvas = Image.new("RGB", (len(order) * (T + 4) + 70, 2 * (T + 4) + L), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    dr.text((4, 2), "上段：今（横30・傾+25＝底側が見える） 下段：案（物ごと）", fill=(0, 0, 0), font=FONT)
    for i, lab in enumerate(("今", "案")):
        dr.text((4, L + i * (T + 4) + 100), lab, fill=(0, 0, 0), font=FONT)
        for j, slot in enumerate(order):
            yaw, tilt = (30, 25) if i == 0 else (_G.PRESENT_ANGLES[SLOT_WORD[slot]]["yaw"], _G.PRESENT_ANGLES[SLOT_WORD[slot]]["tilt"])
            img = shot(slot, yaw, tilt)
            x, y = 70 + j * (T + 4), L + i * (T + 4)
            canvas.paste(Image.fromarray(img), (x, y))
            dr.text((x + 2, y + 2), "%s 横%d 傾%d" % (SLOT_WORD[slot], yaw, tilt), fill=(255, 255, 0), font=FONT)
    p = "F/logs/F2-49_実物スキャン/図_提示角度_案.png"
    canvas.save(p); print("保存:", p)
    ren.close()


PROPOSAL = {"toy1": {"yaw": 0, "tilt": -20},     # くつ：横顔＋少し上から
            "toy5": {"yaw": 60, "tilt": -25},    # コップ：取っ手と口が見える
            "toy6": {"yaw": 30, "tilt": -60},    # おさら：面を向ける
            "toy7": {"yaw": 30, "tilt": -55},    # おわん：中が見える
            "toy8": {"yaw": 30, "tilt": -25},    # つみき（素材が不適・別途判断）
            "toy9": {"yaw": 30, "tilt": -20},    # かばん：正面やや上
            "toy10": {"yaw": 0, "tilt": -10},    # がおー：横顔（縮まない）
            "toy11": {"yaw": -30, "tilt": -15},  # バス：斜め前から車輪と顔
            "toy12": {"yaw": 30, "tilt": -25}}   # ボール：どこからでも同じ


if __name__ == "__main__":
    main()
