"""段階2の合格テストを動画で見る（2026-10-16、統合担当）。

1試行（10秒）を core_v2 の脳で回し、脳1回（0.1秒）ごとに1コマ描いて、実時間（10コマ/秒）の mp4 にする。
* 左：外から見たカメラ（太郎の頭と3つの物が入る固定の画角）。視線を黄色い線で描く
* 右：太郎の左目の実際の入力（脳が見ている画像）。真ん中の＋が視線の先。物の名前を添える
* 下：時刻、親の発話（4秒から）、目的（探している物）、反射を抑えているか、3つの物への角度の時間変化
描き方は run/plugins/common/view_video.py（実験用の動画の道具）と同じ考え（MuJoCo のオフスクリーン描画・
自由カメラ・ffmpeg）。あちらは旧太郎の学習ループ（ctx）につながっているので、core_v2 用にここで書いた。

使い方：python G/scripts/make_video.py <場面> <語> <種> <出力.mp4>
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, REPO)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core_v2 import run_v2
from core_v2.brain import Brain
from core_v2.tools.register_appearance import project

FONT = "C:/Windows/Fonts/meiryo.ttc"
W, H = 1280, 720
TP_W, EYE = 720, 560          # 外のカメラの幅・目の映像の一辺
NAMES = {"ball": "ボール", "cup": "コップ", "distractor": "光る物"}
COL = {"ball": (230, 80, 80), "cup": (200, 200, 200), "distractor": (90, 150, 255)}
WORD_TARGET = {"ボールはどこ？": "ボール", "コップはどこ？": "コップ"}


def main(scene, word, seed, out):
    import mujoco
    seed = int(seed)
    brain = Brain()
    brain.boxes["eyein"].setup()
    brain.boxes["semantic"].load_table(os.path.join(REPO, "G", "models", "semantic_table.pt"))
    env = run_v2.build_env(scene, word, seed=seed)
    u = env.unwrapped
    m = u.model
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), TP_W)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), EYE)
    ren = mujoco.Renderer(m, height=EYE, width=TP_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cid = int(m.camera("eye_left").id)
    font = ImageFont.truetype(FONT, 24)
    font_s = ImageFont.truetype(FONT, 18)
    font_b = ImageFont.truetype(FONT, 30)
    tmp = tempfile.mkdtemp(prefix="stage2_video_")
    state = {"n": 0, "heard_t": None, "fire_t": -9.0, "hist": []}
    target_name = WORD_TARGET.get(word, "")

    def third_person():
        head = np.array(u.data.body("head").xpos, dtype=float)
        toys = np.array([u.data.body(b).xpos for b in run_v2.OBJECTS.values()], dtype=float)
        toy = toys.mean(axis=0)
        mid = head * 0.45 + toy * 0.55
        cam.lookat[:] = mid
        cam.distance = 0.62
        cam.azimuth = math.degrees(math.atan2(toy[1] - head[1], toy[0] - head[0])) + 125.0
        cam.elevation = -18
        ren.update_scene(u.data, camera=cam)
        scn = ren.scene
        if scn.ngeom < scn.maxgeom:      # 視線（左目から 0.32m 先まで）
            eye = np.array(u.data.cam_xpos[cid], dtype=float)
            g = u._gaze_dir()
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                np.zeros(3), np.eye(3).flatten(), np.array([1.0, 0.85, 0.1, 0.9], np.float32))
            mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025,
                                 eye, eye + g * 0.32)
            scn.ngeom += 1
        return ren.render().copy()

    def on_tick(row):
        rec = run_v2.path_record(brain)
        t = row["t"]
        if row["heard"] and state["heard_t"] is None:
            state["heard_t"] = t
        if row["cmd_h"] is not None:
            state["fire_t"] = t
        state["hist"].append((t, row["ang_ball"], row["ang_cup"], row["ang_distractor"]))
        goal = brain.boxes["goal"]
        seeking = goal.held is not None
        inhibit = float(rec["gate3>orienteye"]) > 0.5
        # どれを見ているか（10度以内で一番近い物）
        angs = {k: row[f"ang_{k}"] for k in run_v2.OBJECTS}
        k_look, a_look = min(angs.items(), key=lambda kv: kv[1])
        looking = NAMES[k_look] if a_look <= 10.0 else "（どれも見ていない）"

        canvas = Image.new("RGB", (W, H), (18, 18, 22))
        canvas.paste(Image.fromarray(third_person()), (0, 0))
        img = np.asarray(row["_eye"]).astype(np.uint8)
        eye_img = Image.fromarray(img).resize((EYE, EYE), Image.BILINEAR)
        de = ImageDraw.Draw(eye_img)
        k = EYE / 224.0
        for key, body in run_v2.OBJECTS.items():
            px, py = project(u, body)
            if 0 <= px < 224 and 0 <= py < 224:
                x, y = px * k, py * k
                de.text((x + 26, y - 14), NAMES[key], fill=COL[key], font=font_s,
                        stroke_width=2, stroke_fill=(0, 0, 0))
        c = EYE / 2
        de.line([c - 22, c, c + 22, c], fill=(255, 220, 0), width=4)
        de.line([c, c - 22, c, c + 22], fill=(255, 220, 0), width=4)
        de.text((c + 8, c + 6), "視線", fill=(255, 220, 0), font=font_s, stroke_width=2, stroke_fill=(0, 0, 0))
        canvas.paste(eye_img, (TP_W, 0))
        dr = ImageDraw.Draw(canvas)
        dr.text((10, 8), "外から見た様子（黄色い線＝視線）", fill=(255, 255, 255), font=font_s,
                stroke_width=2, stroke_fill=(0, 0, 0))
        dr.text((TP_W + 10, 8), "太郎の左目に映っているもの", fill=(255, 255, 255), font=font_s,
                stroke_width=2, stroke_fill=(0, 0, 0))

        # 下の帯
        y0 = EYE + 8
        dr.text((14, y0), f"{t:4.1f} 秒", fill=(230, 230, 230), font=font_b)
        if state["heard_t"] is not None and t - state["heard_t"] < 2.0:
            dr.text((160, y0), f"親：「{word}」", fill=(255, 225, 110), font=font_b)
        elif state["heard_t"] is None:
            dr.text((160, y0 + 4), "親：（まだ黙っている。4秒に話しかける）", fill=(150, 150, 150), font=font)
        dr.text((14, y0 + 50), "探している物：" + (target_name if seeking else "なし"),
                fill=(120, 230, 140) if seeking else (170, 170, 170), font=font)
        dr.text((14, y0 + 86), "目立つ物への反射：" + ("抑えている" if inhibit else "ふつう"),
                fill=(255, 160, 90) if inhibit else (170, 170, 170), font=font)
        dr.text((14, y0 + 122), "いま見ている物：" + looking, fill=(230, 230, 230), font=font)
        if t - state["fire_t"] < 0.3:
            dr.text((420, y0 + 122), "← 目を跳ばした", fill=(255, 220, 0), font=font)

        # 角度の時間変化（0〜10秒、0〜30度）
        gx0, gx1, gy0, gy1 = 700, 1265, y0 + 30, H - 10
        dr.rectangle([gx0, gy0, gx1, gy1], outline=(80, 80, 80))
        def X(tt): return gx0 + (gx1 - gx0) * min(tt, 10.5) / 10.5
        def Y(a): return gy1 - (gy1 - gy0) * min(a, 30.0) / 30.0
        dr.line([gx0, Y(10), gx1, Y(10)], fill=(90, 90, 90), width=1)
        dr.text((gx1 - 170, Y(10) + 2), "↓10度以内＝見ている", fill=(130, 130, 130), font=font_s)
        if state["heard_t"] is not None:
            dr.line([X(state["heard_t"]), gy0, X(state["heard_t"]), gy1], fill=(255, 225, 110), width=2)
        hist = state["hist"]
        for j, key in enumerate(("ball", "cup", "distractor")):
            pts = [(X(h[0]), Y(h[1 + j])) for h in hist]
            if len(pts) > 1:
                dr.line(pts, fill=COL[key], width=3)
        dr.text((gx0, gy0 - 28), "視線と各物の角度（赤＝ボール 白＝コップ 青＝光る物）",
                fill=(170, 170, 170), font=font_s)
        canvas.save(os.path.join(tmp, "f%05d.png" % state["n"]))
        state["n"] += 1

    # run_trial は目の画像を row に入れないので、環境の最新の観測を横取りする
    orig_step = env.step

    def step_and_keep(a):
        res = orig_step(a)
        step_and_keep.last_obs = res[0]
        return res
    env.step = step_and_keep

    def tick_hook(row):
        row["_eye"] = step_and_keep.last_obs["eye_left"]
        on_tick(row)
        row.pop("_eye")

    try:
        rows = run_v2.run_trial(brain, env, seed=seed, duration_s=10.0, on_tick=tick_hook)
    finally:
        ren.close()
        env.step = orig_step
        run_v2.close_env(env)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10",
                    "-i", os.path.join(tmp, "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-preset", "medium", out], check=True)
    shutil.copy(os.path.join(tmp, "f%05d.png" % 60), out.replace(".mp4", "_6秒のコマ.png"))
    shutil.rmtree(tmp, ignore_errors=True)
    # この試行の判定（合格テストと同じ判定）
    sys.path.insert(0, HERE)
    from pass_test import judge
    cfg = {"look_deg": 10.0, "word_t": 4.0, "b_within_s": 3.0, "b_hold_s": 1.0}
    tgt = {"ボールはどこ？": "ball", "コップはどこ？": "cup"}[word]
    res = judge(rows, tgt, cfg)
    print(f"保存: {out}  {os.path.getsize(out) / 1e6:.1f}MB  判定: A={res['A']} B={res['B']} "
          f"既に見ていた={res['already_on_target']} 着いてから={res['dwell_first_s']}秒 発話前={res['pre_counts']}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
