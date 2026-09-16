# -*- coding: utf-8 -*-
"""動きの向きの検出（2026-09-03・下見）：太郎の目で上下左右を見分けられるか。

今の太郎は「動いている強さ」だけを持ち、向きを持たない
（`taro_core/src/brain/midbrain/orienting.py` の `_detect_motion` は差の絶対値）。
人間は生まれつき動きの向きを見る回路を持つので、そこを足したい。

やり方はライヒャルト型：**少し前の像を上下左右にずらして今の像と比べ、
一番よく重なるずらし方＝動いた向き**とする。動きのある画素だけで比べる
（背景が静止していると、全画面の平均では向きが消えるため）。

    .venv/Scripts/python.exe F/scripts/f57_motion_direction_probe.py
出力: F/logs/F2-57_動きの向き/図_動きの向きの検出.png
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

OUT = "F/logs/F2-57_動きの向き"
MAX_SHIFT = 14         # 何画素まで探すか（224画素の目で、1歩あたりの動きに合わせる）


def gray(img):
    a = np.asarray(img, dtype=np.float32)
    return (a.mean(axis=-1) if a.ndim == 3 else a) / 255.0


def flow(prev, cur, max_shift=MAX_SHIFT, thresh=0.02):
    """少し前の像をずらして今の像と比べ、最も合う変位を返す。
    動いた画素だけで比べる（静止した背景に引っ張られないように）。
    返り値: (dx, dy, 動きの強さ)。dx>0 は右、dy>0 は下（画像の行方向）。"""
    diff = np.abs(cur - prev)
    mask = diff > thresh
    strength = float(diff[mask].sum() / diff.size) if mask.any() else 0.0
    if mask.sum() < 20:
        return 0.0, 0.0, strength
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - max_shift, 0), min(ys.max() + max_shift + 1, cur.shape[0])
    x0, x1 = max(xs.min() - max_shift, 0), min(xs.max() + max_shift + 1, cur.shape[1])
    c = cur[y0:y1, x0:x1]
    best = None
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            p = prev[y0 - dy:y1 - dy, x0 - dx:x1 - dx] if (y0 - dy >= 0 and x0 - dx >= 0
                                                           and y1 - dy <= prev.shape[0]
                                                           and x1 - dx <= prev.shape[1]) else None
            if p is None or p.shape != c.shape:
                continue
            err = float(np.abs(c - p).mean())
            if best is None or err < best[0]:
                best = (err, dx, dy)
    return (float(best[1]), float(best[2]), strength) if best else (0.0, 0.0, strength)


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
    base = eye + fwd * 0.30; base[2] = max(base[2], 0.10)
    ren = mujoco.Renderer(m, height=224, width=224)

    def shot(pos):
        d.qpos[a:a + 3] = pos
        d.qpos[a + 3:a + 7] = present_quat(u, ang["yaw"], ang["tilt"])
        mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
        return ren.render().copy()

    # 物を1歩あたり 1.2cm ずつ動かし、太郎の目で向きが取れるか見る
    step = 0.012
    moves = [("下へ（落ちる）", -up), ("上へ", up), ("右へ", right), ("左へ", -right),
             ("止まっている", np.zeros(3))]
    rows, imgs = [], []
    for name, v in moves:
        prev_img = shot(base)
        cur_img = shot(base + v * step)
        dx, dy, st = flow(gray(prev_img), gray(cur_img))
        rows.append((name, dx, dy, st))
        imgs.append((name, prev_img, cur_img))
        print("%-14s → 横 %+.0f 画素・縦 %+.0f 画素（下が+）・強さ %.4f" % (name, dx, dy, st))
    ren.close()
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 14)
    f2 = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 12)
    W, H = 232, 300
    canvas = Image.new("RGB", (W * len(imgs), H), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    for i, ((name, p_im, c_im), (nm, dx, dy, st)) in enumerate(zip(imgs, rows)):
        x = i * W
        dr.text((x + 4, 4), name, fill=(0, 0, 120), font=font)
        canvas.paste(Image.fromarray(c_im).resize((224, 224)), (x + 4, 26))
        judge = "→右" if dx > 0.5 else ("←左" if dx < -0.5 else "")
        judge += "↓下" if dy > 0.5 else ("↑上" if dy < -0.5 else "")
        dr.text((x + 4, 256), "検出：横%+.0f 縦%+.0f" % (dx, dy), fill=(0, 0, 0), font=f2)
        dr.text((x + 4, 274), "判定：%s（強さ %.3f）" % (judge or "動きなし", st), fill=(0, 0, 0), font=f2)
    p = OUT + "/図_動きの向きの検出.png"; canvas.save(p); print("図:", p)
    env.close()


if __name__ == "__main__":
    main()
