"""【目視用】体型補正の前後を並べた静止画を撮る。

【なぜ要るか】
体格の数値（上肢/下肢比など）が合っても、**見た目が壊れていないか**は別問題。
mimoGrowth の custom は geom の大きさしか変えないため、body の位置は動かない。
つまり「geom だけ縮んで関節の間に隙間ができる」ことがありうる。
数値が良くなったからと言って採用する前に、必ず絵で確かめる。

使い方: python e_body_shot.py [leg_scale] [trunk_scale]
出力: E/logs/video/bodyshot_leg{..}_trunk{..}.png
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
OUT_DIR = os.path.join(_HERE, os.pardir, "logs", "video")

# 子プロセスで1条件ずつ撮る（MuJoCoのモデルはプロセス内で作り直せないため）
CHILD = r'''
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"{here}")
sys.path.insert(0, os.path.join(r"{root}", "D", "scripts"))
sys.path.insert(0, os.path.join(r"{root}", "C", "scripts"))
sys.path.insert(0, os.path.join(r"{root}", "taro_core"))
import paths; paths.setup_brain_path(); sys.path.insert(0, paths.MIMO_DIR)
import mimoEnv, mujoco, numpy as np, cv2
import d_c5_motor_quality as mq
env, *_ = mq.build("off", age=0)
m, d = env.unwrapped.model, env.unwrapped.data
mujoco.mj_forward(m, d)
# 柵を消す（太郎が見えないため）。物理は動かさず描画だけ透明にする
for i in range(m.ngeom):
    nm = m.body(m.geom_bodyid[i]).name
    if nm in ("world",):
        m.geom_rgba[i, 3] = 0.0
m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 900)
m.vis.global_.offheight = max(int(m.vis.global_.offheight), 700)
ren = mujoco.Renderer(m, height=700, width=900)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
cam.lookat[:] = d.body("upper_body").xpos
cam.distance = 0.95
cam.elevation = -89.0      # 真上から（仰向けの全身が見える）
cam.azimuth = 90.0
ren.update_scene(d, camera=cam)
img = ren.render().copy()
cv2.imwrite(r"{out}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("saved", r"{out}")
env.close()
'''


def shot(leg, trunk, path, width=1.0):
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "E_TOY": "1", "E_VOR": "1", "C5_AGE": "0",
                "E_TOY_OBJ": "0", "E_FENCE": "1", "E_PLAIN": "1", "C_TOUCH": "1",
                "C_TOUCH_MODE": "input", "OMP_NUM_THREADS": "2",
                "E_LEG_SCALE": str(leg), "E_TRUNK_LEN": str(trunk),
                "E_TRUNK_WIDTH": str(width)})
    code = CHILD.format(here=_HERE, root=_ROOT, out=path.replace("\\", "\\\\"))
    r = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    ok = os.path.exists(path)
    print(f"  leg={leg} trunk={trunk} → {'OK' if ok else 'FAIL'}")
    if not ok:
        print((r.stdout or "") + (r.stderr or ""))
    return ok


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = [(1.0, 1.0, "補正なし")]
    if len(sys.argv) > 1:
        cases.append((float(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 1.0, "補正あり"))
    else:
        cases += [(0.62, 1.0, "脚のみ"), (0.62, 0.80, "脚+胴")]
    for leg, trunk, name in cases:
        p = os.path.join(OUT_DIR, f"bodyshot_leg{leg}_trunk{trunk}.png")
        shot(leg, trunk, p)


if __name__ == "__main__":
    main()
