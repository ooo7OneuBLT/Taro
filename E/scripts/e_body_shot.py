"""【目視用】体型の候補を並べた静止画を撮る。

【なぜ要るか】
体格の数値（身長・体重・上肢/下肢比）が合っても、**見た目が壊れていないか**は別問題。
mimoGrowth の custom は geom の大きさしか変えないため、body の位置は動かない。
つまり「geom だけ縮んで関節の間に隙間ができる」ことがありうる。
数値が良くなったからと言って採用する前に、必ず絵で確かめる。
逆に、2026-07-21 の体型は**目視だけで決めて数値を一度も測らなかった**結果、
身長35.6cm・体重1.44kg・頭が体重の49.5% という壊れた体になっていた。
＝**数値と絵の両方**を見ないと決められない。

【2026-07-25 改修】
- 任意の体型係数を渡せるようにした（旧版は leg/trunk しか振れなかった）。
- **`e_body_measure.measure()` とまったく同じ経路で体を作る**ようにした。
  旧版は `d_c5_motor_quality.build`（おもちゃ環境）を使っており、
  「測った体」と「写した体」が同じである保証がなかった。
- 姿勢は測定と同じ `qpos0`（基準姿勢）に固定＝物理の落ち着き方で見え方が変わらない。
- 真上と真横の2アングルを撮り、条件を縦に並べた1枚にまとめる。

使い方:
    python E/scripts/e_body_shot.py                  # v2（今の既定）と探索結果を比較
    python E/scripts/e_body_shot.py leg=0.8 arm=1.3  # 係数を指定して1条件だけ
出力: E/logs/video/bodyshot_compare.png
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
OUT_DIR = os.path.join(_HERE, os.pardir, "logs", "video")

# 子プロセスで1条件ずつ撮る（MuJoCoのモデルはプロセス内で作り直せないため）
CHILD = r'''
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"{here}")
import numpy as np, mujoco, cv2
import e_body_measure as M        # 測定とまったく同じ経路で体を作る
from d_supine_env import SupineMimoEnv
from mimoActuation.muscle import MuscleModel
from infant_body import body_scale_custom, HEAD_ELONGATION

scales = json.loads(os.environ["E_SHAPE_JSON"])
kw = {{}}
custom = body_scale_custom(0.0, scales)
if custom:
    kw["custom_measurements"] = custom
env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=0.0,
                    head_elongation=HEAD_ELONGATION, **kw)
m, d = env.unwrapped.model, env.unwrapped.data
# 測定と同じく基準姿勢に固定（物理の落ち着き方で見え方が変わらない）
d.qpos[:] = m.qpos0
d.qvel[:] = 0
mujoco.mj_forward(m, d)

# 環境物（床・柵）を透明にして太郎だけ見せる
for i in range(m.ngeom):
    nm = m.body(m.geom_bodyid[i]).name
    gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    if nm == "world" or gn.startswith("fence_post"):
        m.geom_rgba[i, 3] = 0.0

H, W = 640, 520
m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W)
m.vis.global_.offheight = max(int(m.vis.global_.offheight), H)
ren = mujoco.Renderer(m, height=H, width=W)

imgs = []
for elev, azim in [(-89.0, 90.0), (-5.0, 90.0)]:   # 真上から / 真横から
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = d.body("upper_body").xpos
    cam.distance = 0.95
    cam.elevation = elev
    cam.azimuth = azim
    ren.update_scene(d, camera=cam)
    imgs.append(ren.render().copy())
img = np.concatenate(imgs, axis=1)
cv2.imwrite(r"{out}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("saved", r"{out}")
env.close()
'''


def shot(scales, path):
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "2",
                "E_SHAPE_JSON": json.dumps(scales)})
    code = CHILD.format(here=_HERE, out=path.replace("\\", "\\\\"))
    r = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    ok = os.path.exists(path)
    print(f"  {os.path.basename(path):<28} {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print((r.stdout or "") + (r.stderr or ""))
    return ok


def _label(img, text, cv2, np):
    """画像の上に帯を足して条件名を書く（日本語は出ないので英数字で書く）。"""
    bar = np.full((34, img.shape[1], 3), 245, dtype=img.dtype)
    cv2.putText(bar, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1,
                cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.path.insert(0, _HERE)
    import e_body_measure as M

    # v3（現在の既定）と、比較用に旧v2（目視だけで決めていた版）
    v3 = dict(M.NEWBORN_SHAPE_DEFAULTS)
    v2 = dict(v3)
    v2.update({"leg": 0.45, "leg_thick": 0.60, "arm": 0.62, "arm_thick": 0.62,
               "trunk_len": 0.74, "trunk_width": 1.0, "foot_width": 1.0})

    cases = [("v2 (old): 35.6cm 1.44kg 2.8heads head-mass 49.5%", v2, "v2"),
             ("v3 (new): 49.3cm 3.54kg 3.9heads head-mass 25.0%", v3, "v3")]

    if len(sys.argv) > 1:                       # 係数を直接指定する使い方
        s = dict(v2)
        for a in sys.argv[1:]:
            k, v = a.split("=")
            s[k] = float(v)
        cases = [("custom", s, "custom")]

    paths = []
    for _, s, tag in cases:
        p = os.path.join(OUT_DIR, f"bodyshot_{tag}.png")
        if shot(s, p):
            paths.append(p)

    if len(paths) < 2:
        return
    import cv2
    import numpy as np
    rows = [_label(cv2.imread(p), cases[i][0], cv2, np) for i, p in enumerate(paths)]
    out = os.path.join(OUT_DIR, "bodyshot_compare.png")
    cv2.imwrite(out, np.concatenate(rows, axis=0))
    print(f"\n比較画像: {out}")


if __name__ == "__main__":
    main()
