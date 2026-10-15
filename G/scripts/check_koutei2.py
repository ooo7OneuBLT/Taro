"""段階2・工程2の確かめ（仕様書 §2 の終わり）。

1. 環境と一緒に30秒回して落ちない
2. eyein から物ごとの荷札が出る（3つの物がそれぞれ切り出されているかを、物の画素位置と突き合わせる）
3. earin に発話が届く（4秒の「ボールはどこ？」）
4. eyeout へ手で方向を入れると目が動く（右0.5 → 10秒、上0.5 → 20秒。_gaze_dir の変化で見る）

7つの仮部品はまだ素通しのまま（工程3の前）。
出力：G/logs/工程2/check.csv と要約の表示。
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, REPO)

import numpy as np
import torch

from core_v2 import run_v2
from core_v2.brain import Brain

OUT = os.path.join(REPO, "G", "logs", "工程2")
os.makedirs(OUT, exist_ok=True)


def project(u, body):
    cid = int(u.model.camera("eye_left").id)
    eye = u.data.cam_xpos[cid]; R = u.data.cam_xmat[cid].reshape(3, 3)
    v = u.data.body(body).xpos - eye
    x, y, z = v @ R[:, 0], v @ R[:, 1], -(v @ R[:, 2])
    f = 112.0 / np.tan(np.radians(u.model.cam_fovy[cid]) / 2)
    return 112 + f * x / z, 112 - f * y / z


def main():
    brain = Brain()
    brain.boxes["eyein"].setup()
    env = run_v2.build_env("段階2_3物体_ボール左_2026-10-15", "ボールはどこ？", seed=0)
    u = env.unwrapped
    eyeout = brain.boxes["eyeout"]
    kicks = {100: (0.5, 0.0), 200: (0.0, 0.5)}

    def manual(i):
        if i in kicks:
            r = eyeout.step({"orienteye": torch.tensor(kicks[i], dtype=torch.float32)})
            return r.get("env")
        return None

    def on_tick(row):
        objs = brain.buffer[("eyein", "gaze")]
        row["n_objs"] = len(objs)
        dets = brain.boxes["eyein"].last_dets
        for k, body in run_v2.OBJECTS.items():
            px, py = project(u, body)
            d = min([np.hypot(dd["pos"][0] - px, dd["pos"][1] - py) for dd in dets] or [999])
            row[f"det_{k}"] = int(d < 15)
        row["listen_in"] = brain.buffer[("earin", "listen")]
        row["sal_max"] = float(brain.buffer[("eyein", "prio")].max())

    rows = run_v2.run_trial(brain, env, seed=0, duration_s=30.0, on_tick=on_tick,
                            manual_eye=manual)
    with open(os.path.join(OUT, "check.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n = len(rows)
    print(f"tick {n}（{rows[-1]['t']:.1f}秒）落ちずに完走")
    print("物が切り出された割合:", {k: round(np.mean([r[f'det_{k}'] for r in rows]), 2)
                              for k in run_v2.OBJECTS})
    print("物の数（平均）:", round(np.mean([r["n_objs"] for r in rows]), 1))
    print("earin→listen:", [(r["t"], r["listen_in"]) for r in rows if r["listen_in"]])
    print("heard:", [(r["t"], r["heard"]) for r in rows if r["heard"]])
    for i in kicks:
        a0 = {k: rows[i][f"ang_{k}"] for k in run_v2.OBJECTS}
        a1 = {k: rows[i + 5][f"ang_{k}"] for k in run_v2.OBJECTS}
        print(f"tick {i} に方向 {kicks[i]}：物への角度 {a0} → 0.5秒後 {a1}")

    # 向きの確認：環境だけで、右0.5・上0.5 を入れて視線の変化をカメラの軸へ射影
    for hv in ((0.5, 0.0), (0.0, 0.5)):
        obs, _ = env.reset(seed=0)
        act = run_v2.zero_action(env)
        for _ in range(100):
            env.step(act)
        cid = int(u.model.camera("eye_left").id)
        R0 = u.data.cam_xmat[cid].reshape(3, 3).copy(); g0 = u._gaze_dir().copy()
        u._orienting.set_map_target(*hv)
        for _ in range(50):
            env.step(act)
        g1 = u._gaze_dir()
        right = np.degrees(np.arcsin(np.clip(g1 @ R0[:, 0], -1, 1)))
        up = np.degrees(np.arcsin(np.clip(g1 @ R0[:, 1], -1, 1)))
        print(f"方向 {hv} → 視線 右へ {right:+.1f}度・上へ {up:+.1f}度")
    run_v2.close_env(env)


if __name__ == "__main__":
    main()
