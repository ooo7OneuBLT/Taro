"""段階2bの場面づくりと確かめ（2026-10-15、統合担当）。

最初の場面（段階2_3物体_ボール左/右）では、①赤いボールが目立ちの地図の1位になり、条件A
（黙っている間、妨害物を一番長く見る）が成り立たなかった（ボール左で 39/39 がボール）
②ボールとコップが約36度離れ、ボールを見るとコップが視野（片側30度）の外に出た。
決定（スレッドの責任者、2026-10-15）：借り物の目立ちの重みではなく**場面**を直す。
  1. 3つの物を最初の視線から約±12度以内に寄せる（どの2つも25度以内）
  2. まずボールの赤をくすませる。足りなければ妨害物の揺れ・点滅を大きく・速く
  3. 本番の前に、目を動かさない0〜4秒で目立ちの1位を数え、両方の置き方で妨害物がはっきり多数なら進む。3回でだめなら止める

使い方：python G/scripts/stage2b_scenes.py <試行番号 1〜3>
  場面を作り直して（run/scene_tools/make_stage2_scene.py）、試行の頭の視線から見た物の角度と、
  目立ちの1位の数を G/logs/場面2b/試行<番号>.json に書く。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, REPO)
import core_v2  # noqa: E402,F401  負荷の決まり（core_v2/perf.py：スレッド1本など）を numpy・torch より先に効かせる

import numpy as np

NAMES = {+1: "段階2b_3物体_ボール左_2026-10-15", -1: "段階2b_3物体_ボール右_2026-10-15"}
# 試行ごとの設定。角度は make_stage2_scene の意味（ボール・コップは左右に±angle、妨害物は上下に toy3_up）
ATTEMPTS = {
    1: {"angle_deg": 11.0, "toy3_up_deg": -11.0, "ball_rgba": [0.62, 0.40, 0.38, 1.0]},
    # 試行1の実測：左右の中心が左目から +4.5度ずれる（物は両目の真ん中を基準に置かれ、測るのは左目。
    #   30cm先で目の間の半分≒2.4cm → 約4.5度の視差）。左の物が -6.4度＝最初から「見ている」線の内側。
    #   ボールの1位は 30/39・22/39 で足りない。→ 左右を詰め（8度）、左側の物だけ上へ10度上げて10度の外へ出し、赤をもっとくすませる
    2: {"angle_deg": 8.0, "toy3_up_deg": -10.5, "ball_rgba": [0.55, 0.43, 0.41, 1.0],
        "left_elev_deg": 10.0},
    # 試行2の実測：置き方は狙いどおり（左上の物 10.6度・右の物 12.5度・妨害物 11.4度、組の最大 22.1度）。
    #   目立ちの1位は 妨害物20/ボール19（ボール左）・17/22（ボール右）。ボール右では妨害物の揺れがボールに当たり、
    #   ボールが「触ると光る」仕掛けで黄色く光っていた（39回中40回点灯、接触相手 test_object3）。
    #   場面を作り直さずに環境の値だけ変えて下見した（scratchpad の probe7〜10）：
    #   右の物を上へ4度・妨害物を下へ11.5度にすると当たらなくなる。くすませた赤 (0.42,0.38,0.37) は
    #   照明で茶色に写り、青い机との色の差で目立ち続けた（色の特徴 0.91）。机に近い くすんだ桃色
    #   (0.58,0.46,0.50) と、点滅を 2→4.5回/秒（ほぼ毎コマ色が変わり「動き」の特徴が強まる）にすると
    #   妨害物 26/39・29/39。揺れを大きく(4cm)・速く(1回/秒)しても良くならなかった
    3: {"angle_deg": 8.0, "toy3_up_deg": -11.5, "ball_rgba": [0.58, 0.46, 0.50, 1.0],
        "left_elev_deg": 10.0, "right_elev_deg": 4.0,
        "flash": {"amp_m": 0.03, "hz": 0.5, "blink_hz": 4.5}},
}
OUT = os.path.join(REPO, "G", "logs", "場面2b")


def measure(scene, n_ticks=40):
    from core_v2 import run_v2
    from core_v2.tools.register_appearance import project
    from brain.midbrain.salience_map import SalienceMap
    env = run_v2.build_env(scene, None, seed=0)
    try:
        u = env.unwrapped
        env.reset(seed=0)
        R0 = run_v2.frame_at(env)
        d0 = run_v2.directions(env, R0)
        start = {k: (round(d0[f"x_{k}"], 1), round(d0[f"y_{k}"], 1)) for k in run_v2.OBJECTS}
        ang0 = {k: round(v, 1) for k, v in run_v2.gaze_angles(env).items()}
        sm = SalienceMap(cell=28, weights=(1, 1, 1, 1), dog_iters=10)
        act = run_v2.zero_action(env)
        cnt = {}
        for i in range(n_ticks):
            for _ in range(run_v2.K):
                obs, *_ = env.step(act)
            res = sm.update(np.asarray(obs["eye_left"]).astype(np.uint8))
            if i == 0:
                continue        # 1コマ目は動きの特徴が無い
            r, c = np.unravel_index(np.argmax(res["salience"]), (28, 28))
            px, py = (c + .5) * 8, (r + .5) * 8
            pos = {k: project(u, b) for k, b in run_v2.OBJECTS.items()}
            dist = {k: np.hypot(px - p[0], py - p[1]) for k, p in pos.items()}
            k = min(dist, key=dist.get)
            k = k if dist[k] < 20 else "other"
            cnt[k] = cnt.get(k, 0) + 1
        pair = {}
        names = list(start)
        for a in range(3):
            for b in range(a + 1, 3):
                pa, pb = np.array(start[names[a]]), np.array(start[names[b]])
                pair[f"{names[a]}-{names[b]}"] = round(float(np.linalg.norm(pa - pb)), 1)
        return {"start_xy_deg": start, "start_angle_deg": ang0, "pair_deg": pair,
                "salience_rank1": cnt, "n": n_ticks - 1}
    finally:
        run_v2.close_env(env)


def main(attempt):
    from run.scene_tools import make_stage2_scene as mk
    cfg = ATTEMPTS[attempt]
    os.makedirs(OUT, exist_ok=True)
    mk.make_xml()
    for side in (+1, -1):
        kw = {k: v for k, v in cfg.items() if k not in ("left_elev_deg", "right_elev_deg")}
        le = cfg.get("left_elev_deg", 0.0)
        ri = cfg.get("right_elev_deg", 0.0)
        # side=+1（ボール左）は toy1＝ボールが左、-1（ボール右）は toy2＝コップが左
        kw["toy1_elev_deg"] = le if side == +1 else ri
        kw["toy2_elev_deg"] = ri if side == +1 else le
        mk.make_scene(side, names=NAMES, **kw)
    res = {"attempt": attempt, "config": cfg}
    for side, scene in NAMES.items():
        res[scene] = measure(scene)
        print(scene, json.dumps(res[scene], ensure_ascii=False))
    with open(os.path.join(OUT, f"試行{attempt}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]))
