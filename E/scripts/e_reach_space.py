"""E1準備：仰向け・新生児体の太郎の「腕が届く範囲」を実測する。

【なぜ測るか】
E1は「手の届く所におもちゃを置く」設計。だが**どこが"届く所"かは体格で決まる**
（新生児体=mimoGrowthのage=0は18ヶ月体より小さい）。勘で置くと
  ・遠すぎ → 一生触れない（随伴性を経験できず連鎖が始まらない）
  ・近すぎ → 何もしなくても当たる（＝リーチの創発を測れない／自己接触と混ざる）
のどちらかになる。**実測してから置く**のが落とし穴チェックの筋。

【測り方の要点＝体の移動を除く】
第1版はワールド座標で手の範囲を測ったが、**太郎は暴れて体ごと移動する**ため
（実測：頭・肩のxが±0.25〜0.3動く）、手の範囲に体の移動が混ざって過大評価になった。
そこで**肩(upper_arm)から手への相対ベクトル**で測る＝純粋な腕の到達範囲。
おもちゃの置き場所は「体のどこから見て何cm」で決めたいので、これが正しい基準。

使い方:
  python e_reach_space.py [n_steps] [age]
    n_steps: ランダム行動の回数（既定300）
    age    : 体の月齢 0-24（既定0＝新生児）
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import gymnasium as gym  # noqa: E402
import mimoEnv  # noqa: F401,E402
from gymnasium.envs.registration import register  # noqa: E402
from d_supine_env import SupineMimoEnv  # noqa: F401,E402

register(id="TaroSupine-v0", entry_point="d_supine_env:SupineMimoEnv",
         max_episode_steps=6000)

# 手とその付け根（肩）の対。相対ベクトル＝純粋な腕の到達。
ARMS = [("right_hand", "right_upper_arm"), ("left_hand", "left_upper_arm")]


def summarize(name, arr):
    a = np.asarray(arr)
    lo, hi, mean = a.min(0), a.max(0), a.mean(0)
    print(f"  {name:26s} x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}] "
          f"z[{lo[2]:+.3f},{hi[2]:+.3f}]")
    return lo, hi, mean


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    age = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    print(f"=== supine / body age {age:.0f}mo : reachable arm space ===")
    print(f"random actions = {n_steps} (motor babbling)\n")

    env = gym.make("TaroSupine-v0", vision_params=None, touch_params=None, age=age)
    obs, _ = env.reset(seed=0)
    mj = env.unwrapped

    # --- 初期姿勢（何もしない時）の肩→手 ---
    print("[initial pose : shoulder -> hand vector]")
    init_rel = {}
    for hand, arm in ARMS:
        rel = mj.data.body(hand).xpos - mj.data.body(arm).xpos
        init_rel[hand] = rel.copy()
        print(f"  {hand:12s} rel=({rel[0]:+.3f},{rel[1]:+.3f},{rel[2]:+.3f}) "
              f"dist={np.linalg.norm(rel):.3f} m")

    # --- 体の寸法（囲い＝ベビーサークルの設計に使う）---
    # 太郎の体は実乳児と完全に同じではないので、**実測して**囲いの大きさを決める。
    names = [mj.model.body(i).name for i in range(mj.model.nbody)]
    body_pts = np.array([mj.data.body(i).xpos.copy() for i in range(mj.model.nbody)
                         if not names[i].startswith("test_object")
                         and names[i] not in ("world",)])
    blo, bhi = body_pts.min(0), body_pts.max(0)
    print("\n[body extent at rest (world coords) - for fence design]")
    print(f"  x[{blo[0]:+.3f},{bhi[0]:+.3f}] (len {bhi[0]-blo[0]:.3f} m)  "
          f"y[{blo[1]:+.3f},{bhi[1]:+.3f}] (width {bhi[1]-blo[1]:.3f} m)  "
          f"z[{blo[2]:+.3f},{bhi[2]:+.3f}]")
    print(f"  center=({(blo[0]+bhi[0])/2:+.3f},{(blo[1]+bhi[1])/2:+.3f})")

    # --- ランダム行動（＝運動性喃語）中の到達 ---
    rel_track = {h: [] for h, _ in ARMS}
    dist_track = {h: [] for h, _ in ARMS}
    body_track = []           # 体（頭）のワールド位置＝どれだけ移動したか
    n_act = env.action_space.shape[0]
    for i in range(n_steps):
        a = np.random.uniform(-1.0, 1.0, size=n_act)
        te = tr = False
        for _ in range(20):   # 1判断=20物理ステップ(0.1秒)
            obs, _, te, tr, _ = env.step(a)
            for hand, arm in ARMS:
                rel = mj.data.body(hand).xpos - mj.data.body(arm).xpos
                rel_track[hand].append(rel.copy())
                dist_track[hand].append(float(np.linalg.norm(rel)))
            body_track.append(mj.data.body("head").xpos.copy())
            if te or tr:
                break
        if te or tr:
            obs, _ = env.reset()

    n = len(rel_track["right_hand"])
    print(f"\n[shoulder -> hand vector during babbling]  samples={n}")
    stats = {h: summarize(h, rel_track[h]) for h, _ in ARMS}

    print("\n[arm extension = |shoulder -> hand| ]")
    for hand, _ in ARMS:
        d = np.asarray(dist_track[hand])
        print(f"  {hand:12s} min={d.min():.3f} mean={d.mean():.3f} "
              f"max={d.max():.3f} p95={np.percentile(d,95):.3f} m")

    b = np.asarray(body_track)
    print("\n[body drift (head world pos) = how much Taro moves while babbling]")
    print(f"  x[{b[:,0].min():+.3f},{b[:,0].max():+.3f}] "
          f"y[{b[:,1].min():+.3f},{b[:,1].max():+.3f}] "
          f"z[{b[:,2].min():+.3f},{b[:,2].max():+.3f}]")

    # --- おもちゃ配置の指針 ---
    print("\n=== toy placement guide ===")
    for hand, _ in ARMS:
        d = np.asarray(dist_track[hand])
        d0 = np.linalg.norm(init_rel[hand])
        print(f"  {hand}: rest={d0:.3f} m, babbling max={d.max():.3f} m, "
              f"p95={np.percentile(d,95):.3f} m")
    print("  -> place the toy so that: dist(shoulder,toy) > rest-distance")
    print("     (not touching by default) AND < p95 reach (reachable when moving).")
    print("  NOTE: this range is what the hand SWEEPS while flailing, not what it can")
    print("        AIM at. Aiming is exactly what E1 must make emerge.")
    print("  NOTE: the toy has a freejoint -> it falls/rolls. Put it on the floor")
    print("        beside the hand, or give it friction/shape so it stays.")

    env.close()


if __name__ == "__main__":
    main()
