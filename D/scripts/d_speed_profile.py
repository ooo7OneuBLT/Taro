"""
診断：太郎の1秒はどこに消えているのか（＝学習量を買うために、どこを削ればいいか）。

【なぜ必要か】2026-07-15
発達研究に照らすと、**太郎の経験量は胎児の約700分の1**：
  胎児の自己接触の練習：14週〜出生の約700時間（胎動が全時間の2割として概算）
    ・自己接触は14週から観察される（Zoia et al. 2007, n=8, 4D超音波）
    ※【訂正・2026-07-16】以前ここには「19週で先読みまで完成して生まれてくる」と書いていたが
      **過大主張**だった。口の開きが接触に先行するのは事実（Myowa-Yamakoshi & Takeshita 2006）
      だが「予期」は著者の解釈で、反射との弁別はできていない（詳細は d1_env.py の訂正メモ）。
      なお**700時間という概算自体は「経験量の桁」の話**なので、この訂正では揺らがない。
  太郎の人生：3600判断 × 1 sim秒 = **1時間**
このままでは、どんな設計でも人間の発達に追いつけない。**まず速度を測る。**

【今日の教訓】思いつきで手を打って6回外した。「触覚が重そう」も憶測。**測ってから削る。**

【測る対象】太郎の実構成そのもの（仰向け・筋肉モデル・乳児触覚・視覚なし）で、
  ①触覚あり ②触覚なし を比べる＝触覚のコストが分かる（0コストの引き算）
さらに cProfile で関数ごとの内訳を出す。

使い方: python d_speed_profile.py [n_step]
"""
import os, sys, warnings, time, cProfile, pstats, io
warnings.filterwarnings("ignore")
os.environ.setdefault("C_SUPINE", "1")
import numpy as np
import torch
torch.set_num_threads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import gymnasium as gym
import mimoEnv  # noqa
from hybrid_env import HybridEnv
from mimoActuation.muscle import MuscleModel
from mimoActuation.actuation import SpringDamperModel
from d_supine_env import infant_touch_params


def build(touch, muscle=True, touch_factor=2.0):
    return HybridEnv(gym.make("TaroSupine-v0", vision_params=None,
                              touch_params=infant_touch_params(touch_factor) if touch else None,
                              actuation_model=MuscleModel if muscle else SpringDamperModel))


def timeit(env, n):
    env.reset(seed=0)
    a = env.action_space.sample() * 0
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(a)
    dt = time.perf_counter() - t0
    sim_sec = n * env.unwrapped.dt
    return dt, sim_sec


def main():
    from gymnasium.envs.registration import register
    from d_supine_env import SupineMimoEnv  # noqa
    register(id="TaroSupine-v0", entry_point="d_supine_env:SupineMimoEnv", max_episode_steps=100000)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

    print("=== 太郎の1秒はどこに消えているか ===")
    print(f"{n} env.step を実測（1 env.step = {0.01}sim秒 = 物理2step）\n")
    print(f"{'構成':34s} {'実時間':>9s} {'sim秒':>7s} {'倍率':>8s} {'1時間の学習に':>12s}")
    rows = []
    for label, kw in (("① 太郎の実構成（触覚あり・筋肉）", dict(touch=True, muscle=True)),
                      ("② 触覚なし（筋肉）", dict(touch=False, muscle=True)),
                      ("③ 触覚あり・バネダンパー", dict(touch=True, muscle=False)),
                      ("④ 触覚を粗く(factor=4)", dict(touch=True, muscle=True, touch_factor=4.0))):
        env = build(**kw)
        dt, sim = timeit(env, n)
        env.close()
        ratio = sim / dt          # 実時間1秒でsim何秒進むか
        hours = 3600 / ratio / 60  # sim 1時間(=太郎の一生)に必要な実時間[分]
        rows.append((label, dt, sim, ratio, hours))
        print(f"{label:34s} {dt:7.1f}秒 {sim:6.1f}秒 {ratio:6.2f}倍 {hours:9.1f}分")

    base = rows[0][3]
    print(f"\n=== 引き算で分かること ===")
    print(f"触覚のコスト   : {rows[0][1]:.1f}秒 → {rows[1][1]:.1f}秒  "
          f"（触覚を外すと **{rows[0][1]/max(rows[1][1],1e-9):.1f}倍速い**）")
    print(f"筋肉のコスト   : {rows[0][1]:.1f}秒 → {rows[2][1]:.1f}秒  "
          f"（バネダンパーにすると {rows[0][1]/max(rows[2][1],1e-9):.1f}倍）")
    print(f"触覚を粗くする : {rows[0][1]:.1f}秒 → {rows[3][1]:.1f}秒  "
          f"（factor 2→4 で {rows[0][1]/max(rows[3][1],1e-9):.1f}倍）")

    print(f"\n=== 関数ごとの内訳（太郎の実構成・cProfile）===")
    env = build(touch=True, muscle=True)
    env.reset(seed=0)
    a = env.action_space.sample() * 0
    pr = cProfile.Profile(); pr.enable()
    for _ in range(max(200, n // 5)):
        env.step(a)
    pr.disable(); env.close()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
    for line in s.getvalue().splitlines():
        if "mimo" in line or "mujoco" in line or "cumtime" in line or "function calls" in line:
            print("  " + line.strip()[:150])

    print(f"\n=== 学習量の見積もり ===")
    r = rows[0][4]
    print(f"太郎の一生(sim 1時間) = 実時間 {r:.1f}分  （並列22本なら {r:.1f}分で22シード）")
    print(f"胎児の自己接触の練習 ≈ 700時間 → 同じ量を積むには 実時間 {700 * r / 60:.0f}時間 "
          f"= **{700 * r / 60 / 24:.1f}日**（1シードあたり）")


if __name__ == "__main__":
    main()
