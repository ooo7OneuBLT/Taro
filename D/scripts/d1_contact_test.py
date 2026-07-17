"""
0コスト診断：アルファの手はベータに届くのか。支えと座位は本当に要るのか。

【問い】目標は介護AI＝太郎が触る側（ユーザー判断）。設計書は「座位＋腕の届く位置」だが、
**座らせるのが最初なのは発達順序に反する**（自己接触は胎児14週＝姿勢制御ゼロで成立。Zoia et al. 2007）。
さらに「触る＝正確な到達が要る」も早とちりかもしれない：**腕の振れる範囲に置けば、今の
バタバタでも接触は起きる**かもしれない。＝支えも座位も要らないかもしれない。

**憶測で決めず、両方を測る**：
  A案 supine : 仰向けのアルファの隣に仰向けのベータ。支え無し（胎児〜新生児の条件）
  B案 seated : 座位のアルファの前にベータ。骨盤支持あり（Rochat & Goubet の実験条件）

【必ず基準線と比べる】2026-07-15の教訓：`hand_touch_sum>0` を自己接触の指標にしたら、
**何もしない太郎ですら100%**だった（初期姿勢で指が前腕に載っていただけ）。
だから今回は必ず
  ①何もしない（ctrl=0）②バタバタ（ランダム行動）
の両方を測り、**接触ペアを直接読んで「アルファの体 - ベータの体」だけを数える**。

使い方: python d1_contact_test.py [n_decision]
"""
import os, sys, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np
import torch
torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import gymnasium as gym
import mimoEnv  # noqa
from gymnasium.envs.registration import register
from hybrid_env import HybridEnv
from mimoActuation.muscle import MuscleModel
from test_phase8_motor_learning import rescale_action
from d1_env import D1Env, BETA  # noqa
from d_supine_env import infant_touch_params

register(id="TaroD1-v0", entry_point="d1_env:D1Env", max_episode_steps=100000)
K = 100


def classify(m, gid):
    """このジオメトリは アルファ / ベータ / 世界 のどれに属するか。"""
    name = m.body(m.geom_bodyid[gid]).name or ""
    n = name.lower()
    if n.startswith("world") or "floor" in n or "ground" in n or n == "":
        return "world", name
    if name.startswith(BETA):
        return "beta", name
    return "alpha", name


def hand_like(n):
    return any(k in n.lower() for k in ("hand", "finger", "distal", "thumb", "thhub"))


def probe(layout, sep, n_dec, random_action):
    env = HybridEnv(gym.make("TaroD1-v0", layout=layout, sep=sep, actuation_model=MuscleModel,
                             touch_params=infant_touch_params(2.0), vision_params=None))
    obs, _ = env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data
    na = env.action_space.shape[0]
    nb = len(env.unwrapped.beta_actuators)

    ab_pairs = collections.Counter()
    n_ab, n_ab_hand = 0, 0
    for i in range(n_dec):
        a = (torch.empty(na).uniform_(-1, 1) if random_action else torch.zeros(na))
        ctrl = rescale_action(a, env.action_space)
        env.unwrapped.set_beta_ctrl(np.zeros(nb))     # ベータは今回は動かさない（配置の検証）
        te = tr = False
        for _ in range(K):
            obs, r, te, tr, info = env.step(ctrl)
            if te or tr:
                break
        hit = hand = False
        for c in range(d.ncon):
            k1, n1 = classify(m, d.contact[c].geom1)
            k2, n2 = classify(m, d.contact[c].geom2)
            if {k1, k2} == {"alpha", "beta"}:
                ab_pairs[tuple(sorted((n1, n2)))] += 1
                hit = True
                if hand_like(n1) or hand_like(n2):
                    hand = True
        n_ab += int(hit); n_ab_hand += int(hand)
        if te or tr:
            obs, _ = env.reset()
    env.close()
    return n_ab, n_ab_hand, ab_pairs


def main():
    n_dec = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print("=== アルファの手はベータに届くか（学習なし・0コスト診断）===")
    print(f"判断{n_dec}回ぶん。**接触ペアを直接読み、「アルファの体 - ベータの体」だけを数える**")
    print("（2026-07-15の教訓：hand_touch>0 のような指標は、何もしない太郎で100%立って無意味だった）\n")
    print(f"{'配置':28s} {'行動':8s} {'体の接触':>9s} {'手の接触':>9s}")
    results = {}
    for layout, seps in (("seated", (0.20, 0.25, 0.30, 0.35, 0.40)),):
        for sep in seps:
            for rnd in (False, True):
                try:
                    ab, hd, pairs = probe(layout, sep, n_dec, rnd)
                except Exception as e:
                    print(f"{layout} sep={sep}: 失敗 {type(e).__name__}: {e}")
                    continue
                tag = f"{layout} sep={sep}m"
                act = "バタバタ" if rnd else "何もしない"
                print(f"{tag:28s} {act:8s} {ab:4d}/{n_dec:<4d} {hd:4d}/{n_dec:<4d}")
                results[(layout, sep, rnd)] = (ab, hd, pairs)

    print("\n--- 接触していた部位（バタバタ時・上位）---")
    for (layout, sep, rnd), (ab, hd, pairs) in results.items():
        if not rnd or not pairs:
            continue
        print(f"[{layout} sep={sep}m]")
        for k, v in pairs.most_common(4):
            print(f"   {k[0]:26s} - {k[1]:26s} {v}回")

    print("\n=== 判定 ===")
    best_supine = max((v[1] for k, v in results.items() if k[0] == "supine" and k[2]), default=0)
    best_seated = max((v[1] for k, v in results.items() if k[0] == "seated" and k[2]), default=0)
    print(f"仰向け（支え無し）で手がベータに触れた最良: {best_supine}/{n_dec}")
    print(f"座位（骨盤支持）で手がベータに触れた最良  : {best_seated}/{n_dec}")
    if best_supine >= n_dec * 0.3:
        print("\n→ **仰向けで接触が起きる**＝座らせる必要も支える必要も無い。")
        print("   発達順序（自己接触は胎児14週・姿勢制御ゼロ）とも整合する。A案で進める。")
    elif best_seated >= n_dec * 0.3:
        print("\n→ **座位＋骨盤支持でのみ接触が起きる**＝支えが必要。")
        print("   Rochat & Goubet（支えるとリーチできるようになる）と整合する。B案で進める。")
    else:
        print("\n→ **どちらでも接触が起きない**＝配置の問題。sepを振り直すか、配置を設計し直す。")
        print("   学習を回しても意味が無い（触覚に相手が写らない）。")


if __name__ == "__main__":
    main()
