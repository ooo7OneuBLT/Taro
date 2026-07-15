"""
診断：「自己接触」の指標が本物か、重力の産物かを切り分ける（0コスト診断＝落とし穴チェック項5/6）。

背景：d0_selftouch.py の hand_touch_sum() は
    「座位では手は床から浮いている → 手に触覚が立つ＝自分の体に触れた」
という**未検証の仮定**の上に立っている。ところがD0ログでは hand_touch_pct=41〜96% と高いのに、
録画を目視すると太郎はほとんど自分の体を触っていない。**数字と目視が矛盾**している。

やること：MuJoCoの接触ペア(data.contact)を直接読み、接触を
    ①自己接触（太郎の体 - 太郎の体）
    ②環境接触（太郎の体 - 床/座面）
に厳密に分ける。これを**何もしない太郎（ctrl=中立）**で測る。
何もしない太郎で hand_touch_pct が既に高いなら、その指標は自己接触ではなく重力を測っている。

使い方: python d0_contact_truth.py [n_decision]
"""
import os, sys, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np
import mujoco
torch_threads = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import gymnasium as gym
import mimoEnv  # noqa
from hybrid_env import HybridEnv
from mimoActuation.muscle import MuscleModel
from d0_selftouch import hand_touch_sum, touch_sum, K


def body_of_geom(model, gid):
    return model.body(model.geom_bodyid[gid]).name or f"body{model.geom_bodyid[gid]}"


def is_mimo(name):
    """世界(床/座面)以外＝太郎の体。MIMoの世界側ジオメトリは world/floor 等に属する。"""
    n = name.lower()
    return not (n.startswith("world") or "floor" in n or "ground" in n or n == "")


def main():
    n_dec = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    env = HybridEnv(gym.make("MIMoSelfBody-v0", actuation_model=MuscleModel,
                             done_active=False, max_episode_steps=6000))
    obs, _ = env.reset(seed=0)
    m = env.unwrapped.model
    d = env.unwrapped.data
    na = env.action_space.shape[0]

    # 「何もしない」＝行動0を rescale した中立の司令。学習も探索ノイズも一切なし。
    from test_phase8_motor_learning import rescale_action
    import torch
    ctrl = rescale_action(torch.zeros(na), env.action_space)

    self_pairs = collections.Counter()
    env_pairs = collections.Counter()
    hand_self, hand_env, hand_any = 0, 0, 0

    for _ in range(n_dec):
        for _ in range(K):
            obs, r, te, tr, info = env.step(ctrl)
            if te or tr:
                break
        hs = he = False
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            b1, b2 = body_of_geom(m, g1), body_of_geom(m, g2)
            hand1 = any(k in b1.lower() for k in ("hand", "finger", "distal", "thumb"))
            hand2 = any(k in b2.lower() for k in ("hand", "finger", "distal", "thumb"))
            if is_mimo(b1) and is_mimo(b2):
                self_pairs[tuple(sorted((b1, b2)))] += 1
                if hand1 or hand2:
                    hs = True
            else:
                env_pairs[tuple(sorted((b1, b2)))] += 1
                if hand1 or hand2:
                    he = True
        hand_self += int(hs)
        hand_env += int(he)
        hand_any += int(hand_touch_sum(env) > 0)

    env.close()
    print("=== 何もしない太郎（ctrl=中立・学習なし・探索なし）の接触の真実 ===")
    print(f"判断{n_dec}回ぶん（1回=K={K} env.step）\n")
    print(f"手が【自分の体】に触れた判断        : {hand_self}/{n_dec} ({hand_self/n_dec*100:.0f}%)  ←これが本物の自己接触")
    print(f"手が【床/座面】に触れた判断          : {hand_env}/{n_dec} ({hand_env/n_dec*100:.0f}%)")
    print(f"hand_touch_sum>0 だった判断（現指標）: {hand_any}/{n_dec} ({hand_any/n_dec*100:.0f}%)  ←D0のログに出していた数字")
    print("\n--- 実際に接触していた体のペア ---")
    print("[自己接触 太郎-太郎]")
    for k, v in self_pairs.most_common(10):
        print(f"   {k[0]:28s} - {k[1]:28s} {v}回")
    if not self_pairs:
        print("   （なし）")
    print("[環境接触 太郎-床/座面]")
    for k, v in env_pairs.most_common(10):
        print(f"   {k[0]:28s} - {k[1]:28s} {v}回")

    print("\n=== 判定 ===")
    if hand_any > 0 and hand_self == 0:
        print("→ **現指標は自己接触を測っていない**。何もしない太郎ですら"
              f"{hand_any/n_dec*100:.0f}%立つ＝重力で手が床/体に乗っているだけ。")
        print("   D0の hand_touch_pct 41〜96% は『学習の成果』ではなく**受動的な接触**。指標を接触ペアで作り直すべき。")
    elif hand_self > 0:
        print(f"→ 何もしない太郎でも自己接触が{hand_self/n_dec*100:.0f}%起きる＝"
              "初期姿勢で既に手が体に触れている。**学習の成果と受動的接触を分離する基準線が必要**。")
    else:
        print("→ 何もしない太郎では手の接触は起きない＝現指標は自己接触の指標として妥当。")


if __name__ == "__main__":
    main()
