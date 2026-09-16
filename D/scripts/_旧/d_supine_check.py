"""
0コスト診断：仰向け環境が意図どおりかを、学習を1stepも回さずに確認する。

確認すること（どれか1つでも外れたら6本の本番ランは無駄になる）：
  ①本当に仰向けで安定するか（転ばない・転がらない）
  ②手が体に触れていないところから始まるか（座位版は指が前腕に載りっぱなしだった）
  ③何もしない太郎の接触が、行動なしでどれだけ起きるか＝**学習の成果を測るための基準線**
  ④触覚の値が動くのか（動かないなら学ぶものが無い＝D0と同じ轍）

使い方: python d_supine_check.py [n_decision] [--record]
"""
import os, sys, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import gymnasium as gym
import mimoEnv  # noqa
import torch
from gymnasium.envs.registration import register
from hybrid_env import HybridEnv
from mimoActuation.muscle import MuscleModel
from test_phase8_motor_learning import rescale_action
from d_supine_env import SupineMimoEnv, infant_touch_params  # noqa

register(id="TaroSupine-v0", entry_point="d_supine_env:SupineMimoEnv", max_episode_steps=6000)

K = 100


def body_of_geom(model, gid):
    return model.body(model.geom_bodyid[gid]).name or f"body{model.geom_bodyid[gid]}"


def is_mimo(name):
    n = name.lower()
    return not (n.startswith("world") or "floor" in n or "ground" in n or n == "")


def main():
    n_dec = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    record = "--record" in sys.argv

    env = HybridEnv(gym.make("TaroSupine-v0", actuation_model=MuscleModel,
                             touch_params=infant_touch_params(2.0), vision_params=None,
                             render_mode="rgb_array" if record else None))
    obs, _ = env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data
    na = env.action_space.shape[0]
    ctrl = rescale_action(torch.zeros(na), env.action_space)   # 何もしない＝中立の司令

    self_pairs, env_pairs = collections.Counter(), collections.Counter()
    hand_self = 0
    zpos, touch_tot, frames = [], [], []
    for _ in range(n_dec):
        for k in range(K):
            obs, r, te, tr, info = env.step(ctrl)
            if record and k % 4 == 0:
                f = env.render()
                if f is not None:
                    frames.append(f)
            if te or tr:
                break
        hs = False
        for c in range(d.ncon):
            b1 = body_of_geom(m, d.contact[c].geom1)
            b2 = body_of_geom(m, d.contact[c].geom2)
            hand = any(k_ in (b1 + b2).lower() for k_ in ("hand", "finger", "distal", "thumb"))
            if is_mimo(b1) and is_mimo(b2):
                self_pairs[tuple(sorted((b1, b2)))] += 1
                if hand:
                    hs = True
            else:
                env_pairs[tuple(sorted((b1, b2)))] += 1
        hand_self += int(hs)
        zpos.append(float(d.body("hip").xpos[2]))
        touch_tot.append(float(np.abs(np.asarray(obs["touch"])).sum()))

    print("=== 仰向け環境の0コスト診断（何もしない太郎・学習なし・探索なし） ===")
    print(f"判断{n_dec}回ぶん（1回=K={K} env.step）\n")
    print(f"①腰の高さ  : 平均{np.mean(zpos):.3f}m  幅{np.min(zpos):.3f}〜{np.max(zpos):.3f}m"
          f"  ← 一定なら安定して寝ている")
    print(f"②手が自分の体に触れた判断 : {hand_self}/{n_dec} ({hand_self/n_dec*100:.0f}%)"
          f"  ← 座位版は100%だった。低いほど『触りにいく余地』がある")
    print(f"③触覚の合計 : 平均{np.mean(touch_tot):.1f}  幅{np.min(touch_tot):.1f}〜{np.max(touch_tot):.1f}"
          f"  変動係数={np.std(touch_tot)/max(np.mean(touch_tot),1e-9)*100:.1f}%")
    print(f"   ← 何もしなくても触覚が動くなら、その分は『行動の成果』ではない（基準線）")
    print(f"④触覚の次元数 : {len(obs['touch'])}")
    print("\n--- 接触していた体のペア ---")
    print("[自己接触 太郎-太郎]")
    for k_, v in self_pairs.most_common(8):
        print(f"   {k_[0]:24s} - {k_[1]:24s} {v}回")
    if not self_pairs:
        print("   （なし）← 理想。ここから自分で触りにいけば、それは純粋に行動の成果")
    print("[環境接触 太郎-床]")
    for k_, v in env_pairs.most_common(8):
        print(f"   {k_[0]:24s} - {k_[1]:24s} {v}回")

    if record and frames:
        import cv2
        outdir = os.path.join(_HERE, os.pardir, "logs", "video"); os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, "supine_donothing.mp4")
        dt = env.unwrapped.dt
        fps = (1.0 / dt) / 4
        hh, ww, _ = frames[0].shape
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"\n録画: {p} ({len(frames)}コマ, 等速{fps:.0f}fps)")
    env.close()


if __name__ == "__main__":
    main()
