"""
養育者の手の4動作を記録し、触覚の反応を測る（太郎は何もしない）。

【この実験が答える問い】
触覚は、**他者の行動**を運べるのか。太郎に能力を一切要求しない最も甘い条件で確かめる
（接触は最初から与えられる・相手が動く・太郎は無動作）。ここで駄目なら、触覚だけの
他者理解は本当に駄目＝視覚/聴覚を足す根拠になる。

【2026-07-16 の作り直しについて】
旧版（/tmp の使い捨てスクリプト）は**2つのバグに汚染されていた**ので数値を破棄した：
  ①シーンのパスをハードコード → 年齢調整を素通りし、体のスケールが他の環境と食い違った
    （触覚センサ点数 1202→1543）
  ②手の定位置(z=0.50)が**太郎の落下経路のど真ん中**だった → 落ちてくる太郎を受け止め、
    腰が 0.050m→0.163m に浮いた（＝対照群の姿勢が既に汚染＝比較が成立しない）
両方 `d1_carer_env.py` 側で修正済み。修正後は**手があっても太郎の状態が単体版と完全一致**
（触覚次元・腰・頭・触覚合計すべて差ゼロ）＝ still が対照群として機能する。

さらに旧版は胸の位置を **CHEST=[0.27, 0, 0.17]** としていたが、これは**汚染された環境で
測った値**（浮いた太郎の胸）。修正後の実測は **[0.263, -0.001, 0.062]**＝旧版は
そもそも狙う場所を間違えていた。

【手を降ろす高さの実測（修正後・接触の立ち上がり）】
    目標z 0.20 → 接触0・触覚142.08（＝基準値と同じ＝触れていない）
    目標z 0.16 → 接触1・触覚153.30（手は z=0.177 で太郎に当たって止まる）
    目標z 0.08 → 接触1・触覚204.28（押し込むほど触覚が増える＝物理的に正しい）
  どの高さでも腰は 0.0505 で不動＝太郎を押し潰しても弾いてもいない。

使い方: python d1_carer_record.py [動作名...]   （既定＝4動作すべて）
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import numpy as np
import torch
import cv2
import gymnasium as gym
import mimoEnv  # noqa
from gymnasium.envs.registration import register

from hybrid_env import HybridEnv
from mimoActuation.muscle import MuscleModel
from test_phase8_motor_learning import rescale_action
from d1_carer_env import CarerEnv, CARER  # noqa
from d_supine_env import infant_touch_params

register(id="TaroCarerRec-v0", entry_point="d1_carer_env:CarerEnv", max_episode_steps=100000)

# 太郎の胸（修正後の実測。旧版の [0.27,0,0.17] は汚染された環境の値だった）
CHEST = np.array([0.2626, -0.0011, 0.062])
TOUCH_Z = 0.14      # 触れる高さ（実測：手は z≈0.177 で当たって止まる＝軽く触れる）
AWAY_Z = 0.30       # 触れない高さ（実測：接触0・触覚は基準値のまま）
N_DEC = 20          # 判断20回ぶん＝20 sim秒


def hand_target(action, t):
    """養育者の手の目標位置（**ワールド座標**）。t は経過sim秒。"""
    if action == "still":     # 触れずに静止＝対照群
        return np.array([CHEST[0], 0.0, AWAY_Z])
    if action == "press":     # 胸を押し続ける
        return np.array([CHEST[0], 0.0, TOUCH_Z])
    if action == "stroke":    # 胸を一定速度で撫でる（Ackerley et al. 2014 のC触覚線維の条件）
        x = CHEST[0] + 0.13 * np.sin(2 * np.pi * t / 6.0)
        return np.array([x, 0.0, TOUCH_Z])
    if action == "rock":      # 周期的に揺らす（縦方向）
        return np.array([CHEST[0], 0.0, TOUCH_Z + 0.04 * np.sin(2 * np.pi * t / 2.0)])
    raise ValueError(action)


def count_taro_hand_contacts(m, d):
    """太郎の体と養育者の手の接触だけを数える（床や自己接触は除く）。

    MIMo公式 `catch.py` に倣い、**接触ペアの有無だけでなく実際に力がかかっているか**まで
    見る（力ゼロの接触はMuJoCoでは"非活性"＝触れていないのと同じ）。
    2026-07-15の教訓：`touch>0` のような素朴な指標は、**何もしない太郎ですら100%**立った。
    """
    n = 0
    for c in range(d.ncon):
        n1 = m.body(m.geom_bodyid[d.contact[c].geom1]).name or ""
        n2 = m.body(m.geom_bodyid[d.contact[c].geom2]).name or ""
        if n1.startswith(CARER) == n2.startswith(CARER):
            continue                                  # 手同士 or 太郎同士
        if "world" in (n1 + n2).lower() or "floor" in (n1 + n2).lower():
            continue                                  # 床は他者ではない
        f = np.zeros(6, dtype=np.float64)
        import mujoco
        mujoco.mj_contactForce(m, d, c, f)
        if abs(f[0]) > 1e-9:                          # 力がかかっている＝本当に触れている
            n += 1
    return n


def run(action):
    env = HybridEnv(gym.make("TaroCarerRec-v0", actuation_model=MuscleModel,
                             touch_params=infant_touch_params(2.0), vision_params=None,
                             render_mode="rgb_array"))
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    ctrl = rescale_action(torch.zeros(env.action_space.shape[0]), env.action_space)  # 太郎は無動作

    frames, tsum, ncon, hips = [], [], [], []
    t = 0.0
    for _ in range(N_DEC):
        for k in range(100):
            u.set_hand_target(hand_target(action, t))
            obs, r, te, tr, info = env.step(ctrl)
            t += u.dt
            if k % 4 == 0:
                f = env.render()
                if f is not None:
                    frames.append(f)
        tsum.append(float(np.abs(np.asarray(obs["touch"])).sum()))
        ncon.append(count_taro_hand_contacts(u.model, u.data))
        hips.append(float(u.data.body("hip").xpos[2]))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       "logs", "video", f"carer_{action}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    h, w, _ = frames[0].shape
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), (1.0 / u.dt) / 4, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    env.close()

    cv = np.std(tsum) / max(np.mean(tsum), 1e-9) * 100
    return dict(action=action, mean=np.mean(tsum), cv=cv, ncon=np.mean(ncon),
                hip=np.mean(hips), hip_sd=np.std(hips), out=out)


def main():
    actions = sys.argv[1:] or ["still", "press", "stroke", "rock"]
    print("=== 養育者の手の4動作（太郎は何もしない・各20sim秒）===")
    print("【基準】手なしの仰向け単体：触覚合計 142.08 / 腰z 0.0505 / 変動係数 1.6%\n")
    print(f"{'動作':8s} {'触覚平均':>9s} {'変動係数':>9s} {'手との接触':>10s} {'腰z平均':>9s} {'腰zのばらつき':>13s}")
    rows = []
    for a in actions:
        r = run(a)
        rows.append(r)
        print(f"{r['action']:8s} {r['mean']:9.2f} {r['cv']:8.1f}% {r['ncon']:10.2f} "
              f"{r['hip']:9.4f} {r['hip_sd']:13.5f}", flush=True)
    print("\n動画：")
    for r in rows:
        print("  ", os.path.normpath(r["out"]))


if __name__ == "__main__":
    main()
