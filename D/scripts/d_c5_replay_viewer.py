"""taro-C5：履歴を等速再生するビューア（ユーザー仕様）。

普段は早く：まず脳を早送りで動かし、各コマの姿勢(qpos)を裏で記録する（重い処理は最速で終わる）。
確認は等速：記録した履歴を実時間ぴったりで再生する。物理計算は再生時はしない（姿勢を並べるだけ）
ので軽い。スペースで一時停止／←→で1コマずつ送り戻し。ループ再生。

使い方（体・モデルは環境変数で d_c5_motor_quality と共通）:
  C5_CKPT=<pt> C5_AGE=0 python d_c5_replay_viewer.py on [ticks]
    on/off = 活性化ダイナミクス、ticks = 記録するティック数（既定20＝20秒ぶん）
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import mujoco
import mujoco.viewer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import d_c5_motor_quality as mq  # build/make_policy/AGE/CKPT/K を共有（環境変数で設定）
from test_phase8_motor_learning import rescale_action


def main():
    actuation = sys.argv[1] if len(sys.argv) > 1 else "on"
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    # ── フェーズ1：早送りで脳を動かし、姿勢を記録 ──
    env, brain, fusion, emb_proj, cereb, n_act = mq.build(actuation, age=mq.AGE)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=False)
    m, d = env.unwrapped.model, env.unwrapped.data
    real_dt = m.opt.timestep * env.unwrapped.frame_skip  # 1 env.step の実時間相当(秒)
    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    qpos_buf = []
    print(f"記録中（{ticks}ティック=約{ticks}秒ぶんを早送りで生成）...")
    for tick in range(ticks):
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = rescale_action(a, env.action_space)
        for k in range(mq.K):
            obs, r, te, tr, info = env.step(ctrl)
            qpos_buf.append(d.qpos.copy())
            if te or tr:
                break
        prev_a = a
    print(f"記録完了：{len(qpos_buf)}コマ（実時間 約{len(qpos_buf)*real_dt:.1f}秒ぶん）")

    # ── フェーズ2：等速再生（実時間・一時停止・コマ送り）──
    state = {"paused": False, "i": 0, "step": 0}

    def key_cb(keycode):
        if keycode == 32:            # スペース＝一時停止/再生
            state["paused"] = not state["paused"]
        elif keycode == 262:         # → 1コマ進める（一時停止中）
            state["step"] = 1
        elif keycode == 263:         # ← 1コマ戻す（一時停止中）
            state["step"] = -1

    def show(i):
        d.qpos[:] = qpos_buf[i]
        mujoco.mj_forward(m, d)

    print("等速再生：スペース=一時停止／再生、←→=1コマ送り戻し（一時停止中）、閉じるまでループ。")
    show(0)
    with mujoco.viewer.launch_passive(m, d, key_callback=key_cb) as viewer:
        while viewer.is_running():
            if state["step"] != 0:               # コマ送り（一時停止中の←→）
                state["i"] = (state["i"] + state["step"]) % len(qpos_buf)
                show(state["i"]); state["step"] = 0
            elif not state["paused"]:             # 通常再生
                state["i"] = (state["i"] + 1) % len(qpos_buf)
                show(state["i"])
            viewer.sync()
            time.sleep(real_dt)                   # ★実時間に合わせる（等速）


if __name__ == "__main__":
    main()
