"""taro-C5：新生児の体（成長モジュール）に学習済みモデルを載せ、関節速度・ジャークが
乳児域(1〜3 rad/s)に下がるかを実測する（学習なし・数値のみ）。

弱さ(トルク↓)と軽さ(慣性↓)のどちらが勝つかは測らないと分からない＝実測で確かめる。
比較対象：18ヶ月児(デフォルト)＝速度平均max 9.2/最大88・mean|jerk| 4690（既測）。

使い方: python d_c5_newborn_test.py [age=1] [ticks=20]
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
sys.path.insert(0, paths.MIMO_DIR)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from d_c5_motor_quality import build, make_policy, actuated_dofs, JerkMeter, K
from test_phase8_motor_learning import rescale_action


def measure(age, ticks, act_mode="off"):
    env, brain, fusion, emb_proj, cereb, n_act = build(act_mode, age=age)
    m, d = env.unwrapped.model, env.unwrapped.data
    dofs = actuated_dofs(m)
    dt_env = m.opt.timestep * env.unwrapped.frame_skip
    meter = JerkMeter(dofs, dt_env)
    policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble=False)
    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    qvel_max = []
    for tick in range(ticks):
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = rescale_action(a, env.action_space)
        for k in range(K):
            obs, r, te, tr, info = env.step(ctrl)
            meter.observe(d.qacc.copy(), is_boundary=(k == 0))
            qvel_max.append(float(np.abs(d.qvel[6:]).max()))
            if te or tr:
                obs, _ = env.reset(); hidden = brain.init_motor_hidden(); break
        prev_a = a
    s = meter.summary()
    return s, np.mean(qvel_max), np.max(qvel_max)


if __name__ == "__main__":
    age = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    act_mode = sys.argv[3] if len(sys.argv) > 3 else "off"  # on=活性化ダイナミクスも入れて測る
    print(f"=== taro-C5 新生児テスト: 月齢{age}ヶ月・{ticks}ティック・活性化={act_mode.upper()}"
          f"（モデル={os.path.basename(os.environ.get('C5_CKPT','c_pred_abs_seed0.pt'))}）===")
    s, vmean, vmax = measure(age, ticks, act_mode)
    print(f"\n{'':>16} {'関節速度 平均max':>14} {'最大':>7} | {'mean|jerk|':>10} {'境界':>8} {'内部':>8}")
    print(f"{'18ヶ月(既測)':>16} {9.20:>14.2f} {88.1:>7.1f} | {4690:>10.0f} {16948:>8.0f} {4572:>8.0f}")
    print(f"{f'{age}ヶ月(新生児)':>16} {vmean:>14.2f} {vmax:>7.1f} | {s['mean']:>10.0f} {s['boundary']:>8.0f} {s['interior']:>8.0f}")
    print(f"\n参考：乳児の関節速度はおよそ1〜3 rad/s。")
