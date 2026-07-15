"""
対照実験：アルファの触覚のうち「ベータ由来」はどれだけか。
MIMoの本物の触覚は**床も自己接触も拾う**ので、「触覚が立った」だけではベータ知覚の証拠にならない。

 ① sep=3.0（ベータ遠方＝ベータ接触ゼロ）… 触覚は床由来のみ ＝ベースライン
 ② sep=0.22（近接）           … 床＋ベータ
 ③ 真値：MuJoCoの接触ペアを見て「アルファ×ベータ」の接触力だけ集計（分析用の正解ラベル）

②−① と ③ が一致すれば「ベータ由来の触覚が実在する」と確認できる。
使い方: python d_touch_control.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, mujoco
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from d_env import TwoMimoEnv, infant_touch_params, BETA


def beta_contact_force(env):
    """真値：アルファの体×ベータの体 の接触力の総和（分析用。モデルには渡さない）。"""
    m, d = env.model, env.data
    f6 = np.zeros(6); tot = 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        n1 = m.body(m.geom_bodyid[con.geom1]).name or ''
        n2 = m.body(m.geom_bodyid[con.geom2]).name or ''
        b1, b2 = n1.startswith(BETA), n2.startswith(BETA)
        if b1 != b2:  # 片方がベータ、もう片方がベータでない(=アルファ or 床)
            other = n2 if b1 else n1
            if other and other != 'world' and not other.startswith('floor'):
                mujoco.mj_contactForce(m, d, c, f6)
                tot += float(np.linalg.norm(f6[:3]))
    return tot


def run(sep, n=60, factor=2.0, seed=0):
    env = TwoMimoEnv(sep=sep, vision_params=None, touch_params=infant_touch_params(factor))
    obs, _ = env.reset()
    rng = np.random.default_rng(seed)
    touch_sums, beta_true, hits_beta = [], [], 0
    for _ in range(n):
        env.set_beta_ctrl(rng.uniform(-1, 1, env.n_beta_actuators) * 0.5)
        obs, *_ = env.step(rng.uniform(-1, 1, env.action_space.shape[0]) * 0.5)
        touch_sums.append(float(np.abs(obs["touch"]).sum()))
        bt = beta_contact_force(env)
        beta_true.append(bt)
        if bt > 0:
            hits_beta += 1
    env.close()
    return np.mean(touch_sums), np.mean(beta_true), hits_beta / n * 100


if __name__ == "__main__":
    print("=== 対照：アルファの触覚に占める『ベータ由来』 ===", flush=True)
    t_far, b_far, h_far = run(sep=3.0)
    print(f"① ベータ遠方 sep=3.0 : 触覚総和(平均)={t_far:8.2f}  ベータ接触(真値)={b_far:6.2f}  ベータ接触率={h_far:4.0f}%", flush=True)
    t_near, b_near, h_near = run(sep=0.22)
    print(f"② 近接      sep=0.22: 触覚総和(平均)={t_near:8.2f}  ベータ接触(真値)={b_near:6.2f}  ベータ接触率={h_near:4.0f}%", flush=True)
    print(f"\n触覚総和の差（②−①）= {t_near - t_far:+.2f}   ベータ接触の真値差 = {b_near - b_far:+.2f}")
    frac = (t_near - t_far) / max(t_near, 1e-9) * 100
    print(f"→ 近接時の触覚のうち、ベータ由来とみられる割合 ≈ {frac:.0f}%")
    print("=> ①でも触覚が立つなら、それは『床接触』。Dはこの床の背景からベータ分を"
          "取り出す必要がある（人間も同じ＝触覚に発信元のラベルは付かない）", flush=True)
