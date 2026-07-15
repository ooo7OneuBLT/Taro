"""
触覚C案の実測：MIMo本物の触覚を「乳児相当の粗さ」にしたときの次元数とコスト。
併せて「アルファがベータに触れたら触覚が立つか」を確認（＝Dの前提）。

比較する粗さ factor（scale倍率。大きいほど粗い＝点が少ない）:
  3.0 / 2.0 / 1.0(=成人相当・本物フル解像度=A案)
使い方: python d_touch_check.py
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from d_env import TwoMimoEnv, infant_touch_params, BETA


def measure(factor, sep=3.0, n_step=20):
    t0 = time.time()
    env = TwoMimoEnv(sep=sep, vision_params=None, touch_params=infant_touch_params(factor))
    obs, _ = env.reset()
    t_init = time.time() - t0
    dim = obs["touch"].shape[0]
    a = np.zeros(env.action_space.shape[0])
    t1 = time.time()
    for _ in range(n_step):
        obs, *_ = env.step(a)
    t_step = (time.time() - t1) / n_step
    # 脳1回の判断=K=100 env.step 相当のコスト
    print(f"factor={factor:<4} 触覚次元={dim:>6}  構築={t_init:5.1f}s  1step={t_step*1000:6.2f}ms  "
          f"→ 脳1判断(K=100)={t_step*100:5.2f}s  1時間の人生(3600判断)≈{t_step*100*3600/60:5.1f}分", flush=True)
    env.close()
    return dim, t_step


def contact_test(factor=2.0, sep=0.22):
    """ベータを近くに置き、アルファの触覚が立つか＋どの部位が触られたか。"""
    env = TwoMimoEnv(sep=sep, vision_params=None, touch_params=infant_touch_params(factor))
    obs, _ = env.reset()
    beta_ids = env.beta_actuators
    rng = np.random.default_rng(0)
    hits = 0; best = 0.0
    for t in range(60):
        env.set_beta_ctrl(rng.uniform(-1, 1, len(beta_ids)) * 0.5)   # ベータがジタバタ
        obs, *_ = env.step(rng.uniform(-1, 1, env.action_space.shape[0]) * 0.5)
        s = float(np.abs(obs["touch"]).sum())
        if s > 0:
            hits += 1; best = max(best, s)
    print(f"\n[接触テスト sep={sep}] 触覚が立ったstep = {hits}/60   最大触覚総和 = {best:.2f}")
    print("=> 立っていれば『アルファはベータを触覚で感じられる』(Dの前提が成立)")
    env.close()


if __name__ == "__main__":
    print("=== 触覚の粗さ別 次元とコスト（ベータ遠方・接触なしの純コスト） ===", flush=True)
    for f in (3.0, 2.0, 1.0):
        try:
            measure(f)
        except Exception as e:
            print(f"factor={f}: 失敗 {type(e).__name__}: {e}", flush=True)
    contact_test(factor=2.0)
