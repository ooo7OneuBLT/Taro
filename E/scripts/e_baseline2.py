"""【段階2の正しいベースライン】段階1を終えた太郎で「手が視野に入る割合」と「持続の長さ」を測る。

【なぜ測り直すか＝比較の土台が違っていた】
最初のベースライン 0.7%（4000tick×3シード）は **c_pred_abs_seed0.pt** で測ったが、
段階1の学習は **c5_progress_seed0.pt** から始めた＝**別のモデル**だった。
学習後に値が増えても「学習の効果」か「元々このモデルはそうだった」かを区別できない。
→ **段階1を終えたモデル（e1_stage1_seed*.pt）そのもの**でベースラインを取り直す。

【もう1つ測るもの＝持続の長さ】
「手が視野に入る」が何tick続くかで、観測される割合の**揺れ方が全く変わる**：
   独立(1tick)なら 600tick区間のSDは 0.34% → 2.17%は 4.3σ（極めて稀）
   10tick続くなら SDは 1.08%             → 2.17%は 1.4σ（普通に起こる）
＝**この数字が分からないと「何%増えたら本物か」を決められない**。
実効サンプル数 n_eff ≒ n / (平均持続tick) として、成功条件の閾値を決めるのに使う。

【出すもの】
 ・hand_in_view の割合（全体・シードごと）
 ・エピソード（連続して視野内だった区間）の**平均の長さ・分布**
 ・600tick区間ごとの割合のばらつき（＝実測のSD。理論値と照合する）

使い方: python e_baseline2.py [n_ticks] [seed]
  環境変数 C5_CKPT で測るモデルを指定（例：C/models/e1_stage1_seed0.pt）
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import mimoEnv  # noqa: F401,E402
import d_c5_motor_quality as mq  # noqa: E402
import e_toy_env as te  # noqa: E402
from e_hand_in_view import hand_in_view  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    half_fov = te.VISION_FOVY / 2.0
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    raw = env.unwrapped
    m, d = raw.model, raw.data

    torch.manual_seed(seed); np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    seq = []
    for t in range(n):
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = mq.rescale_action(a, env.action_space); prev_a = a
        for k in range(mq.K):
            obs, r, term, trunc, info = env.step(ctrl)
            if term or trunc:
                break
        seq.append(int(hand_in_view(m, d)))
        if term or trunc:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    s = np.asarray(seq, dtype=int)

    # --- 連続して視野内だった区間（エピソード）の長さ ---
    runs = []
    cur = 0
    for v in s:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    runs = np.asarray(runs, dtype=float)

    # --- 600tick区間ごとの割合（＝実測のばらつき） ---
    W = 600
    chunks = np.array([s[i:i+W].mean()*100 for i in range(0, len(s)-W+1, W)])

    print(f"\n=== 段階2のベースライン（{os.path.basename(os.environ.get('C5_CKPT','?'))} / seed{seed}）===")
    print(f"  hand_in_view       : {s.mean()*100:.2f}%  ({s.sum()}/{len(s)} tick)")
    if len(runs):
        print(f"  ★視野内が続く長さ  : mean {runs.mean():.2f} tick / median {np.median(runs):.0f}"
              f" / max {runs.max():.0f}  （{len(runs)} 回）")
        n_eff = len(s) / max(runs.mean(), 1.0)
        p = s.mean()
        sd_eff = np.sqrt(p*(1-p)/max(n_eff/(len(s)/W), 1.0))*100
        print(f"  → 実効サンプル数    : {n_eff:.0f} / {len(s)} tick"
              f"（600tick区間なら実効 {n_eff/(len(s)/W):.0f}）")
        print(f"  → 理論上のSD(600tick区間) : {sd_eff:.2f}%")
    else:
        print("  ★視野内になった回数 : 0")
    if len(chunks) > 1:
        print(f"  実測のSD(600tick区間): {chunks.std(ddof=1):.2f}%   区間値 {np.round(chunks,2)}")
    env.close()


if __name__ == "__main__":
    main()
