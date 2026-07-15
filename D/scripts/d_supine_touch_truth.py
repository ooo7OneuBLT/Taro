"""
診断：学習した太郎は、触覚を「何もしない太郎」より動かせているのか。

【なぜ必要か】
仰向け・触覚ありの3シードは、分けて測ると 触覚 persist 97〜104% / margin +7〜13
＝**触覚を学べていない**。理由の候補は2つあり、どちらかで打ち手が正反対になる：
  ①触覚が行動で変化していない（＝学ぶ対象が存在しない）→ 環境・姿勢を変えるべき
  ②変化しているのに学習が失敗している        → 学習側を変えるべき

基準線はすでに取ってある（d_supine_check.py・何もしない太郎）：
    触覚の合計 平均224.8（222.0〜234.0）・**変動係数1.6%**
    接触ペア：指と親指が同じ手の中で触れ合っている／腕が頭の横にある
学習した太郎がこれを超えていなければ①。

【仮説】仰向け＝背中全体が床に触れっぱなし＝座位より接触面積が大きく、より変化しない信号に
なる。腕は自由になったが触覚はより一定になった、という可能性がある（＝仰向けは触覚には逆効果）。

使い方: python d_supine_touch_truth.py [model_path] [n_decision]
"""
import os, sys, warnings, collections
warnings.filterwarnings("ignore")
os.environ.setdefault("C_TOUCH", "1")
os.environ.setdefault("C_SUPINE", "1")
import numpy as np, torch, torch.nn as nn
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
from taro_brain_motor import TaroBrainWithMotor
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import rescale_action
from run_c_metrics_ac_lr import MinimalFusion, _ENV_ID, _touch_params

K = 100
BASE_MEAN, BASE_CV = 224.8, 1.6   # 何もしない太郎の基準線（d_supine_check.py で実測）


def body_of_geom(m, gid):
    return m.body(m.geom_bodyid[gid]).name or f"body{m.geom_bodyid[gid]}"


def is_world(name):
    n = name.lower()
    return n.startswith("world") or "floor" in n or "ground" in n or n == ""


def hand_like(n):
    return any(k in n.lower() for k in ("hand", "finger", "distal", "thumb", "thhub", "mf", "lf", "rf"))


def same_hand(b1, b2):
    """同じ手の中の指どうしの接触か（＝自己接触とは呼べない。指が親指に載っているだけ）。"""
    s1 = b1.startswith("left_") and b2.startswith("left_")
    s2 = b1.startswith("right_") and b2.startswith("right_")
    return (s1 or s2) and hand_like(b1) and hand_like(b2)


def main():
    mp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, os.pardir, "models", "supine_touch1_seed0.pt")
    n_dec = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    ck = torch.load(mp, weights_only=False); cfg = ck["config"]
    T = cfg["touch_dim"]
    print(f"=== 学習した太郎は触覚を動かせているか ===\nmodel={os.path.basename(mp)} seed={cfg['seed']}\n")

    env = HybridEnv(gym.make(_ENV_ID, vision_params=None, touch_params=_touch_params()))
    fusion = MinimalFusion(T); tfusion = MinimalFusion(T).freeze()
    n_act = env.action_space.shape[0]
    obs, _ = env.reset(seed=cfg["seed"])
    m, d = env.unwrapped.model, env.unwrapped.data
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=cfg["sdim"], n_actuators=n_act)
    emb_proj = nn.Linear(cfg["sdim"] + n_act, brain.sensory_proj.out_features)
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    brain.load_state_dict(ck["brain"])
    fusion.insula.load_state_dict(ck["fusion_insula"]); fusion.proprio.load_state_dict(ck["fusion_proprio"])
    fusion.vestibular.load_state_dict(ck["fusion_vestibular"]); fusion.touch.load_state_dict(ck["fusion_touch"])
    emb_proj.load_state_dict(ck["emb_proj"]); cereb.load_state_dict(ck["cereb"])
    for mm in (brain, emb_proj, cereb):
        for p in mm.parameters():
            p.requires_grad_(False)
    fusion.freeze()

    h = brain.init_motor_hidden(); pa = torch.zeros(n_act)
    tsum, floor_share = [], []
    real_self, intra_hand, floor_pairs = collections.Counter(), collections.Counter(), collections.Counter()
    n_real_self = 0
    for _ in range(n_dec):
        sv = fusion.encode(obs); cf = tfusion.encode(obs).detach()
        emb = emb_proj(torch.cat([sv, pa], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, hn = brain.motor_gru(emb, h)
        z, _, _ = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf); z = z.detach()
        pm = torch.tanh(brain.motor_head(z)); w, ca, _ = cereb.gate(z, pm)
        a = torch.clamp((1 - w) * pm + w * ca, -1, 1).detach()
        ctrl = rescale_action(a, env.action_space)
        te = tr = False
        for _ in range(K):
            obs, r, te, tr, info = env.step(ctrl)
            if te or tr:
                break
        tsum.append(float(np.abs(np.asarray(obs["touch"])).sum()))
        hit = False
        nfloor = ncon = 0
        for c in range(d.ncon):
            b1, b2 = body_of_geom(m, d.contact[c].geom1), body_of_geom(m, d.contact[c].geom2)
            ncon += 1
            if is_world(b1) or is_world(b2):
                floor_pairs[tuple(sorted((b1, b2)))] += 1; nfloor += 1
            elif same_hand(b1, b2):
                intra_hand[tuple(sorted((b1, b2)))] += 1     # 同じ手の中＝自己接触ではない
            else:
                real_self[tuple(sorted((b1, b2)))] += 1
                if hand_like(b1) or hand_like(b2):
                    hit = True
        n_real_self += int(hit)
        floor_share.append(nfloor / max(ncon, 1))
        h = hn.detach(); pa = a
        if te or tr:
            obs, _ = env.reset(); h = brain.init_motor_hidden(); pa = torch.zeros(n_act)
    env.close()

    cv = np.std(tsum) / max(np.mean(tsum), 1e-9) * 100
    print(f"触覚の合計 : 平均{np.mean(tsum):7.1f}  幅{np.min(tsum):.1f}〜{np.max(tsum):.1f}  変動係数={cv:.1f}%")
    print(f"  基準線（何もしない太郎）: 平均{BASE_MEAN:.1f}  変動係数={BASE_CV:.1f}%")
    print(f"  → 学習した太郎は 何もしない太郎の {cv/BASE_CV:.1f}倍 だけ触覚を動かせている")
    print(f"\n接触に占める床の割合: {np.mean(floor_share)*100:.0f}%  ← 高いほど『寝ているだけ』の一定信号")
    print(f"手が【本当に】自分の体に触れた判断: {n_real_self}/{n_dec} ({n_real_self/n_dec*100:.0f}%)"
          f"  ※同じ手の中の指どうしは除外して数えた")
    print("\n[本物の自己接触（別部位どうし）]")
    for k, v in real_self.most_common(6):
        print(f"   {k[0]:22s} - {k[1]:22s} {v}回")
    if not real_self:
        print("   （なし）")
    print("[床との接触]")
    for k, v in floor_pairs.most_common(6):
        print(f"   {k[0]:22s} - {k[1]:22s} {v}回")

    print("\n=== 判定 ===")
    if cv < BASE_CV * 3:
        print(f"→ **①学ぶ対象が存在しない**。学習した太郎の触覚の変動({cv:.1f}%)は"
              f"何もしない太郎({BASE_CV}%)とほぼ同じ＝触覚は行動でほとんど変化していない。")
        print("   打ち手は学習側ではなく**環境・姿勢側**。触覚が行動で変わる状況を作らないと、"
              "どんな動機・どんな学習を入れても学べるものが無い。")
    else:
        print(f"→ **②変化はしている({cv:.1f}% vs 基準{BASE_CV}%)のに学べていない**＝学習側の問題。")


if __name__ == "__main__":
    main()
