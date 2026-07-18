"""taro-C5：運動の質(なめらかさ)を、クリーンな仰向け環境で測る／見るハーネス。

【なぜ】太郎の運動が非人間的（Prechtl's GMAの基準ではカクつき過大＝異常寄り）と判明した。
最初の実験①（活性化ダイナミクス）はD視覚環境で試したが、交絡（倒れた姿勢・ベータ・視覚
レンダリング）が多すぎ、指標（眼球コマ間差分）も100ms間隔で35msの効果を検出できず判定不能
だった。ここでは**視覚なし・ベータなし・仰向け**のクリーンな環境で、GMAの"jerkiness"に直接
対応する**ジャーク（＝加速度の時間変化率）**を物理ステップ解像度で測る。これなら①の効果も
原理的に見える。

【測るもの】
  ・mean|jerk|：関節角加速度(qacc)の時間微分の絶対値平均。小さいほどなめらか（Flash&Hogan
    1985の最小ジャークの発想＝ヒトの滑らかな運動はジャークを最小化する）。
  ・境界ジャーク vs 内部ジャーク：1ティック(K=100)の「切り替わり目」と「保持中」でジャークを
    分けて集計。スナップ＆ホールドなら境界で跳ねるはず。①はこの境界の跳ねを抑えるのが狙い。
  ・per-tick 行動変化量：毎ティック行動がどれだけ"ジャンプ"するか（スナップの大きさ）。

【変えないもの】脳・方策・1秒に1回の判断。ACTION_SCALEのようなD側の後付けは使わない
（＝Cで実際に学習した方策そのものの運動を、素で測る）。

使い方:
  python d_c5_motor_quality.py view off          # 見る（従来トルク SpringDamperModel）
  python d_c5_motor_quality.py view on           # 見る（活性化ダイナミクス SmoothTorqueModel）
  python d_c5_motor_quality.py measure off [n]   # 測る（ヘッドレス, n=60ティック既定, 従来）
  python d_c5_motor_quality.py measure on  [n]   # 測る（活性化ダイナミクス）
  末尾に babble を付けると探索ノイズ(運動性喃語)込みで動かす（既定は決定的な方策平均）。
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import mujoco

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))  # fusion再エクスポート等
import paths
paths.setup_brain_path()
sys.path.insert(0, os.path.join(paths.SRC, "body"))   # smooth_actuation
sys.path.insert(0, paths.MIMO_DIR)

import mujoco.viewer
from hybrid_env import HybridEnv
from fusion import MinimalFusion, to_tensor
from taro_brain_motor import TaroBrainWithMotor
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import rescale_action
from d_supine_env import SupineMimoEnv
from mimoActuation.actuation import SpringDamperModel
from smooth_actuation import SmoothTorqueModel

CKPT = os.path.join(_HERE, os.pardir, os.pardir, "C", "models", "c_pred_abs_seed0.pt")
K = 100


def load_matching(module, sd, tag):
    own = module.state_dict()
    matched = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in own if k not in matched]
    module.load_state_dict(matched, strict=False)
    note = f"（作り直し: {sorted(set(k.split('.')[0] for k in skipped))}）" if skipped else "（全層一致）"
    print(f"  [{tag}] ロード{len(matched)}層/作り直し{len(skipped)}層 {note}")


def build(mode_actuation):
    seed = 0
    torch.manual_seed(seed); np.random.seed(seed)
    act_model = SmoothTorqueModel if mode_actuation == "on" else SpringDamperModel
    env = HybridEnv(SupineMimoEnv(vision_params=None, actuation_model=act_model))
    obs, _ = env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    fusion = MinimalFusion(touch_dim=0)
    sdim = fusion.encode(obs).shape[0]
    prop_dim = to_tensor(obs["observation"]).shape[0]
    print(f"融合次元 sdim={sdim}／固有感覚 prop_dim={prop_dim}／行動 n_act={n_act}")
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    emb_proj = nn.Linear(sdim + n_act, brain.sensory_proj.out_features)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"チェックポイント読込: {os.path.basename(CKPT)}")
    load_matching(brain, blob["brain"], "脳")
    fusion.insula.load_state_dict(blob["fusion_insula"])
    fusion.proprio.load_state_dict(blob["fusion_proprio"])
    fusion.vestibular.load_state_dict(blob["fusion_vestibular"])
    load_matching(emb_proj, blob["emb_proj"], "emb_proj")
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    load_matching(cereb, blob["cereb"], "小脳")
    return env, brain, fusion, emb_proj, cereb, n_act


def actuated_dofs(model):
    """アクチュエータが駆動する関節のDOFアドレス（＝脳が動かす関節の集合）。"""
    dofs = []
    for i in range(model.nu):
        if model.actuator(i).name.startswith("beta_"):
            continue
        jid = model.actuator_trnid[i, 0]
        if jid >= 0:
            dofs.append(int(model.jnt_dofadr[jid]))
    return np.array(sorted(set(dofs)), dtype=int)


def make_policy(brain, fusion, emb_proj, cereb, n_act, babble):
    """1ティックぶんの決定的（または喃語込み）行動を返すクロージャ。"""
    ne_level = 0.095  # 学習後期のNE水準（ログ実測値）。babble時のノイズ幅に使う。

    def policy(obs, prev_a, hidden):
        sv = fusion.encode(obs); cf = sv.detach()
        emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, nh = brain.motor_gru(emb, hidden)
        z, _, _ = brain.pc_latent.infer(hidden[-1, 0], out[0, -1], cf)
        z = z.detach()
        policy_m = torch.tanh(brain.motor_head(z))            # ★ACTION_SCALEなし＝Cで学習した素の方策
        w_c, cere_a, _ = cereb.gate(z, policy_m)
        mean = (1.0 - w_c) * policy_m + w_c * cere_a
        if babble:
            std = (0.05 + ne_level * 0.45) * (1.0 - w_c)
            a = torch.clamp(torch.distributions.Normal(mean, std).sample(), -1, 1)
        else:
            a = torch.clamp(mean, -1, 1)
        return a.detach(), nh.detach()

    return policy


class JerkMeter:
    """物理ステップごとのqaccを受け取り、ジャーク（qaccの時間微分）を集計する。"""

    def __init__(self, dofs, dt):
        self.dofs = dofs; self.dt = dt
        self.prev_qacc = None
        self.boundary, self.interior = [], []

    def observe(self, qacc, is_boundary):
        a = qacc[self.dofs]
        if self.prev_qacc is not None:
            jerk = np.abs(a - self.prev_qacc) / self.dt
            (self.boundary if is_boundary else self.interior).append(float(jerk.mean()))
        self.prev_qacc = a

    def summary(self):
        b = np.array(self.boundary) if self.boundary else np.array([0.0])
        it = np.array(self.interior) if self.interior else np.array([0.0])
        allj = np.concatenate([b, it])
        return {"mean": float(allj.mean()), "boundary": float(b.mean()),
                "interior": float(it.mean()), "max": float(allj.max())}


def run_view(mode_actuation, babble):
    env, brain, fusion, emb_proj, cereb, n_act = build(mode_actuation)
    policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble)
    m, d = env.unwrapped.model, env.unwrapped.data
    dofs = actuated_dofs(m)
    dt_env = m.opt.timestep * env.unwrapped.frame_skip
    meter = JerkMeter(dofs, dt_env)

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    print(f"\nビューア起動。仰向けの太郎が学習済みの脳で動きます（活性化ダイナミクス={mode_actuation.upper()}"
          f"／{'喃語込み' if babble else '決定的'}）。スペース=一時停止、左History=巻き戻し。")
    tick = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            a, hidden = policy(obs, prev_a, hidden)
            ctrl = rescale_action(a, env.action_space)
            for k in range(K):
                obs, r, te, tr, info = env.step(ctrl)
                meter.observe(d.qacc.copy(), is_boundary=(k == 0))
                viewer.sync()
                if te or tr:
                    break
            prev_a = a; tick += 1
            if tick % 20 == 0:
                s = meter.summary()
                print(f"  tick {tick}: mean|jerk|={s['mean']:.1f} 境界={s['boundary']:.1f} "
                      f"内部={s['interior']:.1f} 最大={s['max']:.1f}", flush=True)


def run_measure(mode_actuation, n, babble):
    env, brain, fusion, emb_proj, cereb, n_act = build(mode_actuation)
    policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble)
    m, d = env.unwrapped.model, env.unwrapped.data
    dofs = actuated_dofs(m)
    dt_env = m.opt.timestep * env.unwrapped.frame_skip
    meter = JerkMeter(dofs, dt_env)

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    action_jumps = []
    print(f"\n測定開始（{n}ティック・活性化ダイナミクス={mode_actuation.upper()}"
          f"／{'喃語込み' if babble else '決定的'}）")
    for tick in range(n):
        a, hidden = policy(obs, prev_a, hidden)
        action_jumps.append(float(np.abs((a - prev_a).numpy()).mean()))
        ctrl = rescale_action(a, env.action_space)
        for k in range(K):
            obs, r, te, tr, info = env.step(ctrl)
            meter.observe(d.qacc.copy(), is_boundary=(k == 0))
            if te or tr:
                obs, _ = env.reset()
                hidden = brain.init_motor_hidden()
                break
        prev_a = a
    s = meter.summary()
    print(f"\n===== 結果（{mode_actuation.upper()}）=====")
    print(f"mean|jerk|      = {s['mean']:.2f}   (小さいほどなめらか)")
    print(f"  境界ジャーク  = {s['boundary']:.2f}  (ティック切替の瞬間)")
    print(f"  内部ジャーク  = {s['interior']:.2f}  (行動保持中)")
    print(f"  最大ジャーク  = {s['max']:.2f}")
    print(f"per-tick 行動ジャンプ = {np.mean(action_jumps):.3f}  (毎秒どれだけ行動が跳ぶか)")
    return s


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "view"
    actuation = sys.argv[2] if len(sys.argv) > 2 else "off"
    rest = sys.argv[3:]
    babble = "babble" in rest
    nums = [int(x) for x in rest if x.isdigit()]
    n = nums[0] if nums else 60
    print(f"=== taro-C5 運動の質: mode={mode} 活性化ダイナミクス={actuation.upper()} "
          f"{'喃語込み' if babble else '決定的'} ===")
    if mode == "measure":
        run_measure(actuation, n, babble)
    else:
        run_view(actuation, babble)
