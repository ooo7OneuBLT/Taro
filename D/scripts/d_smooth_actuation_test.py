"""taro-C5 実験①：筋活性化ダイナミクス(一次遅れ)の ON/OFF 比較（学習なし・録画のみ）。

同じ学習済みチェックポイント(c_pred_abs_seed0.pt)を読み、アクチュエータモデルだけを
差し替えて、太郎の運動が「カクカク(1秒ごとに瞬時ジャンプ)」から「なめらか」に変わるかを
第三者視点＋一人称(眼球)視点の両方で録画する。

変数はアクチュエータモデル1つだけ（脳の重み・行動生成ロジック・座位環境・ベータの動き・
乱数シード・ACTION_SCALE=0.3 はすべて同一）＝1機構ずつ検証の原則に従う。学習はしない
（＝OFFとONで脳の重みが完全に同一。差は"力の渡し方"だけ）。

使い方:
    python d_smooth_actuation_test.py off [n]   # 従来（瞬時トルク SpringDamperModel）
    python d_smooth_actuation_test.py on  [n]   # 一次遅れ（活性化ダイナミクス SmoothTorqueModel）
出力: D/logs/video/smooth_{off,on}_{eye,thirdperson}.mp4 ＋ 対応する sheet.png、
      眼球コマ間差分の統計（小さいほど滑らか）。
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import mujoco
import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))  # fusion再エクスポート
import paths
paths.setup_brain_path()
sys.path.insert(0, os.path.join(paths.SRC, "body"))   # smooth_actuation
sys.path.insert(0, paths.MIMO_DIR)

from hybrid_env import HybridEnv
from fusion import MinimalFusion, to_tensor
from taro_brain_motor import TaroBrainWithMotor
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import rescale_action
from d_beta_sitting_env import BetaSittingEnv
from d1_carer_vision_env import lean_vision_params
from mimoActuation.actuation import SpringDamperModel
from smooth_actuation import SmoothTorqueModel

CKPT = os.path.join(_HERE, os.pardir, os.pardir, "C", "models", "c_pred_abs_seed0.pt")
OUT = os.path.abspath(os.path.join(_HERE, os.pardir, "logs", "video"))
K = 100
RES = 64
SUB = 10           # 1ティック(=1秒)あたり10フレーム描画＝滑らかな動画
ACTION_SCALE = 0.3  # d_vision_train.py の録画と同一（＝発覚時のカクカク映像と同条件）


def load_matching(module, sd):
    own = module.state_dict()
    matched = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    module.load_state_dict(matched, strict=False)


def build(mode):
    seed = 0
    torch.manual_seed(seed); np.random.seed(seed)
    act_model = SmoothTorqueModel if mode == "on" else SpringDamperModel
    env = HybridEnv(BetaSittingEnv(vision_params=lean_vision_params(RES, fovy=60),
                                   actuation_model=act_model))
    obs, _ = env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    fusion = MinimalFusion(vision_res=RES)
    sdim = fusion.encode(obs).shape[0]
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    emb_proj = nn.Linear(sdim + n_act, brain.sensory_proj.out_features)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    load_matching(brain, blob["brain"])
    fusion.insula.load_state_dict(blob["fusion_insula"])
    fusion.proprio.load_state_dict(blob["fusion_proprio"])
    fusion.vestibular.load_state_dict(blob["fusion_vestibular"])
    load_matching(emb_proj, blob["emb_proj"])
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    load_matching(cereb, blob["cereb"])
    return env, brain, fusion, emb_proj, cereb, n_act


def zc(brain, emb_proj, sv, pa, cf, h):
    emb = emb_proj(torch.cat([sv, pa], dim=-1)).unsqueeze(0).unsqueeze(0)
    out, nh = brain.motor_gru(emb, h)
    z, _, _ = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf)
    return z, nh


def record(mode, env, brain, fusion, emb_proj, cereb, n_act, n=40):
    m = env.unwrapped.model
    m.vis.global_.offwidth = 640; m.vis.global_.offheight = 480
    ren = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.9; cam.elevation = -12; cam.azimuth = 90; cam.lookat = [0.2, 0.0, 0.25]

    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    beta_z = env.unwrapped.BETA_HOME[2]
    third, eye = [], []
    frame_diffs = []
    prev_eye = None
    obs, _ = env.reset(seed=0)
    for i in range(n):
        y = 0.3 * np.sin(2 * np.pi * i / 60.0)
        env.unwrapped.set_beta_target([0.3, y, beta_z])
        sv = fusion.encode(obs); cf = sv.detach()
        z, hn = zc(brain, emb_proj, sv, prev_a, cf, hidden.detach())
        policy_m = torch.tanh(brain.motor_head(z.detach())) * ACTION_SCALE
        w_c, cere_a, _ = cereb.gate(z.detach(), policy_m)
        a = torch.clamp((1.0 - w_c) * policy_m + w_c * cere_a, -1, 1)
        ctrl = rescale_action(a, env.action_space)
        for k in range(K):
            o, r, te, tr, info = env.step(ctrl)
            if (k % (K // SUB)) == 0:
                ren.update_scene(env.unwrapped.data, camera=cam)
                third.append(ren.render())
                ef = cv2.resize(np.asarray(o["eye_left"]).copy(), (240, 240),
                                interpolation=cv2.INTER_NEAREST)
                eye.append(ef)
                if prev_eye is not None:
                    frame_diffs.append(float(np.abs(ef.astype(np.float32) - prev_eye).mean()))
                prev_eye = ef.astype(np.float32)
            if te or tr:
                break
        obs = o; hidden = hn.detach(); prev_a = a.detach()

    os.makedirs(OUT, exist_ok=True)

    def save_mp4(frames, name, size):
        vw = cv2.VideoWriter(os.path.join(OUT, name), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()

    save_mp4(eye, f"smooth_{mode}_eye.mp4", (240, 240))
    save_mp4(third, f"smooth_{mode}_thirdperson.mp4", (640, 480))
    idxs4 = [0, len(third) // 3, 2 * len(third) // 3, len(third) - 1]
    row = np.concatenate([cv2.resize(third[k], (240, 240)) for k in idxs4], axis=1)
    cv2.imwrite(os.path.join(OUT, f"smooth_{mode}_third_sheet.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    idxs8 = np.linspace(0, len(eye) - 1, 8).astype(int)
    eye_row = np.concatenate([eye[k] for k in idxs8], axis=1)
    cv2.imwrite(os.path.join(OUT, f"smooth_{mode}_eye_sheet.png"), cv2.cvtColor(eye_row, cv2.COLOR_RGB2BGR))

    fd = np.array(frame_diffs)
    top10 = np.sort(fd)[int(len(fd) * 0.9):]
    print(f"[{mode}] 眼球コマ間差分: 平均={fd.mean():.2f} 最大={fd.max():.2f} "
          f"上位10%平均={top10.mean():.2f} 急変コマ率(>15)={100*np.mean(fd>15):.1f}%  (小さいほど滑らか)")
    print(f"[{mode}] mp4(第三者): {os.path.join(OUT, f'smooth_{mode}_thirdperson.mp4')}")
    print(f"[{mode}] mp4(眼球)  : {os.path.join(OUT, f'smooth_{mode}_eye.mp4')}")
    return fd


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "off"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    print(f"=== taro-C5 実験①: 活性化ダイナミクス {mode.upper()}（学習なし・録画のみ）===")
    env, brain, fusion, emb_proj, cereb, n_act = build(mode)
    dt = env.unwrapped.model.opt.timestep
    print(f"物理dt={dt*1000:.1f}ms／frame_skip={env.unwrapped.frame_skip}／"
          f"1ティック={K}×env.step＝{K*env.unwrapped.frame_skip}物理ステップ＝{K*env.unwrapped.frame_skip*dt:.2f}秒")
    record(mode, env, brain, fusion, emb_proj, cereb, n_act, n)
