"""
D0学習後の太郎を録画する（学習前＝ランダムな"右手ぴくぴく"との before/after 比較用）。

学習済みモデル(self_touch_seed0.pt)を読み、決定的な行動（探索ノイズなし＝act_mean）で
MIMoSelfBody-v0 を動かして mp4 に保存。触覚がどれだけ立ったかも併記する。

使い方: python d0_record_trained.py [n_step] [model_path]
出力: D/logs/video/d0_trained_<日時>.mp4
"""
import os, sys, datetime, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import gymnasium as gym
import mimoEnv  # noqa
import cv2
from hybrid_env import HybridEnv
from taro_brain_motor import TaroBrainWithMotor
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import rescale_action, to_tensor
from d0_selftouch import SelfTouchFusion, ln_sens, touch_sum, hand_touch_sum, K



def main():
    n_step = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    mp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, os.pardir, "models", "self_touch_muscle_seed0.pt")
    # 実時間モード：脳の判断(K=100step=0.5秒)の"間"のコマも撮る。
    # 既定(0)は判断ごとに1枚＝0.5秒/枚を15fpsで再生＝7.5倍速。
    # realtime_every=N で N env.step ごとに1枚撮り、fps=(1/dt)/N で再生＝等速。
    realtime_every = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ck = torch.load(mp, weights_only=False); cfg = ck["config"]
    print(f"=== D0学習後の録画 ===\nmodel={os.path.basename(mp)} n_train={cfg['n_train']}", flush=True)

    # 学習時と同じ駆動モデルで再現する（configから復元）
    from mimoActuation.actuation import SpringDamperModel
    from mimoActuation.muscle import MuscleModel
    act_model = MuscleModel if cfg.get("actuation") == "MuscleModel" else SpringDamperModel
    print(f"駆動モデル={cfg.get('actuation', 'SpringDamperModel')}", flush=True)
    # 学習時と同じ条件で再現する。done_active/max_episode_steps を渡し忘れると、
    # 録画中だけ太郎が500 env.step（＝判断5回）ごとにリセットされ、動画に「ビクッ」という
    # 不連続が入る（＝学習時と違う太郎を見せることになる）。d0_selftouch.py と必ず揃える。
    env = HybridEnv(gym.make("MIMoSelfBody-v0", render_mode="rgb_array", actuation_model=act_model,
                             done_active=False, max_episode_steps=6000))
    obs, _ = env.reset(seed=0)
    n_act = env.action_space.shape[0]
    fusion = SelfTouchFusion(cfg["prop_dim"], cfg["touch_dim"]); tfusion = SelfTouchFusion(cfg["prop_dim"], cfg["touch_dim"]).freeze()
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=cfg["sdim"], n_actuators=n_act)
    emb_proj = nn.Linear(cfg["sdim"] + n_act, brain.sensory_proj.out_features)
    nat_head = nn.Sequential(nn.Linear(brain.latent_dim + n_act, 128), nn.SiLU(),
                             nn.LayerNorm(128), nn.Linear(128, cfg["out_dim"]))
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    brain.load_state_dict(ck["brain"]); fusion.insula.load_state_dict(ck["fusion_insula"])
    fusion.proprio.load_state_dict(ck["fusion_proprio"]); fusion.touch.load_state_dict(ck["fusion_touch"])
    emb_proj.load_state_dict(ck["emb_proj"]); nat_head.load_state_dict(ck["nat_head"]); cereb.load_state_dict(ck["cereb"])
    for m in (brain, emb_proj, nat_head, cereb):
        for p in m.parameters():
            p.requires_grad_(False)
    fusion.freeze()

    # 注意：pc_latent.infer は推論時に内部で勾配を使う（予測符号化の誤差回帰）ので
    # torch.no_grad() で囲んではいけない。重みは上で凍結済みなので学習は起きない。
    h = brain.init_motor_hidden(); pa = torch.zeros(n_act)
    frames, tsum, hand_tsum, qvmax, amag, asat = [], [], [], [], [], []
    for t in range(n_step):
        sv = fusion.encode(obs); cf = tfusion.encode(obs).detach()
        emb = emb_proj(torch.cat([sv, pa], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, hn = brain.motor_gru(emb, h)
        z, _, _ = brain.pc_latent.infer(h[-1, 0], out[0, -1], cf)
        z = z.detach()
        pm = torch.tanh(brain.motor_head(z))
        w, ca, _ = cereb.gate(z, pm)
        a = torch.clamp((1.0 - w) * pm + w * ca, -1.0, 1.0).detach()   # 学習した決定的な行動
        ctrl = rescale_action(a, env.action_space)
        te = tr = False
        for k in range(K):
            obs, r, te, tr, info = env.step(ctrl)
            if realtime_every and (k % realtime_every == 0):
                fr = env.render()          # 判断の"間"のコマも撮る＝実時間で見える
                if fr is not None:
                    frames.append(fr)
            if te or tr:
                break
        tsum.append(touch_sum(obs))
        hand_tsum.append(hand_touch_sum(env))
        qvmax.append(float(np.abs(env.unwrapped.data.qvel).max()))   # 暴れの指標
        amag.append(float(a.abs().mean()))
        asat.append(float((a.abs() > 0.9).float().mean()))           # 飽和＝限界に振り切った次元の割合
        if not realtime_every:
            f = env.render()
            if f is not None:
                frames.append(f)
        h = hn.detach(); pa = a
        if te or tr:
            obs, _ = env.reset()
            h = brain.init_motor_hidden(); pa = torch.zeros(n_act)
    env.close()

    outdir = os.path.join(_HERE, os.pardir, "logs", "video"); os.makedirs(outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "muscle" if cfg.get("actuation") == "MuscleModel" else "spring"
    dt = env.unwrapped.dt if hasattr(env.unwrapped, "dt") else 0.005
    if realtime_every:
        fps = (1.0 / dt) / realtime_every          # 等速再生になるfps
        speed = "realtime"
    else:
        fps = 15; speed = f"{dt*K*15:.0f}x"        # 判断ごと1枚を15fpsで再生
    path = os.path.join(outdir, f"d0_trained_{tag}_{speed}_{stamp}.mp4")
    hh, ww, _ = frames[0].shape
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    nz = sum(1 for x in hand_tsum if x > 0)
    print(f"フレーム={len(frames)}")
    print(f"暴れの指標: 関節速度 平均={np.mean(qvmax):.2f} 最大={np.max(qvmax):.2f} | |a|平均={np.mean(amag):.3f} "
          f"飽和率={np.mean(asat)*100:.0f}%  ← 限界に振り切った行動次元の割合(固まりの兆候)")
    print(f"  ※参考: バネダンパー版の学習後は 関節速度 平均11.1/最大18.3、|a|=0.833、飽和65% だった")
    print(f"触覚(全身) 平均={np.mean(tsum):.1f} 最大={np.max(tsum):.1f}  ← 大半は座面(尻・脚)")
    print(f"触覚(手・指) 平均={np.mean(hand_tsum):.2f} 最大={np.max(hand_tsum):.2f} "
          f"立ったstep={nz}/{len(hand_tsum)} ({nz/max(len(hand_tsum),1)*100:.0f}%)  ← これが自己接触")
    print(f"再生速度: {'等速(realtime)' if realtime_every else f'{dt*K*15:.1f}倍速'}  fps={fps:.0f}  "
          f"sim時間={len(frames)/fps if realtime_every else n_step*dt*K:.1f}秒ぶん")
    print(f"保存先: {os.path.abspath(path)}", flush=True)


if __name__ == "__main__":
    main()
