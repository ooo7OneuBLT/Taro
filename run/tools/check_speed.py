# -*- coding: utf-8 -*-
"""「学習済みの方が倍速感がある」を実測で確かめる。

ユーザーの目視（2026-07-31）：
> 学習モデルの方がなんか倍速感がある。デフォルトのもがき運動に比べて

注意：過去に同じ型の問題があった（学習は関節モード・Viewerは筋肉モードで
  動きが人間の新生児の3.3倍速だった）。推測でなく数字で確かめる。

測るもの（同じ環境・同じ体・同じステップ数で）:
  ・環境へ送る値の変化量   同じ物差しで比べられる唯一の量
  ・手と頭の速度[m/s]      「速い」の直接の指標
  ・体の重心の移動距離     柵を外したので「遠くへ行くか」も同時に見る
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
for p in ("E/scripts", "D/scripts", "taro_core/src/body", "taro_core/src/brain",
          "taro_core/src/senses", "taro_core/src/wrapper", ""):
    sys.path.insert(0, os.path.join(_R, p))
os.chdir(_R)

import numpy as np, torch, mujoco
import e_scene

STEPS = 3000          # 物理ステップ（K=10 なので判断は300回）
# 測るモデル。コマンドラインの第1引数で差し替えられる。
#   .venv/Scripts/python.exe run/tools/check_speed.py E/logs/.../model.pt
MODEL = (sys.argv[1] if len(sys.argv) > 1
         else "E/logs/selfmodel_v3/model_柵なし_線形_seed0.pt")


def make_env(hybrid):
    sc = e_scene.load("新生児_仰向け_柵なし")
    sc["body"]["age_months"] = 4.0
    sc["fingerprint"] = None
    env, _ = e_scene.build(sc, seed=0, verbose=False)
    if hybrid:
        from hybrid_env import HybridEnv
        env = HybridEnv(env)
    env.reset(seed=0)
    return env, sc


def measure(name, env, get_action):
    """get_action(obs) -> 環境へ送る値（0〜1 の筋活性化）"""
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    m, d = u.model, u.data
    hand = m.body("right_hand").id
    head = m.body("head").id
    hip = m.body("hip").id
    act = np.zeros(env.action_space.shape[0], dtype=np.float32)
    prev = act.copy()
    dact, v_hand, v_head, pos0 = [], [], [], None
    obs = None
    for i in range(STEPS):
        if i % 10 == 0:
            prev = act.copy()
            act = get_action(obs)
            dact.append(float(((act - prev) ** 2).mean()))
        out = env.step(act)
        obs = out[0]
        v_hand.append(float(np.linalg.norm(d.cvel[hand][3:])))   # 並進速度
        v_head.append(float(np.linalg.norm(d.cvel[head][3:])))
        if pos0 is None:
            pos0 = d.xpos[hip].copy()
    moved = float(np.linalg.norm(d.xpos[hip] - pos0))
    print(f"--- {name}")
    print(f"  環境へ送る値の変化量  {np.mean(dact[1:]):.5f}")
    print(f"  右手の速度[m/s]      平均{np.mean(v_hand):.4f}  最大{np.max(v_hand):.4f}")
    print(f"  頭の速度[m/s]        平均{np.mean(v_head):.4f}  最大{np.max(v_head):.4f}")
    print(f"  腰の移動距離         {moved*100:.2f} cm（{STEPS}ステップ＝"
          f"{STEPS*0.002*5:.0f}秒相当）")
    return {"dact": np.mean(dact[1:]), "vh": np.mean(v_hand), "moved": moved}


# ---- ① もがき運動（学習していないノイズ）--------------------------------
env, sc = make_env(hybrid=False)
from spinal_cord.cpg import ColoredNoiseGenerator
n_act = env.action_space.shape[0]
gen = ColoredNoiseGenerator(n_act, seed=0)
r1 = measure("もがき運動（ColoredNoise・学習なし）", env,
             lambda _o: np.clip(0.5 + 0.174 * gen.sample(0.7), 0.0, 1.0).astype(np.float32))
env.close()

# ---- ② 学習した脳 ---------------------------------------------------------
env, sc = make_env(hybrid=True)
from run.config import Config, touch_setting_of
from run.taro_setup import Taro, rescale_action
# 注意：触覚の設定はモデルから読み取る。食い違うと**触覚の層だけ白紙**の
#   別の脳を測ることになる（例外は出ない）。
_spec = {"actuation": "muscle", "age_months": 4.0, "model": MODEL}
_spec.update(touch_setting_of(MODEL))
print(f"\n測るモデル: {MODEL}")
cfg = Config(_spec, {"seed": 0, "K": 10}, scene=sc["name"], name="speed-test")
t = Taro(cfg, env, seed=0, verbose=False)
st = t.init_state(t.first_obs)


def brain_action(obs):
    if obs is not None:
        st["obs"] = obs
    sv = t.fusion.encode(st["obs"])
    cf = t.target_fusion.encode(st["obs"]).detach()
    z, _k, _r, hn = t.infer_latent(sv, st["prev_a"], cf, st["hidden"])
    z = z.detach()
    mean = t.act_mean(z)
    a, _lp = t.brain.explore(mean, torch.full_like(mean, 0.174))
    a = a.detach()
    st["hidden"], st["prev_a"] = hn.detach(), a
    return rescale_action(t.brain.to_env_action(a), env.action_space).astype(np.float32)


r2 = measure("学習した脳（18000回・柵なし線形成長 seed0）", env, brain_action)
env.close()

print()
print("=" * 70)
print(f"比べ  変化量 {r2['dact']/max(r1['dact'],1e-9):.2f}倍   "
      f"手の速度 {r2['vh']/max(r1['vh'],1e-9):.2f}倍   "
      f"移動 {r2['moved']/max(r1['moved'],1e-9):.2f}倍")
print("=" * 70)

# ---- ③ 白紙の脳（学習前）＝「学習で速くなったのか」の切り分け -------------
env, sc = make_env(hybrid=True)
# 注意：白紙の脳も**同じ触覚設定**で作る。揃えないと「学習の差」でなく
#   「脳の作りの差」を測ってしまう。
_spec0 = {"actuation": "muscle", "age_months": 4.0}
_spec0.update(touch_setting_of(MODEL, verbose=False))
cfg0 = Config(_spec0, {"seed": 0, "K": 10}, scene=sc["name"], name="speed-test-blank")
t0 = Taro(cfg0, env, seed=0, verbose=False)
st0 = t0.init_state(t0.first_obs)


def blank_action(obs):
    if obs is not None:
        st0["obs"] = obs
    sv = t0.fusion.encode(st0["obs"])
    cf = t0.target_fusion.encode(st0["obs"]).detach()
    z, _k, _r, hn = t0.infer_latent(sv, st0["prev_a"], cf, st0["hidden"])
    z = z.detach()
    mean = t0.act_mean(z)
    a, _lp = t0.brain.explore(mean, torch.full_like(mean, 0.174))
    a = a.detach()
    st0["hidden"], st0["prev_a"] = hn.detach(), a
    return rescale_action(t0.brain.to_env_action(a), env.action_space).astype(np.float32)


r3 = measure("白紙の脳（学習前・同じ揺らぎ0.174）", env, blank_action)
env.close()

print()
print("=" * 70)
print("切り分け（もがき運動を1とした比）")
print(f"  もがき運動   変化量 1.00   手の速度 1.00")
print(f"  白紙の脳   変化量 {r3['dact']/r1['dact']:.2f}   "
      f"手の速度 {r3['vh']/r1['vh']:.2f}")
print(f"  学習した脳 変化量 {r2['dact']/r1['dact']:.2f}   "
      f"手の速度 {r2['vh']/r1['vh']:.2f}")
print()
print(f"  ⇒ 学習した脳 ÷ 白紙の脳 = 変化量 {r2['dact']/r3['dact']:.2f}倍  "
      f"手の速度 {r2['vh']/r3['vh']:.2f}倍")
print("     ここが1に近ければ「速いのは学習のせいではなく脳の作りのせい」")
print("=" * 70)
