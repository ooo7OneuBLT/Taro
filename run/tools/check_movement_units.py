# -*- coding: utf-8 -*-
"""「動きの区切り（movement units）」の数え方が正しいか確かめる。

【なぜ要るか、2026-07-31】この指標はリーチングができたかの判定に使う予定なので、
**数え方そのものが合っているか**を先に確かめる。実環境で回して
「それらしい数字が出た」で済ませない（落とし穴 項86）。

やり方：**答えの分かっている速度**を流し込んで、出てくる個数と突き合わせる。

    正弦波（周期0.5秒）を10秒 → 山は20個 → 20個 ちょうど出るはず
    一定の速度            → 山が無い → 0個
    細かいノイズだけ        → 深さのしきい値で落ちる → 0個

使い方:
    .venv/Scripts/python.exe run/tools/check_movement_units.py
    .venv/Scripts/python.exe run/tools/check_movement_units.py MODEL.pt   （実環境でも測る）
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
for p in ("E/scripts", "D/scripts", "taro_core/src/body", "taro_core/src/brain",
          "taro_core/src/senses", "taro_core/src/wrapper", "MIMo", ""):
    sys.path.insert(0, os.path.join(_R, p))
os.chdir(_R)

import numpy as np                                          # noqa: E402
from run.plugins.common.movement_units import MovementUnits  # noqa: E402


def make(dt=0.01, prominence=0.02, min_gap_ms=100):
    """環境を作らずにプラグインだけ用意する（数え方だけ試すため）。"""
    p = MovementUnits({"prominence": prominence, "min_gap_ms": min_gap_ms})
    p.prominence = prominence
    p.min_gap_ms = min_gap_ms
    p.dt = dt
    p.min_gap_steps = max(1, int(min_gap_ms / 1000.0 / dt))
    p.units = 0
    p.steps = 0
    p.peak = 0.0
    p._seg_reset()
    p._prev_v = None
    p._rising = False
    p._valley_v = None
    p._peak_since_valley = 0.0
    p._steps_since_unit = 10 ** 9
    return p


print("=" * 78)
print(" 動きの区切りの数え方を、答えの分かる速度で確かめる")
print("=" * 78)
ok = True

# ---- ① 正弦波：山の数がそのまま答え --------------------------------------
for period, sec in [(0.5, 10.0), (0.25, 10.0), (1.0, 10.0)]:
    dt = 0.01
    n = int(sec / dt)
    t = np.arange(n) * dt
    v = 0.15 + 0.10 * np.sin(2 * np.pi * t / period)   # 振幅0.10 → 谷から山まで0.20
    p = make(dt=dt)
    for x in v:
        p.feed(float(x))
    expect = int(sec / period)
    hit = abs(p.units - expect) <= 1        # 端の切れ方で±1はずれうる
    ok &= hit
    print(f"  正弦波 周期{period}秒 × {sec:.0f}秒   数えた{p.units:3d} / "
          f"あるべき{expect:3d}  {'合格' if hit else '不合格'}")

# ---- ② 一定の速度：山が無い ------------------------------------------------
p = make()
for _ in range(1000):
    p.feed(0.2)
print(f"  一定の速度                     数えた{p.units:3d} / あるべき  0  "
      f"{'合格' if p.units == 0 else '不合格'}")
ok &= (p.units == 0)

# ---- ③ 細かいノイズ：深さのしきい値で落ちるはず ----------------------------
rng = np.random.default_rng(0)
p = make(prominence=0.02)
for _ in range(1000):
    p.feed(0.2 + float(rng.normal(0, 0.002)))     # 揺れ幅がしきい値よりずっと小さい
print(f"  細かいノイズ(標準偏差0.002)      数えた{p.units:3d} / あるべき  0  "
      f"{'合格' if p.units == 0 else '不合格'}")
ok &= (p.units == 0)

# ---- ④ 最短間隔のしきい値が効くか ------------------------------------------
#   周期0.05秒（＝50ms）の山を流す。最短間隔100ms なので半分以下しか数えないはず
dt = 0.005
n = int(10.0 / dt)
t = np.arange(n) * dt
v = 0.15 + 0.10 * np.sin(2 * np.pi * t / 0.05)
p = make(dt=dt, min_gap_ms=100)
for x in v:
    p.feed(float(x))
expect_max = int(10.0 / 0.100)      # 100ms 間隔なら最大100個
hit = p.units <= expect_max
ok &= hit
print(f"  50ms周期の山（最短間隔100ms）   数えた{p.units:3d} / 上限{expect_max:3d}以下  "
      f"{'合格' if hit else '不合格'}")

print()
print(f"  数え方の検証：{'すべて合格' if ok else '不合格あり'}")

# ---- 実環境で測る（モデルを渡したときだけ）---------------------------------
if len(sys.argv) > 1:
    MODEL = sys.argv[1]
    print()
    print("=" * 78)
    print(" 実際の太郎で測る")
    print("=" * 78)
    import torch                                    # noqa: E402
    import e_scene                                  # noqa: E402
    from hybrid_env import HybridEnv                # noqa: E402
    from run.config import Config, touch_setting_of  # noqa: E402
    from run.taro_setup import Taro, rescale_action  # noqa: E402

    sc = e_scene.load("新生児_仰向け_柵なし")
    sc["body"]["age_months"] = 4.0
    sc["fingerprint"] = None
    env0, _ = e_scene.build(sc, seed=0, verbose=False)
    env = HybridEnv(env0)
    env.reset(seed=0)

    spec = {"actuation": "muscle", "age_months": 4.0, "model": MODEL}
    spec.update(touch_setting_of(MODEL))
    cfg = Config(spec, {"seed": 0, "K": 10}, scene=sc["name"], name="mu-test")
    t = Taro(cfg, env, seed=0, verbose=False)
    st = t.init_state(t.first_obs)

    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.env, ctx.model, ctx.data = env, env0.unwrapped.model, env0.unwrapped.data
    plug = MovementUnits({})
    plug.setup(ctx)
    print(f"  1ステップ = {plug.dt * 1000:.1f} ミリ秒 / "
          f"最短間隔 {plug.min_gap_steps} ステップ")

    act = np.zeros(env.action_space.shape[0], dtype=np.float32)
    obs = None
    STEPS = 3000
    speeds = []          # 速度そのものを取っておき、あとでしきい値を振って数え直す
    for i in range(STEPS):
        if i % 10 == 0:
            if obs is not None:
                st["obs"] = obs
            sv = t.fusion.encode(st["obs"])
            cf = t.target_fusion.encode(st["obs"]).detach()
            z, _k, _r, hn = t.infer_latent(sv, st["prev_a"], cf, st["hidden"])
            mean = t.act_mean(z.detach())
            a, _lp = t.brain.explore(mean, torch.full_like(mean, 0.174))
            a = a.detach()
            st["hidden"], st["prev_a"] = hn.detach(), a
            act = rescale_action(t.brain.to_env_action(a),
                                 env.action_space).astype(np.float32)
        obs = env.step(act)[0]
        plug.on_step(ctx)
        d = ctx.data
        speeds.append(float(np.linalg.norm(
            d.cvel[plug.hand_id][3:] - d.cvel[plug.hip_id][3:])))

    rep = plug.report(ctx)
    print(f"\n  モデル {os.path.basename(MODEL)}（{STEPS}ステップ＝"
          f"{STEPS * plug.dt:.0f}秒相当）")
    for k, v in rep.items():
        print(f"    {k:22s} {v}")

    # ---- しきい値を振って、数がしきい値に張り付いていないか見る --------------
    #   注意：出た値がしきい値の近くだと、「本当にその細かさ」なのか
    #     「しきい値で頭打ちになっている」のか区別がつかない。振って確かめる。
    print("\n  --- しきい値を振ったときの変化（張り付いていないかの確認）")
    print(f"    {'最短間隔ms':>10} {'深さm/s':>9} {'区切り/秒':>10} {'間隔ms':>9}")
    for gap_ms in (200, 100, 50, 20, 10):
        for prom in (0.02,):
            q = make(dt=plug.dt, prominence=prom, min_gap_ms=gap_ms)
            for x in speeds:
                q.feed(x)
            ps = q.units / (len(speeds) * plug.dt)
            print(f"    {gap_ms:>10} {prom:>9} {ps:>10.2f} "
                  f"{1000.0 / max(ps, 1e-9):>9.1f}")
    print("    ⇒ 最短間隔を短くしても数が増えないなら、その細かさが本物")
    print(f"\n    {'深さm/s':>10} {'区切り/秒':>10} {'間隔ms':>9}  （最短間隔100msで固定）")
    for prom in (0.05, 0.02, 0.01, 0.005):
        q = make(dt=plug.dt, prominence=prom, min_gap_ms=100)
        for x in speeds:
            q.feed(x)
        ps = q.units / (len(speeds) * plug.dt)
        print(f"    {prom:>10} {ps:>10.2f} {1000.0 / max(ps, 1e-9):>9.1f}")
    sp = np.asarray(speeds)
    print(f"\n    参考：手の速度[cm/s] 平均{100 * sp.mean():.1f} "
          f"中央{100 * np.median(sp):.1f} 最大{100 * sp.max():.1f}")
    env0.close()
