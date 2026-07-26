"""視線誘導反射の発火の閾値を、実測で決める（間隔は700msに固定）。

【なぜ実測で決めるのか】閾値に対応する文献値が無い（上丘の発火閾値の数値は
2026-07-26の調査4本で出てこなかった）。そこで**信号を拾うかノイズを切るかの
線引き**として、実測から決める。

【間隔＝700ms の根拠】新生児のサッケード間隔 500〜900ms（1〜2ヶ月児）の**中央**。
    Aslin RN, Salapatek P (1975) "Saccadic localization of visual targets by the
    very young human infant" Perception & Psychophysics 17:293-302
⚠️[Tier2] 範囲の中央を採ったのは判断。500/700/900 の感度分析は別途行う。

【閾値の決め方】
    拾いたい    おもちゃが揺れている（＝外界の動き）
    拾いたくない おもちゃが静止／存在しない（＝自己運動由来の流れだけ）

  閾値を振って、
      正しく撃つ率  揺らした条件で撃った回数 ÷ 撃てる最大回数
      間違って撃つ率 止めた・無しの条件で撃った回数 ÷ 同
  を測り、**間違って撃つ率が5%以下になる最小の閾値**を選ぶ。

⚠️閾値を変えると撃つタイミングが変わり、自己運動も変わる（循環する）。
  だから「strengthの分布を測って線を引く」のではなく、**実際に閾値を入れて
  撃った回数を数える**。これが唯一まっとうな測り方。
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"),
          _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np
import mujoco
import e_visibility as VIS

LATENCY = 0.7          # ★新生児のサッケード間隔 500〜900ms の中央（Aslin & Salapatek 1975）
SEC = 40.0
SHAKE_HZ = 2.5
SHAKE_AMP = 0.015
THRESHOLDS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
FAR = np.array([3.0, 3.0, 0.05])


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    import e_orienting_v2 as OR
    import e_toy_env as TE

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, vor=True, orient=True, **kw)
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    toy_bid = int(m.body("test_object1").id)
    toy_jid = next(j for j in range(m.njnt)
                   if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                   and m.body(m.jnt_bodyid[j]).name == "test_object1")
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])
    reflex = u._orienting

    max_shots = int(SEC / LATENCY)      # 撃てる最大回数

    def run(thresh, mode):
        """mode: shake（揺らす）／still（止める）／none（おもちゃ無し）"""
        OR.SACCADE_LATENCY = LATENCY
        OR.SACCADE_MIN_STRENGTH = thresh
        env.reset(seed=0)
        reflex.reset()
        for _ in range(int((TE.TOY_APPEAR_DELAY + TE.TOY_APPROACH_SEC + 0.2) / dt)):
            env.step(a)
        base = (FAR.copy() if mode == "none" else np.array(u._rest_pos, dtype=float))
        strengths, seen, devs = [], [], []
        t = 0.0
        for i in range(int(SEC / dt)):
            wob = (np.array([0.0, SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t), 0.0])
                   if mode == "shake" else np.zeros(3))
            u._rest_pos = base + wob
            d.qpos[toy_qadr:toy_qadr + 3] = u._rest_pos
            d.qvel[toy_dof:toy_dof + 6] = 0.0
            env.step(a)
            t += dt
            strengths.append(float(reflex.strength))
            if i % 20 == 0:
                u._vision_t = None
                u._vision_cache = None
                imgs = u.get_vision_obs()
                img = imgs.get("eye_left") if isinstance(imgs, dict) else None
                if img is not None:
                    v = VIS.visible_in_image(img)
                    seen.append(1.0 if v["seen"] else 0.0)
                    if v["seen"]:
                        devs.append(float(np.hypot(v["cx"], v["cy"])))
        s = np.array(strengths)
        return dict(n=reflex.n_saccades,
                    rate=reflex.n_saccades / max(max_shots, 1) * 100,
                    s_mean=float(s.mean()), s_p95=float(np.percentile(s, 95)),
                    s_max=float(s.max()),
                    seen=(float(np.mean(seen)) * 100 if seen else 0.0),
                    dev=(float(np.mean(devs)) if devs else float("nan")))

    print("=== 視線誘導反射：発火の閾値を実測で決める ===")
    print(f"  間隔 {LATENCY*1000:.0f}ms（新生児 500〜900ms の中央／Aslin & Salapatek 1975）")
    print(f"  {SEC:.0f}秒・自発運動なし・撃てる最大 {max_shots}回\n")

    print(f"{'閾値':>6} | {'揺らす':^26} | {'止める':^18} | {'無し':^18}")
    print(f"{'':>6} | {'撃った':>6}{'率':>7}{'見えた':>7}{'ずれ':>6} | "
          f"{'撃った':>6}{'率':>7}{'反応95%':>9} | {'撃った':>6}{'率':>7}{'反応95%':>9}")
    print("-" * 82)
    rows = {}
    for th in THRESHOLDS:
        r_sh = run(th, "shake")
        r_st = run(th, "still")
        r_no = run(th, "none")
        rows[th] = (r_sh, r_st, r_no)
        print(f"{th:>6.2f} | {r_sh['n']:>6d}{r_sh['rate']:>6.0f}%"
              f"{r_sh['seen']:>6.0f}%{r_sh['dev']:>6.2f} | "
              f"{r_st['n']:>6d}{r_st['rate']:>6.0f}%{r_st['s_p95']:>9.3f} | "
              f"{r_no['n']:>6d}{r_no['rate']:>6.0f}%{r_no['s_p95']:>9.3f}")

    env.close()
    print("\n=== 判定 ===")
    print("  ★間違って撃つ率（止める・無し）が 5% 以下になる最小の閾値を選ぶ")
    best = None
    for th in THRESHOLDS:
        r_sh, r_st, r_no = rows[th]
        fp = max(r_st["rate"], r_no["rate"])
        if fp <= 5.0 and best is None:
            best = th
        print(f"  閾値 {th:.2f} : 間違って撃つ率 {fp:5.1f}%  "
              f"正しく撃つ率 {r_sh['rate']:5.1f}%  "
              f"見えた {r_sh['seen']:5.1f}%  ずれ {r_sh['dev']:.2f}"
              + ("   ← 条件を満たす最小" if best == th else ""))
    if best is None:
        print("\n  ★★どの閾値でも間違って撃つ率が5%を超える。")
        print("     ＝自己運動由来の反応が強すぎて、閾値では切り分けられない。")
    else:
        print(f"\n  → 閾値 {best:.2f} を採用の候補とする")
    print("\n  ⚠️『ずれ』が反射OFF（0.48）より小さくならなければ、"
          "中心には寄せられていない（サッケードの大きさが未実装のため）")


if __name__ == "__main__":
    main()
