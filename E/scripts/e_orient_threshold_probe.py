"""視線誘導反射の「撃つ閾値」を実測で決める。

【なぜ実測で決めるのか】上丘の発火閾値に対応する文献値が無い（複数回の調査で
出てこなかった）。そこで**拾いたい信号と、拾いたくない雑音を分ける線**として、
実測から決める。

  拾いたい      おもちゃが点滅している（＝外界の変化）
  拾いたくない  おもちゃが静止している／そもそも存在しない
                （＝自分が動いたことで生じる視野の流れだけ）

【なぜ点滅か】おもちゃを物理的に動かすと顔と当たって物理が破綻する。
文献の定位実験も点滅光を使い（Lewis & Maurer 系）、上丘のニューロンは
静止した点滅ドットにも動く刺激とほぼ同等に応答する（J Neurophysiol 2004）。

【間隔】新生児のサッケード間隔 500〜900ms（Aslin & Salapatek 1975）。
既定は 900ms（実測でこの範囲の最良）。

注意：閾値を変えると撃つタイミングが変わり、自己運動も変わる（循環する）。
だから「strength の分布を測って線を引く」のではなく、**実際に閾値を入れて
撃った回数を数える**。これが唯一まっとうな測り方。
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np
import mujoco
import e_visibility as VIS

LATENCY = float(os.environ.get("E_SACC_LATENCY", "0.9"))
SEC = 20.0
BLINK_HZ = 2.5
# 注意：閾値の候補は実測に合わせた。動きマップの最大値は 0.036〜0.047 だったので、
#   従来の 0.05〜0.40 では全部「撃たない」になってしまう。
THRESHOLDS = [float(x) for x in os.environ.get(
    "E_THRESHOLDS", "0,0.005,0.010,0.015,0.020,0.030,0.045").split(",")]
# 「中心に十分近ければ撃たない」の境界（固視）。視野の半角に対する比。
#   中心窩の直径 約5度／視野半角30度 → 2.5/30 ≒ 0.083 が目安。
MIN_DIRS = [float(x) for x in os.environ.get(
    "E_MIN_DIRS", "0.0,0.083,0.15,0.25").split(",")]
FAR = np.array([3.0, 3.0, 0.05])


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    os.environ.setdefault("E_TOY_SHAPE", "sphere")
    os.environ.setdefault("E_TOY_RADIUS", "0.0056")
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
    reflex = u._orienting
    toy_bid = int(m.body("test_object1").id)
    toy_gadr = int(m.body("test_object1").geomadr[0])
    toy_jid = next(j for j in range(m.njnt)
                   if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                   and m.body(m.jnt_bodyid[j]).name == "test_object1")
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])
    max_shots = int(SEC / LATENCY)

    base_rgba = m.geom_rgba[toy_gadr].copy()
    dim_rgba = base_rgba.copy()
    dim_rgba[:3] *= 0.25

    def run(thresh, mode, min_dir=0.0):
        """mode: blink（点滅）／still（点滅なし）／none（おもちゃ無し）"""
        OR.SACCADE_LATENCY = LATENCY
        OR.SACCADE_MIN_STRENGTH = thresh
        OR.SACCADE_MIN_DIR = min_dir
        env.reset(seed=0)
        reflex.reset()
        for _ in range(int((TE.TOY_APPEAR_DELAY + TE.TOY_APPROACH_SEC + 0.3) / dt)):
            env.step(a)
        if mode == "none":
            u._rest_pos = FAR.copy()
            d.qpos[toy_qadr:toy_qadr + 3] = FAR
            d.qvel[toy_dof:toy_dof + 6] = 0.0
        strengths, devs = [], []
        for i in range(int(SEC / dt)):
            if mode == "blink":
                on = (np.sin(2 * np.pi * BLINK_HZ * i * dt) >= 0)
                m.geom_rgba[toy_gadr] = base_rgba if on else dim_rgba
            else:
                m.geom_rgba[toy_gadr] = base_rgba
            env.step(a)
            strengths.append(float(reflex.strength))
            if i % 50 == 0 and mode != "none":
                # 描き分けで測る（色の判定は本当の46%しか拾えない：項55）
                v = VIS.visible_by_segment(m, d, toy_bid, "eye_left", size=64)
                if v["seen"]:
                    devs.append(float(np.hypot(v["cx"], v["cy"])))
        m.geom_rgba[toy_gadr] = base_rgba
        s = np.array(strengths) if strengths else np.zeros(1)
        return dict(n=reflex.n_saccades,
                    rate=reflex.n_saccades / max(max_shots, 1) * 100,
                    s_mean=float(s.mean()), s_p95=float(np.percentile(s, 95)),
                    dev=(float(np.mean(devs)) if devs else float("nan")))

    print("=== 視線誘導反射：撃つ閾値を実測で決める ===")
    print(f"  間隔 {LATENCY*1000:.0f}ms（新生児 500〜900ms／Aslin & Salapatek 1975）")
    print(f"  {SEC:.0f}秒・自発運動なし・撃てる最大 {max_shots}回")
    print("  拾いたい：点滅（外界の変化）／拾いたくない：点滅なし・おもちゃ無し\n")
    print(f"{'動きの閾値':>11}{'固視の境界':>11}{'点滅で撃つ':>12}{'静止で撃つ':>12}"
          f"{'無しで撃つ':>12}{'点滅時のずれ':>14}")

    rows = {}
    for md in MIN_DIRS:
        for th in THRESHOLDS:
            r_b = run(th, "blink", md)
            r_s = run(th, "still", md)
            r_n = run(th, "none", md)
            rows[(md, th)] = (r_b, r_s, r_n)
            print(f"{th:>11.3f}{md:>11.3f}{r_b['rate']:>11.0f}%"
                  f"{r_s['rate']:>11.0f}%{r_n['rate']:>11.0f}%{r_b['dev']:>14.3f}")
    env.close()

    print("\n=== 判定 ===")
    print("  間違って撃つ率（静止・無しの大きい方）が 5% 以下で、")
    print("    正しく撃つ率がいちばん高い閾値を選ぶ")
    best = None
    for key in sorted(rows):
        md, th = key
        r_b, r_s, r_n = rows[key]
        fp = max(r_s["rate"], r_n["rate"])
        ok = fp <= 5.0
        if ok and (best is None or r_b["rate"] > rows[best][0]["rate"]):
            best = key
        if ok or fp <= 20.0:
            print(f"  動き {th:.3f} 固視 {md:.3f} : 間違って撃つ {fp:5.0f}%   "
                  f"正しく撃つ {r_b['rate']:5.0f}%   ずれ {r_b['dev']:.3f}"
                  + ("   ← 条件を満たす" if ok else ""))
    if best is None:
        print("\n  どの閾値でも間違って撃つ率が5%を超えた。")
        print("    ＝自己運動由来の反応が強すぎて、閾値だけでは分けられない。")
        print("    strength の作り方そのもの（動きの最大値でよいか）を見直すこと。")
    else:
        print(f"\n  → 閾値 {best:.3f} を採用の候補とする")


if __name__ == "__main__":
    main()
