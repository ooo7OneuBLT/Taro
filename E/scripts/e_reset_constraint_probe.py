"""リセット直後の巨大な拘束反力（首に 6.31 Nm）の出どころを特定する。

【前提】`e_neck_force_breakdown.py` の実測：
  t=0.00 の首の力の内訳
    bias(重力)      0.00157
    passive(バネ)  -0.00002
    actuator(筋)    0.00000
    constraint      6.31246   ← ★他の4000倍。首の筋力 0.066Nm の95倍
  → 重力でも筋でもバネでもなく、**拘束反力**が太郎を弾いている。

【constraint が出る条件は2つ】
  ①関節角度が可動域（jnt_range）の外にある → MuJoCo が押し戻す
  ②物体どうしがめり込んでいる（接触の貫入）→ 押し出す

★①が疑わしい。生理的屈曲を「伸展の限界＝jnt_range」で実装したとき
（2026-07-26、FLEXION_MODE="range"）、**初期姿勢がその範囲の外に出た**可能性がある。
そうならリセットのたびに太郎は弾き飛ばされ、以降の全実験が汚染される。
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


def check(env, label):
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    mujoco.mj_forward(m, d)

    print(f"\n{'='*78}\n=== {label} ===")

    # ①可動域の外にいる関節
    out = []
    for j in range(m.njnt):
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        if not m.jnt_limited[j]:
            continue
        q = float(d.qpos[int(m.jnt_qposadr[j])])
        lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
        if q < lo - 1e-9 or q > hi + 1e-9:
            over = (q - hi) if q > hi else (q - lo)
            out.append((m.joint(j).name, np.degrees(q), np.degrees(lo),
                        np.degrees(hi), np.degrees(over)))
    print(f"\n[1] ★可動域の外にある関節: {len(out)} 個")
    if out:
        print(f"  {'関節':<34}{'現在':>9}{'下限':>9}{'上限':>9}{'はみ出し':>11}")
        for nm, q, lo, hi, ov in sorted(out, key=lambda x: -abs(x[4])):
            print(f"  {nm:<34}{q:>9.2f}{lo:>9.2f}{hi:>9.2f}{ov:>11.2f}")

    # ②めり込んでいる接触
    def gname(gid):
        """geom に名前が無いことがあるので、無ければ所属ボディ名で示す。"""
        nm = m.geom(gid).name
        if nm:
            return nm
        b = int(m.geom_bodyid[gid])
        return f"<{m.body(b).name}の無名geom#{gid}>"

    print(f"\n[2] 接触の数: {d.ncon}")
    deep = []
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist < -1e-5:      # 負の距離＝貫入
            deep.append((gname(c.geom1), gname(c.geom2), float(c.dist)))
    print(f"    ★めり込んでいる接触: {len(deep)} 個")
    for g1, g2, dist in sorted(deep, key=lambda x: x[2])[:15]:
        print(f"      {g1:<32} × {g2:<32} 貫入 {-dist*1000:7.2f} mm")

    # ③拘束反力の大きい自由度
    f = np.abs(np.asarray(d.qfrc_constraint, dtype=float))
    idx = np.argsort(-f)[:10]
    print(f"\n[3] 拘束反力の大きい自由度（上位10）")
    print(f"    合計 {f.sum():.3f} Nm  最大 {f.max():.3f} Nm")
    for i in idx:
        if f[i] < 1e-6:
            break
        jid = int(np.searchsorted(m.jnt_dofadr, i, side="right") - 1)
        print(f"      {m.joint(jid).name:<34}{d.qfrc_constraint[i]:>12.5f}")
    return len(out), len(deep), float(f.sum())


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    res = {}
    for label, flex, toy in [("生理的屈曲ON・おもちゃあり（現状）", True, True),
                             ("生理的屈曲OFF・おもちゃあり", False, True),
                             ("生理的屈曲ON・★おもちゃなし", True, False)]:
        kw = body_kwargs_from_env(0.0, verbose=False)
        kw["flexion"] = flex
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(),
                           age=0.0, toy=toy, vor=True, orient=False, **kw)
        res[label] = check(env, label)
        env.close()

    print(f"\n\n{'='*78}\n=== まとめ ===")
    print(f"{'条件':<38}{'範囲外':>8}{'めり込み':>10}{'拘束反力の合計':>16}")
    for k, (a, b, c) in res.items():
        print(f"{k:<38}{a:>8}{b:>10}{c:>16.3f}")
    print("\n  屈曲をOFFにして範囲外が消える → 原因は生理的屈曲の jnt_range")
    print("  どの条件でもめり込みが多い     → 原因は初期姿勢と床・柵の配置")


if __name__ == "__main__":
    main()
