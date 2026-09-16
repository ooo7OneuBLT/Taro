"""左右の眼球が同じように動いているかを測る。

【なぜ要るか、2026-07-28】Viewer での目視でユーザーから報告：
> おもちゃは中心で捉えようとして、ちゃんと中心で捉えられている
> **だが、左目だけが動いて右目が動かない**

コードを読む限り、視線誘導反射は左右のアクチュエータに**同じ指令**を書いている
（`e_orienting_v2.py` の `eye_idx["h"]` は名前に "eye" と "horizontal" を含む
アクチュエータを**すべて**集める）。にもかかわらず片目しか動かないなら、
指令より後（アクチュエータの強さ・関節の可動域・別の制御との競合）に原因がある。

【人間ではどうか】成人の眼球運動は**両眼が共同して動く**（Hering の等神経支配の法則）。
片目だけを随意に動かすことはできない。例外は輻輳（寄り目）で、これは左右が逆向きに動く。
新生児の両眼協調は3〜4ヶ月まで未熟だが、太郎は体年齢4ヶ月なので協調しているのが自然。
＝**片目だけ動くのは明確な異常**。

【測ること】
  1. 眼球のアクチュエータと関節の対応（名前・gear・fmax・可動域）
  2. 反射を動かしたときの左右の関節角度の時系列
  3. アクチュエータへ書かれた指令値（左右で同じか）
  4. 関節にかかる力の内訳（何が動きを止めているか）

使い方:
    .venv/Scripts/python.exe E/scripts/e_eye_pair_check.py
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
import os
import sys
import warnings
import contextlib
import io

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
           os.path.join(_ROOT, "taro_core"),
           os.path.join(_ROOT, "taro_core", "src", "body"),
           os.path.join(_ROOT, "taro_core", "src", "brain"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

AGE = float(os.environ.get("E_AGE", "4.0"))
SECONDS = float(os.environ.get("E_SECONDS", "10.0"))
OFFSET = float(os.environ.get("E_OFFSET", "0.03"))   # おもちゃを横へずらす量[m]


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    os.environ.setdefault("E_TOY_SHAPE", "sphere")
    os.environ.setdefault("E_TOY_RADIUS", "0.0056")
    os.environ.setdefault("E_FENCE", "0")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from e_head_hold import CaregiverHands
    from infant_body import actuator_strength
    import e_toy_env as TE

    kw = body_kwargs_from_env(AGE, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=AGE),
                           age=AGE, toy=True, vor=True, orient=True, **kw)
        env.reset(seed=0)
    u = env.unwrapped
    m, d = u.model, u.data
    am = getattr(u, "actuation_model", None)
    hands = CaregiverHands(m, d)
    hands.hold()
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)

    # ---- 1. 眼球のアクチュエータと関節 ----
    print("=" * 78)
    print(" 1. 眼球のアクチュエータと関節（体年齢 %.0fヶ月）" % AGE)
    print("=" * 78)
    eyes = []
    for i in range(m.nu):
        nm = m.actuator(i).name
        if "eye" not in nm:
            continue
        jid = int(m.actuator_trnid[i, 0])
        j = m.joint(jid)
        eyes.append(dict(aid=i, name=nm, jid=jid, jname=j.name,
                         qadr=int(m.jnt_qposadr[jid]), dof=int(m.jnt_dofadr[jid]),
                         lo=float(np.degrees(m.jnt_range[jid, 0])),
                         hi=float(np.degrees(m.jnt_range[jid, 1])),
                         strength=actuator_strength(m, i, am),
                         gear=abs(float(m.actuator_gear[i, 0])),
                         damping=float(m.dof_damping[int(m.jnt_dofadr[jid])]),
                         stiffness=float(m.jnt_stiffness[jid])))
    print(f"{'アクチュエータ':<28}{'関節':<26}{'筋力':>9}{'可動域[度]':>18}{'減衰':>9}")
    for e in eyes:
        print(f"{e['name']:<28}{e['jname']:<26}{e['strength']:>9.3f}"
              f"{e['lo']:>9.1f}〜{e['hi']:>7.1f}{e['damping']:>9.4f}")

    # ---- 2. おもちゃを横へずらして反射を働かせる ----
    reflex = u._orienting
    reflex.reset()
    moved = False
    cam_id = int(m.camera("eye_left").id)
    R = np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    right = R[:, 0]
    while getattr(u, "_toy_pending", False):
        env.step(a)
        if getattr(u, "_toy_arriving", False) and not moved and u._rest_pos is not None:
            u._rest_pos = np.array(u._rest_pos, dtype=float) + right * OFFSET
            u._carry_from = u._rest_pos + np.array([0.0, 0.0, 1.0]) * TE.TOY_APPROACH_DIST
            moved = True
    for _ in range(int(0.3 / dt)):
        env.step(a)

    toy_gadr = int(m.body("test_object1").geomadr[0])
    base_rgba = m.geom_rgba[toy_gadr].copy()
    dim_rgba = base_rgba.copy(); dim_rgba[:3] *= 0.25

    rows = []
    n = int(SECONDS / dt)
    for k in range(n):
        m.geom_rgba[toy_gadr] = base_rgba if (np.sin(2*np.pi*2.5*k*dt) >= 0) else dim_rgba
        env.step(a)
        if k % max(1, int(0.25 / dt)):
            continue
        rows.append(dict(
            t=k * dt,
            ang={e["name"]: float(np.degrees(d.qpos[e["qadr"]])) for e in eyes},
            ctrl={e["name"]: float(d.ctrl[e["aid"]]) for e in eyes},
            frc={e["name"]: float(d.qfrc_actuator[e["dof"]]) for e in eyes},
            passive={e["name"]: float(d.qfrc_passive[e["dof"]]) for e in eyes},
            constraint={e["name"]: float(d.qfrc_constraint[e["dof"]]) for e in eyes},
        ))

    # ---- 3. 左右の比較（水平） ----
    hs = [e["name"] for e in eyes if "horizontal" in e["name"]]
    vs = [e["name"] for e in eyes if "vertical" in e["name"]]
    for title, group in (("水平（左右を見る）", hs), ("垂直（上下を見る）", vs)):
        if not group:
            continue
        print("\n" + "=" * 78)
        print(f" 2. {title} — 左右の眼球は同じように動いているか")
        print("=" * 78)
        print(f"{'t[s]':>6}" + "".join(f"{g.replace('act:',''):>26}" for g in group))
        print(f"{'':>6}" + "".join(f"{'角度   指令   筋の力':>26}" for _ in group))
        for r in rows:
            line = f"{r['t']:>6.2f}"
            for g in group:
                line += (f"{r['ang'][g]:>9.2f}{r['ctrl'][g]:>8.3f}"
                         f"{r['frc'][g]:>9.4f}")
            print(line)

        # 動いた量
        print("-" * 78)
        for g in group:
            a0 = rows[0]["ang"][g]
            amax = max(abs(r["ang"][g] - a0) for r in rows)
            cmax = max(abs(r["ctrl"][g]) for r in rows)
            fmax_ = max(abs(r["frc"][g]) for r in rows)
            pmax = max(abs(r["passive"][g]) for r in rows)
            xmax = max(abs(r["constraint"][g]) for r in rows)
            print(f"  {g.replace('act:',''):<26} 最大の動き {amax:6.2f}度  "
                  f"指令の最大 {cmax:5.3f}  筋の力 {fmax_:7.4f}  "
                  f"受動 {pmax:7.4f}  拘束 {xmax:7.4f}")

    print("\n" + "=" * 78)
    print("読み方")
    print("-" * 78)
    print("  ・指令が左右で同じなのに動きが違う → 筋力・減衰・可動域・拘束のどれかが違う")
    print("  ・指令自体が片方だけ → 反射の配線の問題（eye_idx にもう片方が入っていない）")
    print("  ・拘束（constraint）が大きい → 可動域の限界に張り付いている")
    print("  ・受動（passive）が大きい → 筋の受動張力が引き戻している")
    env.close()


if __name__ == "__main__":
    main()
