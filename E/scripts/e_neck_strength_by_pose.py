"""首の「持ち上げ能力比」が姿勢と方向でどう変わるかを測る。

【なぜ要るか、2026-07-28】4ヶ月の太郎の比が 47.5倍と出たが、この数字には条件がある。
  ①比の分母（頭の重力モーメント）は**首の角度で変わる**。仰向けで頭が中立だと
    頭の重心が首関節のほぼ真上に来るので分母が小さくなり、比が大きく出る。
  ②MIMoの筋肉は1関節に**曲げる筋と伸ばす筋の2本**があり、力（FMAX）が別々。
    `infant_body.actuator_strength` は**大きい方**を返すので、うつ伏せで頭を上げる
    （＝伸展）ときに使われる力とは限らない。
＝「47.5倍が強すぎる」と言う前に、**首座りの本番の姿勢と方向**で測り直す必要がある。

【測ること】
  1. head_tilt の FMAX を neg（負方向）/ pos（正方向）に分けて出す
  2. 首の角度を振り、各角度での頭の重力モーメントと比を出す
  3. 月齢ごとに比較する

⚠️うつ伏せ環境（prone）はまだ無い。ここでは**仰向けのまま首の角度を振る**ことで
  「頭の重心が首関節の真上から外れたとき」の比を出す。重力の向きに対する頭の位置関係が
  同じになる角度が、うつ伏せで頭を上げる場面に相当する。

使い方:
    E_AGES=0,4,18 .venv/Scripts/python.exe E/scripts/e_neck_strength_by_pose.py
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
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
           os.path.join(_ROOT, "taro_core", "src", "body"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import mujoco       # noqa: E402

AGES = [float(x) for x in os.environ.get("E_AGES", "0,4,18").split(",")]
ANGLES = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
G = 9.81
TILT_JOINT = "robot:head_tilt"
TILT_ACT = "act:head_tilt"


def _head_bodies(model):
    ids = []
    for b in range(model.nbody):
        p = b
        while p != 0:
            if model.body(p).name == "head":
                ids.append(b)
                break
            p = int(model.body_parentid[p])
    return ids


def run(age, corrections):
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(age, verbose=False)
    kw["body_corrections"] = corrections
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = SupineMimoEnv(vision_params=None, age=age,
                            actuation_model=MuscleModel, **kw)
        env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data
    am = getattr(env.unwrapped, "actuation_model", None)

    jid = [j for j in range(m.njnt) if m.joint(j).name == TILT_JOINT][0]
    aid = [i for i in range(m.nu) if m.actuator(i).name == TILT_ACT][0]
    qadr = int(m.jnt_qposadr[jid])

    # --- FMAX を neg / pos に分けて読む ---
    f = np.asarray(getattr(am, "fmax", np.array([])), dtype=float)
    n_act = int(getattr(am, "n_actuators", f.size // 2)) if f.size else 0
    if f.size:
        f_neg, f_pos = float(f[aid]), float(f[aid + n_act])
    else:
        f_neg = f_pos = abs(float(m.actuator_gear[aid, 0]))

    head_ids = _head_bodies(m)
    head_mass = float(sum(m.body_mass[i] for i in head_ids))
    lo = float(np.degrees(m.jnt_range[jid, 0]))
    hi = float(np.degrees(m.jnt_range[jid, 1]))

    rows = []
    for ang in ANGLES:
        if not (lo <= ang <= hi):
            continue
        d.qpos[qadr] = np.radians(ang)
        mujoco.mj_forward(m, d)
        anchor = d.xanchor[jid]
        com = sum(m.body_mass[i] * d.xipos[i] for i in head_ids) / head_mass
        arm = float(np.linalg.norm((com - anchor)[:2]))   # 水平＝重力モーメントの腕
        tau = head_mass * G * arm
        rows.append(dict(ang=ang, arm=arm, tau=tau,
                         r_neg=f_neg / max(tau, 1e-9),
                         r_pos=f_pos / max(tau, 1e-9)))
    env.close()
    return dict(age=age, f_neg=f_neg, f_pos=f_pos, head_mass=head_mass,
                lo=lo, hi=hi, rows=rows)


def main():
    print("=" * 74)
    print(" 首の力は姿勢と方向でどう変わるか（2026-07-28）")
    print("=" * 74)
    print(" ⚠️太郎の補正は入れたまま（＝実際に使う体）")

    outs = []
    for age in AGES:
        print(f"  [{age:g}ヶ月...]")
        outs.append(run(age, corrections=True))

    print("\n" + "=" * 74)
    print("1. head_tilt の筋力（曲げる筋と伸ばす筋は別）")
    print("-" * 74)
    print(f"{'月齢':>6}{'負方向 [N·m]':>16}{'正方向 [N·m]':>16}"
          f"{'頭 [kg]':>10}{'可動域 [度]':>16}")
    for o in outs:
        print(f"{o['age']:>6g}{o['f_neg']:>16.3f}{o['f_pos']:>16.3f}"
              f"{o['head_mass']:>10.3f}   {o['lo']:>6.1f} 〜 {o['hi']:>5.1f}")

    print("\n2. 首の角度ごとの「頭の重さが首を回そうとする力」と持ち上げ能力比")
    print("-" * 74)
    for o in outs:
        print(f"\n  --- {o['age']:g}ヶ月 ---")
        print(f"{'角度[度]':>10}{'腕[cm]':>10}{'頭の負荷[N·m]':>16}"
              f"{'比(負方向)':>14}{'比(正方向)':>14}")
        for r in o["rows"]:
            print(f"{r['ang']:>10}{r['arm']*100:>10.2f}{r['tau']:>16.4f}"
                  f"{r['r_neg']:>14.1f}{r['r_pos']:>14.1f}")

    print("\n" + "=" * 74)
    print("3. 姿勢によって比が何倍変わるか")
    print("-" * 74)
    print(f"{'月齢':>6}{'最小の比':>12}{'最大の比':>12}{'最大/最小':>12}"
          f"{'最小になる角度':>16}")
    for o in outs:
        rs = [min(r["r_neg"], r["r_pos"]) for r in o["rows"]]
        if not rs:
            continue
        i = int(np.argmin(rs))
        print(f"{o['age']:>6g}{min(rs):>12.1f}{max(rs):>12.1f}"
              f"{max(rs)/max(min(rs),1e-9):>12.1f}{o['rows'][i]['ang']:>14}度")

    print("\n" + "=" * 74)
    print("読み方")
    print("-" * 74)
    print("  ・負方向と正方向で値が違えば、うつ伏せ（伸ばす）と仰向け（曲げる）で")
    print("    使える力が変わる＝どちらで測るかで結論が変わる")
    print("  ・角度で比が大きく変わるなら、『47.5倍』は姿勢に依存した数字であって、")
    print("    首の強さそのものを表していない")


if __name__ == "__main__":
    main()
