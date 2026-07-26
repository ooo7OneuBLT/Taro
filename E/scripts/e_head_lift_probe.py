"""首すわりの力学ゲート＝首の筋力で頭を持ち上げられるか、を月齢別に測る。

【なぜ】E1（首すわり相当＋リーチ）に進む前に、そもそも太郎の身体が
「頭を持ち上げる」という物理条件を満たしているかを、学習なしで確認する。

【測るもの】
  1. 静的な指標：lift_ratio ＝ 首の筋力 / 頭の重力モーメント（既存関数）
  2. 動的な実測：首の伸筋だけを最大活性化して、頭の角度が何度上がるか
     ・仰向けから開始、他の筋は0
     ・3秒間 activation=1.0、その後3秒 free
     ・head_tilt の qpos の最大値（絶対値）を記録
     ・引き起こしテスト：仰向けから頭を上げる方向は head_tilt が
       負なら顎を引く／正なら仰け反る、モデル依存

【期待】
  age 0     ： ratio=1.0 の設計だが、動的には持ち上がらない可能性
  age 5     ： ratio=236 で余裕あり、しっかり上がる
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

AGES = [0.0, 1.5, 3.0, 4.0, 5.0]
HOLD_SEC = 3.0
FREE_SEC = 3.0


def run(age):
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from infant_neck import lift_ratio, head_gravity_torque, TILT_ACTUATOR, TILT_JOINT

    kw = body_kwargs_from_env(age, verbose=False)
    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=age, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    am = getattr(env.unwrapped, "actuation_model", None)
    env.reset(seed=0)

    ratio, gear, tau = lift_ratio(m, d, am)
    tau_grav, head_mass, arm = head_gravity_torque(m, d)

    aid = [i for i in range(m.nu) if m.actuator(i).name == TILT_ACTUATOR][0]
    jid = [j for j in range(m.njnt) if m.joint(j).name == TILT_JOINT][0]
    qadr = int(m.jnt_qposadr[jid])

    n_act = env.action_space.shape[0]
    n_joint = n_act // 2
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)

    initial_head = float(np.degrees(d.qpos[qadr]))

    # 首の伸筋（頭を上げる向き）を最大活性化。他は 0.5（無指令＝中立）
    # MuscleModelは前半n_jointが「負方向筋」、後半n_jointが「正方向筋」
    # → まず両方向を試す
    results = {}
    for label, side in [("neg(顎引き)", 0), ("pos(仰け反り)", 1)]:
        env.reset(seed=0)
        # ★首以外の全筋は最小活性化(0)にする＝完全脱力
        a_hold = np.zeros(n_act, dtype=np.float32)
        if side == 0:
            a_hold[aid] = 1.0  # 前半n_joint内なら負方向筋
        else:
            a_hold[aid + n_joint] = 1.0  # 後半なら正方向筋

        n_hold = int(HOLD_SEC / dt)
        n_free = int(FREE_SEC / dt)
        head_angles = []
        for step in range(n_hold + n_free):
            a = a_hold if step < n_hold else np.zeros(n_act, dtype=np.float32)
            env.step(a)
            head_angles.append(float(np.degrees(d.qpos[qadr])))

        head_angles = np.array(head_angles)
        peak = head_angles[:n_hold].max()
        trough = head_angles[:n_hold].min()
        # 初期姿勢からの変化量
        delta_up = peak - initial_head
        delta_down = initial_head - trough
        results[label] = dict(delta_up=delta_up, delta_down=delta_down,
                              peak=peak, trough=trough)

    env.close()
    return dict(ratio=ratio, strength=gear, torque=tau,
                head_mass=head_mass, arm_m=arm, initial=initial_head,
                results=results)


def main():
    print("=== 首の力学ゲート測定 ===")
    print("  静的な指標（lift_ratio）＋ 動的な実測（首伸筋を3秒最大活性化）")
    print("  ratio = 首の最大筋力 ÷ 頭の重力モーメント")
    print("  ratio < 1 : 物理的に頭を持ち上げられない")
    print("  ratio = 1 : 境界（動的にはたぶん無理）")
    print("  ratio > 1 : 上げられる（余裕は倍率次第）\n")

    print(f"{'age':>5}{'ratio':>8}{'strength(Nm)':>14}{'grav_torq(Nm)':>15}"
          f"{'head_mass(kg)':>15}{'arm(cm)':>10}{'init(deg)':>11}")
    print("-" * 78)
    all_r = {}
    for age in AGES:
        r = run(age)
        all_r[age] = r
        print(f"{age:>5.1f}{r['ratio']:>8.2f}{r['strength']:>14.3f}"
              f"{r['torque']:>15.4f}{r['head_mass']:>15.3f}"
              f"{r['arm_m']*100:>10.2f}{r['initial']:>11.2f}")

    print("\n=== 3秒間 最大活性化 → 頭の角度変化（deg）===")
    print(f"{'age':>5}  {'方向':<18}{'△上向き':>12}{'△下向き':>12}{'peak':>10}")
    print("-" * 66)
    for age in AGES:
        r = all_r[age]
        for lbl, v in r["results"].items():
            print(f"{age:>5.1f}  {lbl:<18}{v['delta_up']:>12.2f}{v['delta_down']:>12.2f}"
                  f"{v['peak']:>10.2f}")

    print("\n  --- 判定の目安 ---")
    print("  仰向けから顔を持ち上げる（＝首すわり相当）に必要な角度差：")
    print("    5-10度  ： 頭を軽く動かせる（新生児レベル）")
    print("    15-30度 ： 見回せる（2ヶ月〜）")
    print("    45度以上： 首すわりに向かう（2ヶ月末〜）")
    print("    90度    ： 完全な首すわり（3-4ヶ月）")


if __name__ == "__main__":
    main()
