"""眼球はそもそもどれだけ動かせるのか（筋力と可動域の健康診断）。

【なぜ】視線誘導反射のサッケードが、指令 0.64 を出しているのに眼球が 0.1度しか
動かなかった。反射の側の問題か、身体の側の問題かを切り分ける。

【測ること】
  1. 眼球の水平・垂直アクチュエータに**最大指令**を入れて、何度動くか
  2. 可動域（モデル上の上限・下限）
  3. VOR ON / OFF で違うか（VORが眼球指令を上書きしている疑い）
  4. 月齢による筋力スケーリングが眼球にも掛かっているか
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"),
          os.path.join(_ROOT, "taro_core", "src", "brain"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from spinal_cord.cpg import write_joint_command

    for vor_on in (False, True):
        kw = body_kwargs_from_env(0.0, verbose=False)
        kw["flexion"] = True
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(),
                           age=0.0, toy=False, vor=vor_on, orient=False, **kw)
        u = env.unwrapped
        m, d = u.model, u.data
        env.reset(seed=0)
        dt = float(m.opt.timestep) * int(u.frame_skip)
        nu = int(m.nu)
        dim = env.action_space.shape[0]

        eye_acts = [i for i in range(nu)
                    if "eye" in m.actuator(i).name and "horizontal" in m.actuator(i).name]
        if vor_on is False:
            print("=== 眼球はどれだけ動かせるか ===")
            print(f"  アクチュエータ数 {nu}   行動の次元 {dim}"
                  f"   （{'筋肉モデル＝2倍' if dim >= 2*nu else 'ばね系'}）\n")
            for i in eye_acts:
                jid = int(m.actuator_trnid[i, 0])
                lo, hi = np.degrees(m.jnt_range[jid])
                gear = float(m.actuator_gear[i, 0])
                print(f"  {m.actuator(i).name:32} 可動域 {lo:+6.1f}〜{hi:+6.1f}度"
                      f"   gear={gear:.4g}")
            print()

        # ★VORは「方策の眼球出力を捨てる」設計（皮質は反射弓に介入しない）。
        #   視線誘導反射は override の**後**に加算されるので消えない。
        #   ここでも同じ経路を再現しないと、VOR ON の測定は
        #   「VORだけの動き」になってしまう（最初にこれで測り違えた）。
        if vor_on and u._vor is not None:
            _orig = u._vor.override
            _hold = {"cmd": 0.0}

            def _patched(action, model, data, dt_, _o=_orig, _h=_hold):
                out = _o(action, model, data, dt_)
                for i in eye_acts:
                    write_joint_command(out, i, _h["cmd"], nu,
                                        co_activation=0.0, additive=True)
                return out
            u._vor.override = _patched
        else:
            _hold = None

        print(f"--- VOR {'ON' if vor_on else 'OFF'}"
              f"{'（反射と同じ経路で加算）' if vor_on else ''} ---")
        print(f"{'指令':>8}{'0.5秒後の角度':>15}{'1.0秒後':>12}{'到達速度':>12}")
        for cmd in (0.25, 0.5, 1.0):
            env.reset(seed=0)
            if _hold is not None:
                _hold["cmd"] = cmd
            angs = []
            for step in range(int(1.0 / dt)):
                a = np.zeros(dim, dtype=np.float32)
                if _hold is None:
                    for i in eye_acts:
                        write_joint_command(a, i, cmd, nu, co_activation=0.0)
                env.step(a)
                if abs((step + 1) * dt - 0.5) < dt / 2 or step == int(1.0 / dt) - 1:
                    jid = int(m.actuator_trnid[eye_acts[0], 0])
                    angs.append(float(np.degrees(d.qpos[m.jnt_qposadr[jid]])))
            spd = (angs[-1] - angs[0]) / 0.5 if len(angs) >= 2 else float("nan")
            print(f"{cmd:>8.2f}{angs[0]:>14.2f}度{angs[-1]:>11.2f}度{spd:>10.1f}度/秒")
        env.close()
        print()

    print("=== 読み方 ===")
    print("  最大指令でも数度しか動かない → 身体側（筋力・可動域）の問題")
    print("  VOR ON で動かなくなる        → VOR が眼球指令を打ち消している")
    print("  どちらでもよく動く           → 反射側（指令の作り方）の問題")


if __name__ == "__main__":
    main()
