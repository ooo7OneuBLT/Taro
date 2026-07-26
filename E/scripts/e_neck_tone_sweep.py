"""首のバネの剛性を3軸それぞれ振って、頭が落ち着く角度と視界を測る。

【なぜ】前後の傾き（head_tilt）だけにバネを入れたら、残る2軸
（左右のひねり head_swivel／横倒し head_tilt_side）が自由に倒れ、
**顔が横を向いておもちゃが視界から消えた**（実測：45→47秒で side が 11→52度）。

顔が横を向くこと自体は人間的（新生児は覚醒時に右65%・左15%が顔を横に向ける
／Michel 1981 Science 212:685-687）。だが実験でおもちゃが視界から消えるのは困る。

【軸ごとに重力の効き方が違う】
    前後の傾き   仰向けで頭が後屈する方向に重力が効く      → 強め
    横倒し       頭が横に転がる方向。これも重力が効く      → 中くらい
    ひねり       軸がほぼ鉛直。重力の影響は小さい          → 弱め
ユーザーの目視「3軸すべてONにすると行き過ぎる」は、3軸に同じ0.4を入れたため。

【目標角】前後だけ -45度（重力込みの実効値）。他の2軸は0度（正中）。
⚠️「正中を目標にする」根拠は無い[Tier3]。新生児は顔を横に向けるのが普通だが、
  脱力時の物理的な釣り合い角の実測は文献に存在しない（2026-07-26の調査）。
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

SEC = 60.0
TARGET_TILT = -45.0
AXES = ("head_tilt", "head_swivel", "head_tilt_side")
JP = {"head_tilt": "前後", "head_swivel": "ひねり", "head_tilt_side": "横倒し"}

# (前後, ひねり, 横倒し) の剛性
COMBOS = [
    (0.4, 0.0, 0.0),     # 今の実装（前後だけ）
    (0.4, 0.1, 0.1),
    (0.4, 0.2, 0.2),
    (0.4, 0.4, 0.4),     # ユーザーが「行き過ぎる」と言った条件
    (0.4, 0.1, 0.4),     # ひねりは弱く、横倒しは強く
    (0.4, 0.05, 0.2),
]


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    os.environ.setdefault("E_TOY_MODE", "hold")
    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, vor=True, orient=False, **kw)
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)

    ids = {}
    for j in range(m.njnt):
        s = m.joint(j).name.split(":")[-1]
        if s in AXES:
            ids[s] = j
    hb = int(m.body("head").id)
    arm = float(np.linalg.norm(np.array(d.xpos[hb])
                               - np.array(d.xpos[int(m.body_parentid[hb])])))
    I = float(m.body_inertia[hb][0]) + float(m.body_mass[hb]) * arm ** 2

    def run(ks):
        env.reset(seed=0)
        for s, k in zip(AXES, ks):
            j = ids[s]
            m.jnt_stiffness[j] = k
            m.qpos_spring[int(m.jnt_qposadr[j])] = (np.radians(TARGET_TILT)
                                                    if s == "head_tilt" else 0.0)
            m.dof_damping[int(m.jnt_dofadr[j])] = max(
                0.0145, 2.0 * float(np.sqrt(max(k, 1e-12) * I)))
        seen = []
        for i in range(int(SEC / dt)):
            env.step(a)
            if i % 50 == 0:
                u._vision_t = None
                u._vision_cache = None
                imgs = u.get_vision_obs()
                img = imgs.get("eye_left") if isinstance(imgs, dict) else None
                if img is not None:
                    seen.append(1.0 if VIS.visible_in_image(img)["seen"] else 0.0)
        ang = {s: float(np.degrees(d.qpos[int(m.jnt_qposadr[ids[s]])])) for s in AXES}
        vel = {s: abs(float(np.degrees(d.qvel[int(m.jnt_dofadr[ids[s]])]))) for s in AXES}
        return ang, vel, (float(np.mean(seen)) * 100 if seen else 0.0)

    print("=== 首のバネ：3軸の剛性を振る ===")
    print(f"  {SEC:.0f}秒・action=0（完全脱力）・目標角は前後だけ {TARGET_TILT:+.0f}度、"
          f"他は0度（正中）\n")
    print(f"{'前後':>5}{'ひねり':>7}{'横倒し':>7} | "
          f"{'前後':>8}{'ひねり':>9}{'横倒し':>9}{'最大速さ':>11}{'見えた割合':>12}")
    print("-" * 76)
    for ks in COMBOS:
        ang, vel, seen = run(ks)
        print(f"{ks[0]:>5.2f}{ks[1]:>7.2f}{ks[2]:>7.2f} | "
              f"{ang['head_tilt']:>8.1f}{ang['head_swivel']:>9.1f}"
              f"{ang['head_tilt_side']:>9.1f}"
              f"{max(vel.values()):>9.3f}度/s{seen:>11.1f}%")
    env.close()
    print("\n=== 読み方 ===")
    print("  見えた割合が高く、最大速さが小さい（＝落ち着いている）組み合わせを選ぶ")
    print("  ★ただし『顔が横を向く』のは人間的なので、正中に固めすぎるのも逸脱になる")


if __name__ == "__main__":
    main()
