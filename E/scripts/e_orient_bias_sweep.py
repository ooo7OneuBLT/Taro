"""中心視野バイアスの強さを振って、対象の位置をどれだけ正しく出せるかを測る。

【なぜ】設計図ステップ2で入れた「画像の中心ほど重み大」のガウス窓（σ=0.32）が、
視野の端にある対象の反応を潰している可能性がある。

  実測（2026-07-27）：おもちゃの重心 +0.687（右） → 反射の出す方向 +0.176
  ＝かなり右にあるのに「ちょっと右」としか判断していない

⚠️この測定は「動き検出が方向を持たない」という別の問題とは**独立**。
  縁しか光らないことによるずれと、中心バイアスによる圧縮は、原因が違う。

【測り方】太郎を完全に静止させ（物理を回さない）、おもちゃだけを動かす。
おもちゃを視野のいろいろな位置に置き、
    ★真の位置（描き分けで取得。色の判定は本当の46%しか拾えないため）
    反射が出す方向 h_dir
を比べる。理想は 真の位置 ≈ h_dir。

【根拠】中心視野バイアスは上丘の中心視野マグニフィケーション（中心視野が広い面積を
占める）の模倣。σ の値自体は [Tier3・ARBITRARY]。
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

SHAKE_HZ = 2.5
SHAKE_AMP = 0.015
N_FRAMES = 40
SIGMAS = [0.20, 0.32, 0.50, 0.80, 2.00]     # 2.00 ＝ ほぼバイアス無し
OFFSETS = [-0.05, -0.03, -0.015, 0.0, 0.015, 0.03, 0.05]


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    import e_orienting_v2 as OR

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
    toy_jid = next(j for j in range(m.njnt)
                   if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                   and m.body(m.jnt_bodyid[j]).name == "test_object1")
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    cam_id = int(m.camera("eye_left").id)

    import e_toy_env as TE
    for _ in range(int((TE.TOY_APPEAR_DELAY + TE.TOY_APPROACH_SEC + 0.2) / dt)):
        env.step(a)
    frozen = d.qpos.copy()
    base = np.array(u._rest_pos, dtype=float)
    right = np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)[:, 0]
    # ★左目のカメラで測るので、左目の正面を基準にする（両目の中点だと右にずれる）
    eye = np.array(d.cam_xpos[cam_id], dtype=float)
    fwd = -np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)[:, 2]
    base = eye + fwd * float(np.linalg.norm(base - eye))

    print("=== 中心視野バイアスの強さを振る ===")
    print("  ★太郎は完全に静止。おもちゃだけを動かす")
    print("  真の位置＝描き分けで取得（色の判定は本当の46%しか拾えない）")
    print("  理想は 真の位置 ≒ h_dir\n")

    hdr = "".join(f"{o*100:+6.1f}cm" for o in OFFSETS)
    print(f"{'σ':>6}  {'真の位置':>10}{hdr}")
    truths = []
    rows = {}
    for si, sg in enumerate(SIGMAS):
        OR.CENTER_BIAS_SIGMA_FRAC = sg
        hs, ts = [], []
        for off in OFFSETS:
            reflex.reset()
            reflex._center_weight = None     # ★念のため明示（reset でも消えるが）
            pos0 = base + right * off
            for i in range(N_FRAMES):
                wob = right * (SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * i * dt))
                d.qpos[:] = frozen
                d.qpos[toy_qadr:toy_qadr + 3] = pos0 + wob
                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
                u._vision_t = None
                u._vision_cache = None
                imgs = u.get_vision_obs()
                img = imgs.get("eye_left") if isinstance(imgs, dict) else None
                if img is not None:
                    reflex.update(img)
            sv = VIS.visible_by_segment(m, d, toy_bid, "eye_left", size=128)
            ts.append(sv["cx"] if sv["seen"] else float("nan"))
            hs.append(float(reflex.h_dir))
        if si == 0:
            truths = ts
            print(f"{'':>6}  {'（真値）':>10}"
                  + "".join(f"{v:+8.2f}" if not np.isnan(v) else "     --"
                            for v in ts))
            print("-" * (18 + 8 * len(OFFSETS)))
        rows[sg] = hs
        # 真値との一致度（符号が合っているか・相関）
        ok = sum(1 for t, h in zip(truths, hs)
                 if not np.isnan(t) and np.sign(t) == np.sign(h) and abs(t) > 0.05)
        n = sum(1 for t in truths if not np.isnan(t) and abs(t) > 0.05)
        print(f"{sg:>6.2f}  {'h_dir':>10}"
              + "".join(f"{v:+8.2f}" for v in hs)
              + f"   符号が合う {ok}/{n}")

    VIS.close_renderers()
    env.close()
    print("\n=== 読み方 ===")
    print("  σが大きいほどバイアスは弱い（2.00 はほぼ無し）")
    print("  ★σを大きくして h_dir が真の位置に近づくなら、バイアスが強すぎる")
    print("  ★どのσでも近づかないなら、原因は別（＝方向を検出していないこと）")


if __name__ == "__main__":
    main()
