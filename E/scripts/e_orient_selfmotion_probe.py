"""視線誘導反射が「自分の動き」に反応していないかを切り分ける。

【なぜ】反射ONにするとおもちゃが視界から消える（画素 3255→268）のに、
サッケードは**撃ちっぱなし**（6.7秒で32発＝潜時0.2秒の上限）。
見えていないのに撃ち続けるなら、おもちゃ以外の何かに反応している。

いちばん疑わしいのは自分の動き：
    首が動く → 視界全体が流れる → それを「動き」と検出 → さらに首が動く → 発散

【切り分け】4条件でサッケードの数と反応の強さを比べる。

  A おもちゃを揺らす・反射ON       … 通常
  B ★おもちゃを止める・反射ON      … 動くものが何も無い。撃ったら自分の動きが原因
  C ★おもちゃを消す・反射ON        … 対象そのものが無い
  D おもちゃを揺らす・反射OFF      … 首が動かない状態での動き検出だけを見る
     （反射OFFでも update() は呼ばれるので strength は測れる）

Bで撃つなら**残差法が破れている**（自分で動かした結果を外界の動きと誤認している）。
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

SEC = 5.0
SHAKE_HZ = 2.5
SHAKE_AMP = 0.015


def run(env, shake, reflex_on, toy_visible=True):
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

    rf = u._orienting
    if rf is not None:
        rf.reset()
    if not reflex_on:
        u._orienting = None

    import e_toy_env as TE
    for _ in range(int((TE.TOY_APPEAR_DELAY + TE.TOY_APPROACH_SEC + 0.2) / dt)):
        env.step(a)
    base = np.array(u._rest_pos, dtype=float)
    if not toy_visible:
        base = np.array([3.0, 3.0, 0.05])       # 遠くへ退避＝視界に何も無い

    # ★眼球の角速度も測る。頭が止まっていても眼が動けば視界は流れる。
    eye_dofs = [int(m.jnt_dofadr[j]) for j in range(m.njnt)
                if "eye" in m.joint(j).name]
    # 視線そのものの流れ（カメラのz軸が1tickで何度回ったか）＝反射が見る「動き」の元
    cam_id = int(m.camera("eye_left").id)

    strengths, hs, vs, heads, pixels, eyes, gazes = [], [], [], [], [], [], []
    prev_fwd = [None]
    t = 0.0
    for _ in range(int(SEC / dt)):
        off = SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t) if shake else 0.0
        u._rest_pos = base + np.array([0.0, off, 0.0])
        d.qpos[toy_qadr:toy_qadr + 3] = u._rest_pos
        d.qvel[toy_dof:toy_dof + 6] = 0.0
        env.step(a)
        t += dt
        # 反射OFFのときも動き検出だけは回して strength を見る
        if not reflex_on and rf is not None:
            imgs = u.get_vision_obs()
            if isinstance(imgs, dict) and "eye_left" in imgs:
                rf.update(imgs["eye_left"])
        if rf is not None:
            strengths.append(float(rf.strength))
            hs.append(float(rf.h_dir))
            vs.append(float(rf.v_dir))
        heads.append(float(np.linalg.norm(d.cvel[int(m.body("head").id)][:3])))
        eyes.append(float(np.max(np.abs(d.qvel[eye_dofs]))) if eye_dofs else 0.0)
        fwd = -np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)[:, 2]
        if prev_fwd[0] is not None:
            c = float(np.clip(np.dot(fwd, prev_fwd[0]), -1, 1))
            gazes.append(float(np.degrees(np.arccos(c))) / dt)   # 度/秒
        prev_fwd[0] = fwd
        imgs = u.get_vision_obs()
        img = imgs.get("eye_left") if isinstance(imgs, dict) else None
        if img is not None:
            pixels.append(VIS.visible_in_image(img)["n_pixels"])

    n_sacc = getattr(rf, "n_saccades", 0)
    u._orienting = rf
    return dict(n_sacc=n_sacc,
                strength=float(np.mean(strengths)) if strengths else float("nan"),
                strength_max=float(np.max(strengths)) if strengths else float("nan"),
                h=float(np.mean(hs)) if hs else float("nan"),
                v=float(np.mean(vs)) if vs else float("nan"),
                head=float(np.mean(heads)), head_max=float(np.max(heads)),
                eye=float(np.mean(eyes)) if eyes else 0.0,
                gaze=float(np.mean(gazes)) if gazes else 0.0,
                gaze_max=float(np.max(gazes)) if gazes else 0.0,
                pix=float(np.mean(pixels)) if pixels else 0.0)


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, vor=True, orient=True, **kw)

    cases = [
        ("A 揺らす・反射ON     ", dict(shake=True,  reflex_on=True,  toy_visible=True)),
        ("B ★止める・反射ON    ", dict(shake=False, reflex_on=True,  toy_visible=True)),
        ("C ★おもちゃ無し・ON  ", dict(shake=False, reflex_on=True,  toy_visible=False)),
        ("D 揺らす・反射OFF    ", dict(shake=True,  reflex_on=False, toy_visible=True)),
    ]

    print("=== 視線誘導反射は何に反応しているか ===")
    print(f"  {SEC:.0f}秒・自発運動なし\n")
    print(f"{'条件':<22}{'サッケード':>11}{'反応の強さ':>12}{'最大':>9}"
          f"{'頭ω':>8}{'眼ω':>8}{'視線の流れ':>12}{'最大':>9}{'画素':>7}")
    print("-" * 98)
    res = {}
    for label, kwargs in cases:
        r = run(env, **kwargs)
        res[label] = r
        print(f"{label:<22}{r['n_sacc']:>11d}{r['strength']:>12.4f}"
              f"{r['strength_max']:>9.4f}{r['head']:>8.3f}{r['eye']:>8.3f}"
              f"{r['gaze']:>10.1f}°/s{r['gaze_max']:>8.0f}{r['pix']:>7.0f}")
    env.close()

    print("\n=== 読み方 ===")
    b = res["B ★止める・反射ON    "]
    c = res["C ★おもちゃ無し・ON  "]
    print(f"  B（動くものが何も無い）でサッケード {b['n_sacc']} 発")
    print(f"  C（そもそも対象が無い）でサッケード {c['n_sacc']} 発")
    if b["n_sacc"] > 3 or c["n_sacc"] > 3:
        print("\n  ★★動くものが無いのに撃っている＝**自分の動きに反応している**。")
        print("     残差法（自分で動かしたぶんを差し引く）が働いていない。")
    else:
        print("\n  動くものが無ければ撃たない＝外界の動きに正しく反応している。")
        print("     逆効果の原因は別（方向の符号、サッケードの大きさなど）。")


if __name__ == "__main__":
    main()
