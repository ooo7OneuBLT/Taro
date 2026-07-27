"""★視線誘導反射が「おもちゃを視野の中心へ寄せられるか」を測る（本番の判定）。

【なぜこれが本当の判定か】それまでは「1発で真の位置と一致するか」を見ていたが、
人間の新生児はそうならない。**第一サッケードは著しく低振幅で目標に届かず、
同じ振幅の追加サッケードが階段状に続く**（Aslin & Salapatek 1975）。
＝正しい判定は「繰り返して近づくか」。

【測り方】
  ・太郎は仰向けで脱力（action=0）。反射だけが目と首を動かす
  ・おもちゃを視野の端に置き、親が揺らしている想定で振動させる
    （文献の定位実験も出現・点滅・移動のいずれかの時間変化を伴う）
  ・毎ステップ、描き分け（segmentation）でおもちゃの真の重心を測る
  ・反射ON / OFF を同じ条件で比べる

【読み方】
  ずれが単調に減る               → 反射が働いている
  減らない・増える               → 働いていない（向きが逆なら符号の誤り）
  階段状に減る                   → 新生児らしい（1発では届かない）
"""
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for _f in ("Meiryo", "Yu Gothic", "MS Gothic", "IPAexGothic"):
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

import e_visibility as VIS

START_OFFSETS = [-0.04, -0.02, 0.02, 0.04]   # おもちゃの初期の横位置[m]
BLINK_HZ = 2.5     # おもちゃを点滅させる周期[Hz]
SECONDS = 6.0
OUT_DIR = os.path.join(_ROOT, "E", "logs", "orient_converge")


def run(orient_on, off, env, u, m, d, dt, toy_gadr, right, toy_bid, n_steps):
    """1条件を走らせ、時系列（時刻・ずれ・眼球角度・サッケード数）を返す。"""
    env.reset(seed=0)
    reflex = u._orienting
    if reflex is not None:
        reflex.reset()
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    ts, errs, eyes, sacc, cmds, tgts = [], [], [], [], [], []
    import e_toy_env as TE

    # ★おもちゃの置き場所は、親が運び始める瞬間に環境が決める（視線の正面）。
    #   そこを横へずらす。**一度だけ**動かす（毎ステップ動かすと、顔と重なった
    #   ときに hold が位置を強制して巨大な反力になり、物理が破綻した）。
    moved = False
    while getattr(u, "_toy_pending", False):
        env.step(a)
        if getattr(u, "_toy_arriving", False) and not moved and u._rest_pos is not None:
            u._rest_pos = np.array(u._rest_pos, dtype=float) + right * off
            u._carry_from = u._rest_pos + np.array([0.0, 0.0, 1.0]) * TE.TOY_APPROACH_DIST
            moved = True
    for _ in range(int(0.2 / dt)):
        env.step(a)                       # 到着後に落ち着かせる

    # ★動きを作る手段は「点滅」。おもちゃを物理的に動かすと顔と当たって壊れる。
    #   文献の定位実験も点滅光を使い（Lewis & Maurer 系）、上丘のニューロンは
    #   静止した点滅ドットにも動く刺激とほぼ同等に応答する（J Neurophysiol 2004）。
    base_rgba = m.geom_rgba[toy_gadr].copy()
    dim_rgba = base_rgba.copy()
    dim_rgba[:3] *= 0.25
    every = max(1, int(round(0.02 / dt)))
    for i in range(n_steps):
        phase = (np.sin(2 * np.pi * BLINK_HZ * i * dt) >= 0)
        m.geom_rgba[toy_gadr] = base_rgba if phase else dim_rgba
        env.step(a)
        if i % every:
            continue
        sv = VIS.visible_by_segment(m, d, toy_bid, "eye_left", size=64)
        ts.append(i * dt)
        errs.append(sv["cx"] if sv["seen"] else float("nan"))   # ★符号つき
        eyes.append(reflex._angle_deg(reflex.eye_qadr["h"])
                    if (reflex is not None and orient_on) else 0.0)
        sacc.append(reflex.n_saccades if (reflex is not None and orient_on) else 0)
        if reflex is not None and orient_on:
            cmds.append(float(getattr(reflex, "last_eye_cmd", 0.0)))
            tgts.append(float(reflex._tgt["eye_h"]))
        else:
            cmds.append(0.0); tgts.append(0.0)
    m.geom_rgba[toy_gadr] = base_rgba
    return (np.array(ts), np.array(errs), np.array(eyes), np.array(sacc),
            np.array(cmds), np.array(tgts))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    os.environ.setdefault("E_TOY_SHAPE", "sphere")
    os.environ.setdefault("E_TOY_RADIUS", "0.0056")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    import e_orienting_v2 as OR

    results = {}
    for orient_on in (True, False):
        kw = body_kwargs_from_env(0.0, verbose=False)
        kw["flexion"] = True
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(),
                           age=0.0, toy=True, vor=True, orient=orient_on, **kw)
        u = env.unwrapped
        m, d = u.model, u.data
        env.reset(seed=0)
        dt = float(m.opt.timestep) * int(u.frame_skip)
        n_steps = int(SECONDS / dt)
        toy_bid = int(m.body("test_object1").id)
        toy_gadr = int(m.body("test_object1").geomadr[0])
        cam_id = int(m.camera("eye_left").id)
        eye = np.array(d.cam_xpos[cam_id], dtype=float)
        R = np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)
        right, fwd = R[:, 0], -R[:, 2]
        dist = float(np.linalg.norm(np.array(u._rest_pos, dtype=float) - eye))
        base = eye + fwd * dist

        if orient_on:
            print("=== 視線誘導反射：おもちゃを中心へ寄せられるか ===")
            print(f"  顔からおもちゃまで {dist*100:.1f} cm   {SECONDS} 秒間")
            print(f"  1発で詰める割合 SACCADE_FRAC={OR.SACCADE_FRAC}"
                  f"  間隔 {OR.SACCADE_LATENCY}s  閾値 {OR.SACCADE_MIN_STRENGTH}")
            print(f"  首の分担 {OR.NECK_SHARE}（暫定で首は動かさない）\n")

        for off in START_OFFSETS:
            key = (orient_on, off)
            results[key] = run(orient_on, off, env, u, m, d, dt,
                               toy_gadr, right, toy_bid, n_steps)
        env.close()

    # ---- 表 ----------------------------------------------------------------
    print("  ★ずれは符号つき（正＝おもちゃが視野の右）。眼球の動きも符号つき。")
    print("    正しければ、おもちゃが右にあるとき眼球も右へ動いてずれが減る\n")
    print(f"{'初期位置':>10}{'反射':>6}{'最初のずれ':>11}{'最後のずれ':>11}"
          f"{'|ずれ|の最小':>12}{'サッケード':>11}{'眼球の動き':>11}")
    for off in START_OFFSETS:
        for on in (True, False):
            ts, errs, eyes, sacc, cmds, tgts = results[(on, off)]
            good = errs[~np.isnan(errs)]
            e0 = float(np.nanmean(errs[:5])) if len(errs) >= 5 else float("nan")
            e1 = float(np.nanmean(errs[-5:])) if len(errs) >= 5 else float("nan")
            emin = float(np.abs(good).min()) if len(good) else float("nan")
            de = float(eyes[-1] - eyes[0]) if len(eyes) else 0.0
            print(f"{off*100:>+9.1f}cm{'ON' if on else 'OFF':>6}"
                  f"{e0:>11.3f}{e1:>11.3f}{emin:>11.3f}"
                  f"{int(sacc[-1]):>11d}{de:>10.1f}度"
                  + (f"   指令 max{np.abs(cmds).max():.2f}"
                     f"  目標の振れ幅{tgts.max()-tgts.min():.1f}度" if on else ""))

    # ---- グラフ ------------------------------------------------------------
    fig, axes = plt.subplots(2, len(START_OFFSETS),
                             figsize=(4.0 * len(START_OFFSETS), 6.4), sharex=True)
    for j, off in enumerate(START_OFFSETS):
        ax = axes[0, j]
        for on, c, lb in ((True, "tab:red", "反射ON"), (False, "tab:gray", "反射OFF")):
            ts, errs, eyes, sacc, cmds, tgts = results[(on, off)]
            ax.plot(ts, errs, color=c, lw=1.4, label=lb)
        ax.set_title(f"初期位置 {off*100:+.1f} cm")
        ax.set_ylabel("中心からのずれ")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=9)
        ax2 = axes[1, j]
        ts, errs, eyes, sacc, cmds, tgts = results[(True, off)]
        ax2.plot(ts, eyes, color="tab:blue", lw=1.4, label="実際の角度")
        ax2.plot(ts, tgts, color="tab:orange", lw=1.0, ls="--", label="目標角度")
        if j == 0:
            ax2.legend(fontsize=8)
        ax2.set_xlabel("時刻[秒]")
        ax2.set_ylabel("眼球の水平角[度]")
        ax2.grid(alpha=0.3)
    fig.suptitle("視線誘導反射：おもちゃを視野の中心へ寄せられるか"
                 "（上＝ずれ／下＝眼球の角度）", fontsize=12)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "converge.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"\n  グラフを保存: {png}")
    VIS.close_renderers()


if __name__ == "__main__":
    main()
