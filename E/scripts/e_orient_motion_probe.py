"""動き検出（ステップ1）が対象の位置をずらす原因を切り分ける。

【なぜ】揺れ幅を振る実験（e_orient_shake_probe.py）で、**中心バイアスを掛ける前**の
段階ですでに重心が真の位置から 0.20〜0.38 ずれていることが分かった。
しかもずれの向きが**いつも画面の中心向き**。原因の候補が3つある：

  候補1  縁だけが光るので、おもちゃの中心でなく縁の平均を指している
  候補2  おもちゃ以外にも動いているものがある（影・背景の描画ゆらぎ）
  候補3  重心を全画素で取っているので、弱い雑音が画面全体で効いている
         ＝**私の測り方の問題**であって、反射の欠陥ではない

【測り方】太郎は完全に静止（物理を回さない）。おもちゃだけを揺らす。
おもちゃの位置を7通りに変え、各条件で

  ・重心を閾値を変えて測る（0 / 0.10 / 0.20 / 0.35 / 0.50）
      閾値を上げて真の位置に近づく → 候補3（雑音）
  ・おもちゃの領域の内と外で、動きの合計を比べる
      外が大きい → 候補2（他に動くものがある）
  ・動きマップを画像に保存して目視
      縁だけが光り、内と外の比が正常なら → 候補1

おもちゃの領域は描き分け（segmentation rendering）で取る。色の判定では
本当の46%しか拾えないことが分かっているため（落とし穴チェックリスト 項55）。
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for _f in ("Meiryo", "Yu Gothic", "MS Gothic", "TakaoGothic", "IPAexGothic"):
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

import e_visibility as VIS

SHAKE_HZ = 2.5
SHAKE_AMP = 0.008
N_FRAMES = 40
OFFSETS = [-0.05, -0.03, -0.015, 0.0, 0.015, 0.03, 0.05]
THRESHES = [0.0, 0.10, 0.20, 0.35, 0.50]
OUT_DIR = os.path.join(_ROOT, "E", "logs", "orient_motion")


def centroid_x(a, thresh_frac=0.0):
    """平らな画像で重心の横位置を -1〜1 で返す。"""
    a = np.clip(np.asarray(a, dtype=float), 0, None)
    if a.max() <= 1e-12:
        return float("nan")
    if thresh_frac > 0:
        a = np.where(a >= a.max() * thresh_frac, a, 0.0)
    if a.sum() <= 1e-12:
        return float("nan")
    w = a.shape[1]
    cx = float((np.arange(w)[None, :] * a).sum() / a.sum())
    hx = (w - 1) / 2.0
    return (cx - hx) / hx


def fmt(v):
    return f"{v:+7.2f}" if not np.isnan(v) else "     --"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
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
    right = np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)[:, 0]
    eye = np.array(d.cam_xpos[cam_id], dtype=float)
    fwd = -np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)[:, 2]
    dist = float(np.linalg.norm(np.array(u._rest_pos, dtype=float) - eye))
    base = eye + fwd * dist

    print("=== 動き検出（ステップ1）の切り分け ===")
    print(f"  顔からおもちゃまで {dist*100:.1f} cm  揺れ幅 {SHAKE_AMP*100:.1f} cm")
    print("  ★中心バイアス・競合を通す**前**の生の動きマップだけを見る\n")

    print("--- 1. 閾値を変えて重心を測る ---")
    hdr = "".join(f"{t:>8.2f}" for t in THRESHES)
    print(f"{'置いた位置':>10}{'真の位置':>9}{hdr}    おもちゃ外の動きの割合")
    maps, truths, masks, rgbs, sides = [], [], [], [], []
    errs = {t: [] for t in THRESHES}
    for off in OFFSETS:
        reflex.reset()
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
                last_rgb = np.asarray(img)
        mp = np.asarray(reflex.motion_map, dtype=float)
        rgbs.append(last_rgb)
        sv = VIS.visible_by_segment(m, d, toy_bid, "eye_left", size=mp.shape[0])
        truth = sv["cx"] if sv["seen"] else float("nan")
        mask = VIS.segment_mask(m, d, toy_bid, "eye_left", size=mp.shape[0])
        mask = np.asarray(mask, dtype=bool)
        # おもちゃは揺れるので、領域を少し広げてから内外を比べる
        from scipy.ndimage import binary_dilation
        grown = binary_dilation(mask, iterations=6)
        inside = float(mp[grown].sum())
        outside = float(mp[~grown].sum())
        frac_out = outside / max(inside + outside, 1e-12)

        # ★おもちゃの真の重心を境に、動きを左右に分けて明るさを比べる。
        #   おもちゃが揺れて光るのは「左の縁が掃いた帯」と「右の縁が掃いた帯」。
        #   本来この2本は同じ明るさのはずで、重心はおもちゃの中心に来る。
        if not np.isnan(truth):
            xc = (truth + 1) / 2 * (mp.shape[1] - 1)
            xs_grid = np.arange(mp.shape[1])[None, :]
            lft = float(mp[:, :].sum(axis=0)[xs_grid[0] < xc].sum())
            rgt = float(mp[:, :].sum(axis=0)[xs_grid[0] >= xc].sum())
        else:
            lft = rgt = float("nan")
        sides.append((lft, rgt))

        cs = [centroid_x(mp, t) for t in THRESHES]
        for t, c in zip(THRESHES, cs):
            if not (np.isnan(c) or np.isnan(truth)):
                errs[t].append(abs(c - truth))
        rat = lft / max(rgt, 1e-12) if not np.isnan(lft) else float("nan")
        print(f"{off*100:>+9.1f}cm{fmt(truth)}" + "".join(fmt(c) for c in cs)
              + f"{frac_out*100:>13.1f} %"
              + (f"   左:右 = {lft/(lft+rgt)*100:4.0f}:{rgt/(lft+rgt)*100:<4.0f}"
                 if not np.isnan(rat) else ""))
        maps.append(mp)
        truths.append(truth)
        masks.append(grown)

    print(f"\n{'':>19}" + "".join(f"{t:>8.2f}" for t in THRESHES))
    print(f"{'平均のずれ':>19}"
          + "".join(f"{np.mean(errs[t]):>8.3f}" if errs[t] else "      --"
                    for t in THRESHES))

    # ---- 2. 動きマップを画像に保存 ----------------------------------------
    fig, axes = plt.subplots(1, len(OFFSETS), figsize=(3.0 * len(OFFSETS), 3.6))
    for ax, mp, tr, mk, off in zip(axes, maps, truths, masks, OFFSETS):
        ax.imshow(mp, cmap="inferno", vmin=0, vmax=max(mp.max(), 1e-9))
        ys, xs = np.nonzero(mk)
        if len(xs):
            ax.contour(mk.astype(float), levels=[0.5], colors="cyan",
                       linewidths=0.8)
        w = mp.shape[1]
        if not np.isnan(tr):
            ax.axvline((tr + 1) / 2 * (w - 1), color="cyan", lw=1.2, ls="--")
        c0 = centroid_x(mp, 0.0)
        c3 = centroid_x(mp, 0.35)
        if not np.isnan(c0):
            ax.axvline((c0 + 1) / 2 * (w - 1), color="lime", lw=1.2)
        if not np.isnan(c3):
            ax.axvline((c3 + 1) / 2 * (w - 1), color="white", lw=1.2, ls=":")
        ax.set_title(f"{off*100:+.1f} cm\n真{tr:+.2f} / 閾0 {c0:+.2f} / 閾.35 {c3:+.2f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("動き検出の生の出力（水色破線＝おもちゃの真の重心／緑＝閾値なしの重心／"
                 "白点線＝閾値0.35の重心／水色の輪郭＝おもちゃの領域）", fontsize=11)
    fig.tight_layout()
    import e_toy_env as _TE
    tag = f"{_TE.TOY_SHAPE}_r{_TE.TOY_RADIUS*1000:.1f}mm"
    png = os.path.join(OUT_DIR, f"motion_maps_{tag}.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"\n  画像を保存: {png}")

    # ---- 3. 目に映っている生の画像も保存（照明・背景・遮蔽を目で見る）------
    fig, axes = plt.subplots(1, len(OFFSETS), figsize=(3.0 * len(OFFSETS), 3.4))
    for ax, rgb, tr, off in zip(axes, rgbs, truths, OFFSETS):
        im = rgb
        if im.dtype != np.uint8:
            im = np.clip(im, 0, 1)
        ax.imshow(im)
        w = im.shape[1]
        if not np.isnan(tr):
            ax.axvline((tr + 1) / 2 * (w - 1), color="cyan", lw=1.2, ls="--")
        ax.set_title(f"{off*100:+.1f} cm", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("左目に映っている生の画像（水色破線＝おもちゃの真の重心）", fontsize=11)
    fig.tight_layout()
    png2 = os.path.join(OUT_DIR, f"eye_rgb_{tag}.png")
    fig.savefig(png2, dpi=110)
    plt.close(fig)
    print(f"  画像を保存: {png2}")

    VIS.close_renderers()
    env.close()
    print("\n=== 読み方 ===")
    print("  閾値を上げて真の位置に近づく          → 候補3（弱い雑音）＝私の測り方の問題")
    print("  おもちゃ外の割合が大きい              → 候補2（他に動くものがある）")
    print("  外は小さく、閾値を上げても近づかない  → 候補1（縁だけが光る性質）")


if __name__ == "__main__":
    main()
