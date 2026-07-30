"""おもちゃの揺れ幅と、中心バイアスの効き方を同時に切り分ける。

【なぜ】上丘の変換を入れたあとも、おもちゃの位置と反射の出す向きの符号が
一致しない条件が残った（+1.5cm に置いて h_dir = -0.30）。原因の候補が
**二層**あり、片方だけ直しても測定は良くならないので、同時に測って分ける。

  層1（実装）  中心バイアスが急峻すぎる。マグニフィケーション M∝1/(R+A)² は
               視野の端で中心の 1.05%（旧ガウス σ=0.32 は 33.6%）。
               中心付近のわずかな動きが、周辺の本物の対象に勝ちうる。
  層2（測定）  揺れの振幅 1.5cm ＝ 顔から 8.6cm で ±9.9度。
               ★条件の刻み（1.5cm）と同じ大きさなので、
               「+1.5cm に置いたおもちゃ」が揺れて中心を通過する。
               動き検出は 20フレーム前まで遡るので、動きマップは
               **揺れた軌跡全体**に広がり、条件どうしが混ざる。

【測り方】太郎は完全に静止（物理を回さない）。おもちゃだけを動かす。
揺れ幅を 3通り、置く位置を 7通り。各条件で**段階ごとの重心**を出す：

    真の位置       描き分け（segmentation）で取ったおもちゃの重心
    ①動き          motion_map をそのまま平らに重心（バイアス前）
    ②バイアス後    biased_map を平らに重心
    ③競合後        competed_map を平らに重心
    ④h_dir         実際の出力（上丘座標での重心 → 視野角へ逆変換）

【読み方】
  揺れ幅を小さくして①が真の位置に近づく  → 層2が効いている
  どの揺れ幅でも ①→② で中心へ寄る        → 層1が効いている
  ②→③ で大きく飛ぶ                       → 競合が別の山を選んでいる
  ③→④ で中心へ寄る                       → 上丘座標での重心の潰れ（これは仕様）
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

SHAKE_HZ = 2.5
SHAKES = [float(s) for s in
          os.environ.get("E_SHAKES", "0.003,0.008,0.015").split(",")]
N_FRAMES = 40
OFFSETS = [-0.05, -0.03, -0.015, 0.0, 0.015, 0.03, 0.05]


def flat_centroid_x(a, thresh_frac=0.0):
    """平らな画像で重心の横位置を -1〜1 で返す（バイアスや変換を通さない素の値）。"""
    if a is None:
        return float("nan")
    a = np.clip(np.asarray(a, dtype=float), 0, None)
    if a.max() <= 1e-12:
        return float("nan")
    if thresh_frac > 0:
        a = np.where(a >= a.max() * thresh_frac, a, 0.0)
    if a.sum() <= 1e-12:
        return float("nan")
    w = a.shape[1]
    xs = np.arange(w)[None, :]
    cx = float((xs * a).sum() / a.sum())
    hx = (w - 1) / 2.0
    return (cx - hx) / hx


def fmt(v):
    return f"{v:+7.2f}" if not np.isnan(v) else "     --"


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
    eye = np.array(d.cam_xpos[cam_id], dtype=float)
    fwd = -np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)[:, 2]
    dist = float(np.linalg.norm(base - eye))
    base = eye + fwd * dist

    print("=== 揺れ幅と中心バイアスの切り分け ===")
    print(f"  顔からおもちゃまで {dist*100:.1f} cm")
    for s in SHAKES:
        print(f"  揺れ幅 {s*100:.1f} cm → 角度で ±{np.degrees(np.arctan(s/dist)):.1f} 度")
    print(f"  条件の刻み 1.5 cm → 角度で ±{np.degrees(np.arctan(0.015/dist)):.1f} 度")
    print(f"  中心バイアス方式 = {OR.CENTER_BIAS_MODE}\n")

    for shake in SHAKES:
        ang = np.degrees(np.arctan(shake / dist))
        print(f"--- 揺れ幅 {shake*100:.1f} cm（±{ang:.1f}度）---")
        lbl2 = "②上丘+受容野" if OR.CENTER_BIAS_MODE == "grid" else "②バイアス後"
        print(f"{'置いた位置':>10}{'真の位置':>9}{'①動き':>9}{lbl2:>10}"
              f"{'③競合後':>9}{'④h_dir':>9}   ①の誤差")
        err1, err4, n = 0.0, 0.0, 0
        for off in OFFSETS:
            reflex.reset()
            pos0 = base + right * off
            for i in range(N_FRAMES):
                wob = right * (shake * np.sin(2 * np.pi * SHAKE_HZ * i * dt))
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
            truth = sv["cx"] if sv["seen"] else float("nan")
            c1 = flat_centroid_x(reflex.motion_map)
            if OR.CENTER_BIAS_MODE == "grid":
                # ★格子方式では②③の中身が変わる。
                #   ②＝上丘の地図へ写して受容野でまとめた直後（重みは掛けない）
                #   ③＝上丘の上での競合のあと
                sm = reflex._smap
                c2 = (sm.grid_direction(reflex.sc_input)[0]
                      if (sm is not None and reflex.sc_input is not None)
                      else float("nan"))
                c3 = (sm.grid_direction(reflex.competed_map,
                                        thresh_frac=OR.CENTROID_THRESH_FRAC)[0]
                      if sm is not None else float("nan"))
            else:
                c2 = flat_centroid_x(reflex.biased_map)
                c3 = flat_centroid_x(reflex.competed_map,
                                     thresh_frac=OR.CENTROID_THRESH_FRAC)
            c4 = float(reflex.h_dir)
            if not np.isnan(truth):
                if not np.isnan(c1):
                    err1 += abs(c1 - truth); n += 1
                err4 += abs(c4 - truth)
            print(f"{off*100:>+9.1f}cm{fmt(truth)}{fmt(c1)}{fmt(c2)}"
                  f"{fmt(c3)}{fmt(c4)}"
                  + (f"{abs(c1-truth):>10.2f}" if not (np.isnan(c1) or np.isnan(truth))
                     else "        --"))
        if n:
            print(f"  平均のずれ  ①動き {err1/n:.3f}   ④h_dir {err4/max(n,1):.3f}")
        print()

    VIS.close_renderers()
    env.close()
    print("=== 読み方 ===")
    print("  揺れ幅を小さくして①が真の位置に近づく → 層2（測り方）が効いている")
    print("  どの揺れ幅でも ①→② で中心へ寄る       → 層1（バイアスが急峻）が効いている")
    print("  ②→③ で大きく飛ぶ                      → 競合が別の山を選んでいる")


if __name__ == "__main__":
    main()
