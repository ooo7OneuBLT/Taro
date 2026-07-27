"""視線誘導反射の内部を段階ごとに覗いて、左右の情報がどこで失われるかを特定する。

【なぜ】人工画像のテスト（`e_orient_v2_test.py`）は 11/11 通過しているのに、
環境では左右を区別できない（実測：目の左右差 0.063度／上下差 7.402度）。
どこかの段階で情報が落ちている。

【測り方】★自己運動をゼロにして測る。
物理を回さず（`mj_forward` だけ）、**おもちゃだけを手で動かす**。
これで「太郎は完全に静止、おもちゃだけが動く」という理想条件になり、
自己運動の交絡なしに反射の中身を見られる。

【段階ごとの重心を比べる】どれも「画像の中で右が正、左が負」で -1〜1 に正規化。
    ①画像       赤い画素の重心          ＝おもちゃの本当の位置
    ②動き検出    motion_map の重心       ステップ1の出力
    ③中心バイアス biased_map の重心      ステップ2の出力
    ④競合        competed_map の重心     ステップ3の途中
    ⑤出力        h_dir                   反射が使う値
①〜⑤が一致していれば正常。途中でずれたら、そこが壊れている場所。
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
N_FRAMES = 60          # 動き検出に必要なフレーム数（time_scales の最大 20 より多く）
OUT_DIR = os.path.join(_ROOT, "E", "logs", "orient_stage")

# おもちゃを置く向き（視線の正面からの横ずれ[m]）
OFFSETS = {
    "左へ 6cm": -0.06, "左へ 3cm": -0.03, "正面": 0.0,
    "右へ 3cm": +0.03, "右へ 6cm": +0.06,
}


def centroid_x(a):
    """2次元マップの重心のx座標。画像中心を0・右端を+1にそろえる。"""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.max() <= 1e-12:
        return float("nan")
    w = np.clip(a, 0, None)
    tot = w.sum()
    if tot < 1e-12:
        return float("nan")
    xs = np.arange(a.shape[1])
    cx = float((w.sum(axis=0) * xs).sum() / tot)
    half = (a.shape[1] - 1) / 2.0
    return (cx - half) / half


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    os.makedirs(OUT_DIR, exist_ok=True)
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

    # おもちゃが設置されるまで物理を回す
    import e_toy_env as TE
    for _ in range(int((TE.TOY_APPEAR_DELAY + TE.TOY_APPROACH_SEC + 0.2) / dt)):
        env.step(a)
    base = np.array(u._rest_pos, dtype=float)
    # 太郎の姿勢をここで固定する（以後 env.step を呼ばない＝自己運動ゼロ）
    frozen_qpos = d.qpos.copy()

    # 目の「右方向」＝カメラのローカルx軸（画像の右がどちらの世界方向か）
    R = np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    right = R[:, 0]

    print("=== 視線誘導反射：左右の情報はどこで失われるか ===")
    print("  ★太郎は完全に静止（物理を回さない）。おもちゃだけを動かす")
    print("  各段階の重心（画像の中で 右が正・左が負、-1〜1）\n")
    print(f"{'置き場所':<11}{'①画像':>9}{'②動き検出':>11}{'③中心バイアス':>15}"
          f"{'④競合':>9}{'⑤h_dir':>9}{'画素':>7}")
    print("-" * 74)

    saved_imgs = {}
    for label, dy in OFFSETS.items():
        reflex.reset()
        pos0 = base + right * dy
        c1 = c2 = c3 = c4 = h = float("nan")
        npix = 0
        for i in range(N_FRAMES):
            t = i * dt
            wob = right * (SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t))
            d.qpos[:] = frozen_qpos          # 太郎を固定
            d.qpos[toy_qadr:toy_qadr + 3] = pos0 + wob
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            u._vision_t = None               # 時間が進まないのでキャッシュを毎回捨てる
            u._vision_cache = None
            imgs = u.get_vision_obs()
            img = imgs.get("eye_left") if isinstance(imgs, dict) else None
            if img is None:
                continue
            reflex.update(img)
            if i == N_FRAMES - 1:
                v = VIS.visible_in_image(img)
                npix = v["n_pixels"]
                c1 = v["cx"] if v["seen"] else float("nan")
                c2 = centroid_x(reflex.motion_map)
                c3 = centroid_x(reflex.biased_map)
                c4 = centroid_x(reflex.competed_map)
                h = float(reflex.h_dir)
                saved_imgs[label] = (np.asarray(img).copy(),
                                     np.asarray(reflex.motion_map).copy(),
                                     np.asarray(reflex.competed_map).copy())
        print(f"{label:<11}{c1:>9.3f}{c2:>11.3f}{c3:>15.3f}{c4:>9.3f}{h:>9.3f}{npix:>7d}")

    env.close()

    # 画像を保存（目で見て確かめられるように）
    try:
        from PIL import Image
        for label, (img, mot, comp) in saved_imgs.items():
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            panels = [arr]
            for mp in (mot, comp):
                mp = np.asarray(mp, dtype=float)
                mp = mp / max(mp.max(), 1e-9)
                panels.append(np.stack([(mp * 255).astype(np.uint8)] * 3, axis=-1))
            joined = np.concatenate([np.asarray(p) for p in panels], axis=1)
            name = label.replace(" ", "").replace("へ", "_")
            Image.fromarray(joined).resize(
                (joined.shape[1] * 3, joined.shape[0] * 3),
                Image.NEAREST).save(os.path.join(OUT_DIR, f"{name}.png"))
        print(f"\n  画像を保存しました: {OUT_DIR}")
        print("  （左から：一人称の映像／動き検出／競合のあと）")
    except Exception as e:
        print(f"\n  画像の保存に失敗: {e}")

    print("\n=== 読み方 ===")
    print("  ①〜⑤が同じ符号・近い値なら正常")
    print("  ★①は左右で分かれるのに②で消える → 動き検出の問題")
    print("  ★②までは分かれるのに④で消える → 競合（側方抑制）の問題")
    print("  ★どれも0に近い → そもそもおもちゃの動きが検出できていない")


if __name__ == "__main__":
    main()
