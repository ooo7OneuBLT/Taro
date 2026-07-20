"""【関門②の原因切り分け】視覚エンコーダは「手が映っているか」を区別できているか。

【なぜ調べるか】
関門②（4000tick×3シード）の結果：**手が視野に入っても視覚の予測誤差が動かない**
（効果量 d=+0.044 / -0.151 / -0.070＝符号すらバラバラ＝差が無い）。
一方で「視界が変化したtick」では動く（d≒+0.20、3シードとも正）＝視覚そのものは生きている。
＝**「視界が変わったこと」は分かるが「そこに手があること」は分からない**、という状態。

【切り分ける3つの候補】
 (A) **64次元への圧縮で手の情報が失われている**
     VisionEncoder は**ランダム初期化・未学習のCNN**（正解側は凍結）。手のような小さな対象の
     有無を64次元に残す保証がない。→ 同じ場面で「手あり画像」と「手なし画像」を作り、
     エンコーダの出力がどれだけ離れるかを直接測る。
 (B) **視力フィルタで手が潰れている**（1ヶ月児相当 0.852 cyc/deg ≒ 20/700）
     → acuity ON/OFF で (A) を測り直して比較する。
 (C) **そもそも画像上で手が小さすぎる/映っていない**
     → 手が占めるピクセル数を数える。

【判定】
 生の画像では違うのにエンコーダ出力で差が消えるなら (A)＝圧縮の問題。
 生の画像でも差が小さいなら (B) か (C)＝見え方の問題。
 どちらが効いているかで、次の設計（エンコーダを学習させる／視野や距離を変える）が変わる。

使い方: python e_encoder_probe.py [n_samples]
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import mimoEnv  # noqa: F401,E402
import mujoco  # noqa: E402
import d_c5_motor_quality as mq  # noqa: E402
import e_toy_env as te  # noqa: E402
import e_target as et  # noqa: E402
from e_hand_in_view import eye_angles, EYES, hand_in_view  # noqa: E402

FAR = np.array([5.0, 5.0, 5.0])     # 手を隠すための退避先（視界外）


def hand_geoms(model):
    """左右の手・指に属するgeomのid（＝画像から消す対象）。"""
    keys = ("hand", "ff", "lf", "th", "mf", "rf", "finger")
    ids = []
    for i in range(model.ngeom):
        bid = model.geom_bodyid[i]
        nm = model.body(bid).name
        if any(k in nm for k in keys):
            ids.append(i)
    return np.array(ids, dtype=int)


def render(model, data, ren, cam_name, fovy):
    cid = int(model.camera(cam_name).id)
    model.cam_fovy[cid] = fovy
    c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FIXED; c.fixedcamid = cid
    ren.update_scene(data, camera=c)
    return ren.render().copy()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    raw = env.unwrapped
    m, d = raw.model, raw.data
    half_fov = te.VISION_FOVY / 2.0
    hg = hand_geoms(m)
    frozen = et.make_frozen_fusion(fusion, touch_dim=0, vision_res=te.VISION_RES)
    ren = mujoco.Renderer(m, height=te.VISION_RES, width=te.VISION_RES)

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    rows = []
    print(f"\n手が視野に入った場面を {n} 個集めて、"
          f"「手あり画像」vs「手を消した画像」を比べます…")

    tick = 0
    while len(rows) < n and tick < 20000:
        tick += 1
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = mq.rescale_action(a, env.action_space); prev_a = a
        for k in range(mq.K):
            obs, r, term, trunc, info = env.step(ctrl)
            if term or trunc:
                break
        # 手が視野に入っているか
        inview = None
        for h in ("left_hand", "right_hand"):
            ang = eye_angles(m, d, np.array(d.body(h).xpos, dtype=float))
            if all(ang[c] <= half_fov for c in EYES):
                inview = h
                break
        if inview is not None:
            # ①手あり ②手のgeomを透明にして消す（物理は動かさない＝同じ場面のまま）
            with_hand = render(m, d, ren, "eye_left", te.VISION_FOVY)
            saved = m.geom_rgba[hg].copy()
            m.geom_rgba[hg, 3] = 0.0                    # α=0＝描画されない
            without = render(m, d, ren, "eye_left", te.VISION_FOVY)
            m.geom_rgba[hg] = saved

            g = lambda x: x.astype(float).mean(axis=2)
            px = float((np.abs(g(with_hand) - g(without)) > 2.0).sum())   # 手が占める画素数
            raw_diff = float(np.abs(g(with_hand) - g(without)).mean())

            # acuityをかけた版（＝太郎が実際に受け取る画像）
            ac_w = raw.vision._apply_acuity(with_hand, "eye_left").astype(float)
            ac_o = raw.vision._apply_acuity(without, "eye_left").astype(float)
            ac_diff = float(np.abs(ac_w.mean(axis=2) - ac_o.mean(axis=2)).mean())

            # 凍結エンコーダに通したときの差（＝太郎の脳に届く差）
            with torch.no_grad():
                e_w = frozen.vision(ac_w.astype(np.uint8), ac_w.astype(np.uint8))
                e_o = frozen.vision(ac_o.astype(np.uint8), ac_o.astype(np.uint8))
            enc_diff = float((e_w - e_o).abs().mean())
            enc_scale = float(e_w.abs().mean())
            rows.append((px, raw_diff, ac_diff, enc_diff, enc_scale))
        if term or trunc:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)

    if not rows:
        print("⚠️手が視野に入る場面を1つも集められませんでした")
        env.close(); return
    R = np.asarray(rows, dtype=float)
    print(f"\n--- {len(rows)} 場面（{tick} tick 走査）---")
    print(f"  (C) 手が占める画素数        : mean {R[:,0].mean():7.1f} / {te.VISION_RES**2} px"
          f"  = {100*R[:,0].mean()/te.VISION_RES**2:.2f}%  (最大 {R[:,0].max():.0f})")
    print(f"  (B前) 生画像の差            : mean {R[:,1].mean():.4f}")
    print(f"  (B後) 視力フィルタ後の差    : mean {R[:,2].mean():.4f}"
          f"   （フィルタで {100*(1-R[:,2].mean()/max(R[:,1].mean(),1e-9)):.1f}% 減衰）")
    print(f"  (A)  エンコーダ出力の差     : mean {R[:,3].mean():.5f}"
          f"   （出力自体の大きさ {R[:,4].mean():.5f} ＝ 相対 "
          f"{100*R[:,3].mean()/max(R[:,4].mean(),1e-9):.2f}%）")

    print("\n=== 判定 ===")
    rel = R[:, 3].mean() / max(R[:, 4].mean(), 1e-9)
    if R[:, 0].mean() < 20:
        print("  (C) ★手が画像上で小さすぎる（20px未満）＝そもそも情報がほとんど無い")
    if R[:, 1].mean() > 1e-3 and R[:, 2].mean() < R[:, 1].mean() * 0.3:
        print("  (B) ★視力フィルタが手の情報を大きく削っている")
    if rel < 0.02:
        print("  (A) ★エンコーダの出力がほとんど変わらない＝64次元への圧縮で手が消えている")
    if rel >= 0.02 and R[:, 0].mean() >= 20:
        print("  → エンコーダは手の有無を残している。関門②のNGは別の原因（予測の側）の可能性")
    env.close()


if __name__ == "__main__":
    main()
