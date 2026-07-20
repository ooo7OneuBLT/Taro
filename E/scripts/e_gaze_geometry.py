"""【配置の幾何】太郎の視線がどこを向いているかを実測し、おもちゃの正しい置き場所を求める。

【なぜ要るか＝実測で判明した設計ミス】
おもちゃは頭基準 (-0.05, 0.0, 0.07)＝**後方5cm・上7cm** に吊るしていた。仰向けなので視線は
ほぼ真上(+z)。つまり atan(5/7)=35.5° のずれが**最初から**あり、視野の半角30°(fovy=60°)を
**頭が完全に正面を向いていても超えている**。実測「視線から平均70.1°／視界内は6.5%のtick」の
うち、35.5°は頭の揺れではなく**配置そのもの**が作っていた。
私は「手が届く範囲」だけを見て吊り位置を決め、**視線が通るかを一度も確認しなかった**。

【このスクリプトが出すもの】
 (1) 落ち着いた初期姿勢での視線方向（eye_leftカメラのz軸）と、頭ローカル系での向き
 (2) 今のおもちゃが視線から何度ずれているか
 (3) 「視線の正面・距離D」に置くには頭ローカルでどのオフセットにすべきか
 (4) babbling中に頭がどれだけ振れるか＝正面に置いても何%のtickで視界に残るか

使い方: python e_gaze_geometry.py [n_ticks] [distance_m]
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

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
import torch  # noqa: E402
import d_c5_motor_quality as mq  # noqa: E402
import e_toy_env as te  # noqa: E402


def gaze_dir(model, data, cam="eye_left"):
    """視線方向（カメラのz軸の**逆**＝MuJoCoのカメラは-z方向を見る）。"""
    cid = int(model.camera(cam).id)
    return -np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]


def angle_deg(u, v):
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    dist = float(sys.argv[2]) if len(sys.argv) > 2 else float(np.linalg.norm(te.TOY_OFFSET))
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    raw = env.unwrapped
    m, d = raw.model, raw.data
    half_fov = te.VISION_FOVY / 2.0

    obs, _ = env.reset(seed=0)
    head_bid = int(m.body("head").id)
    g = gaze_dir(m, d)
    hp = np.array(d.xpos[head_bid], dtype=float)
    R = np.array(d.xmat[head_bid], dtype=float).reshape(3, 3)   # 頭ローカル→ワールド
    toy = np.array(d.body("test_object1").xpos, dtype=float)

    print("=== (1) 落ち着いた初期姿勢での視線 ===")
    print(f"  視線方向(world)      : [{g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}]")
    print(f"  真上(+z)からのずれ   : {angle_deg(g, np.array([0, 0, 1.0])):.1f}°")
    print(f"  視線を頭ローカルで   : {np.round(R.T @ g, 3)}")

    print("\n=== (2) 今のおもちゃ ===")
    v = toy - hp
    print(f"  頭→おもちゃ 距離     : {np.linalg.norm(v)*100:.1f} cm")
    print(f"  視線からのずれ       : {angle_deg(g, v):.1f}°   (視野の半角 {half_fov:.0f}°)")
    print(f"  →  {'視界内' if angle_deg(g, v) <= half_fov else '★視界の外（頭が正面でも見えない）'}")
    print(f"  今の TOY_OFFSET(頭ローカル) = {np.round(te.TOY_OFFSET, 3)}")

    print(f"\n=== (3) 視線の正面・距離{dist*100:.0f}cm に置くなら ===")
    new_local = R.T @ (g * dist)
    print(f"  ★ TOY_OFFSET = [{new_local[0]:+.4f}, {new_local[1]:+.4f}, {new_local[2]:+.4f}]  (頭ローカル)")

    print(f"\n=== (4) babbling中に頭が振れても視界に残るか（{n} tick） ===")
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    angs_now, angs_new, tilt = [], [], []
    for t in range(n):
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = mq.rescale_action(a, env.action_space); prev_a = a
        for k in range(mq.K):
            obs, r, te_, tr, info = env.step(ctrl)
            if te_ or tr:
                break
        gg = gaze_dir(m, d)
        hpp = np.array(d.xpos[head_bid], dtype=float)
        RR = np.array(d.xmat[head_bid], dtype=float).reshape(3, 3)
        angs_now.append(angle_deg(gg, np.array(d.body("test_object1").xpos) - hpp))
        angs_new.append(angle_deg(gg, RR @ new_local))   # 頭に固定した「正面」なら常に0のはず
        tilt.append(angle_deg(gg, np.array([0, 0, 1.0])))
        if te_ or tr:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    angs_now = np.asarray(angs_now); tilt = np.asarray(tilt)
    print(f"  頭の傾き(真上から)   : mean {tilt.mean():.1f}°  p50 {np.median(tilt):.1f}°  max {tilt.max():.1f}°")
    print(f"  今の配置のずれ       : mean {angs_now.mean():.1f}°  → 視界内 {(angs_now<=half_fov).mean()*100:.1f}% のtick")
    print(f"  【参考】視線に完全追従なら常に0°＝100%だが、おもちゃは空間に固定なので"
          f"頭が {tilt.mean():.0f}° 振れれば同じだけずれる")
    print(f"  → 空間固定のまま視界に入れ続けるには、視野の半角が {np.percentile(tilt,50):.0f}° 以上必要")
    env.close()


if __name__ == "__main__":
    main()
