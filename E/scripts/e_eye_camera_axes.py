"""左右の目のカメラ座標系が鏡像になっていないかを確かめる。

【なぜ要るか、2026-07-28】眼球は左右とも同じだけ動いているのに（差0.15度）、
カメラ座標系で測った「おもちゃの見かけの方向」の変化が**左右で符号が逆**だった：

    眼球の動き    左 -8.95度 ／ 右 -9.10度   ＝ ほぼ同じ
    見かけの変化  左 -10.26度 ／ 右 +9.56度  ＝ ★符号が逆

＝カメラの right（右方向）の定義が左右で反転している疑い。
もしそうなら「右目の挙動がよくわからん」（ユーザーの目視）は反射の問題ではなく、
**測り方の問題**になる。

【確かめ方】眼球の関節を既知の角度だけ回し、カメラの向きベクトルが
どちらへ動くかを左右で比べる。物理的に同じ向きに回っているなら、
world 座標での向きの変化も同じ側になるはず。

使い方:
    .venv/Scripts/python.exe E/scripts/e_eye_camera_axes.py
"""
import os
import sys
import warnings
import contextlib
import io

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
           os.path.join(_ROOT, "taro_core"),
           os.path.join(_ROOT, "taro_core", "src", "body"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import mujoco       # noqa: E402

AGE = float(os.environ.get("E_AGE", "4.0"))


def main():
    from d_supine_env import SupineMimoEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(AGE, verbose=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = SupineMimoEnv(vision_params=None, age=AGE,
                            actuation_model=MuscleModel, **kw)
        env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data

    qadr, cam = {}, {}
    for side in ("left", "right"):
        j = m.joint(f"robot:{side}_eye_horizontal")
        qadr[side] = int(m.jnt_qposadr[j.id])
        cam[side] = int(m.camera(f"eye_{side}").id)

    def snapshot():
        mujoco.mj_forward(m, d)
        out = {}
        for side in ("left", "right"):
            R = np.array(d.cam_xmat[cam[side]], dtype=float).reshape(3, 3)
            out[side] = dict(right=R[:, 0].copy(), up=R[:, 1].copy(),
                             fwd=(-R[:, 2]).copy(),
                             pos=np.array(d.cam_xpos[cam[side]], dtype=float))
        return out

    print("=" * 76)
    print(" 左右の目のカメラ座標系（体年齢 %.0fヶ月）" % AGE)
    print("=" * 76)

    base = snapshot()
    print("\n1. 中立（関節角0）でのカメラの向き（world座標）")
    print("-" * 76)
    for side in ("left", "right"):
        b = base[side]
        print(f"  {side:<6} 位置 [{b['pos'][0]:+.4f} {b['pos'][1]:+.4f} {b['pos'][2]:+.4f}]")
        print(f"         前方 [{b['fwd'][0]:+.3f} {b['fwd'][1]:+.3f} {b['fwd'][2]:+.3f}]"
              f"   右 [{b['right'][0]:+.3f} {b['right'][1]:+.3f} {b['right'][2]:+.3f}]"
              f"   上 [{b['up'][0]:+.3f} {b['up'][1]:+.3f} {b['up'][2]:+.3f}]")
    dot_fwd = float(np.dot(base["left"]["fwd"], base["right"]["fwd"]))
    dot_right = float(np.dot(base["left"]["right"], base["right"]["right"]))
    dot_up = float(np.dot(base["left"]["up"], base["right"]["up"]))
    print(f"\n  左右の軸の一致度（1.0なら同じ向き、-1.0なら真逆）")
    print(f"    前方 {dot_fwd:+.4f}   右 {dot_right:+.4f}   上 {dot_up:+.4f}")
    if dot_right < 0:
        print("    ★右方向ベクトルが逆＝**カメラ座標系が鏡像**")

    # 2. 関節を +10度 回したとき、前方ベクトルがどちらへ動くか
    print("\n2. 眼球の関節を +10度 回したとき、視線（前方）がどちらへ動くか")
    print("-" * 76)
    for side in ("left", "right"):
        d.qpos[qadr[side]] = np.radians(10.0)
    after = snapshot()
    for side in ("left", "right"):
        dv = after[side]["fwd"] - base[side]["fwd"]
        # world の y 軸（太郎の左右方向）成分で「どちらへ振れたか」を見る
        print(f"  {side:<6} 前方の変化 [{dv[0]:+.4f} {dv[1]:+.4f} {dv[2]:+.4f}]")
    dy_l = float((after["left"]["fwd"] - base["left"]["fwd"])[1])
    dy_r = float((after["right"]["fwd"] - base["right"]["fwd"])[1])
    print(f"\n  左右方向(y)への振れ  左 {dy_l:+.4f}  右 {dy_r:+.4f}")
    if dy_l * dy_r > 0:
        print("  → 同じ側へ振れている＝**関節の符号は左右で揃っている**（共同運動として正しい）")
    else:
        print("  → ★逆へ振れている＝関節の符号が左右で逆＝同じ指令で寄り目/開散になる")

    # 3. 同じ world 座標の点を、左右のカメラ座標で測ると符号が揃うか
    print("\n3. 同じ点を左右のカメラ座標系で測ったとき、符号は揃うか")
    print("-" * 76)
    for side in ("left", "right"):
        d.qpos[qadr[side]] = 0.0
    base = snapshot()
    mid = 0.5 * (base["left"]["pos"] + base["right"]["pos"])
    fwd = 0.5 * (base["left"]["fwd"] + base["right"]["fwd"])
    fwd /= np.linalg.norm(fwd)
    rgt = 0.5 * (base["left"]["right"] + base["right"]["right"])
    rgt /= max(np.linalg.norm(rgt), 1e-9)
    print(f"{'テスト点':<28}{'左目での水平角':>16}{'右目での水平角':>16}")
    for label, offset in (("正面 20cm", 0.0), ("正面20cm＋右へ5cm", +0.05),
                          ("正面20cm＋左へ5cm", -0.05)):
        p = mid + fwd * 0.20 + rgt * offset
        angs = []
        for side in ("left", "right"):
            b = base[side]
            v = p - b["pos"]
            v = v / np.linalg.norm(v)
            angs.append(np.degrees(np.arctan2(np.dot(v, b["right"]),
                                              np.dot(v, b["fwd"]))))
        print(f"{label:<28}{angs[0]:>15.2f}度{angs[1]:>15.2f}度")
    print("\n  ★同じ点なのに左右で符号が逆なら、カメラ座標系が鏡像になっている。")
    print("    その場合『左目から見て右／右目から見て左』は同じ物理方向を指すことがある。")
    env.close()


if __name__ == "__main__":
    main()
