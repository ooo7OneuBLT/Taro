"""【うつぶせ検出】太郎が仰向けからどれだけ転がるかを実測する。

【なぜ測るか】
目視で「長く動かしていると太郎がうつぶせになる」ことが分かった。人間で考えると2つの問題がある：
 ①**新生児は寝返りできない**（寝返りの獲得は4〜6ヶ月）。＝うつぶせになること自体が非人間的で、
   「運動が強すぎる」ことの症状かもしれない。
 ②仮に転がったとしても、**親が仰向けに戻す**（乳児は仰向け寝が推奨されている）。

【何を測るか】
体幹(upper_body)の「腹側」がワールドのどちらを向いているかを角度で追う。
  0°   = 完全に仰向け（腹が真上）
  90°  = 真横（側臥位）
  180° = 完全にうつぶせ
これを全tickで記録し、①どれだけの時間どの姿勢か ②うつぶせに到達する頻度と時刻 を出す。

⚠️「腹側」がモデルのどの軸かは決め打ちせず、**リセット直後（＝仰向けが保証される姿勢）の
   体幹の軸のうち、最もワールド+zに近いもの**を自動で「腹側」と判定する。

使い方: python e_prone_check.py [n_ticks] [seed]
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
import torch  # noqa: E402
import d_c5_motor_quality as mq  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])


def belly_angle(model, data, bid, axis_idx, sign):
    """体幹の腹側とワールド上方向のなす角[度]。0=仰向け、180=うつぶせ。"""
    R = np.array(data.xmat[bid], dtype=float).reshape(3, 3)
    v = sign * R[:, axis_idx]
    return float(np.degrees(np.arccos(np.clip(float(np.dot(v, UP)), -1.0, 1.0))))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    raw = env.unwrapped
    m, d = raw.model, raw.data
    bid = int(m.body("upper_body").id)

    torch.manual_seed(seed); np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    # リセット直後は仰向けが保証されるので、そのとき最も"上"を向いている体幹軸を腹側とみなす
    R0 = np.array(d.xmat[bid], dtype=float).reshape(3, 3)
    dots = [float(np.dot(R0[:, i], UP)) for i in range(3)]
    axis_idx = int(np.argmax(np.abs(dots)))
    sign = 1.0 if dots[axis_idx] > 0 else -1.0
    print(f"  （腹側の軸を自動判定: 体幹ローカル軸{axis_idx} 符号{sign:+.0f}"
          f"／リセット時の角度 {belly_angle(m, d, bid, axis_idx, sign):.1f}°）")

    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    angs = []
    first_prone = None
    for t in range(n):
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = mq.rescale_action(a, env.action_space); prev_a = a
        for k in range(mq.K):
            obs, r, term, trunc, info = env.step(ctrl)
            if term or trunc:
                break
        ang = belly_angle(m, d, bid, axis_idx, sign)
        angs.append(ang)
        if first_prone is None and ang > 135.0:
            first_prone = t
        if term or trunc:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)

    a = np.asarray(angs)
    print(f"\n--- {n} tick（1tick=1sim秒＝約{n/60:.1f}分ぶん）---")
    print(f"  腹側の向き: mean {a.mean():.1f}°  median {np.median(a):.1f}°  max {a.max():.1f}°")
    bins = [("仰向け  (  0- 45°)", (a <= 45)),
            ("やや横  ( 45- 90°)", (a > 45) & (a <= 90)),
            ("横向き超( 90-135°)", (a > 90) & (a <= 135)),
            ("うつぶせ(135-180°)", (a > 135))]
    for name, mask in bins:
        print(f"    {name}: {mask.mean()*100:5.1f}%  ({mask.sum()} tick)")
    if first_prone is not None:
        print(f"  ★初めてうつぶせ(>135°)になったのは {first_prone} tick 目"
              f"（約{first_prone/60:.1f}分）")
    else:
        print(f"  うつぶせ(>135°)には一度もならなかった")
    # 戻ってこられるか（人間の新生児は自力で戻れない）
    over90 = a > 90
    if over90.any():
        idx = np.where(over90)[0]
        print(f"  90°を超えた回数: {np.sum(np.diff(np.concatenate([[0], over90.astype(int)])) > 0)} 回"
              f"／初回 {idx[0]} tick 目／90°超の総時間 {over90.mean()*100:.1f}%")
    env.close()


if __name__ == "__main__":
    main()
