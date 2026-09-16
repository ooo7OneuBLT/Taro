"""おもちゃが「手の届く距離」にあるかを月齢ごとに測る。

【なぜ要るか、2026-07-28】リーチングに進むにあたり、`TOY_DISTANCE = 0.086`（8.6cm）が
**0ヶ月の腕の長さ（18.6cm）を基準に決めた暫定値**のままだと分かった。
体年齢を4ヶ月に上げると腕が伸びるので、同じ距離でよいかを確かめる必要がある。

注意：「見る」だけなら近くてよいが、**リーチは腕を伸ばす動作**なので、
近すぎると伸ばしきる前に触れてしまい「伸ばす」にならない。
遠すぎれば届かない。＝腕の長さとの比で見るべき。

【測ること】
  1. 月齢ごとの腕の長さ（肩の関節から手先まで）
  2. 目からおもちゃまでの距離
  3. 肩からおもちゃまでの距離（実際に届くかはこちらで決まる）
  4. 届くか（肩からの距離 ≦ 腕の長さ か）

使い方:
    E_AGES=0,4 .venv/Scripts/python.exe E/scripts/e_reach_distance_check.py
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import json
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

AGES = [float(x) for x in os.environ.get("E_AGES", "0,4").split(",")]
SAVE_PATH = os.path.join(_HERE, os.pardir, "docs", "viewer_saved.json")


def probe(age):
    from e_toy_env import ToySupineEnv, infant_vision_params, TOY_DISTANCE
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(age, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=max(age, 0.5)),
                           age=age, toy=True, vor=True, orient=False, **kw)
        env.reset(seed=0)
    u = env.unwrapped
    m, d = u.model, u.data
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)

    # おもちゃが環境の規定位置に落ち着くまで進める
    for _ in range(int(3.0 / dt)):
        env.step(a)

    def bpos(name):
        return np.array(d.xpos[int(m.body(name).id)], dtype=float)

    # 腕の長さ＝肩 → 上腕 → 前腕 → 手 の各区間の和（実際に伸ばせる長さ）
    chain = ["right_upper_arm", "right_lower_arm", "right_hand"]
    sh = bpos("right_upper_arm")
    arm_len = 0.0
    prev = sh
    for nm in chain[1:]:
        p = bpos(nm)
        arm_len += float(np.linalg.norm(p - prev))
        prev = p
    # 手の中心から指先まではおよそ手のgeomの半分ぶん。ここでは手body中心までとする
    # 注意：[簡略化] 指先までは含めない＝控えめな見積もり

    eyes = [np.array(d.cam_xpos[int(m.camera(n).id)], dtype=float)
            for n in ("eye_left", "eye_right")]
    eye_mid = np.mean(eyes, axis=0)
    toy = bpos("test_object1")

    env.close()
    return dict(age=age, arm_len=arm_len,
                eye_dist=float(np.linalg.norm(toy - eye_mid)),
                sh_dist=float(np.linalg.norm(toy - sh)),
                toy=toy, shoulder=sh, eye=eye_mid,
                default_toy_dist=float(TOY_DISTANCE))


def main():
    print("=" * 76)
    print(" おもちゃは手の届く距離にあるか")
    print("=" * 76)
    outs = []
    for age in AGES:
        print(f"  [{age:g}ヶ月...]")
        outs.append(probe(age))

    print(f"\n  環境の既定 TOY_DISTANCE = {outs[0]['default_toy_dist']*100:.1f}cm"
          f"（目からの距離として設定される値）")

    print("\n" + "=" * 76)
    print(" 1. 腕の長さと、おもちゃまでの距離")
    print("=" * 76)
    print(f"{'月齢':>5}{'腕の長さ':>12}{'目→おもちゃ':>14}{'肩→おもちゃ':>14}"
          f"{'肩からの距離/腕':>18}{'届くか':>10}")
    for o in outs:
        ratio = o["sh_dist"] / max(o["arm_len"], 1e-9)
        ok = "届く" if ratio <= 1.0 else "届かない"
        print(f"{o['age']:>5g}{o['arm_len']*100:>11.1f}cm{o['eye_dist']*100:>13.1f}cm"
              f"{o['sh_dist']*100:>13.1f}cm{ratio:>17.2f}{ok:>10}")

    print("\n" + "=" * 76)
    print(" 2. 保存された位置（Viewerで調整したもの）ではどうか")
    print("=" * 76)
    try:
        with open(SAVE_PATH, encoding="utf-8") as fp:
            saved = json.load(fp)
        sp = np.array(saved["toy_pos"], dtype=float)
        print(f"  保存位置 {sp}")
        for o in outs:
            e = float(np.linalg.norm(sp - o["eye"]))
            s = float(np.linalg.norm(sp - o["shoulder"]))
            ratio = s / max(o["arm_len"], 1e-9)
            print(f"  {o['age']:>4g}ヶ月  目から {e*100:5.1f}cm   肩から {s*100:5.1f}cm"
                  f"   腕の {ratio*100:5.0f}%   "
                  f"{'届く' if ratio <= 1.0 else '届かない'}")
    except Exception as e:
        print(f"  保存が読めない: {e}")

    print("\n" + "=" * 76)
    print(" 読み方")
    print("=" * 76)
    print("  ・肩からの距離が腕の長さを超えていたら、そもそも届かない")
    print("  ・腕の50%未満なら近すぎ＝『伸ばす』動作にならない（触れてしまう）")
    print("  ・人間のリーチ実験は腕をほぼ伸ばした距離に対象を置く（要文献確認）")
    print("  注意指先までは含めていない（手body中心まで）＝控えめな見積もり")


if __name__ == "__main__":
    main()
