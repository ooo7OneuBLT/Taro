"""体を「起こす」正しい回転を総当たりで見つける。

【なぜ要るか、2026-07-28】リクライニングのために体を y軸まわりに回したが、
+70度でも -70度でも**頭が下がった**（体幹 -37.5度 / -19.4度）。
仰向けの基準は +6.0度で、頭は +x 方向にある。
＝「y軸まわり・world基準で左から掛ける」という仮定のどこかが違う。

【やること】回転軸（x/y/z）× 符号（±）× 合成の順序（左/右）を総当たりし、
**背もたれを置かない状態で mj_forward だけ**して体幹の傾きを測る。
物理を進めないので、接触の影響を受けずに「置き方」だけを見られる。

使い方:
    .venv/Scripts/python.exe E/scripts/e_recline_rot_probe.py
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
TARGET = float(os.environ.get("E_RECLINE", "70"))


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(AGE, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # 背もたれを作らない（recline_deg=0）＝置き方だけを見る
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=AGE),
                           age=AGE, toy=True, vor=True, orient=False,
                           recline_deg=0.0, **kw)
        env.reset(seed=0)
    u = env.unwrapped
    m, d = u.model, u.data

    hip_bid = int(m.body("hip").id)
    head_bid = int(m.body("head").id)
    # 体全体を動かす free joint を探す（hip には無い＝2026-07-28 に判明）。
    #   おもちゃ（test_object1）も free joint を持つので除外する。
    qadr, root_name = None, None
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        bn = m.body(int(m.jnt_bodyid[j])).name
        if "object" in bn or "toy" in bn:
            continue
        qadr = int(m.jnt_qposadr[j]); root_name = bn
        break
    print("=" * 78)
    print(" 体を起こす回転の総当たり（背もたれなし・物理を進めない）")
    print("=" * 78)
    print(f"  体の自由関節: {root_name}（qpos adr={qadr}）" if qadr is not None
          else "  注意自由関節が見つからない")

    q_base = (np.array(d.qpos[qadr + 3:qadr + 7], dtype=float).copy()
              if qadr is not None else np.array(m.body_quat[hip_bid], dtype=float).copy())
    p_base = (np.array(d.qpos[qadr:qadr + 3], dtype=float).copy()
              if qadr is not None else np.array(m.body_pos[hip_bid], dtype=float).copy())

    def trunk_deg():
        mujoco.mj_forward(m, d)
        v = np.array(d.xpos[head_bid]) - np.array(d.xpos[hip_bid])
        return float(np.degrees(np.arctan2(v[2], np.linalg.norm(v[:2])))), \
            np.array(d.xpos[hip_bid]).copy(), np.array(d.xpos[head_bid]).copy()

    # 基準
    if qadr is not None:
        d.qpos[qadr:qadr + 3] = p_base
        d.qpos[qadr + 3:qadr + 7] = q_base
    base, hip0, head0 = trunk_deg()
    print(f"  基準（回転なし）  体幹 {base:+.1f}度   "
          f"骨盤({hip0[0]:+.3f},{hip0[2]:+.3f})  頭({head0[0]:+.3f},{head0[2]:+.3f})")
    print(f"  ＝頭は骨盤の {'+x' if head0[0] > hip0[0] else '-x'} 側にある")

    print("\n" + "=" * 78)
    print(f" {TARGET:g}度 回したときの体幹の傾き（目標＝{TARGET:g}度に近いもの）")
    print("=" * 78)
    print(f"{'軸':>4}{'符号':>6}{'掛ける側':>10}{'体幹の傾き':>14}{'頭の高さ':>12}{'判定':>10}")

    axes = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
            "z": np.array([0, 0, 1.0])}
    best = None
    for an, av in axes.items():
        for sign in (+1, -1):
            th = np.radians(TARGET) * sign
            qt = np.zeros(4)
            mujoco.mju_axisAngle2Quat(qt, av, th)
            for side in ("左", "右"):
                qn = np.zeros(4)
                if side == "左":
                    mujoco.mju_mulQuat(qn, qt, q_base)   # world基準
                else:
                    mujoco.mju_mulQuat(qn, q_base, qt)   # body基準
                if qadr is not None:
                    d.qpos[qadr:qadr + 3] = p_base
                    d.qpos[qadr + 3:qadr + 7] = qn
                t, hp, hd = trunk_deg()
                err = abs(t - TARGET)
                mark = "" if err < 10 else ("〇" if err < 25 else "")
                if best is None or err < best[0]:
                    best = (err, an, sign, side, t)
                print(f"{an:>4}{sign:>+6}{side:>10}{t:>13.1f}度{hd[2]:>11.3f}{mark:>10}")

    print("\n" + "=" * 78)
    if best:
        print(f" 最も近い：{best[1]}軸 まわり {best[2]*TARGET:+.0f}度 を"
              f"{best[3]}から掛ける → 体幹 {best[4]:.1f}度（目標 {TARGET:g}度）")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
