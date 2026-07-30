"""リクライニング環境が成立しているかを測る（Viewerで見る前の数値確認）。

【なぜ要るか、2026-07-28】仰向けでは「見える位置」と「手が届く位置」が両立しない。
人間の実験は体を起こして解決している（Carvalho et al. 2007／Savelsbergh & van der Kamp 1994）。
太郎にもリクライニング環境を作ったので、**まず数値で成立を確かめる**
（落とし穴チェックリスト項60「目視の報告だけで動かない。まず数値で状態を出す」）。

【測ること】
  1. 体が背もたれの角度に沿っているか（体幹の傾き）
  2. ずり落ちていないか（数秒後の位置の変化）
  3. ★肩からおもちゃまでの距離が腕の長さに収まるか＝手が届くか
  4. おもちゃが視野に入っているか

使い方:
    E_RECLINES=0,45,70 .venv/Scripts/python.exe E/scripts/e_recline_check.py
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

AGE = float(os.environ.get("E_AGE", "4.0"))
RECLINES = [float(x) for x in os.environ.get("E_RECLINES", "0,45,70").split(",")]
SETTLE = float(os.environ.get("E_SETTLE", "4.0"))


def probe(rec):
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    import e_visibility as VIS

    kw = body_kwargs_from_env(AGE, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=AGE),
                           age=AGE, toy=True, vor=True, orient=False,
                           recline_deg=rec, **kw)
        env.reset(seed=0)
    u = env.unwrapped
    m, d = u.model, u.data
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)

    def bpos(n):
        return np.array(d.xpos[int(m.body(n).id)], dtype=float)

    hip0 = bpos("hip").copy()
    for _ in range(int(SETTLE / dt)):
        env.step(a)
    hip1 = bpos("hip").copy()

    # 体幹の傾き：hip → head のベクトルが水平から何度上を向いているか
    head = bpos("head")
    hip = bpos("hip")
    v = head - hip
    trunk_deg = float(np.degrees(np.arctan2(v[2], np.linalg.norm(v[:2]))))

    # 腕の長さ（肩→肘→手）と、肩からおもちゃまで
    sh = bpos("right_upper_arm")
    arm = (float(np.linalg.norm(bpos("right_lower_arm") - sh))
           + float(np.linalg.norm(bpos("right_hand") - bpos("right_lower_arm"))))
    toy = bpos("test_object1")
    eyes = [np.array(d.cam_xpos[int(m.camera(n).id)], dtype=float)
            for n in ("eye_left", "eye_right")]
    eye_mid = np.mean(eyes, axis=0)

    seen = VIS.visible_by_segment(m, d, int(m.body("test_object1").id), "eye_left", size=64)
    toy_bid = int(m.body("test_object1").id)
    env.close()
    return dict(rec=rec, trunk_deg=trunk_deg,
                slide=float(np.linalg.norm(hip1 - hip0)),
                hip_z=float(hip1[2]), arm=arm,
                sh_dist=float(np.linalg.norm(toy - sh)),
                eye_dist=float(np.linalg.norm(toy - eye_mid)),
                seen=bool(seen["seen"]),
                cx=float(seen.get("cx", float("nan"))),
                cy=float(seen.get("cy", float("nan"))))


def main():
    print("=" * 78)
    print(f" リクライニング環境の確認（体年齢 {AGE:g}ヶ月・{SETTLE:g}秒落ち着かせてから）")
    print("=" * 78)
    outs = []
    for rec in RECLINES:
        print(f"  [{rec:g}度...]")
        outs.append(probe(rec))

    print("\n" + "=" * 78)
    print(" 1. 姿勢は保てているか")
    print("=" * 78)
    print(f"{'設定[度]':>10}{'体幹の傾き[度]':>16}{'ずれた距離[cm]':>16}{'骨盤の高さ[cm]':>16}")
    for o in outs:
        print(f"{o['rec']:>10.0f}{o['trunk_deg']:>16.1f}{o['slide']*100:>16.2f}"
              f"{o['hip_z']*100:>16.1f}")
    print("  ・体幹の傾きが設定に近ければ、背もたれに沿えている")
    print("  ・ずれた距離が大きければ、ずり落ちている")

    print("\n" + "=" * 78)
    print(" 2. ★手は届くか")
    print("=" * 78)
    print(f"{'設定[度]':>10}{'腕の長さ[cm]':>14}{'肩→おもちゃ':>14}"
          f"{'腕に対する割合':>16}{'判定':>12}")
    for o in outs:
        r = o["sh_dist"] / max(o["arm"], 1e-9)
        v = "★届く" if r <= 1.0 else "届かない"
        print(f"{o['rec']:>10.0f}{o['arm']*100:>13.1f}cm{o['sh_dist']*100:>13.1f}cm"
              f"{r*100:>15.0f}%{v:>12}")

    print("\n" + "=" * 78)
    print(" 3. おもちゃは見えているか")
    print("=" * 78)
    print(f"{'設定[度]':>10}{'目→おもちゃ':>14}{'見える':>10}"
          f"{'視野内の位置(横,縦)':>22}")
    for o in outs:
        pos = (f"({o['cx']:+.2f},{o['cy']:+.2f})" if o["seen"] else "-")
        print(f"{o['rec']:>10.0f}{o['eye_dist']*100:>13.1f}cm"
              f"{('見える' if o['seen'] else '★見えない'):>10}{pos:>22}")
    print("  ・視野内の位置は -1〜+1。0に近いほど中心")

    print("\n" + "=" * 78)
    print(" まとめ")
    print("=" * 78)
    ok = [o for o in outs if o["seen"] and o["sh_dist"] / max(o["arm"], 1e-9) <= 1.0]
    if ok:
        _lst = "、".join(f"{o['rec']:.0f}度" for o in ok)
        print(f"  ★「見えて届く」のは {_lst}")
    else:
        print("  ⚠️「見えて届く」条件がまだ無い。おもちゃの位置か角度の調整が要る")


if __name__ == "__main__":
    main()
