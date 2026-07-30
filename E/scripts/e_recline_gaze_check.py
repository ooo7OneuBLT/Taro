"""リクライニング時に視線がどこを向くかを、首の角度を振って測る。

【なぜ要るか、2026-07-28】リクライニング60度にしたところ、ユーザーの目視で
「視線をもうちょっと下にしないと最初から視界外になっちゃう」。

首のバネの目標角 `NECK_TONE_TARGET = -45度` は**仰向けで重力を織り込んだ実効値**
（`taro_core/src/body/infant_body.py`）。体を起こすと重力の向きが体に対して変わるので、
この値のままでは頭が上を向いてしまう。

【測ること】首の目標角を振って
  1. 視線が水平から何度を向くか（正＝上、負＝下）
  2. おもちゃが視野に入るか
を出し、**おもちゃが見える首の角度**を見つける。

使い方:
    E_RECLINE=60 .venv/Scripts/python.exe E/scripts/e_recline_gaze_check.py
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
REC = float(os.environ.get("E_RECLINE", "60"))
TARGETS = [float(x) for x in
           os.environ.get("E_NECK_TARGETS", "-45,-30,-15,0,15,30,45").split(",")]
SETTLE = float(os.environ.get("E_SETTLE", "4.0"))


def probe(neck_target):
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
                           recline_deg=REC, **kw)
        env.reset(seed=0)
        # 首のバネの目標角を上書きする。
        # ⚠️環境のコンストラクタは neck_tone_target を受け取らないので、
        #   構築後に core の関数を直接呼ぶ。qpos_spring は model の値なので
        #   reset しても消えない。
        from infant_body import apply_neck_tone
        apply_neck_tone(env.unwrapped.model, AGE, target=neck_target,
                        data=env.unwrapped.data, verbose=False)
    u = env.unwrapped
    m, d = u.model, u.data
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for _ in range(int(SETTLE / dt)):
        env.step(a)

    # 視線の向き（両目の中点から見た前方）
    fwds, eyes = [], []
    for nm in ("eye_left", "eye_right"):
        cid = int(m.camera(nm).id)
        R = np.array(d.cam_xmat[cid], dtype=float).reshape(3, 3)
        fwds.append(-R[:, 2])
        eyes.append(np.array(d.cam_xpos[cid], dtype=float))
    fwd = np.mean(fwds, axis=0)
    fwd /= np.linalg.norm(fwd)
    eye_mid = np.mean(eyes, axis=0)
    # 水平から何度上を向いているか（正＝上）
    gaze_deg = float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))))

    toy_bid = int(m.body("test_object1").id)
    toy = np.array(d.xpos[toy_bid], dtype=float)
    seen = VIS.visible_by_segment(m, d, toy_bid, "eye_left", size=64)
    # おもちゃが視線からどれだけずれているか
    v = toy - eye_mid
    v = v / max(np.linalg.norm(v), 1e-9)
    off_deg = float(np.degrees(np.arccos(np.clip(np.dot(v, fwd), -1, 1))))

    sh = np.array(d.xpos[int(m.body("right_upper_arm").id)], dtype=float)
    el = np.array(d.xpos[int(m.body("right_lower_arm").id)], dtype=float)
    hd = np.array(d.xpos[int(m.body("right_hand").id)], dtype=float)
    arm = float(np.linalg.norm(el - sh)) + float(np.linalg.norm(hd - el))
    env.close()
    return dict(target=neck_target, gaze=gaze_deg, seen=bool(seen["seen"]),
                off=off_deg, eye_dist=float(np.linalg.norm(toy - eye_mid)),
                sh_ratio=float(np.linalg.norm(toy - sh)) / max(arm, 1e-9),
                cx=float(seen.get("cx", float("nan"))),
                cy=float(seen.get("cy", float("nan"))))


def main():
    print("=" * 78)
    print(f" リクライニング{REC:g}度：首の角度と視線の向き（体年齢{AGE:g}ヶ月）")
    print("=" * 78)
    print("  首の目標角＝バネが頭を引き寄せる角度。負が後屈（顎を上げる）、正が前屈（顎を引く）")
    print("  視線の向き＝水平から何度上か（正＝上を見ている、負＝下を見ている）")
    outs = []
    for t in TARGETS:
        print(f"  [{t:+.0f}度...]")
        outs.append(probe(t))

    print("\n" + "=" * 78)
    print(f"{'首の目標角':>12}{'視線の向き':>12}{'おもちゃとのずれ':>18}"
          f"{'見える':>10}{'視野内(横,縦)':>18}{'腕の割合':>10}")
    for o in outs:
        pos = (f"({o['cx']:+.2f},{o['cy']:+.2f})" if o["seen"] else "-")
        print(f"{o['target']:>+11.0f}度{o['gaze']:>+11.1f}度{o['off']:>17.1f}度"
              f"{('見える' if o['seen'] else 'ない'):>10}{pos:>18}"
              f"{o['sh_ratio']*100:>9.0f}%")

    ok = [o for o in outs if o["seen"]]
    print("\n" + "=" * 78)
    if ok:
        best = min(ok, key=lambda o: abs(o["cx"]) + abs(o["cy"]))
        print(f" ★おもちゃが見えるのは "
              f"{'、'.join(f'{o["target"]:+.0f}度' for o in ok)}")
        print(f"   最も中心で捉えるのは {best['target']:+.0f}度"
              f"（視野内 {best['cx']:+.2f},{best['cy']:+.2f}）")
    else:
        print(" ⚠️どの角度でもおもちゃが見えない。おもちゃの位置も変える必要がある")
        print(f"   視線は {min(o['gaze'] for o in outs):+.0f}〜"
              f"{max(o['gaze'] for o in outs):+.0f}度 の範囲で振れる")
        print(f"   おもちゃとのずれは最小 {min(o['off'] for o in outs):.1f}度"
              f"（視野の半角は30度）")


if __name__ == "__main__":
    main()
