"""★リクライニングで「見えて届く」おもちゃの位置を探す。

【なぜ要るか、2026-07-28】首の角度を振った測定（`e_recline_gaze_check.py`）で、
    首を後屈（-45度）→ 視線が上 → おもちゃも上 → 肩から126%＝届かない
    首を前屈（+45度）→ 視線が下 → 肩から90%＝**届く**が、視線から57度ずれて見えない
となった。原因は**おもちゃが毎回「視線の正面」に置き直される**こと。
首を動かすとおもちゃも一緒に動くので、いくら首を調整しても両立しない。

【やること】おもちゃを**体に対する固定位置**（`toy_offset`）に置き、
位置を格子状に振って
    ・目から見て視野（半角30度）に入るか
    ・肩から腕の長さ以内か
の両方を満たす場所を探す。

⚠️人間の実験は「その子の肩から手首までの長さ」に相当する距離に置く
  （Carvalho et al. 2007。固定のcm値ではない）。ここでもその比で評価する。

使い方:
    E_RECLINE=60 E_NECK_TARGET=45 .venv/Scripts/python.exe E/scripts/e_recline_place_search.py
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
import mujoco       # noqa: E402

AGE = float(os.environ.get("E_AGE", "4.0"))
REC = float(os.environ.get("E_RECLINE", "60"))
NECK = float(os.environ.get("E_NECK_TARGET", "45"))
HALF_FOV = 30.0     # 視野の半角[度]
SETTLE = float(os.environ.get("E_SETTLE", "4.0"))


def check_visible(env, pos):
    """★おもちゃを実際にその位置へ置いて、体に隠れずに見えるかを確かめる。

    ⚠️【2026-07-28 に判明した穴】この探索は当初「視線からの角度」と
      「肩からの距離」だけで判定していた。しかしリクライニング姿勢では
      **視野の半分近くを自分の胴体が占める**（実測52.8%）ので、
      角度が合っていても体に隠れて見えないことがある。
      → 描き分け（segmentation）で実際に見えるかを確かめる。
    """
    import e_visibility as VIS
    u = env.unwrapped
    m, d = u.model, u.data
    bid = int(m.body("test_object1").id)
    jid = int(m.body_jntadr[bid])
    qadr = int(m.jnt_qposadr[jid])
    dadr = int(m.jnt_dofadr[jid])
    d.qpos[qadr:qadr + 3] = pos
    d.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(m, d)
    return VIS.visible_by_segment(m, d, bid, "eye_left", size=64)


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from infant_body import apply_neck_tone

    kw = body_kwargs_from_env(AGE, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=AGE),
                           age=AGE, toy=True, vor=True, orient=False,
                           recline_deg=REC, **kw)
        env.reset(seed=0)
        apply_neck_tone(env.unwrapped.model, AGE, target=NECK,
                        data=env.unwrapped.data, verbose=False)
        # ★実験者が頭を抑える（3軸とも固定）。
        #   リクライニングでは首の**左右回転**にも重力が効くが、首のバネは
        #   前後（head_tilt）にしか入っていない（仰向け前提の設計）。
        #   そのため頭が横を向いて倒れ、視線が真横になっていた（2026-07-28 に発覚）。
        #   人間の乳児実験でも実験者が頭を支えているので、条件としても正しい。
        if os.environ.get("E_HEAD_HOLD", "1") == "1":
            from e_head_hold import CaregiverHands
            _hands = CaregiverHands(env.unwrapped.model, env.unwrapped.data)
            # ⚠️目標角を渡さないと「今の角度」で固定されるので、首の角度を変えても
            #   視線が動かない（2026-07-28 に踏んだ）。前後だけ指定し、
            #   左右のひねり・傾きは今の姿勢（ほぼ0度）で保つ。
            _hands.hold(target={"head_tilt": NECK})
    u = env.unwrapped
    m, d = u.model, u.data
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for _ in range(int(SETTLE / dt)):
        env.step(a)

    # 基準となる体の各点
    def bpos(n):
        return np.array(d.xpos[int(m.body(n).id)], dtype=float)

    eyes, fwds = [], []
    for nm in ("eye_left", "eye_right"):
        cid = int(m.camera(nm).id)
        eyes.append(np.array(d.cam_xpos[cid], dtype=float))
        fwds.append(-np.array(d.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2])
    eye = np.mean(eyes, axis=0)
    fwd = np.mean(fwds, axis=0); fwd /= np.linalg.norm(fwd)
    sh_r = bpos("right_upper_arm")
    sh_l = bpos("left_upper_arm")
    el = bpos("right_lower_arm")
    hd = bpos("right_hand")
    arm = float(np.linalg.norm(el - sh_r)) + float(np.linalg.norm(hd - el))
    chest = bpos("upper_body")

    print("=" * 78)
    print(f" 「見えて届く」おもちゃの位置を探す"
          f"（リクライニング{REC:g}度・首{NECK:+.0f}度・{AGE:g}ヶ月）")
    print("=" * 78)
    print(f"  目      ({eye[0]:+.3f}, {eye[1]:+.3f}, {eye[2]:+.3f})")
    print(f"  視線    ({fwd[0]:+.3f}, {fwd[1]:+.3f}, {fwd[2]:+.3f})"
          f"  水平から {np.degrees(np.arcsin(fwd[2])):+.1f}度")
    print(f"  右肩    ({sh_r[0]:+.3f}, {sh_r[1]:+.3f}, {sh_r[2]:+.3f})")
    print(f"  腕の長さ {arm*100:.1f}cm   視野の半角 {HALF_FOV:.0f}度")

    # 格子で探す（体のまわり）
    best = []
    for dx in np.arange(-0.05, 0.26, 0.01):
        for dz in np.arange(-0.10, 0.26, 0.01):
            for dy in (0.0,):
                p = chest + np.array([dx, dy, dz])
                v = p - eye
                r = float(np.linalg.norm(v))
                if r < 0.05:
                    continue
                ang = float(np.degrees(np.arccos(np.clip(np.dot(v / r, fwd), -1, 1))))
                dr = float(np.linalg.norm(p - sh_r)) / arm
                dl = float(np.linalg.norm(p - sh_l)) / arm
                reach = min(dr, dl)
                if ang <= HALF_FOV and reach <= 1.0:
                    best.append(dict(p=p, ang=ang, reach=reach, dist=r,
                                     dx=dx, dz=dz))

    print("\n" + "=" * 78)
    if not best:
        print(" ⚠️「見えて届く」位置が見つからない")
        print("    首の角度かリクライニング角を変える必要がある")
        env.close()
        return

    print(f" ★見えて届く位置が {len(best)} 通り見つかった")
    print("=" * 78)
    # 「視線の中心に近く」かつ「腕の70〜95%」を良い位置とする
    #   ⚠️近すぎ（<50%）は伸ばす動作にならない。人間の実験は腕をほぼ伸ばした距離
    good = [b for b in best if 0.70 <= b["reach"] <= 0.95]
    pool = good if good else best
    pool.sort(key=lambda b: b["ang"])
    print(f"{'胸からのずれ(x,z)':>22}{'目からの距離':>14}"
          f"{'視線からの角度':>16}{'腕に対する割合':>16}")
    for b in pool[:8]:
        print(f"   ({b['dx']:+.2f}, {b['dz']:+.2f})      {b['dist']*100:>9.1f}cm"
              f"{b['ang']:>15.1f}度{b['reach']*100:>15.0f}%")

    # ★上位の候補を実際に置いて、体に隠れずに見えるかを確かめる（2026-07-28 追加）
    print("\n" + "=" * 78)
    print(" ★実際に置いて確かめる（体に隠れていないか）")
    print("=" * 78)
    print(f"{'胸からのずれ':>16}{'視線から':>10}{'腕の割合':>10}"
          f"{'実際に見えるか':>16}{'視野内(横,縦)':>18}")
    verified = []
    for b in pool[:12]:
        sv = check_visible(env, b["p"])
        b["seen"] = bool(sv["seen"])
        b["cx"] = float(sv.get("cx", float("nan")))
        b["cy"] = float(sv.get("cy", float("nan")))
        if b["seen"]:
            verified.append(b)
        pos_s = (f"({b['cx']:+.2f},{b['cy']:+.2f})" if b["seen"] else "-")
        print(f"   ({b['dx']:+.2f},{b['dz']:+.2f}) {b['ang']:>9.1f}度"
              f"{b['reach']*100:>9.0f}%"
              f"{('見える' if b['seen'] else '★隠れている'):>16}{pos_s:>18}")

    if not verified:
        print("\n ⚠️★上位の候補はすべて体に隠れて見えない")
        print("   ＝角度と距離だけでは足りない。首の角度を変えるか、")
        print("     おもちゃを体から離れた方向へ置く必要がある")
        env.close()
        return
    pool = verified

    top = pool[0]
    print("\n" + "=" * 78)
    print(" ★おすすめの位置（視線の中心に近く、実際に見える）")
    print("=" * 78)
    print(f"   world座標   ({top['p'][0]:+.3f}, {top['p'][1]:+.3f}, {top['p'][2]:+.3f})")
    print(f"   胸からのずれ ({top['dx']:+.3f}, 0.000, {top['dz']:+.3f})")
    print(f"   目から {top['dist']*100:.1f}cm   視線から {top['ang']:.1f}度"
          f"   腕の {top['reach']*100:.0f}%")
    print(f"\n   Viewerで使うなら「おもちゃ」区画の X/Y/Z に")
    print(f"     X={top['p'][0]:.3f}  Y={top['p'][1]:.3f}  Z={top['p'][2]:.3f}")
    env.close()


if __name__ == "__main__":
    main()
