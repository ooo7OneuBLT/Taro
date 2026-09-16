"""保存された設定（Viewerで人が調整した位置）が成立しているかを測る。

【なぜ要るか、2026-07-28】Viewerで人が目視で位置を決めたあと、
「本当に見えているか」「手が届くか」を数値で確かめる。
注意：私（Claude）は探索スクリプトが計算した候補を測って「見えて届く」と報告したが、
  遮蔽を計算に入れておらず、実際には全部体に隠れていた。
  ＝**幾何的な角度だけで『見える』と言ってはいけない**。描き分けで確かめる。

【測ること】
  1. おもちゃが実際に見えるか（描き分け＝segmentation。遮蔽込み）
  2. 視野内のどこにあるか
  3. 肩から腕の長さ以内か（＝手が届くか）
  4. 両目で見るのに要る寄り目の角度

使い方:
    .venv/Scripts/python.exe E/scripts/e_saved_pos_check.py
    E_TOY_POS=x,y,z で位置を直接指定することもできる
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
import mujoco       # noqa: E402

SAVE_PATH = os.path.join(_HERE, os.pardir, "docs", "viewer_saved.json")
AGE = float(os.environ.get("E_AGE", "4.0"))
REC = float(os.environ.get("E_RECLINE", "60"))
HOLD_TILT = os.environ.get("E_HOLD_TILT", "60")
SETTLE = float(os.environ.get("E_SETTLE", "4.0"))


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    os.environ.setdefault("E_TOY_SHAPE", "sphere")
    saved = {}
    try:
        with open(SAVE_PATH, encoding="utf-8") as fp:
            saved = json.load(fp)
    except Exception as e:
        print(f"注意保存が読めない: {e}")

    if os.environ.get("E_TOY_POS"):
        pos = np.array([float(x) for x in os.environ["E_TOY_POS"].split(",")])
        src = "環境変数"
    elif "toy_pos" in saved:
        pos = np.array(saved["toy_pos"], dtype=float)
        src = "保存された設定（Viewerで調整したもの）"
    else:
        print("位置が分からない")
        return
    radius = float(saved.get("toy_half_size", 0.0056))
    os.environ.setdefault("E_TOY_RADIUS", str(radius))

    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from e_head_hold import CaregiverHands
    import e_visibility as VIS

    kw = body_kwargs_from_env(AGE, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    # 条件は保存された設定に合わせる（Viewerで人が見ていた状態を再現するため）。
    #   注意：これを揃えずに測ると、まったく別の姿勢の太郎を測ることになる
    #     （2026-07-28 に実際に起きた：保存は首30度なのに60度で測っていた）。
    from infant_body import apply_neck_tone
    _neck_spring = saved.get("neck_target")
    _use_hold = os.environ.get("E_HEAD_HOLD", "0") == "1"
    with contextlib.redirect_stdout(buf):
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=AGE),
                           age=AGE, toy=True, vor=True, orient=False,
                           recline_deg=REC, **kw)
        env.reset(seed=0)
        if _neck_spring is not None:
            apply_neck_tone(env.unwrapped.model, AGE, target=float(_neck_spring),
                            stiffness=saved.get("neck_k"),
                            data=env.unwrapped.data, verbose=False)
        if _use_hold:
            hands = CaregiverHands(env.unwrapped.model, env.unwrapped.data)
            hands.hold(target=({"head_tilt": float(HOLD_TILT)} if HOLD_TILT else None))
    u = env.unwrapped
    m, d = u.model, u.data
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)

    bid = int(m.body("test_object1").id)
    jid = int(m.body_jntadr[bid])
    qadr = int(m.jnt_qposadr[jid])
    dadr = int(m.jnt_dofadr[jid])
    for _ in range(int(SETTLE / dt)):
        d.qpos[qadr:qadr + 3] = pos
        d.qvel[dadr:dadr + 6] = 0.0
        env.step(a)
    d.qpos[qadr:qadr + 3] = pos
    d.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(m, d)

    def bpos(n):
        return np.array(d.xpos[int(m.body(n).id)], dtype=float)

    eyes, fwds = [], []
    for nm in ("eye_left", "eye_right"):
        cid = int(m.camera(nm).id)
        eyes.append(np.array(d.cam_xpos[cid], dtype=float))
        fwds.append(-np.array(d.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2])
    eye = np.mean(eyes, axis=0)
    ipd = float(np.linalg.norm(eyes[0] - eyes[1]))
    fwd = np.mean(fwds, axis=0); fwd /= np.linalg.norm(fwd)

    print("=" * 74)
    print(f" 保存された位置は成立しているか（{src}）")
    print("=" * 74)
    print(f"  おもちゃ  ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  半径{radius*100:.1f}cm")
    print(f"  条件      リクライニング{REC:g}度  体年齢{AGE:g}ヶ月")
    print(f"            首のバネ 目標{_neck_spring}度 / 強さ{saved.get('neck_k')}"
          f"   実験者の手 {'あり(' + str(HOLD_TILT) + '度)' if _use_hold else 'なし'}")

    # 1. 見えるか（左右それぞれ・遮蔽込み）
    print("\n" + "=" * 74)
    print(" 1. 実際に見えるか（体に隠れていないか。描き分けで判定）")
    print("=" * 74)
    for cam in ("eye_left", "eye_right"):
        sv = VIS.visible_by_segment(m, d, bid, cam, size=96)
        lbl = "左目" if cam == "eye_left" else "右目"
        if sv["seen"]:
            print(f"  {lbl}  見える   視野内の位置 ({sv['cx']:+.2f}, {sv['cy']:+.2f})"
                  f"   （0が中心、±1が端）")
        else:
            print(f"  {lbl}  見えない（体に隠れているか視野の外）")

    # 2. 幾何的な位置関係
    v = pos - eye
    r = float(np.linalg.norm(v))
    ang = float(np.degrees(np.arccos(np.clip(np.dot(v / r, fwd), -1, 1))))
    sh_r, sh_l = bpos("right_upper_arm"), bpos("left_upper_arm")
    el, hd = bpos("right_lower_arm"), bpos("right_hand")
    arm = float(np.linalg.norm(el - sh_r)) + float(np.linalg.norm(hd - el))
    reach = min(float(np.linalg.norm(pos - sh_r)),
                float(np.linalg.norm(pos - sh_l))) / arm
    verg = 2.0 * np.degrees(np.arctan2(ipd / 2.0, max(r, 1e-4)))

    print("\n" + "=" * 74)
    print(" 2. 位置関係")
    print("=" * 74)
    print(f"  目からの距離     {r*100:6.1f} cm")
    print(f"  視線からのずれ   {ang:6.1f} 度   （視野の半角は30度）")
    print(f"  腕の長さ         {arm*100:6.1f} cm")
    print(f"  肩からの距離     {min(float(np.linalg.norm(pos - sh_r)), float(np.linalg.norm(pos - sh_l)))*100:6.1f} cm"
          f"  ＝ 腕の {reach*100:.0f}%   {'届く' if reach <= 1.0 else '届かない'}")
    print(f"  要る寄り目       {verg:6.1f} 度   （太郎は寄り目ができない）")
    print(f"  見かけの大きさ   {2*np.degrees(np.arctan2(radius, r)):6.1f} 度")

    # 3. 視野の内訳
    print("\n" + "=" * 74)
    print(" 3. 視野の内訳（自分の体がどれだけ占めるか）")
    print("=" * 74)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 96)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 96)
    mujoco.mj_forward(m, d)     # 注意レンダリング前に状態を確定させる
    ren = mujoco.Renderer(m, 96, 96)
    ren.enable_segmentation_rendering()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = int(m.camera("eye_left").id)
    ren.update_scene(d, cam)
    seg = ren.render()[:, :, 0].copy()
    ren.close()
    agg = {}
    print(f"   （描き分けの値の範囲 {int(seg.min())}〜{int(seg.max())}"
          f"、種類 {len(np.unique(seg))}）")
    for gid in np.unique(seg):
        if gid < 0 or int(gid) >= m.ngeom:
            continue
        bn = (m.body(int(m.geom_bodyid[int(gid)])).name or "").lower()
        if "object" in bn:
            k = "おもちゃ"
        elif any(x in bn for x in ("foot", "toe", "leg")):
            k = "足"
        elif any(x in bn for x in ("hand", "arm", "finger", "thumb")):
            k = "手・腕"
        elif any(x in bn for x in ("head", "eye")):
            k = "頭"
        elif any(x in bn for x in ("body", "hip", "chest", "cb", "ub", "lb")):
            k = "胴"
        else:
            k = "環境"
        agg[k] = agg.get(k, 0) + int((seg == gid).sum())
    tot = seg.size
    for k in sorted(agg, key=lambda x: -agg[x]):
        print(f"  {k:<8}{agg[k]/tot*100:6.2f}%")
    own = sum(v for k, v in agg.items() if k in ("足", "手・腕", "頭", "胴"))
    print(f"  ---- 自分の体 {own/tot*100:.1f}% ／ おもちゃ {agg.get('おもちゃ',0)/tot*100:.2f}% ----")
    env.close()


if __name__ == "__main__":
    main()
