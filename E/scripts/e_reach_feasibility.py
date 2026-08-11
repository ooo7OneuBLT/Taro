"""リーチング環境で「そもそも腕が上がるか・おもちゃに届くか」を測る。

【なぜ要るか、2026-07-30】ユーザーの問い：
> Goal Babblingの仕組みってもう完璧になってたんだっけ？
> 前流用出来ているけど完全じゃないみたいな話じゃなかった？

Goal Babbling 以前に、**物理的に届くか**が未確認だった。
昨日の `e_reach_babble_check.py` の冒頭には壁の候補が3つ挙がっている：
  1. 自発運動の振幅が小さい（Viewer 実効値 0.174）
  2. 四肢の筋力が弱すぎる（4ヶ月で ×0.047＝1/21 と記録されていた）
  3. おもちゃが遠い（手から22.7cm・可動域の端でやっと届く）
注意：2 は 2026-07-29 の**基準の汚染バグ**（落とし穴 項76）の影響を受けている
  可能性があるので、直した後の値を測り直す。

【測ること／判定を先に決めておく】
  ① 筋力 ÷ 重力モーメント（＝relative strength）
       > 1  重力に逆らって動かせる
       ≦ 1  その関節は自重で動かせない＝リーチング以前の問題
       注意：重力モーメントは「軸に垂直な距離×重さ」の**姿勢によらない上界**で計算。
         ＝ここで > 1 なら、どんな姿勢でも動かせる（控えめな見積もり）
  ② 肩からおもちゃまでの距離 と 腕の長さ
       距離 ≦ 腕の長さ    幾何学的には届く
       距離 > 腕の長さ    絶対に届かない
       注意：幾何学的に届いても**可動域**で届かないことがあるので、必ず絵を見る
         （落とし穴 項70「位置・姿勢は数値より先に絵を撮る」）

使い方:
    .venv/Scripts/python.exe E/scripts/e_reach_feasibility.py
    E_SCENE=名前 で環境を変えられる
出力:
    E/logs/reach/届くかの確認_<シーン名>.png
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

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in ("E/scripts", "D/scripts", "taro_core/src/body",
           "taro_core/src/wrapper", "taro_core/src/senses", "run/scene_tools"):
    sys.path.insert(0, os.path.join(_ROOT, *_p.split("/")))

import numpy as np      # noqa: E402
import mujoco           # noqa: E402
import mimoEnv          # noqa: E402,F401
import e_scene          # noqa: E402
from infant_limbs import actuator_ratios   # noqa: E402
from mimoActuation.actuation import SpringDamperModel   # noqa: E402

SCENE = os.environ.get("E_SCENE", "リーチング_リクライニング60度")
OUT = os.path.join(_ROOT, "E", "logs", "reach")
# 腕の**アクチュエータ**（左右）。この順で表示する。
# 注意：関節名とアクチュエータ名は違う（2026-07-30 にここで踏んだ）：
#     関節 robot:right_shoulder_ad_ab   ← アクチュエータ act:right_shoulder_abduction
#     関節 robot:right_shoulder_rotation ← アクチュエータ act:right_shoulder_internal
#   関節名で照合すると全部 nan になり、「動かせない」と読み違える。
ARM_KEYS = [("shoulder_horizontal", "肩：水平の前後"),
            ("shoulder_abduction", "肩：開き（横へ上げる）"),
            ("shoulder_internal", "肩：ひねり（内旋）"),
            ("elbow", "ひじ"),
            ("wrist_flexion", "手首：曲げ伸ばし"),
            ("wrist_rotation", "手首：ひねり"),
            ("wrist_ulnar", "手首：左右")]


def _body_pos(m, d, name):
    return np.array(d.xpos[int(m.body(name).id)], dtype=float)


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = e_scene.load(SCENE)
    sc["fingerprint"] = None
    env, _ = e_scene.build(sc, orient=False, vor=True, seed=0, verbose=False,
                           actuation_model=SpringDamperModel)
    u = env.unwrapped
    m, d = u.model, u.data
    mujoco.mj_forward(m, d)

    print("=" * 78)
    print(f" リーチングの成立条件（シーン: {SCENE} / 月齢 {sc['body']['age_months']}ヶ月）")
    print("=" * 78)

    # ---- ① 筋力 ÷ 重力モーメント -------------------------------------------
    ratios = actuator_ratios(m, d, SpringDamperModel)
    print("\n【①】筋力 ÷ 重力モーメント（>1 なら重力に逆らって動かせる）")
    print(f"  {'関節':<20}{'右':>12}{'左':>12}   判定")
    ng = 0
    for base, jp in ARM_KEYS:
        vals = {}
        for side in ("right", "left"):
            hit = [v[2] for k, v in ratios.items()
                   if k.split(":")[-1] == f"{side}_{base}"]
            vals[side] = hit[0] if hit else float("nan")
        # 注意：nan は「動かせない」ではなく「測れなかった」。名前の照合ミスを
        #   「動かせない」と読み違えると、原因の切り分けを丸ごと間違える。
        if any(np.isnan(v) for v in vals.values()):
            judge = "測れなかった（名前を確認）"
            ng += 1
        elif all(v > 1.0 for v in vals.values()):
            judge = "OK"
        else:
            judge = "動かせない"
            ng += 1
        print(f"  {jp:<24}{vals['right']:>10.2f}{vals['left']:>10.2f}   {judge}")
    print(f"  → OK でない関節: {ng} 個")

    # ---- ② 距離と腕の長さ ---------------------------------------------------
    print("\n【②】おもちゃまでの距離 と 腕の長さ")
    toy_bid = next((b for b in range(m.nbody)
                    if (m.body(b).name or "").startswith("test_object1")), None)
    if toy_bid is None:
        print("  注意おもちゃ(test_object1)が見つからない＝この環境にはおもちゃが無い")
        toy = None
    else:
        toy = np.array(d.xpos[toy_bid], dtype=float)
        print(f"  おもちゃの位置: [{toy[0]:+.3f}, {toy[1]:+.3f}, {toy[2]:+.3f}] m")
    for side, jp in (("right", "右腕"), ("left", "左腕")):
        sh = _body_pos(m, d, f"{side}_upper_arm")
        el = _body_pos(m, d, f"{side}_lower_arm")
        ha = _body_pos(m, d, f"{side}_hand")
        arm = np.linalg.norm(el - sh) + np.linalg.norm(ha - el)
        line = f"  {jp}  腕の長さ(肩→ひじ→手) {arm*100:6.1f}cm"
        if toy is not None:
            dist = float(np.linalg.norm(toy - sh))
            hand_d = float(np.linalg.norm(toy - ha))
            mark = "届かない" if dist > arm else f"届く（余裕 {(arm-dist)*100:.1f}cm）"
            line += (f" / 肩→おもちゃ {dist*100:6.1f}cm  {mark}"
                     f" / いまの手→おもちゃ {hand_d*100:5.1f}cm")
        print(line)
    if toy is not None:
        print("  注意幾何学的に届いても**可動域**で届かないことがある。必ず絵を見る（項70）")

    # ---- 絵 -----------------------------------------------------------------
    from PIL import Image
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 640)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 480)
    ren = mujoco.Renderer(m, 480, 640)
    shots = []
    for tag, elev, azim in (("真横", -6.0, 90.0), ("真上", -80.0, 90.0),
                            ("斜め前", -18.0, 40.0)):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = _body_pos(m, d, "upper_body")
        cam.distance = 1.0
        cam.elevation = elev
        cam.azimuth = azim
        ren.update_scene(d, cam)
        shots.append((tag, ren.render().copy()))
    ren.close()
    W = 640 * len(shots)
    canvas = Image.new("RGB", (W, 480 + 22), (250, 250, 252))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(canvas)
    for i, (tag, im) in enumerate(shots):
        canvas.paste(Image.fromarray(im), (i * 640, 22))
        dr.text((i * 640 + 6, 5), f"{tag}（おもちゃと手の位置を見る）", fill=(20, 20, 30))
    path = os.path.join(OUT, f"届くかの確認_{SCENE}.png")
    canvas.save(path)
    print(f"\n  → 絵を保存: {os.path.relpath(path, _ROOT)}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
