"""サッケード1発で視線が何度動くかを測る（設計「目標の30%」との突き合わせ）。

【なぜ】ユーザーの目視（2026-07-26）：
  「視界の端っこにおもちゃを置くと、視界がちょっとぴくぴく動こうとしているが、
    対象物を真ん中に持ってくるとかはない」

設計図（`E/docs/視線誘導反射_設計図.md` ステップ4）は
  「重心方向へ小さくジャンプ（例：目標の30%）／届かなければ200ms後にまた撃つ＝階段状」
と書いているが、実装に `SACCADE_FRAC` が**存在しない**。
実装は方向 h,v（-1〜1）に固定ゲイン（EYE_GAIN=0.15）を掛けて 0.05 秒出すだけで、
**どれだけ動かすかという距離の概念が無い**。

【測ること】おもちゃを視線から離れた位置に置いて激しく揺らし、
  ・サッケードが何発撃たれるか
  ・1発ごとに視線とおもちゃのなす角が何度縮むか
  ・眼球の角度が1発で何度動くか
を追う。設計通りなら、ズレ角が 30% ずつ階段状に縮んでいくはず。

【自発運動は流さない】人間でも、親がおもちゃを見せるのは
バタバタしている最中ではない（ユーザーの指摘 2026-07-26）。action=0 で測る。
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
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"),
          _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np
import mujoco

SEC = 6.0
SHAKE_HZ = 2.5          # 「小さく激しく」＝2.5Hz
SHAKE_AMP = 0.015       # 1.5cm
HALF_FOV = 30.0         # 視野の半角（fovy=60）

# ★おもちゃの基準位置は、ユーザーが Viewer で調整して保存したもの（＝視界に入る位置）を使う。
#   最初の版は環境のデフォルト位置に 8cm ずらして測ったが、ズレ角が 54〜109度になり
#   **視野（半角30度）の外**だった。見えていないものに反応できるはずがない。
#   → 落とし穴チェックリスト 項1（交絡）・項6（測定器の健康診断）
SAVED_POSE = os.path.join(_HERE, os.pardir, "docs", "pose_editor_saved.json")

# 視界に入れたうえで、少しだけずらす。8cm は大きすぎた。
OFFSETS = {
    "そのまま  ": (0.0, 0.0, 0.0),
    "右へ 2cm  ": (0.0, +0.02, 0.0),
    "左へ 2cm  ": (0.0, -0.02, 0.0),
    "上へ 2cm  ": (0.0, 0.0, +0.02),
    "下へ 2cm  ": (0.0, 0.0, -0.02),
    "右へ 4cm  ": (0.0, +0.04, 0.0),
    "左へ 4cm  ": (0.0, -0.04, 0.0),
}


def load_saved_toy_pos():
    """Viewer で調整・保存したおもちゃの位置を読む。無ければ None。"""
    import json
    p = os.path.abspath(SAVED_POSE)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fp:
        return np.array(json.load(fp)["toy_pos"], dtype=float)


def gaze_angle(m, d, cam_id, toy_bid):
    cpos = d.cam_xpos[cam_id]
    fwd = -d.cam_xmat[cam_id].reshape(3, 3)[:, 2]
    v = d.xpos[toy_bid] - cpos
    n = np.linalg.norm(v)
    if n < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(fwd, v / n), -1, 1))))


def run_case(env, label, offset, dt, n_act, eye_qadr, toy_base=None):
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    rf = env.unwrapped._orienting
    if rf is not None:
        rf.reset()

    cam_id = next(c for c in range(m.ncam) if "eye_left" in (m.camera(c).name or ""))
    toy_bid = int(m.body("test_object1").id)
    toy_jid = next(j for j in range(m.njnt)
                   if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                   and m.body(m.jnt_bodyid[j]).name == "test_object1")
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])
    ref = d.qpos[toy_qadr:toy_qadr + 3].copy() if toy_base is None else np.array(toy_base)
    base = ref + np.array(offset)

    a = np.zeros(n_act, dtype=np.float32)
    n = int(SEC / dt)
    t = 0.0
    prev_sacc = 0
    rows = []          # (t, ズレ角, 眼球h, 眼球v, サッケード発火)
    for step in range(n):
        # おもちゃを固定位置で小さく激しく揺らす（環境側から動かす）
        off = SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t)
        d.qpos[toy_qadr:toy_qadr + 3] = base + np.array([0.0, off, 0.0])
        d.qvel[toy_dof:toy_dof + 6] = 0.0
        env.step(a)
        t += dt
        cur = rf.n_saccades if rf is not None else 0
        fired = cur > prev_sacc
        prev_sacc = cur
        rows.append((t, gaze_angle(m, d, cam_id, toy_bid),
                     float(np.degrees(d.qpos[eye_qadr["h"]])),
                     float(np.degrees(d.qpos[eye_qadr["v"]])),
                     fired))
    return rows, (rf.n_saccades if rf is not None else 0)


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    os.environ.setdefault("E_ORIENT_V", "2")
    kw = body_kwargs_from_env(0.0, verbose=False)
    kw["flexion"] = True
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, vor=True, orient=True, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    n_act = env.action_space.shape[0]
    eye_qadr = {}
    for j in range(m.njnt):
        nm = m.joint(j).name
        if nm == "robot:left_eye_horizontal":
            eye_qadr["h"] = int(m.jnt_qposadr[j])
        elif nm == "robot:left_eye_vertical":
            eye_qadr["v"] = int(m.jnt_qposadr[j])

    toy_base = load_saved_toy_pos()
    print("=== サッケード1発で視線が何度動くか ===")
    print(f"  {SEC:.0f}秒・action=0（自発運動なし）・おもちゃは {SHAKE_HZ}Hz で "
          f"{SHAKE_AMP*100:.1f}cm 揺らす")
    print(f"  設計（設計図ステップ4）：1発で目標の30%だけ寄る → 階段状に中央へ")
    print(f"  視野の半角 {HALF_FOV:.0f}度")
    if toy_base is not None:
        print(f"  おもちゃの基準位置：保存された設定を使う {toy_base}")
    else:
        print("  ★保存された設定が見つからない。環境のデフォルト位置を使う"
              "（視界の外にある可能性が高い）")
    print()

    for label, off in OFFSETS.items():
        rows, n_sacc = run_case(env, label, off, dt, n_act, eye_qadr, toy_base)
        arr = np.array([(r[0], r[1], r[2], r[3]) for r in rows])
        fires = [i for i, r in enumerate(rows) if r[4]]

        seen = float((arr[:, 1] < HALF_FOV).mean()) * 100
        print(f"--- {label} ---")
        print(f"  ズレ角 : 開始 {arr[0,1]:6.1f}度 → 終了 {arr[-1,1]:6.1f}度  "
              f"（最小 {np.nanmin(arr[:,1]):.1f}度）")
        print(f"  ★視界に入っていた時間 {seen:5.1f}%"
              + ("" if seen > 5 else "  ← 見えていない。この行の結果は無効"))
        print(f"  サッケード発火 {n_sacc} 発")
        if not fires:
            print("  ★1発も撃っていない（strength が閾値 0.02 に届いていない）\n")
            continue

        # 1発ごとに「撃つ直前」と「0.15秒後」を比べる
        win = int(0.15 / dt)
        d_gaze, d_eye_h, d_eye_v, d_expect = [], [], [], []
        for i in fires:
            j = min(i + win, len(rows) - 1)
            d_gaze.append(arr[i, 1] - arr[j, 1])          # 縮んだ角度（正なら近づいた）
            d_eye_h.append(arr[j, 2] - arr[i, 2])
            d_eye_v.append(arr[j, 3] - arr[i, 3])
            d_expect.append(0.30 * arr[i, 1])             # 設計値＝目標の30%
        print(f"  1発あたり ズレ角の縮み  平均 {np.mean(d_gaze):+7.3f}度  "
              f"（設計の期待値 {np.mean(d_expect):.2f}度＝目標の30%）")
        print(f"  1発あたり 眼球の動き    横 {np.mean(np.abs(d_eye_h)):6.3f}度  "
              f"縦 {np.mean(np.abs(d_eye_v)):6.3f}度")
        ratio = np.mean(d_gaze) / max(np.mean(d_expect), 1e-9)
        print(f"  ★設計に対する達成率     {ratio*100:6.1f}%")
        # 階段状になっているか＝毎発ちゃんと縮んでいるか
        good = sum(1 for x in d_gaze if x > 0.1)
        print(f"  0.1度以上縮んだ発 : {good}/{len(d_gaze)}\n")

    env.close()
    print("=== 読み方 ===")
    print("  達成率が数%以下 → 1発の大きさが未実装（SACCADE_FRAC が無い）ことが原因")
    print("  発火数が0        → 動き検出の閾値が高すぎる（別の原因）")
    print("  縮む発が半分以下 → 方向が正しく出ていない（左右の弁別の問題）")


if __name__ == "__main__":
    main()
