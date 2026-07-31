"""視線誘導反射が今の体（筋力バグ修正後）でも機能しているかを確認する。

【なぜ】E1 でこの反射に頼るが、実装済みの検証は 2026-07-21（150tickで反応率94%）まで。
その後 MuscleModel化・階層化・筋力バグ修正が入っており、機能保証がない。
「実装済み ≠ 機能している」（項18）。

【何を測るか】
  おもちゃを右／左／上／下に小さく激しく揺らし、
  首と目がその方向に向くか、を「動かす前後の変化」で見る。

  期待：
    右揺れ → head_swivel が右向きに変化、eye_h が右向きに変化
    上揺れ → head_tilt が上向きに変化、eye_v が上向きに変化

  失敗の見え方：
    ・首も目も動かない → 反射が実質OFF
    ・全方向に同じ量動く → 方向の情報が失われている
    ・自分の動きに引き込まれる → 残差法が破れている

【動かし方】ユーザー指示：小さく激しく。
  振幅 3cm・周期 0.4秒（2.5Hz）のサイン波。qpos を直接書き換える（物理を上書き）。
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
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"), _HERE]:
    if p not in sys.path: sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np

HOLD_STILL_SEC = 2.0   # まず動かさずに基準の首・目の位置を測る
SHAKE_SEC = 3.0        # 動かして反応を見る
AMP = 0.03             # 振幅 3cm
FREQ = 2.5             # 2.5Hz（小さく激しく）


def run(direction, action_zero=True):
    """direction: 'right','left','up','down'。おもちゃを1軸方向に揺らす。"""
    from e_toy_env import ToySupineEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from infant_neck import TILT_JOINT

    kw = body_kwargs_from_env(0.0, verbose=False)
    from e_toy_env import infant_vision_params
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, orient=True, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)

    # おもちゃの qpos の位置成分（freejoint なので先頭3つが位置）
    toy_bid = m.body("test_object1").id
    toy_jid = m.body_jntadr[toy_bid]           # 対応する自由関節
    toy_qadr = int(m.jnt_qposadr[toy_jid])     # qpos 内の先頭
    toy_center = d.qpos[toy_qadr:toy_qadr+3].copy()

    # 首と目の関節を特定
    head_swivel = int(m.jnt_qposadr[[j for j in range(m.njnt)
                                      if m.joint(j).name == "robot:head_swivel"][0]])
    head_tilt = int(m.jnt_qposadr[[j for j in range(m.njnt)
                                    if m.joint(j).name == TILT_JOINT][0]])

    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    n_act = env.action_space.shape[0]
    zero = np.zeros(n_act, dtype=np.float32) if action_zero else \
           np.full(n_act, 0.5, dtype=np.float32)

    # 方向ベクトル（オフセット中心。0 を中心の対称振動ではなく、
    # その方向にずらした位置で小さく揺らす）
    OFFSET = 0.08   # 中心位置を 8cm 一方向にオフセット
    JITTER = 0.015  # 振幅 1.5cm で小さく激しく揺らす
    axis = {"right": [0, -1, 0], "left":  [0, +1, 0],
            "up":    [0,  0, 1], "down":  [0,  0, -1]}[direction]
    axis = np.array(axis, dtype=float)

    # 目の関節も測る（視線誘導は目にも加算するので、こちらが主）
    def find(name):
        for j in range(m.njnt):
            if m.joint(j).name == name:
                return int(m.jnt_qposadr[j])
        return None
    eye_h_r = find("robot:right_eye_horizontal")
    eye_v_r = find("robot:right_eye_vertical")
    eye_h_l = find("robot:left_eye_horizontal")
    eye_v_l = find("robot:left_eye_vertical")

    def snapshot():
        s = float(np.degrees(d.qpos[head_swivel]))
        t = float(np.degrees(d.qpos[head_tilt]))
        eh = np.mean([np.degrees(d.qpos[eye_h_r]) if eye_h_r else 0,
                       np.degrees(d.qpos[eye_h_l]) if eye_h_l else 0])
        ev = np.mean([np.degrees(d.qpos[eye_v_r]) if eye_v_r else 0,
                       np.degrees(d.qpos[eye_v_l]) if eye_v_l else 0])
        return s, t, float(eh), float(ev)

    # フェーズ1：静止で基準値
    hist_still = []
    for _ in range(int(HOLD_STILL_SEC / dt)):
        env.step(zero)
        hist_still.append(snapshot())
    base = np.mean(hist_still[-20:], axis=0)

    # フェーズ2：おもちゃを「オフセットした位置で小さく揺らす」
    hist_shake = []
    reflex_dir = []  # 反射の内部状態を直接見る
    n_shake = int(SHAKE_SEC / dt)
    toy_dofadr = int(m.jnt_dofadr[toy_jid])
    reflex = env.unwrapped._orienting
    for step in range(n_shake):
        t = step * dt
        pos_offset = (OFFSET + JITTER * np.sin(2 * np.pi * FREQ * t)) * axis
        d.qpos[toy_qadr:toy_qadr+3] = toy_center + pos_offset
        d.qvel[toy_dofadr:toy_dofadr+6] = 0.0
        env.step(zero)
        hist_shake.append(snapshot())
        reflex_dir.append((getattr(reflex, "h_dir", 0.0),
                            getattr(reflex, "v_dir", 0.0)))
    env.close()
    reflex_dir = np.array(reflex_dir)

    end = np.mean(hist_shake[-50:], axis=0)   # 最後の 0.5s 平均
    # 反射の内部方向の統計
    h_mean = float(np.mean(reflex_dir[:, 0]))
    v_mean = float(np.mean(reflex_dir[:, 1]))
    h_max = float(np.max(np.abs(reflex_dir[:, 0])))
    v_max = float(np.max(np.abs(reflex_dir[:, 1])))
    nonzero_frac = float(np.mean(np.abs(reflex_dir).sum(axis=1) > 1e-6))
    return dict(base=base, end=end,
                d_swivel=end[0] - base[0], d_tilt=end[1] - base[1],
                d_eye_h=end[2] - base[2], d_eye_v=end[3] - base[3],
                h_mean=h_mean, v_mean=v_mean, h_max=h_max, v_max=v_max,
                nonzero=nonzero_frac)


def main():
    print("=== 視線誘導反射の動作確認 ===")
    print(f"  amp={AMP*100:.1f}cm  freq={FREQ}Hz  shake={SHAKE_SEC}s（静止基準{HOLD_STILL_SEC}s後）")
    print(f"  action=zero（脱力）で反射だけを見る")
    print(f"  head_swivel: 右手側が正か負かはMIMo定義依存、方向差を見る\n")

    print(f"{'向き':<10}{'△swivel':>10}{'△tilt':>10}{'△eye_h':>10}{'△eye_v':>10}"
          f"{'h_mean':>10}{'v_mean':>10}{'h_max':>8}{'nz%':>7}")
    print("-" * 82)
    results = {}
    for direction in ["right", "left", "up", "down"]:
        r = run(direction)
        results[direction] = r
        print(f"{direction:<10}{r['d_swivel']:>10.3f}{r['d_tilt']:>10.3f}"
              f"{r['d_eye_h']:>10.3f}{r['d_eye_v']:>10.3f}"
              f"{r['h_mean']:>10.4f}{r['v_mean']:>10.4f}"
              f"{r['h_max']:>8.3f}{r['nonzero']*100:>7.1f}")

    print("\n--- 判定 ---")
    # 目の反応（首の力学ゲート未満の age 0 でも目は動くはず）
    dr_eye = results["right"]["d_eye_h"] - results["left"]["d_eye_h"]
    du_eye = results["up"]["d_eye_v"] - results["down"]["d_eye_v"]
    dr_neck = results["right"]["d_swivel"] - results["left"]["d_swivel"]
    du_neck = results["up"]["d_tilt"] - results["down"]["d_tilt"]
    print(f"  目：右-左 の eye_h 差: {dr_eye:+.3f}度  （符号が反転すれば反射が働いている）")
    print(f"  目：上-下 の eye_v 差: {du_eye:+.3f}度  （縦は文献比 10/30 で弱いはず）")
    print(f"  首：右-左 の swivel 差: {dr_neck:+.3f}度")
    print(f"  首：上-下 の tilt 差  : {du_neck:+.3f}度")
    print()
    if abs(dr_eye) < 0.3 and abs(dr_neck) < 0.3:
        print("  注意 横方向は目も首も反応が弱い（<0.3度）＝反射が実質OFF")
    elif abs(dr_eye) > abs(dr_neck) * 3:
        print("  → 目主導で反射が働いている（首は age 0 で動きにくい設定と整合）")


if __name__ == "__main__":
    main()
