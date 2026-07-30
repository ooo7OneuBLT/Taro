"""空中スポーンの確認 — 太郎は「落ちて」から学習を始めていないか。

【なぜ測るか】2026-07-25、ユーザーがViewerで「Resetして最初に空中にいる」と指摘。
実際の新生児に「毎回20cm落下してから動き始める」という経験は無いので、
毎エピソードの冒頭に落下の衝撃が入っているなら**人間模倣からの逸脱**になる。

【測るもの】3つの時点で、太郎の各部位の高さと接地状態を測る。
  ①構築直後（settle_steps=0＝落ち着かせる前の生の状態）
  ②settle後（＝init_position、これが毎エピソードの出発点）
  ③reset後 と ④reset後300step（＝学習が実際に見る状態）

使い方:
    python E/scripts/e_spawn_check.py          # 仰向け環境（学習で使う方）
    python E/scripts/e_spawn_check.py toy      # おもちゃ環境
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
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, os.pardir, os.pardir)
for p in [os.path.join(_ROOT, "D", "scripts"),
          os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core", "src", "body"),
          _HERE]:
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco  # noqa: E402

# 高さを見る代表部位（頭・腰・手・足）
PARTS = ["head", "hip", "left_hand", "right_hand", "left_foot", "right_foot"]


def _lowest_z(model, data):
    """全geomの下端のうち一番低い値[m]。0なら接地、正なら浮いている。"""
    lo = 1e9
    for gi in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if name is None or name.startswith("fence_post") or name in ("floor", "ground"):
            continue
        bid = model.geom_bodyid[gi]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ("world", "test_object1", "test_object2"):
            continue          # おもちゃは吊り下げなので対象外（浮くのが仕様）
        z = float(data.geom_xpos[gi][2]) - float(np.max(model.geom_size[gi]))
        lo = min(lo, z)
    return lo


def _n_floor_contacts(model, data):
    """床との接触点の数。"""
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        for g in (c.geom1, c.geom2):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if nm in ("floor", "ground"):
                n += 1
    return n


def snapshot(model, data, label):
    zs = {}
    for p in PARTS:
        try:
            zs[p] = float(data.body(p).xpos[2])
        except Exception:
            pass
    lo = _lowest_z(model, data)
    nc = _n_floor_contacts(model, data)
    print(f"  [{label:<22}] 最下端 {lo*100:+6.2f}cm  床接触 {nc:2d}点  "
          + " ".join(f"{k}={v*100:5.1f}" for k, v in zs.items()))
    return lo, nc, zs


def build(toy=False, settle=100):
    if toy:
        from e_toy_env import ToySupineEnv, infant_vision_params  # noqa
        return ToySupineEnv(age=0, settle_steps=settle)
    from d_supine_env import SupineMimoEnv
    return SupineMimoEnv(age=0, settle_steps=settle)


def main():
    toy = "toy" in sys.argv
    kind = "ToySupineEnv（おもちゃ）" if toy else "SupineMimoEnv（学習で使う方）"
    print(f"=== 空中スポーンの確認: {kind} ===\n")

    # ①構築直後 = 落ち着かせる前。settle_steps=0 で作れば生の状態が見える
    print("① 構築直後（落ち着かせる前）")
    env0 = build(toy, settle=0)
    mujoco.mj_forward(env0.model, env0.data)
    lo0, _, _ = snapshot(env0.model, env0.data, "構築直後")

    # ①' そこから落下する様子（何step目に着地するか・どれだけ落ちるか）
    z_head0 = float(env0.data.body("head").xpos[2])
    for n in (10, 30, 60, 100):
        while env0.data.time < n * env0.model.opt.timestep * 1:
            mujoco.mj_step(env0.model, env0.data)
        snapshot(env0.model, env0.data, f"  →{n}step後")
    z_head1 = float(env0.data.body("head").xpos[2])
    print(f"    ＝構築直後から100stepで頭が {(z_head0-z_head1)*100:+.2f}cm 落ちた\n")

    # ②〜④ 実運用の設定（settle_steps=100）
    print("②〜④ 実運用の設定（settle_steps=100）")
    env = build(toy, settle=100)
    mujoco.mj_forward(env.model, env.data)
    snapshot(env.model, env.data, "settle後(init_position)")

    env.reset()
    mujoco.mj_forward(env.model, env.data)
    lo_r, nc_r, _ = snapshot(env.model, env.data, "reset直後")

    a = np.zeros(env.action_space.shape)
    for _ in range(300):
        env.step(a)
    mujoco.mj_forward(env.model, env.data)
    lo_3, _, _ = snapshot(env.model, env.data, "reset後300step放置")

    print()
    print("=== 判定 ===")
    # ⚠️絵文字は使わない（Windowsのcp932でエンコードできず途中で止まる。既知の罠）
    if lo_r > 0.005:
        print(f"  [NG] reset直後に {lo_r*100:.2f}cm 浮いている＝毎エピソード落下している")
    elif abs(lo_3 - lo_r) > 0.005:
        print(f"  [NG] 放置で {abs(lo_3-lo_r)*100:.2f}cm 沈んだ＝reset直後は不安定")
    else:
        print(f"  [OK] reset直後から接地している（浮き {lo_r*100:+.2f}cm、"
              f"300step放置で変化 {abs(lo_3-lo_r)*100:.2f}cm）")
        print(f"     ＝学習に落下の衝撃は入っていない。")
    if lo0 > 0.05:
        print(f"  [参考] ただし**環境を作った瞬間だけ** {lo0*100:.1f}cm 浮いている"
              f"（hip.pos=[0,0,0.2]＋MIMoの既定姿勢）。")
        print(f"     settle_steps=100 で落として吸収され、init_position は接地状態。")
        print(f"     Viewerを起動した瞬間に見えるのはこれ。学習には影響しない。")
        print(f"     ＝ただし太郎の初期姿勢は『落下して着地した姿勢』である点は留意。")


if __name__ == "__main__":
    main()
