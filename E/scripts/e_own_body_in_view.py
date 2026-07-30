"""★自分の体（特に足）が視界にどれだけ入っているか、反射の邪魔になるかを測る。

【なぜ要るか、2026-07-28】リクライニング姿勢にしたところ、ユーザーの目視で
「デフォルトの視界に自分の足があるのは大丈夫か？実験に影響しないか」。

【2つの側面】
  人間らしさ  … 正常。リクライニングした乳児は自分の足が見える。
                3〜5ヶ月には自分の足を見て遊ぶ（foot regard）という現象がある。
  実験への影響 … ★交絡になりうる。視線誘導反射は「動くもの」に反応するので、
                足が動けば発火し、おもちゃへの定位を妨げる可能性がある。

【測ること】
  1. 視野の中で自分の体が占める割合（描き分け＝segmentation で数える）
  2. 部位別の内訳（足・手・胴）
  3. ★おもちゃを置かずに自発運動させ、反射が何発撃つか
     （＝自分の体だけで反射が誤発火するか）

使い方:
    E_RECLINE=60 .venv/Scripts/python.exe E/scripts/e_own_body_in_view.py
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
NECK = os.environ.get("E_HOLD_TILT", "60")
EYE_V = float(os.environ.get("E_EYE_REST_V", "-15"))
SECONDS = float(os.environ.get("E_SECONDS", "15.0"))
RES = 96


def build():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from e_head_hold import CaregiverHands

    kw = body_kwargs_from_env(AGE, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=AGE),
                           age=AGE, toy=False, vor=True, orient=True,
                           recline_deg=REC, **kw)
        env.reset(seed=0)
        hands = CaregiverHands(env.unwrapped.model, env.unwrapped.data)
        hands.hold(target=({"head_tilt": float(NECK)} if NECK else None))
    return env


def body_share(m, d, cam_name="eye_left", size=RES):
    """視野の中で自分の体が占める割合を、描き分け（segmentation）で数える。

    ⚠️MuJoCoの segmentation は画素ごとに geom id を返すので、
      「どの部位が映っているか」を正確に数えられる（明度の閾値では区別できない）。
    """
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), size)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), size)
    ren = mujoco.Renderer(m, size, size)
    ren.enable_segmentation_rendering()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = int(m.camera(cam_name).id)
    ren.update_scene(d, cam)
    seg = ren.render()[:, :, 0].copy()      # geom id
    ren.close()

    total = seg.size
    out = {}
    for gid in np.unique(seg):
        if gid < 0:
            continue
        try:
            nm = m.geom(int(gid)).name or ""
            bid = int(m.geom_bodyid[int(gid)])
            bn = m.body(bid).name or ""
        except Exception:
            continue
        n = int((seg == gid).sum())
        out[(nm, bn)] = out.get((nm, bn), 0) + n
    return out, total, seg


def classify(bn):
    """body名から部位を大まかに分ける。"""
    b = bn.lower()
    if any(k in b for k in ("foot", "toe", "lower_leg", "upper_leg")):
        return "足"
    if any(k in b for k in ("hand", "finger", "arm", "thumb", "ff", "mf", "rf", "lf")):
        return "手・腕"
    if any(k in b for k in ("head", "eye")):
        return "頭"
    if any(k in b for k in ("body", "hip", "chest", "cb", "ub", "lb")):
        return "胴"
    if any(k in b for k in ("floor", "world", "recline", "fence")):
        return "環境"
    return "その他"


def main():
    print("=" * 74)
    print(f" 自分の体が視界にどれだけ入っているか"
          f"（リクライニング{REC:g}度・{AGE:g}ヶ月）")
    print("=" * 74)
    env = build()
    u = env.unwrapped
    m, d = u.model, u.data
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for _ in range(int(2.0 / dt)):
        env.step(a)

    parts, total, seg = body_share(m, d)
    agg = {}
    for (gn, bn), n in parts.items():
        agg[classify(bn)] = agg.get(classify(bn), 0) + n

    print("\n1. 静止しているときの視野の内訳")
    print("-" * 74)
    for k in sorted(agg, key=lambda x: -agg[x]):
        print(f"  {k:<8} {agg[k]/total*100:6.2f}%  （{agg[k]:5d} / {total} 画素）")
    own = sum(v for k, v in agg.items() if k in ("足", "手・腕", "頭", "胴"))
    print(f"  ---- 自分の体の合計 {own/total*100:.2f}% ----")

    # 部位の細目（上位）
    print("\n  内訳（上位8つ）")
    for (gn, bn), n in sorted(parts.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {bn:<24}{n/total*100:6.2f}%")

    # ---- 2. おもちゃ無しで自発運動 → 反射が誤発火するか ----
    print("\n" + "=" * 74)
    print("2. ★おもちゃを置かずに自発運動させ、反射が撃つか")
    print("=" * 74)
    print("   （撃つなら、自分の体の動きに反応している＝おもちゃの定位を妨げうる）")
    reflex = u._orienting
    reflex.reset()
    rng = np.random.default_rng(0)
    n_act = env.action_space.shape[0]
    act = np.full(n_act, 0.5, dtype=np.float32)
    shares = []
    for k in range(int(SECONDS / dt)):
        if k % 10 == 0:      # 1Hz くらいで指令を変える＝自発運動
            act = np.clip(0.5 + 0.15 * rng.standard_normal(n_act), 0, 1
                          ).astype(np.float32)
        env.step(act)
        if k % max(1, int(1.0 / dt)) == 0:
            p, t, _ = body_share(m, d, size=64)
            o = sum(v for (gn, bn), v in p.items()
                    if classify(bn) in ("足", "手・腕", "頭", "胴"))
            shares.append(o / t * 100)
    print(f"\n   {SECONDS:g}秒の自発運動で サッケード {reflex.n_saccades} 発")
    print(f"   視野に占める自分の体 平均 {np.mean(shares):.1f}%"
          f"（最小 {min(shares):.1f}% 〜 最大 {max(shares):.1f}%）")
    print(f"   反射の反応の強さ（最後）{reflex.strength:.4f}")

    print("\n" + "=" * 74)
    print(" 読み方")
    print("=" * 74)
    print("  ・自分の体が視野の1〜2割なら、おもちゃ（見かけ7度＝視野の数%）と競合する")
    print("  ・おもちゃ無しでサッケードが多数撃たれるなら、自分の体に定位している")
    print("  ★人間の乳児も自分の手足を見る（hand regard / foot regard）ので、")
    print("    見えること自体は正常。問題は『おもちゃより強く反応するか』")
    env.close()


if __name__ == "__main__":
    main()
