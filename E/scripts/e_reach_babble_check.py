"""自発運動で腕がどれだけ動き、おもちゃに触れうるかを測る。

【なぜ要るか、2026-07-29】ユーザーの目視「自発運動をONにしてみたけど、
腕の動きが小さくておもちゃに触れる雰囲気はない」。

【切り分ける候補】
  1. 自発運動の振幅が小さい      Viewer は 0.174（学習初期の実効値）
  2. 四肢の筋力が弱すぎる        4ヶ月の補正で **×0.047（1/21）** にしている
                                 注意：この補正の目標値には文献の根拠が無い
                                 （`taro_core/src/body/infant_limbs.py` 冒頭）
  3. おもちゃが遠い              手から22.7cm・可動域の端でやっと届く位置

注意：「動かない」はまず計測器を疑う（落とし穴）。指令が本当に関節に届いているかも見る。

使い方:
    .venv/Scripts/python.exe E/scripts/e_reach_babble_check.py
    E_SECONDS=30 E_SCENE=名前 で変えられる
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
# 注意：自発運動の生成器は `taro_core/src/brain/spinal_cord/cpg.py`（脊髄）にある
for _p in [os.path.join(_ROOT, "taro_core", "src", "brain"),
           os.path.join(_ROOT, "run", "scene_tools"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np      # noqa: E402
import mujoco           # noqa: E402
import e_scene          # noqa: E402

SCENE = os.environ.get("E_SCENE", "リーチング_リクライニング60度")
SECONDS = float(os.environ.get("E_SECONDS", "30.0"))

# (ラベル, 筋力の倍率, 自発運動の振幅)
#   筋力の倍率＝`body.limb_scale`。1.0 が現状、大きいほど強い（2.0で補正が半分戻る）
#   振幅＝Viewer と同じ `0.5 + amp * ノイズ`
CASES = [
    ("現状", 1.0, 0.174),
    ("振幅を2倍", 1.0, 0.35),
    ("筋力を5倍", 5.0, 0.174),
    ("筋力5倍＋振幅2倍", 5.0, 0.35),
]

ARM_JOINTS = ("shoulder_horizontal", "shoulder_ad_ab", "shoulder_rotation", "elbow")


def run(label, limb_scale, amp):
    from spinal_cord.cpg import ColoredNoiseGenerator
    sc = e_scene.load(SCENE)
    sc["body"]["limb_scale"] = float(limb_scale)
    sc["fingerprint"] = None          # 条件を変えたので照合しない
    env, hands = e_scene.build(sc, orient=True, vor=True, seed=0)
    u = env.unwrapped
    m, d = u.model, u.data
    dt = float(m.opt.timestep) * int(u.frame_skip)
    n_act = env.action_space.shape[0]
    gen = ColoredNoiseGenerator(n_act, seed=0)
    act = np.full(n_act, 0.5, dtype=np.float32)

    toy_bid = int(m.body("test_object1").id)
    toy = np.array(d.qpos[u._toy_qadr:u._toy_qadr + 3], dtype=float)
    # 腕の関節（左右）の qpos アドレス
    arm_q = []
    for side in ("right", "left"):
        for base in ARM_JOINTS:
            for j in range(m.njnt):
                if (m.joint(j).name or "").split(":")[-1] == f"{side}_{base}":
                    arm_q.append((f"{side}_{base}", int(m.jnt_qposadr[j])))
    hist = {k: [] for k, _ in arm_q}
    hands_pos = {"right": [], "left": []}
    best = {"right": 1e9, "left": 1e9}
    touches = 0
    prev_touch = False

    for i in range(int(SECONDS / dt)):
        if i % 10 == 0:      # Viewer と同じ更新間隔
            act = np.clip(0.5 + amp * gen.sample(0.7), 0.0, 1.0).astype(np.float32)
        env.step(act)
        for k, qa in arm_q:
            hist[k].append(float(np.degrees(d.qpos[qa])))
        for side in ("right", "left"):
            p = np.array(d.xpos[int(m.body(f"{side}_hand").id)], dtype=float)
            hands_pos[side].append(p)
            best[side] = min(best[side], float(np.linalg.norm(toy - p)))
        # おもちゃに触れたか（接触の有無）
        hit = any((int(d.contact.geom1[c]) in _toy_geoms(m, toy_bid)
                   or int(d.contact.geom2[c]) in _toy_geoms(m, toy_bid))
                  for c in range(d.ncon))
        if hit and not prev_touch:
            touches += 1
        prev_touch = hit

    print(f"\n【{label}】 筋力×{limb_scale:g}  振幅{amp:g}")
    print(f"  手→おもちゃ の最接近   右 {best['right']*100:6.2f} cm   "
          f"左 {best['left']*100:6.2f} cm")
    for side in ("right", "left"):
        P = np.array(hands_pos[side])
        span = P.max(axis=0) - P.min(axis=0)
        print(f"  {'右' if side=='right' else '左'}手の動いた範囲       "
              f"x{span[0]*100:5.1f} y{span[1]*100:5.1f} z{span[2]*100:5.1f} cm")
    amps = {k: (np.max(v) - np.min(v)) for k, v in hist.items()}
    print("  腕の関節が動いた幅[度]  "
          + "  ".join(f"{k.replace('right_','右').replace('left_','左')[:6]}"
                      f"{amps[k]:5.1f}" for k in list(amps)[:4]))
    print("                          "
          + "  ".join(f"{k.replace('right_','右').replace('left_','左')[:6]}"
                      f"{amps[k]:5.1f}" for k in list(amps)[4:]))
    print(f"  おもちゃに触れた回数    {touches}")
    env.close()
    return dict(label=label, best_r=best["right"], best_l=best["left"],
                touches=touches, amps=amps)


_TOY_G = {}


def _toy_geoms(m, bid):
    if bid not in _TOY_G:
        _TOY_G[bid] = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid}
    return _TOY_G[bid]


def main():
    print("=" * 74)
    print(f" 自発運動で腕はどれだけ動くか（{SCENE} / {SECONDS:g}秒）")
    print("=" * 74)
    print("  注意おもちゃの半径は0.5cm。最接近が1cm以内なら「触れうる」")
    rows = [run(*c) for c in CASES]
    print("\n" + "=" * 74)
    print(" まとめ")
    print("=" * 74)
    print(f"{'条件':<20}{'右の最接近':>12}{'左の最接近':>12}{'触れた回数':>12}")
    for r in rows:
        print(f"{r['label']:<20}{r['best_r']*100:>10.2f}cm{r['best_l']*100:>10.2f}cm"
              f"{r['touches']:>12}")
    print("\n  読み方")
    print("   ・振幅を上げて改善するなら → 自発運動の振幅が律速")
    print("   ・筋力を上げて改善するなら → 四肢の筋力補正が律速")
    print("   ・どちらでも改善しないなら → おもちゃの位置が届く範囲の端すぎる")


if __name__ == "__main__":
    main()
