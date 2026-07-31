"""【体格の検証】太郎の身体プロポーションが新生児と合っているかを実測する。

【なぜ調べるか】
2026-07-20、Viewerを見た第三者（乳児の育児経験者）から次の指摘を受けた：
  ・動きは赤ちゃんらしい
  ・**手足が長すぎる。赤ちゃんはもっと頭でっかち**
  ・だから手が視界に入りにくいのではないか
＝E1（hand regard）が創発しない原因として、**体格そのもの**という候補が出た。
これは今まで疑っていなかった（予測・報酬・感覚の側ばかり見ていた）。

【人間の新生児の実測値（比較の基準）】
  ・身長 約49〜50cm
  ・頭高（頭頂〜顎）が身長の約 1/4 ＝ 12cm前後。成人は約1/8
  ・上肢長（肩〜指先）は身長の約 32〜35%
＝**頭が大きく四肢が短い**のが新生児の特徴。この比率が合っているかを見る。

【測り方】
MuJoCoのbody位置とgeomのサイズから、
  ①身長（頭頂〜足底）②頭の高さ ③上肢長（肩〜手）④頭高/身長 ⑤上肢長/身長
を出し、人間の新生児の値と並べる。
注意：MIMoは18ヶ月児が基準で、mimoGrowthのageで縮尺する。**ageが効いているか**も同時に確認する
（今日、視力の設定値が効いていなかった前例があるため）。

使い方: python e_body_proportion.py [age]
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
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import mimoEnv  # noqa: F401,E402
import mujoco  # noqa: E402
import d_c5_motor_quality as mq  # noqa: E402

# 人間の新生児（満期産）の実測値。出典は doc/参考文献リスト.md に追記すること。
HUMAN = {
    "身長": (49.9, "cm", "WHO Child Growth Standards 2006 男児0ヶ月 中央値"),
    "頭高/身長": (0.25, "", "頭高は身長の約1/4（新生児）。成人は約1/8"),
    "上肢長/身長": (0.335, "", "上肢長＝肩峰〜指尖。新生児で身長の約1/3"),
}


def span(model, data, names, axis=2):
    """指定bodyのgeomが占める範囲（min, max）を軸ごとに返す。"""
    lo, hi = np.inf, -np.inf
    for i in range(model.ngeom):
        bid = model.geom_bodyid[i]
        nm = model.body(bid).name
        if not any(k in nm for k in names):
            continue
        c = data.geom_xpos[i][axis]
        r = float(np.max(model.geom_size[i][model.geom_size[i] > 0], initial=0.0))
        lo, hi = min(lo, c - r), max(hi, c + r)
    return lo, hi


def main():
    age = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    os.environ["C5_AGE"] = str(age)
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=age)
    raw = env.unwrapped
    m, d = raw.model, raw.data
    mujoco.mj_forward(m, d)

    # ---- 全身の上下範囲（仰向けなので「身長」は前後方向=x） ----
    # 注意：第1版は「world/fence/toyを除く」という**除外リスト**で書いたが、
    # `test_object1/2`（遠方に置かれた別body）が漏れて身長382cmになった。
    # ＝除外リストは漏れる。太郎に属するbodyを**ホワイトリスト**で取る。
    taro_bodies = set()
    for i in range(m.nbody):
        nm = m.body(i).name
        if nm in ("world", "test_object1", "test_object2", "mimo_location"):
            continue
        taro_bodies.add(i)
    print(f"（太郎のbody {len(taro_bodies)}個で測定）")

    all_lo, all_hi = np.inf, -np.inf
    for i in range(m.ngeom):
        if int(m.geom_bodyid[i]) not in taro_bodies:
            continue
        c = d.geom_xpos[i]
        r = float(np.max(m.geom_size[i][m.geom_size[i] > 0], initial=0.0))
        all_lo, all_hi = min(all_lo, c[0] - r), max(all_hi, c[0] + r)
    height = (all_hi - all_lo) * 100.0

    head_lo, head_hi = span(m, d, ("head",), axis=0)
    head_len = (head_hi - head_lo) * 100.0

    # 上肢長＝肩(upper_arm)の付け根から手の先まで（直線距離）
    arms = []
    for side in ("left", "right"):
        try:
            sh = np.array(d.body(f"{side}_upper_arm").xpos, dtype=float)
            ha = np.array(d.body(f"{side}_hand").xpos, dtype=float)
            # 手の先までを含めるため、手のgeom半径を足す
            arms.append(np.linalg.norm(ha - sh) * 100.0)
        except Exception:
            pass
    arm = float(np.mean(arms)) if arms else float("nan")

    print(f"\n=== 太郎の体格（age={age}ヶ月 相当の設定）===")
    print(f"  身長（頭頂〜足底）  : {height:6.1f} cm")
    print(f"  頭の長さ            : {head_len:6.1f} cm")
    print(f"  上肢長（肩〜手首）  : {arm:6.1f} cm")

    r_head = head_len / height if height else float("nan")
    r_arm = arm / height if height else float("nan")
    print(f"\n=== 人間の新生児との比較 ===")
    print(f"  {'項目':<14}{'太郎':>9}{'新生児':>10}   判定")
    hh, _, hs = HUMAN["身長"]
    print(f"  {'身長':<14}{height:>8.1f}cm{hh:>9.1f}cm   x{height/hh:.2f}")
    hr, _, _ = HUMAN["頭高/身長"]
    print(f"  {'頭高/身長':<14}{r_head:>9.3f}{hr:>10.3f}   "
          f"{'頭が小さい' if r_head < hr * 0.9 else '頭が大きい' if r_head > hr * 1.1 else '一致'}")
    ar, _, _ = HUMAN["上肢長/身長"]
    print(f"  {'上肢長/身長':<14}{r_arm:>9.3f}{ar:>10.3f}   "
          f"{'腕が短い' if r_arm < ar * 0.9 else '腕が長い' if r_arm > ar * 1.1 else '一致'}")

    print(f"\n  ※出典: {hs}")
    print(f"        {HUMAN['頭高/身長'][2]}")
    print(f"        {HUMAN['上肢長/身長'][2]}")

    # ---- 目と手の位置関係（hand regard に直結する） ----
    try:
        eye = np.array(d.body("left_eye").xpos, dtype=float)
    except Exception:
        eye = np.array(d.cam("eye_left").xpos, dtype=float) if hasattr(d, "cam") else None
    if eye is not None and arms:
        ha = np.array(d.body("left_hand").xpos, dtype=float)
        dist = np.linalg.norm(ha - eye) * 100.0
        print(f"\n=== hand regard に効く距離 ===")
        print(f"  目から手まで（安静時）: {dist:6.1f} cm")
        print(f"  ※新生児がはっきり見える距離は 19〜25cm とされる。"
              f"手がそこに届かないと、見えても焦点が合わない")
    env.close()


if __name__ == "__main__":
    main()
