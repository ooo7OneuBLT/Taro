"""ステップ1（複数フレーム動き検出）の人工画像テスト。

各テストケースについて：
  ・frame1 を 20フレームぶん食わせる（履歴を安定させる）
  ・frame2 を 1回食わせる
  ・motion_map を可視化

これで「動きが正しい場所に、正しい強さで出るか」を目視で確認できる。
反射のロジック単体（環境を通さない）なので、失敗しても原因は反射側に絞れる。
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt

from e_orient_test_images import make_cases   # 前に作った10ケース
from e_orienting_v2 import OrientingReflexV2


class DummyModel:
    """アクチュエータ検索を通すためのダミー（テストでは使わない）。"""
    class _Act:
        def __init__(self, name): self.name = name
    def __init__(self):
        self.nu = 0
    def actuator(self, i):
        return DummyModel._Act("")


def run_case(label, f1, f2):
    reflex = OrientingReflexV2(DummyModel())
    # 履歴を frame1 で満たす（20フレームで最大スケール分）
    for _ in range(25):
        reflex.update(f1)
    # ここで frame2
    reflex.update(f2)
    return reflex.motion_map


def judge(label, motion, h_exp, v_exp, note):
    """ステップ1の合否判定（方向はステップ3の役割なのでここでは見ない）：
       ・「動きなし」以外は motion.max() が閾値を超えているか
       ・「動きなし」は max がほぼゼロか
       ・重心（強い画素の平均位置）が期待方向と整合するか（緩い確認）
    """
    res = {"label": label, "max": float(motion.max()),
           "sum": float(motion.sum())}
    if "NO motion" in label:
        res["pass"] = res["max"] < 0.01
        res["reason"] = f"max={res['max']:.4f} < 0.01" if res["pass"] \
                        else f"max={res['max']:.4f} が非ゼロ＝誤検出"
        return res
    if res["max"] < 0.1:
        res["pass"] = False
        res["reason"] = f"max={res['max']:.4f} が閾値 0.1 未満＝動きを検出できていない"
        return res
    # 重心を計算（強さで重み付けした画素位置の平均）
    thresh = motion.max() * 0.3
    strong = motion > thresh
    if strong.sum() == 0:
        res["pass"] = False
        res["reason"] = "閾値超え画素がゼロ"
        return res
    ys, xs = np.where(strong)
    weights = motion[strong]
    cx_map = float((xs * weights).sum() / weights.sum())
    cy_map = float((ys * weights).sum() / weights.sum())
    res["centroid"] = (cx_map, cy_map)
    # 期待方向と重心が矛盾していないか（★緩い確認）
    center = 64.0
    def side_ok(exp, val):
        # 期待が "0" なら中心付近（±10）に収まる
        # 期待が "+" なら中心より正の側
        # 期待が "-" なら中心より負の側
        if exp == "0":
            return abs(val - center) < 10
        if exp in ("+", "+(weak)"):
            return val > center - 3   # 少し余裕
        if exp == "-":
            return val < center + 3
        return True
    # h：画像座標では x が大きい = 右
    h_ok = side_ok(h_exp, cx_map)
    # v：画像座標では y が小さい = 上、大きい = 下（外部では逆向きの符号）
    v_ok = side_ok(v_exp, center * 2 - cy_map)   # y を反転して "上=正" に揃える
    res["pass"] = h_ok and v_ok
    res["reason"] = f"重心(x={cx_map:.1f},y={cy_map:.1f}) 期待(h={h_exp},v={v_exp}): " \
                    f"h_ok={h_ok} v_ok={v_ok}"
    return res


def main():
    cases = make_cases()
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.3 * n))

    results = []
    for i, (label, f1, f2, h_exp, v_exp, note) in enumerate(cases):
        motion = run_case(label, f1, f2)
        r = judge(label, motion, h_exp, v_exp, note)
        results.append(r)

        axes[i, 0].imshow(f1, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"{label}\nframe 1 (x25)", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(f2, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("frame 2", fontsize=9)
        axes[i, 1].axis("off")

        # 参考：単純な frame1→frame2 の差分（ステップ1の変更前の相当）
        simple_diff = np.abs(f2.astype(float) - f1.astype(float))
        axes[i, 2].imshow(simple_diff, cmap="hot",
                          vmin=0, vmax=max(simple_diff.max(), 1e-3))
        axes[i, 2].set_title("simple diff (ref)", fontsize=9)
        axes[i, 2].axis("off")

        # ステップ1の出力
        vmax = max(motion.max(), 1e-3)
        axes[i, 3].imshow(motion, cmap="hot", vmin=0, vmax=vmax)
        axes[i, 3].set_title(f"step1 output\nmax={motion.max():.3f}",
                             fontsize=9)
        axes[i, 3].axis("off")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), os.pardir, "docs", "figures",
                       "orient_v2_step1_20260726.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved:", os.path.abspath(out))

    # 判定
    print("\n=== ステップ1の判定 ===")
    passed = sum(1 for r in results if r["pass"])
    print(f"通過 {passed}/{len(results)}")
    print(f"{'ケース':<28}{'合否':<6}{'max':>8}  理由")
    print("-" * 100)
    for r in results:
        mark = "OK" if r["pass"] else "NG!"
        print(f"{r['label']:<28}{mark:<6}{r['max']:>8.4f}  {r['reason']}")


if __name__ == "__main__":
    main()
