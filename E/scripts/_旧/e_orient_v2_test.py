"""視線誘導反射（新版）ステップ1〜3の人工画像テスト。

【なぜ人工画像か】環境（視覚パイプライン・キャッシュ・物理）を通すと、
失敗したとき原因が反射側か配線側か切り分けられない。まずロジック単体を検査する。

【前回のテストの失敗・2026-07-26】
  ・「上に動く」と「下に動く」が**完全に同じ重心**を返したのに、判定が甘くて両方OKになった
  → 反対向きのペアが【違う結果を返すこと】を必須条件に加える
  ・並進（A地点→B地点）ばかりテストしていたが、E1で実際に使うのは**その場での振動**
  → 振動ケースを追加し、フレーム列を食わせる形に変える
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt

from e_orienting_v2 import OrientingReflexV2

RES = 128
CENTER = (RES - 1) / 2.0


class DummyModel:
    """アクチュエータ検索を通すためのダミー（ステップ1〜3では使わない）。"""
    nu = 0
    def actuator(self, i):
        raise IndexError


def blank():
    return np.zeros((RES, RES), dtype=np.float32)


def square(img, cx, cy, size=15, val=1.0):
    r = size // 2
    img[max(0, cy-r):min(RES, cy+r), max(0, cx-r):min(RES, cx+r)] = val
    return img


def noisy(img, amount, seed):
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.uniform(-amount, amount, img.shape), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# テストケース：(label, frames列, 期待h, 期待v, note)
#   期待は "+" / "-" / "0" / None（判定しない）
# ---------------------------------------------------------------------------
def make_cases():
    cases = []

    # === E1 の実際の使い方＝その場で振動する対象 ===
    def oscillate(cx, cy, amp=4, size=15, n=30, bg=None, seed=None):
        frames = []
        for i in range(n):
            base = blank() if bg is None else bg.copy()
            dx = amp if i % 2 == 0 else -amp
            f = square(base, cx + dx, cy, size)
            if seed is not None:
                f = noisy(f, 0.12, seed + i)
            frames.append(f)
        return frames

    cases.append(("osc RIGHT", oscillate(96, 64), "+", "0",
                  "右で振動 → 右を向く"))
    cases.append(("osc LEFT", oscillate(32, 64), "-", "0",
                  "左で振動 → 左を向く"))
    cases.append(("osc UP", oscillate(64, 32), "0", "+",
                  "上で振動 → 上を向く"))
    cases.append(("osc DOWN", oscillate(64, 96), "0", "-",
                  "下で振動 → 下を向く"))
    cases.append(("osc CENTER", oscillate(64, 64), "0", "0",
                  "中心で振動 → 中心（強度は最大のはず）"))

    # 静止物があっても、動いている方だけを見る
    still_left = square(blank(), 32, 64, 15)
    cases.append(("osc RIGHT + still LEFT", oscillate(96, 64, bg=still_left), "+", "0",
                  "左に静止物、右で振動 → 右（静止は動きゼロ）"))

    # 背景ノイズがあっても対象を拾う
    cases.append(("osc RIGHT + bg noise", oscillate(96, 64, seed=100), "+", "0",
                  "背景ノイズ → 側方抑制と閾値で除去できるか"))

    # === 2標的の分離角度スイープ ===
    # 視野角 60度 / 128画素 → 1画素 = 0.469度。分離角度 θ → θ * 128/60 画素
    # 文献（Van der Stigchel & de Vries 2013, Vision Research）:
    #   ~30度まで   単峰 = 中間点への平均化（global effect）
    #   ~35度付近   単峰と二峰の混在
    #   45度超      二峰 = どちらか一方を選ぶ
    def oscillate_pair(sep_deg, n=30, amp=4, size=13):
        sep_px = int(round(sep_deg * RES / 60.0))
        cx1 = int(round(63.5 - sep_px / 2))
        cx2 = int(round(63.5 + sep_px / 2))
        frames = []
        for i in range(n):
            dx = amp if i % 2 == 0 else -amp
            f = square(square(blank(), cx1 + dx, 64, size), cx2 + dx, 64, size)
            frames.append(f)
        return frames

    cases.append(("pair 10deg", oscillate_pair(10), "0", "0",
                  "10度 → 中間（文献: 30度までは平均化）"))
    cases.append(("pair 30deg", oscillate_pair(30), "0", "0",
                  "30度 → 中間（文献: 境界だがまだ平均化側）"))
    cases.append(("pair 45deg", oscillate_pair(45), "CHOOSE", "0",
                  "45度 → どちらかを選ぶ（文献: 45度超で二峰）"))
    cases.append(("pair 55deg", oscillate_pair(55), "CHOOSE", "0",
                  "55度 → どちらかを選ぶ（文献の二峰域）"))

    # 動きなし
    still = square(blank(), 96, 64, 15)
    cases.append(("NO motion", [still] * 30, "0", "0", "静止のみ → 反応ゼロ"))

    # === 並進（E1では起きないが、限界を明示するために残す）===
    f_a = square(blank(), 32, 64, 15)
    f_b = square(blank(), 96, 64, 15)
    cases.append(("translate L->R (far)", [f_a] * 25 + [f_b], None, "0",
                  "遠い並進 → 側方抑制でどちらかが勝つ（中央でない）"))

    return cases


def run_case(frames):
    reflex = OrientingReflexV2(DummyModel())
    for f in frames:
        reflex.update(f)
    return reflex


def judge(label, reflex, h_exp, v_exp):
    """判定：期待方向と符号が一致するか。"0" は中央付近（|dir| < 0.15）。"""
    h, v, s = reflex.h_dir, reflex.v_dir, reflex.strength
    res = {"label": label, "h": h, "v": v, "strength": s}

    if "NO motion" in label:
        res["pass"] = s < 0.01
        res["reason"] = f"strength={s:.4f}" + (" → 反応なし" if res["pass"] else " → 誤検出")
        return res

    if s < 0.01:
        res["pass"] = False
        res["reason"] = f"strength={s:.4f} → 動きを検出できていない"
        return res

    def ok(exp, val):
        if exp is None:
            return True
        if exp == "0":
            return abs(val) < 0.15
        if exp == "+":
            return val > 0.15
        if exp == "-":
            return val < -0.15
        if exp == "CHOOSE":       # どちらか一方に決着していること（中央でない）
            return abs(val) > 0.15
        return True

    h_ok, v_ok = ok(h_exp, h), ok(v_exp, v)
    res["pass"] = h_ok and v_ok
    res["reason"] = f"h={h:+.3f} v={v:+.3f} 期待(h={h_exp},v={v_exp}) → h_ok={h_ok} v_ok={v_ok}"
    return res


def check_opposite_pairs(results):
    """反対向きのペアが違う結果を返しているかを確認する（前回の失敗への対策）。"""
    pairs = [("osc RIGHT", "osc LEFT", "h"), ("osc UP", "osc DOWN", "v")]
    out = []
    by_label = {r["label"]: r for r in results}
    for a, b, axis in pairs:
        if a not in by_label or b not in by_label:
            continue
        va, vb = by_label[a][axis], by_label[b][axis]
        differ = abs(va - vb) > 0.2
        out.append((f"{a} vs {b} ({axis})", differ, f"{va:+.3f} vs {vb:+.3f}"))
    return out


def main():
    cases = make_cases()
    results = []
    reflexes = []
    for label, frames, h_exp, v_exp, note in cases:
        rf = run_case(frames)
        reflexes.append((label, frames, rf, note))
        results.append(judge(label, rf, h_exp, v_exp))

    # --- 図 ---
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(11.5, 2.35 * n))
    for i, (label, frames, rf, note) in enumerate(reflexes):
        axes[i, 0].imshow(frames[-1], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"{label}\nlast frame", fontsize=8)
        axes[i, 0].axis("off")

        m = rf.motion_map
        axes[i, 1].imshow(m, cmap="hot", vmin=0, vmax=max(m.max(), 1e-3))
        axes[i, 1].set_title(f"step1 motion\nmax={m.max():.3f}", fontsize=8)
        axes[i, 1].axis("off")

        b = rf.biased_map
        axes[i, 2].imshow(b, cmap="hot", vmin=0, vmax=max(b.max(), 1e-3))
        axes[i, 2].set_title("step2 center bias", fontsize=8)
        axes[i, 2].axis("off")

        c = rf.competed_map
        axes[i, 3].imshow(c, cmap="hot", vmin=0, vmax=max(c.max(), 1e-3))
        axes[i, 3].set_title(f"step3 after competition\nh={rf.h_dir:+.2f} v={rf.v_dir:+.2f}",
                             fontsize=8)
        axes[i, 3].axis("off")
        # 重心位置に印
        if abs(rf.h_dir) > 1e-6 or abs(rf.v_dir) > 1e-6:
            px = CENTER + rf.h_dir * CENTER
            py = CENTER - rf.v_dir * CENTER
            axes[i, 3].plot(px, py, "c+", markersize=14, markeredgewidth=2)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), os.pardir, "docs", "figures",
                       "orient_v2_step123_20260726.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print("saved:", os.path.abspath(out))

    # --- 判定 ---
    print("\n=== ステップ1-3 の判定 ===")
    passed = sum(1 for r in results if r["pass"])
    print(f"通過 {passed}/{len(results)}\n")
    print(f"{'ケース':<26}{'合否':<6}{'h_dir':>8}{'v_dir':>8}{'強さ':>8}  理由")
    print("-" * 108)
    for r in results:
        mark = "OK" if r["pass"] else "NG!"
        print(f"{r['label']:<26}{mark:<6}{r['h']:>8.3f}{r['v']:>8.3f}"
              f"{r['strength']:>8.3f}  {r['reason']}")

    print("\n=== 反対向きのペアが区別できているか（前回の失敗への対策）===")
    for name, ok, detail in check_opposite_pairs(results):
        print(f"  [{'OK ' if ok else 'NG!'}] {name}: {detail}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
