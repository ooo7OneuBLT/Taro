"""学習前(白紙) vs 学習後 の「動きの軸」の比較を集計する。

【なぜ・2026-07-25】ユーザーが新生児の自発運動の動画を見て
「足は伸ばす・縮めるの繰り返しで、横方向の動きがあまりない」と指摘した。
寝返りは体を長軸まわりに回す動きなので、**回旋と外転**が無ければ物理的に起きない。
屈曲/伸展（伸ばす・縮める）だけでは、どれだけ激しくても回転モーメントが立たない。

→ 太郎の動きを軸ごとに分け、学習前後でどの軸が増えたかを見る。
   学習後に回旋・横方向が増えていれば「太郎は寝返る動きを学習した」ことになる。

データは e_growth_train.py の E_MEASURE_POSTURE=1 が出す軸別ブロック。

注意：出力は必ずASCII（cp932の端末で日本語が化ける＝チェックリスト項35）。
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
import glob
import os
import re
import sys

import numpy as np

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.join(_HERE, os.pardir, "logs", "E", "axis")

# e_growth_train.py の _AXIS_NAMES と同じ順で出力されている
AXES = ["lateral/abduction", "flexion/extension", "rotation"]
BLOCKS = ["whole body", "TRUNK (roll-over)", "legs (hip+knee)"]

RE_AXIS = re.compile(r"([\d.]+)\s*\(\s*(\d+)\D{1,4}\)\D{0,8}=\s*([\d.\-]+|nan)")
RE_ROT = re.compile(r"=([\d.]+)\D{1,4}\(\D{1,6}([\d.]+)\)\D{1,20}=([\d.]+)%")
RE_TRANS = re.compile(r"(\d+)\D{1,4}\(.*?(\d+)\D{1,4}/.*?(\d+)\D{1,4}\)")


def parse(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    out = {"blocks": []}
    m = RE_ROT.search(t)
    if m:
        out["rot"], out["rot_max"], out["prone"] = (float(m.group(1)),
                                                    float(m.group(2)), float(m.group(3)))
    for line in t.splitlines():
        if "->" in line and "tick=" in line:
            mm = RE_TRANS.search(line)
            if mm:
                out["trans"], out["to_prone"], out["to_supine"] = (int(mm.group(1)),
                                                                   int(mm.group(2)),
                                                                   int(mm.group(3)))
            break
    # 軸別ブロック：「--」で始まる行から次の「--」までの間に、軸の行が出力順に並ぶ
    cur = None
    for line in t.splitlines():
        s = line.strip()
        if s.startswith("--") and "|" not in s:
            cur = []
            out["blocks"].append(cur)
            continue
        if cur is not None and "=" in s and "#" not in s.split("=")[0]:
            mm = RE_AXIS.search(s)
            if mm:
                cur.append(float(mm.group(1)))
    return out


def collect(prefix):
    rows = [parse(f) for f in sorted(glob.glob(os.path.join(_DIR, prefix + "_s*.txt")))]
    return [r for r in rows if r.get("blocks")]


def main():
    # 【2026-07-25】3群比較。シナジーを筋肉モードへ配線した効果を見る。
    #   pre  = 学習前（白紙の脳）
    #   post = 学習後・シナジーOFF（従来＝筋肉モードでは強制OFFだった）
    #   syn  = 学習後・シナジーON（pair_offset=90 で拮抗筋ペアに符号反転で適用）
    # 判定は「寝返りが減ったか」だけでは足りない。回旋が潰れていたら
    #   それは poor repertoire（異常GM）の方向で、改善ではない（Prechtl系の文献）。
    pre, post, syn = collect("pre"), collect("post"), collect("syn")
    print(f"3 groups: BEFORE(blank) n={len(pre)} / AFTER synOFF n={len(post)} "
          f"/ AFTER synON n={len(syn)}")
    print("  deterministic policy (act_mean), K=10, 6000 steps, muscle model\n")

    def mean_of(rows, key):
        v = [r[key] for r in rows if key in r]
        return float(np.mean(v)) if v else float("nan")

    def sd_of(rows, key):
        v = [r[key] for r in rows if key in r]
        return float(np.std(v)) if v else float("nan")
    print(f"{'':<24}{'BEFORE':>10}{'synOFF':>10}{'synON':>10}{'  d(ON-OFF)':>12}")
    for k, nm in [("prone", "prone %"), ("rot", "trunk rotation deg"),
                  ("rot_max", "max rotation deg"), ("to_supine", "roll-backs"),
                  ("trans", "posture switches")]:
        sd = ((sd_of(post,k)**2 + sd_of(syn,k)**2)/2) ** 0.5
        d = (mean_of(syn,k) - mean_of(post,k)) / sd if sd > 1e-9 else 0.0
        print(f"  {nm:<22}{mean_of(pre,k):>10.2f}{mean_of(post,k):>10.2f}"
              f"{mean_of(syn,k):>10.2f}{d:>12.2f}")

    nb = min([len(r["blocks"]) for r in pre + post + syn] or [0])
    for bi in range(min(nb, len(BLOCKS))):
        print(f"\n  == {BLOCKS[bi]} ==   |joint angular velocity|")
        print(f"{'    axis':<26}{'BEFORE':>10}{'synOFF':>10}{'synON':>10}{'ON/OFF':>9}")
        for ai, ax in enumerate(AXES):
            g = lambda rows: [r["blocks"][bi][ai] for r in rows
                              if len(r["blocks"][bi]) > ai]
            b, a, sy = g(pre), g(post), g(syn)
            if not b or not a:
                continue
            mb, ma = float(np.mean(b)), float(np.mean(a))
            ms = float(np.mean(sy)) if sy else float("nan")
            print(f"    {ax:<22}{mb:>10.4f}{ma:>10.4f}{ms:>10.4f}"
                  f"{ms/ma if ma > 1e-9 else float('nan'):>8.2f}x")
        # 屈曲/伸展を1としたときの各軸の比（＝「横方向がどれだけ混ざっているか」）
        for label, rows in [("BEFORE", pre), ("synOFF", post), ("synON", syn)]:
            vals = [r["blocks"][bi] for r in rows if len(r["blocks"][bi]) >= 3]
            if not vals:
                continue
            v = np.mean(np.array(vals), axis=0)
            f = v[1] if v[1] > 1e-9 else float("nan")
            print(f"    -> {label}: flex/ext = 1.00, lateral = {v[0]/f:.2f}, "
                  f"rotation = {v[2]/f:.2f}")


if __name__ == "__main__":
    main()
