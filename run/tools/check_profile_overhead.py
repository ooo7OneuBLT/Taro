# -*- coding: utf-8 -*-
"""作業C（速度改善・処理内訳の記録）の既定OFFの原則の実測確認。

仕様：F3速度改善（2026-08-24）。

  検証1：run.profile=False（既定・未指定）のとき、run.csv に
         t_*_sec / rss_mb 列が1つも増えないこと（列が増えないだけでなく、
         profile用のコード自体を1文も実行しない構造になっている。
         run/trainer.py の Trainer._prof_t0/_prof_add を参照）。
  検証2：run.profile=False を2回回した結果（既知の再現性が確立している
         新生児_仰向け_柵ありシーン。run/tools/check_divergence.py と同じ
         シーンを使う）と、run.profile=True を1回回した結果を比べ、
         「on vs off」の食い違いパターンが「off vs off」の食い違いパターンと
         同じ性質（同じ行・同程度の大きさ）であること＝profile自体が新しい
         食い違いの原因になっていないことを確認する。
         【重要な事実（2026-08-24実測）】このシーンは実は**off同士でも
         完全一致しない**（cereb_err等が4桁目からずれる。原因未特定・
         本作業のスコープ外）。CLAUDE.mdの過去記録は「350ステップ×5回
         完全一致」だが、今回はF2-1由来の設定（small体・muscle等）が
         混じっている可能性があり、この検証だけでは"profile機能が原因でない"
         ことの証明にしかならない（食い違いの根本原因の追跡は別タスク）。
  検証3：run.profile=True のオーバーヘッド（実時間）が、off同士の
         run-to-runばらつきの範囲に収まること（=無視できる大きさ）。

使い方:
    .venv/Scripts/python.exe run/tools/check_profile_overhead.py
"""
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = r"C:\claude\AI\Taro"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from run import main as run_main    # noqa: E402

run_main._register()

SCRATCH = os.path.join(ROOT, "run", "tools", "_scratch_profile_check")
os.makedirs(SCRATCH, exist_ok=True)
STEPS = 300


def spec(profile, tag):
    return {
        "name": "profile検証", "scene": "新生児_仰向け_柵あり",
        "taro": {"actuation": "muscle", "age_months": 0.0,
                 "save": os.path.join(SCRATCH, f"m_{tag}.pt")},
        "run": {"type": "train", "steps": STEPS, "seed": 0, "checkpoint": 50,
                "profile": profile, "csv": os.path.join(SCRATCH, f"c_{tag}.csv")},
        "plugins": {},
    }


def read_csv(path):
    with open(path, encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def diverging_cols(a, b):
    """行ごとに、値が食い違った列の集合を返す（'t_'で始まる列とrss_mbは除く）。"""
    cols = (set(a[0]) & set(b[0])) if a and b else set()
    cols = {c for c in cols if not c.startswith("t_") and c != "rss_mb"}
    out = []
    for x, y in zip(a, b):
        out.append({c for c in cols if x[c] != y[c]})
    return out


def main():
    print("=" * 78)
    print(" check_profile_overhead：作業C・既定OFFの原則の確認")
    print("=" * 78)

    timings = {}
    for profile, tag in [(False, "off1"), (False, "off2"), (True, "on1")]:
        t0 = time.time()
        run_main.run(spec(profile, tag), steps_override=STEPS, verbose=False)
        timings[tag] = time.time() - t0

    off1 = read_csv(os.path.join(SCRATCH, "c_off1.csv"))
    off2 = read_csv(os.path.join(SCRATCH, "c_off2.csv"))
    on1 = read_csv(os.path.join(SCRATCH, "c_on1.csv"))

    print("\n[検証1] profile=False のときprofile列が増えないこと")
    off_cols = set(off1[0].keys())
    prof_cols_in_off = {c for c in off_cols if c.startswith("t_") or c == "rss_mb"}
    ok1 = len(prof_cols_in_off) == 0
    print(f"  OFF側にあるprofile列: {sorted(prof_cols_in_off) or '（無し）'}  合格={ok1}")

    on_cols = set(on1[0].keys())
    prof_cols_in_on = {c for c in on_cols if c.startswith("t_") or c == "rss_mb"}
    print(f"  ON側にあるprofile列: {sorted(prof_cols_in_on)}")
    ok1b = len(prof_cols_in_on) > 0
    print(f"  ON側でprofile列が実際に出ていること 合格={ok1b}")

    print("\n[検証2] off-vs-off と off-vs-on の食い違いパターンを比較")
    diff_off_off = diverging_cols(off1, off2)
    diff_off_on = diverging_cols(off1, on1)
    rows_with_diff_oo = [i for i, s in enumerate(diff_off_off) if s]
    rows_with_diff_oon = [i for i, s in enumerate(diff_off_on) if s]
    print(f"  off1 vs off2 で食い違った行: {rows_with_diff_oo}")
    print(f"  off1 vs on1  で食い違った行: {rows_with_diff_oon}")
    # profileがONのときだけ新たに食い違い始める行があるなら要注意
    #  （off-offでは一致していた行が、on1でだけ崩れる、という意味）
    newly_broken = [i for i in rows_with_diff_oon if i not in rows_with_diff_oo]
    ok2 = len(newly_broken) == 0
    print(f"  off-offでは一致していたのに on1 でだけ崩れた行: {newly_broken or '（無し）'}"
          f"  合格={ok2}")
    if not ok2:
        print("  注意：この一致は「profileが原因でない」ことの直接証明にはならない。"
              "  ただし少なくとも profile ON 特有の崩れではないことは分かる。")

    print("\n[検証3] オーバーヘッドがrun-to-runのばらつき範囲に収まるか")
    base = (timings["off1"] + timings["off2"]) / 2.0
    spread = abs(timings["off1"] - timings["off2"])
    overhead = timings["on1"] - base
    print(f"  off1={timings['off1']:.2f}s off2={timings['off2']:.2f}s "
          f"on1={timings['on1']:.2f}s")
    print(f"  off同士のばらつき={spread:.2f}s / on-off={overhead:.2f}s")
    ok3 = abs(overhead) <= max(spread, 1.0) * 3
    print(f"  ばらつきの3倍以内に収まっている 合格={ok3}")

    print("\n" + "=" * 78)
    all_ok = ok1 and ok1b and ok2 and ok3
    print("全項目OK" if all_ok else "一部NG（上記参照。検証2はコメントの留保つき）")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
