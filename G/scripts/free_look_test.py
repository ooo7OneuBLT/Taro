"""段階3の合格の線①：探し物が無いとき、物を順に見て回るか（仕様書 §2・付録A）。

使い方：
  python G/scripts/free_look_test.py G/experiments/段階3_1_自由に見る_2026-10-15.json [--quick] [--plot-only] [--only ior]

1試行＝親は何も言わない 10秒。段階2の2場面×10種。復帰抑制あり・なしの2回を回して並べる。
判定（脳1回＝0.1秒ごと、「見ている」は段階2と同じく10度以内で角の小さい方）：
* 全部見た＝3つの物それぞれを1回以上見た
* 合格＝復帰抑制ありで「全部見た」が pass_count（18）/20 以上
出力（out フォルダ）：試行ごとの CSV、summary.json、絵（物ごとに見ていた時間の割合）
"""
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, REPO)
import core_v2  # noqa: E402,F401  負荷の決まり（core_v2/perf.py：スレッド1本など）を numpy・torch より先に効かせる
sys.path.insert(0, HERE)

from pass_test import OBJ_JA, load_rows, looking  # noqa: E402

CONDS = {"ior": True, "no_ior": False}
COND_JA = {"ior": "復帰抑制あり", "no_ior": "復帰抑制なし"}


def judge(rows, cfg):
    look = [looking(r, cfg["look_deg"]) for r in rows]
    counts = {k: look.count(k) for k in OBJ_JA}
    first = {}
    for r, l in zip(rows, look):
        if l and l not in first:
            first[l] = round(r["t"], 2)
    switches = sum(1 for a, b in zip(look, look[1:]) if b and a != b)
    return {"counts": counts, "all_seen": all(v > 0 for v in counts.values()),
            "first_t": first, "n_switches": switches,
            "n_saccades": sum(1 for r in rows if r["cmd_h"] not in (None, ""))}


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic", "sans-serif"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    keys = list(OBJ_JA)
    for i, cond in enumerate(CONDS):
        s = summary["per_cond"].get(cond)
        if not s:
            continue
        xs = [j + (i - 0.5) * width for j in range(len(keys))]
        ax.bar(xs, [s["look_share"][k] for k in keys], width,
               label=f"{COND_JA[cond]}（全部見た {s['all_seen']}/{s['n']}）")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([OBJ_JA[k] for k in keys])
    ax.set_ylabel("見ていた時間の割合")
    ax.set_title("段階3 ①：何も言われないとき、どの物をどれだけ見たか")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out, "見ていた割合.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(cfg_path, quick=False, plot_only=False, only=None):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    from core_v2.world_eye.switch import suffix
    out = os.path.join(REPO, suffix(cfg["out"]) + ("_quick" if quick else ""))
    os.makedirs(out, exist_ok=True)
    seeds = cfg["seeds"][:1] if quick else cfg["seeds"]
    trials = []
    brain = None
    if not plot_only:
        from core_v2 import run_v2
        from core_v2.brain import Brain
        brain = Brain()
        brain.boxes["eyein"].setup()
    t_start = time.time()
    for cond, use_ior in CONDS.items():
        if only and cond != only:
            continue
        if brain is not None:
            brain.boxes["orienteye"].use_ior = use_ior
        for scene in cfg["scenes"]:
            env = None if plot_only else run_v2.build_env(scene, None, seed=seeds[0], silent=True)
            try:
                for seed in seeds:
                    name = f"{cond}_{scene}_seed{seed}.csv"
                    path = os.path.join(out, name)
                    if plot_only:
                        rows = load_rows(path)
                    else:
                        t0 = time.time()
                        rows = run_v2.run_trial(brain, env, seed=seed, duration_s=cfg["duration_s"],
                                                on_tick=lambda row: row.update(run_v2.path_record(brain)))
                        with open(path, "w", newline="", encoding="utf-8") as f:
                            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                            w.writeheader()
                            w.writerows(rows)
                    res = judge(rows, cfg)
                    trials.append({"cond": cond, "scene": scene, "seed": seed, "csv": name, "res": res})
                    if not plot_only:
                        print(f"{COND_JA[cond]} {scene} seed{seed}: 全部見た={res['all_seen']} "
                              f"回数={res['counts']} 移った={res['n_switches']} ({time.time() - t0:.0f}s)",
                              flush=True)
            finally:
                if env is not None:
                    run_v2.close_env(env)

    summary = {"experiment": cfg["name"], "quick": quick, "per_cond": {}, "trials": trials}
    for cond in CONDS:
        sel = [t for t in trials if t["cond"] == cond]
        if not sel:
            continue
        tot = {k: sum(t["res"]["counts"][k] for t in sel) for k in OBJ_JA}
        n_ticks = max(1, sum(tot.values()))
        summary["per_cond"][cond] = {
            "n": len(sel), "all_seen": sum(t["res"]["all_seen"] for t in sel),
            "look_share": {k: round(v / n_ticks, 3) for k, v in tot.items()},
            "mean_switches": round(sum(t["res"]["n_switches"] for t in sel) / len(sel), 2),
        }
    s = summary["per_cond"].get("ior")
    summary["pass"] = bool(s and not quick and s["all_seen"] >= cfg["pass_count"])
    summary["plot"] = plot(summary, out)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    for cond, s in summary["per_cond"].items():
        print(f"{COND_JA[cond]}: 全部見た {s['all_seen']}/{s['n']}  見ていた割合 {s['look_share']}  "
              f"平均の移り {s['mean_switches']}")
    print(f"合格: {summary['pass']}  （{time.time() - t_start:.0f}秒）  出力: {out}")
    return summary


if __name__ == "__main__":
    from core_v2.world_eye.switch import pop_world_flag, select
    select(pop_world_flag(sys.argv))      # --world eye|mimo（既定 mimo）
    argv = sys.argv[1:]
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    args = [a for a in argv if not a.startswith("--") and a != only]
    main(args[0], quick="--quick" in sys.argv, plot_only="--plot-only" in sys.argv, only=only)
