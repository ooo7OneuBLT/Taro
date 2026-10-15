"""段階2の合格テスト（仕様書 §4）を回し、判定し、絵にする。

使い方：
  python G/scripts/pass_test.py G/experiments/段階2_合格テスト_2026-10-15.json [--quick]
  --quick … 場面と語ごとに種1つだけ（確かめ用。合格の判定には使わない）
  --plot-only … 走らせず、保存済みの CSV から判定と絵だけ作り直す

判定（脳1回＝0.1秒ごとの記録から）：
* 「見ている」＝視線と物の中心のなす角が look_deg（10度）以内。2つ以上入れば角の小さい方
* 条件A：発話の前（t < word_t）に見ていた回数が、妨害物で一番多い（0回なら不成立）
* 条件B：発話が耳に届いた回から b_within_s（3秒）以内に探し物を見始め、そこから
  b_hold_s（1秒）以上続けて見る（見始めが3秒以内なら、続きは3秒を越えてもよい）
出力（out フォルダ）：試行ごとの CSV、summary.json、判定の表 summary.csv、絵（角度の時間変化・視線の軌跡）
"""
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, REPO)

import numpy as np

OBJ_JA = {"ball": "ボール", "cup": "コップ", "distractor": "妨害物"}
COLORS = {"ball": "#d62728", "cup": "#7f7f7f", "distractor": "#1f77b4"}


def looking(row, look_deg):
    cands = [(row[f"ang_{k}"], k) for k in OBJ_JA if row[f"ang_{k}"] is not None
             and float(row[f"ang_{k}"]) <= look_deg]
    return min(cands)[1] if cands else ""


def judge(rows, target, cfg):
    dt = rows[1]["t"] - rows[0]["t"]
    look = [looking(r, cfg["look_deg"]) for r in rows]
    pre = [l for r, l in zip(rows, look) if r["t"] < cfg["word_t"]]
    counts = {k: pre.count(k) for k in OBJ_JA}
    a_ok = counts["distractor"] > 0 and counts["distractor"] == max(counts.values()) \
        and sum(1 for v in counts.values() if v == counts["distractor"]) == 1
    heard = [i for i, r in enumerate(rows) if r["heard"]]
    res = {"pre_counts": counts, "A": bool(a_ok), "heard_t": None, "B": False,
           "onset_s": None, "dwell_s": 0.0, "already_on_target": False,
           "n_saccades": sum(1 for r in rows if r["cmd_h"] not in (None, "")),
           "n_saccades_after": 0}
    if not heard:
        return res
    i0 = heard[0]
    res["heard_t"] = rows[i0]["t"]
    res["already_on_target"] = look[i0] == target
    res["n_saccades_after"] = sum(1 for r in rows[i0:] if r["cmd_h"] not in (None, ""))
    need = int(round(cfg["b_hold_s"] / dt))
    within = int(round(cfg["b_within_s"] / dt))
    best = 0
    i = i0
    while i < len(rows):
        if look[i] == target:
            j = i
            while j < len(rows) and look[j] == target:
                j += 1
            best = max(best, j - i)
            if i - i0 <= within and j - i >= need and not res["B"]:
                res["B"] = True
                res["onset_s"] = round((i - i0) * dt, 2)
            i = j
        else:
            i += 1
    res["dwell_s"] = round(best * dt, 2)
    return res


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k.startswith(("ang_", "x_", "y_")) or k in ("t",):
                    r[k] = float(v) if v not in ("", "None") else None
            r["cmd_h"] = r["cmd_h"] if r["cmd_h"] not in ("", "None") else None
            rows.append(r)
    return rows


def plot(trials, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic"]
    paths = []
    for word, target in cfg["words"].items():
        sel = [t for t in trials if t["word"] == word]
        n = len(sel)
        cols = 5
        rws = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rws, cols, figsize=(3.2 * cols, 2.4 * rws), sharex=True, sharey=True,
                                 squeeze=False)
        for ax, tr in zip(axes.flat, sel):
            rows = tr["rows"]
            t = [r["t"] for r in rows]
            for k in OBJ_JA:
                ax.plot(t, [r[f"ang_{k}"] for r in rows], color=COLORS[k], lw=1.2, label=OBJ_JA[k])
            ax.axhline(cfg["look_deg"], color="k", lw=0.6, ls=":")
            if tr["res"]["heard_t"] is not None:
                ax.axvline(tr["res"]["heard_t"], color="k", lw=0.8)
            for r in rows:
                if r["cmd_h"] is not None:
                    ax.plot([r["t"]], [0.5], marker="v", color="k", ms=3)
            sc = "左" if "ボール左" in tr["scene"] else "右"
            ax.set_title(f"ボール{sc}・種{tr['seed']}  A{'○' if tr['res']['A'] else '×'} "
                         f"B{'○' if tr['res']['B'] else '×'}", fontsize=9)
            ax.set_ylim(0, 40)
        for ax in axes.flat[n:]:
            ax.axis("off")
        axes[0][0].legend(fontsize=7, loc="upper right")
        fig.supxlabel("時刻（秒）　縦線＝発話が耳に届いた回、▼＝目を跳ばした回、点線＝「見ている」10度")
        fig.supylabel("視線と物のなす角（度）")
        nb = sum(t["res"]["B"] for t in sel)
        na = sum(t["res"]["A"] for t in sel)
        fig.suptitle(f"「{word}」 条件B {nb}/{n}　条件A {na}/{n}")
        fig.tight_layout()
        p = os.path.join(out, f"角度_{word.rstrip('？?')}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

        # 視線の軌跡（基準＝試行の頭の目の向き。右・上が正、度）
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), squeeze=False)
        for ax, scene in zip(axes.flat, cfg["scenes"]):
            ss = [t for t in sel if t["scene"] == scene]
            for tr in ss:
                rows = tr["rows"]
                h = tr["res"]["heard_t"] or 1e9
                pre = [r for r in rows if r["t"] < h]
                post = [r for r in rows if r["t"] >= h]
                ax.plot([r["x_gaze"] for r in pre], [r["y_gaze"] for r in pre], color="#9467bd",
                        lw=1, alpha=0.5)
                ax.plot([r["x_gaze"] for r in post], [r["y_gaze"] for r in post], color="#2ca02c",
                        lw=1, alpha=0.5)
            r0 = ss[0]["rows"][0]
            for k in OBJ_JA:
                xs = [r[f"x_{k}"] for r in ss[0]["rows"]]
                ys = [r[f"y_{k}"] for r in ss[0]["rows"]]
                ax.plot(xs, ys, color=COLORS[k], lw=3, alpha=0.4)
                ax.plot([np.mean(xs)], [np.mean(ys)], "o", color=COLORS[k], ms=10, label=OBJ_JA[k])
            ax.plot([r0["x_gaze"]], [r0["y_gaze"]], "k+", ms=12, label="最初の視線")
            ax.set_aspect("equal")
            ax.set_title(scene.replace("段階2_3物体_", "").replace("_2026-10-15", ""))
            ax.set_xlabel("右へ（度）")
            ax.set_ylabel("上へ（度）")
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"「{word}」視線の軌跡（紫＝発話の前、緑＝発話の後、{len(sel)}試行を重ね描き）")
        fig.tight_layout()
        p = os.path.join(out, f"軌跡_{word.rstrip('？?')}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths


def main(cfg_path, quick=False, plot_only=False):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    out = os.path.join(REPO, cfg["out"] + ("_quick" if quick else ""))
    os.makedirs(out, exist_ok=True)
    seeds = cfg["seeds"][:1] if quick else cfg["seeds"]
    trials = []
    if not plot_only:
        from core_v2 import run_v2
        from core_v2.brain import Brain
        from core_v2.save import save
        brain = Brain()
        brain.boxes["eyein"].setup()
        meta_sem = brain.boxes["semantic"].load_table(os.path.join(REPO, cfg["semantic_table"]))
        t_start = time.time()
        for scene in cfg["scenes"]:
            for word, target in cfg["words"].items():
                env = run_v2.build_env(scene, word, seed=seeds[0], word_t=cfg["word_t"])
                try:
                    for seed in seeds:
                        t0 = time.time()
                        rows = run_v2.run_trial(brain, env, seed=seed, duration_s=cfg["duration_s"],
                                                on_tick=lambda row: row.update(run_v2.path_record(brain)))
                        name = f"{scene}_{word.rstrip('？?')}_seed{seed}.csv"
                        with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as f:
                            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                            w.writeheader()
                            w.writerows(rows)
                        res = judge(rows, target, cfg)
                        trials.append({"scene": scene, "word": word, "seed": seed, "csv": name,
                                       "rows": rows, "res": res})
                        print(f"{scene} {word} seed{seed}: A={res['A']} B={res['B']} "
                              f"onset={res['onset_s']} dwell={res['dwell_s']} pre={res['pre_counts']} "
                              f"({time.time() - t0:.0f}s)", flush=True)
                finally:
                    run_v2.close_env(env)
        # 脳の保存（段階2の仮部品に学習する重みは無いが、型どおり版と配線の指紋を残す）
        save(brain, os.path.join(REPO, "G", "models", "段階2_合格テスト_brain.pt"),
             extra={"experiment": cfg["name"], "semantic_table": meta_sem})
        print(f"全体 {time.time() - t_start:.0f}秒", flush=True)
    else:
        for scene in cfg["scenes"]:
            for word, target in cfg["words"].items():
                for seed in seeds:
                    name = f"{scene}_{word.rstrip('？?')}_seed{seed}.csv"
                    rows = load_rows(os.path.join(out, name))
                    trials.append({"scene": scene, "word": word, "seed": seed, "csv": name,
                                   "rows": rows, "res": judge(rows, target, cfg)})

    summary = {"experiment": cfg["name"], "quick": quick, "per_word": {}, "trials": []}
    for word in cfg["words"]:
        sel = [t for t in trials if t["word"] == word]
        summary["per_word"][word] = {
            "n": len(sel), "A": sum(t["res"]["A"] for t in sel), "B": sum(t["res"]["B"] for t in sel),
            "already_on_target": sum(t["res"]["already_on_target"] for t in sel),
            "dwell_s_median": float(np.median([t["res"]["dwell_s"] for t in sel])),
            "pass": sum(t["res"]["B"] for t in sel) >= cfg["pass_count"] and not quick}
    for t in trials:
        summary["trials"].append({k: t[k] for k in ("scene", "word", "seed", "csv")} | t["res"])
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        keys = ["scene", "word", "seed", "A", "B", "onset_s", "dwell_s", "already_on_target",
                "n_saccades", "n_saccades_after", "heard_t", "pre_counts"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary["trials"])
    paths = plot(trials, cfg, out)
    print(json.dumps(summary["per_word"], ensure_ascii=False, indent=1))
    print("絵:", *paths, sep="\n  ")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], quick="--quick" in sys.argv, plot_only="--plot-only" in sys.argv)
