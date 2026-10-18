"""段階2の合格テスト（仕様書 §4）を回し、判定し、絵にする。

使い方：
  python G/scripts/pass_test.py G/experiments/段階2_合格テスト_2026-10-15.json [--quick]
  --quick … 場面と語ごとに種1つだけ（確かめ用。合格の判定には使わない）
  --plot-only … 走らせず、保存済みの CSV から判定と絵だけ作り直す

判定（脳1回＝0.1秒ごとの記録から）：
* 「見ている」＝視線と物の中心のなす角が look_deg（10度）以内。2つ以上入れば角の小さい方
* 条件A：発話の前（t < word_t）に見ていた回数が、妨害物で一番多い（0回なら不成立）
* 段階3：実験ファイルの semantic_table（手で登録）／semantic_learned（学んだ表）／どちらも無し（空）、
  use_ior（復帰抑制、既定あり）、control（語ごとの「言われていない方の物」）と max_diff
  （書けば「B − B_control がこれ以下で合格」＝③学ぶ前は分からない）
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
import core_v2  # noqa: E402,F401  負荷の決まり（core_v2/perf.py：スレッド1本など）を numpy・torch より先に効かせる

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
    # 段階5のC：発話のあと目的が置かれたか（知らない語では置かれないはず）
    res["goal_set"] = any(r.get("goal.age") not in ("", None) for r in rows[i0:])
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
    # 探し物に着いてから見続けた長さ：発話から b_within_s 以内に始まった最初の「探し物を見ている」続き
    #   （発話の時点で既に見ていたら、発話の回から数える）
    res["dwell_first_s"] = None
    for i in range(i0, min(len(rows), i0 + within + 1)):
        if look[i] == target:
            j = i
            while j < len(rows) and look[j] == target:
                j += 1
            res["dwell_first_s"] = round((j - i) * dt, 2)
            break
    return res


def judge_with_control(rows, word, target, cfg):
    """段階3の③：control（言われていない方の物）でも同じ判定をして B_control に入れる。
    探し物が無くても復帰抑制で物を順に見て回るので、たまたま言われた物を見る試行が出る。
    「分かっていない」は「言われた物を見る数が、言われていない物を見る数と変わらない」で確かめる。"""
    res = judge(rows, target, cfg)
    ctrl = (cfg.get("control") or {}).get(word)
    if ctrl:
        res["control"] = ctrl
        res["B_control"] = judge(rows, ctrl, cfg)["B"]
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
            ax.set_title(scene.replace("段階2b_3物体_", "").replace("段階2_3物体_", "").replace("_2026-10-15", ""))
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
    from core_v2.world_eye.switch import suffix
    out = os.path.join(REPO, suffix(cfg["out"]) + ("_quick" if quick else ""))
    os.makedirs(out, exist_ok=True)
    seeds = cfg["seeds"][:1] if quick else cfg["seeds"]
    trials = []
    if not plot_only:
        from core_v2 import run_v2
        from core_v2.brain import Brain
        from core_v2.save import save
        brain = Brain()
        brain.boxes["eyein"].setup()
        # 段階3：辞書は「手で登録した表」「経験で学んだ表」「空」のどれか。復帰抑制は既定あり
        sem = brain.boxes["semantic"]
        if cfg.get("semantic_table"):
            meta_sem = {"table": sem.load_table(os.path.join(REPO, cfg["semantic_table"]))}
        elif cfg.get("semantic_learned"):
            # 目だけの場面では、目だけの場面で学んだ表（learn_names --world eye の出力 *_目だけ.pt）を読む
            meta_sem = {"learned": sem.load_learned(os.path.join(REPO, suffix(cfg["semantic_learned"][:-3]) + ".pt"))}
        else:
            meta_sem = {"empty": True}
        sem.learning = False
        # 段階5：聞き分ける係を親の話しかけで育てた耳（core_v2/tools/hear_parent.py）。無ければ何も聞き取れない
        if cfg.get("listen_ear"):
            meta_sem["ear"] = brain.boxes["listen"].load_ear(os.path.join(REPO, cfg["listen_ear"]))
        brain.boxes["listen"].learning = False
        brain.boxes["orienteye"].use_ior = bool(cfg.get("use_ior", True))
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
                        res = judge_with_control(rows, word, target, cfg)
                        trials.append({"scene": scene, "word": word, "seed": seed, "csv": name,
                                       "rows": rows, "res": res})
                        print(f"{scene} {word} seed{seed}: A={res['A']} B={res['B']} "
                              f"onset={res['onset_s']} dwell={res['dwell_s']} pre={res['pre_counts']} "
                              f"({time.time() - t0:.0f}s)", flush=True)
                finally:
                    run_v2.close_env(env)
        # 脳の保存（段階2の仮部品に学習する重みは無いが、型どおり版と配線の指紋を残す）
        save(brain, os.path.join(REPO, "G", "models", suffix(cfg.get("brain_out", "段階2_合格テスト_brain.pt")[:-3]) + ".pt"),
             extra={"experiment": cfg["name"], "semantic": meta_sem,
                    "use_ior": bool(cfg.get("use_ior", True))})
        print(f"全体 {time.time() - t_start:.0f}秒", flush=True)
    else:
        for scene in cfg["scenes"]:
            for word, target in cfg["words"].items():
                for seed in seeds:
                    name = f"{scene}_{word.rstrip('？?')}_seed{seed}.csv"
                    rows = load_rows(os.path.join(out, name))
                    trials.append({"scene": scene, "word": word, "seed": seed, "csv": name,
                                   "rows": rows, "res": judge_with_control(rows, word, target, cfg)})

    summary = {"experiment": cfg["name"], "quick": quick, "per_word": {}, "trials": []}
    for word in cfg["words"]:
        sel = [t for t in trials if t["word"] == word]
        summary["per_word"][word] = {
            "n": len(sel), "A": sum(t["res"]["A"] for t in sel), "B": sum(t["res"]["B"] for t in sel),
            "already_on_target": sum(t["res"]["already_on_target"] for t in sel),
            # 発話の時点で探し物を見ていなかった試行だけの条件B（見ていた試行の B は語を聞いた証拠にならない）
            "n_not_already": sum(not t["res"]["already_on_target"] for t in sel),
            "B_not_already": sum(t["res"]["B"] for t in sel if not t["res"]["already_on_target"]),
            "dwell_first_s_median": (float(np.median(d)) if (d := [t["res"]["dwell_first_s"] for t in sel
                                                                     if t["res"]["dwell_first_s"] is not None]) else None),
            "dwell_first_s_min": (min(d) if d else None),
            "n_reached": len(d),
            "dwell_s_median": float(np.median([t["res"]["dwell_s"] for t in sel])),
            "B_control": (sum(t["res"].get("B_control", 0) for t in sel) if cfg.get("control") else None),
            "goal_set": sum(bool(t["res"].get("goal_set")) for t in sel),
            # 段階3の③（学ぶ前は分からない）は「B − B_control が max_diff 以下」で合格
            # 段階5のC（知らない語）は「目的が置かれた試行が0」で合格
            "pass": ((sum(bool(t["res"].get("goal_set")) for t in sel) == 0) if cfg.get("expect_no_goal")
                     else (sum(t["res"]["B"] for t in sel) - sum(t["res"].get("B_control", 0) for t in sel)
                           <= cfg["max_diff"]) if "max_diff" in cfg
                     else sum(t["res"]["B"] for t in sel) >= cfg["pass_count"]) and not quick}
    # 同じ物を見た数を、言葉どうしで比べる（2026-10-16）：物 X について「X を指す言葉のとき X を見た数」
    # − 「ほかの言葉（X を control に持つ）のとき X を見た数」。B − B_control（同じ言葉で物どうし）は物の
    # 目立ちの差を拾うので（③の空の辞書でボール 11・コップ 1）、言葉の効果だけを見るためにこちらも並べる
    if cfg.get("control"):
        cross = {}
        for word, obj in cfg["words"].items():
            others = [w for w, c in cfg["control"].items() if c == obj and w != word]
            hit = sum(t["res"]["B"] for t in trials if t["word"] == word)
            base = sum(t["res"].get("B_control", 0) for t in trials if t["word"] in others)
            cross[obj] = {"言われた言葉": word, "その言葉のとき見た": hit,
                          "ほかの言葉のとき見た": base, "差": hit - base}
        summary["same_object_across_words"] = cross
    for t in trials:
        summary["trials"].append({k: t[k] for k in ("scene", "word", "seed", "csv")} | t["res"])
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        keys = ["scene", "word", "seed", "A", "B", "B_control", "onset_s", "dwell_first_s", "dwell_s", "already_on_target",
                "n_saccades", "n_saccades_after", "heard_t", "pre_counts"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary["trials"])
    paths = plot(trials, cfg, out)
    print(json.dumps(summary["per_word"], ensure_ascii=False, indent=1))
    if "same_object_across_words" in summary:
        print("同じ物を見た数の、言葉どうしの差:")
        print(json.dumps(summary["same_object_across_words"], ensure_ascii=False, indent=1))
    print("絵:", *paths, sep="\n  ")


if __name__ == "__main__":
    from core_v2.world_eye.switch import pop_world_flag, select
    select(pop_world_flag(sys.argv))      # --world eye|mimo（既定 mimo）
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], quick="--quick" in sys.argv, plot_only="--plot-only" in sys.argv)
