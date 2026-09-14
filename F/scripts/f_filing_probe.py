# -*- coding: utf-8 -*-
"""動く場面で、太郎が物をどうファイリングするかを映像で見る（2026-09-13）。

【なぜ】ユーザー指摘「画像からじゃなくて動画からの方が精度良くできそう」。
  太郎には共通運動の手がかりが既にある（Kellman & Spelke 1983 [Tier1]、
  `object_detector.py:178-195`）。静止画1枚で測るとその手がかりを外してしまう。

【本番の経路（2026-09-13に確かめた・ここを間違えると別物を測る）】
  `attend: true` の走行が使うのは **`segment_at_points`（点プロンプト）** であって、
  `mask_generator.generate()`（全画面の自動切り出し）**ではない**。
  `object_files.py:993`。最初この違いを知らずに静止画で自動切り出しを測り、
  本番では起きない現象（机がコップにくっつく等）を報告しかけた。

【横取りの注意（2026-09-13・昨日の二重importと同じ型）】
  プラグインは `setup()` の中で `self._segment_at_points = segment_at_points` と
  **関数を自分の中に抱え込む**。モジュール側を書き換えても効かない。
  ⇒ `setup()` の**後**に、プラグインが持っている方を差し替える。
  さらに「本当に取れているか」を走行中に自己検査し、取れていなければ**中止する**。

【使い方】
    .venv/Scripts/python.exe F/scripts/f_filing_probe.py --steps 240
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np

EXP = "F/experiments/F2-135pre_眼球の利得を3倍に_短い走行_2026-09-12.json"
OUT = os.path.join("F", "logs", "_図", "2026-09-13_ファイリング_動く場面")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--exp", default=EXP)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    from run.log_setup import setup_logging, log_tail, get_logger, warn_if
    paths = setup_logging(a.out)
    log = get_logger("ファイリング測定")

    print("組み立て中（MuJoCo・DINOv2・MobileSAM）...")
    from run import main as run_main
    run_main._register()
    spec = run_main.load_spec(a.exp)

    def _redir(node):
        if isinstance(node, dict):
            return {k: (os.path.join(a.out, os.path.basename(str(v).rstrip("/")))
                        if isinstance(v, str) and (k.endswith("_out") or k in ("csv", "save"))
                        else _redir(v)) for k, v in node.items()}
        return node
    spec["plugins"] = _redir(spec.get("plugins") or {})
    spec["taro"] = _redir(spec.get("taro") or {})
    spec.setdefault("run", {})["csv"] = os.path.join(a.out, "run.csv")

    from run.context import Ctx
    from run.plugins.common import scene as scene_mod
    plugins = run_main.build_plugins(spec)
    env, sc, _ = scene_mod.build(spec["scene"], taro=spec["taro"],
                                 seed=int(spec["run"].get("seed", 0)), verbose=False)
    u = env.unwrapped
    K = int(spec["run"].get("K", 10))
    dt = float(u.model.opt.timestep) * int(u.frame_skip) * K
    ctx = Ctx(env=env, spec=spec, scene=sc, n_steps=a.steps, dt=dt)

    env.reset(seed=int(spec["run"].get("seed", 0)))
    for p in plugins:
        if hasattr(p, "setup"):
            p.setup(ctx)

    OF = [p for p in plugins if hasattr(p, "ofs")]
    if not OF:
        raise SystemExit("物体ファイルのプラグインが見つからない")
    of = OF[0]

    # ---- 横取り：setup の後に、プラグインが抱えた関数を差し替える -----------
    CAP = {"img": None, "dets": None, "pts": None}

    def _wrap_sap(orig):
        def _f(pp, img224, points, *ar, **kw):
            r = orig(pp, img224, points, *ar, **kw)
            CAP["img"] = np.asarray(img224).copy()
            CAP["pts"] = [tuple(q["pos"]) for q in points] if points else []
            dets = r[0] if isinstance(r, tuple) else r
            CAP["dets"] = [{"mask": np.asarray(d["mask"]).copy(), "area": float(d["area"]),
                            "pos": tuple(d["pos"])} for d in dets if d.get("mask") is not None]
            return r
        return _f

    def _wrap_det(orig):
        def _f(p_, n_, *ar, **kw):
            r = orig(p_, n_, *ar, **kw)
            if kw.get("img") is not None:
                CAP["img"] = np.asarray(kw["img"]).copy()
            CAP["pts"] = []
            CAP["dets"] = [{"mask": np.asarray(d["mask"]).copy(), "area": float(d["area"]),
                            "pos": tuple(d["pos"])} for d in r if d.get("mask") is not None]
            return r
        return _f

    n_patched = 0
    if hasattr(of, "_segment_at_points"):
        of._segment_at_points = _wrap_sap(of._segment_at_points); n_patched += 1
    if hasattr(of, "_detect"):
        of._detect = _wrap_det(of._detect); n_patched += 1
    print("横取りを仕掛けた関数: %d 個" % n_patched)
    if n_patched == 0:
        raise SystemExit("横取りできる関数が無い（プラグインの作りが変わった）")

    for p in plugins:
        if hasattr(p, "on_reset"):
            p.on_reset(ctx)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = plt.cm.tab10(np.linspace(0, 1, 10))

    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    rows, n_frame = [], 0
    print("走ります（%d 歩 ＝ %.1f 秒ぶん）..." % (a.steps, a.steps * dt))

    for step in range(a.steps):
        CAP["dets"] = None
        for _ in range(K):
            obs, rew, term, trunc, info = env.step(zero)
        ctx.step = step
        ctx.obs, ctx.info = obs, info
        # 【2026-09-13・これが無いと検出が1回も走らない】プラグインは
        #   `ctx.last["obs_out"]["eye_left"]`（目の画像）を見に行き、
        #   無ければ黙って return する（`object_files.py:452-455`）。
        #   本番は run/trainer.py:2656 がここに置いている。
        ctx.last = {"obs_out": obs}
        # この測定は「ファイリング」だけを見るので、語の産出は動かさない。
        #   本番では run/trainer.py:1412 が「脳が語の選択に使った視覚ベクトル」を
        #   ここに置く（**視界全体**のベクトル。物体ファイル自身の appearance とは
        #   別物だと仕様書が明記している）。None なら注意側はその枝を素通りする。
        if not hasattr(ctx, "last_vision_vec"):
            ctx.last_vision_vec = None
        for p in plugins:
            if hasattr(p, "on_step"):
                p.on_step(ctx)

        # 【自己検査】検出は走っているのに横取りが空、を黙って通さない
        if step >= 8 and n_frame == 0 and getattr(of, "_last_t", -1e18) > -1e17:
            log.error("検出は走っている（_last_t=%.2f）のに横取りが0件。"
                      "この測定は無効なので中止する", of._last_t)
            log_tail(paths)
            raise SystemExit("横取りが効いていない（自己検査で中止）")

        if not CAP["dets"] or CAP["img"] is None:
            continue
        n_frame += 1
        img, dets = CAP["img"], CAP["dets"]
        files = list(of.ofs.files)
        att = getattr(of, "_attended_id", None)
        rows.append({"step": step, "t": round(step * dt, 2),
                     "切り出し": len(dets), "物体ファイル": len(files),
                     "注意": att,
                     "面積": [round(float(d["area"]), 4) for d in dets]})

        fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.0), dpi=110)
        ax[0].imshow(img); ax[0].set_title("1. taro's eye"); ax[0].axis("off")

        ov = np.zeros((224, 224, 4))
        for i, d in enumerate(dets):
            ov[d["mask"].astype(bool)] = (*cols[i % 10][:3], .55)
        ax[1].imshow(img); ax[1].imshow(ov)
        for (px, py) in (CAP["pts"] or []):
            ax[1].plot(px, py, "w*", ms=9, mew=1, mec="k")
        ax[1].set_title("2. MobileSAM (point-prompt): %d" % len(dets)); ax[1].axis("off")
        for i, d in enumerate(dets):
            ax[1].text(4, 14 + i * 15, "%.0f%%" % (100 * d["area"]), color="w", fontsize=9,
                       bbox=dict(fc=cols[i % 10], ec="none", alpha=.9, pad=1))

        ax[2].imshow(img)
        for f in files:
            fx, fy = f.pos
            r = (max(float(f.area), 1e-6) * 224 * 224 / np.pi) ** .5
            is_att = (att is not None and f.id == att)
            c = "red" if is_att else ("yellow" if f.misses == 0 else "gray")
            ax[2].add_patch(plt.Circle((fx, fy), r, fill=False, ec=c,
                                        lw=3 if is_att else 2,
                                        ls="-" if f.misses == 0 else "--"))
            ax[2].text(fx - r, fy - r - 3, "#%d%s" % (f.id, "*" if is_att else ""),
                       color="k", fontsize=8,
                       bbox=dict(fc=c, ec="none", alpha=.85, pad=1))
        ax[2].set_title("3. object files: %d (red=attended)" % len(files)); ax[2].axis("off")
        fig.suptitle("step %d   t=%.1fs" % (step, step * dt), fontsize=11)
        plt.tight_layout(); plt.savefig(os.path.join(a.out, "f_%05d.png" % n_frame))
        plt.close(fig)

    with open(os.path.join(a.out, "ファイリングの推移.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    print("\n絵にしたコマ %d 枚" % n_frame)
    warn_if(log, n_frame < 10, "コマが %d 枚しかない。判断できない", n_frame)
    if rows:
        import collections
        nf = [r["物体ファイル"] for r in rows]
        nd = [r["切り出し"] for r in rows]
        print("物体ファイルの個数:", dict(sorted(collections.Counter(nf).items())))
        print("1コマの切り出し数:", dict(sorted(collections.Counter(nd).items())))
        warn_if(log, len(set(nf)) == 1,
                "物体ファイルの個数が全コマで %d のまま。動いていない疑い", nf[0])
    log_tail(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
