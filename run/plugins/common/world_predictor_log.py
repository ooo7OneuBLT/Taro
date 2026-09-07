# -*- coding: utf-8 -*-
"""測る道具：世界の予測器（M7a）の誤差をCSVに書く。

【仕様】F/docs/二語文/仕様_M7a_世界の予測器_測るだけ_2026-09-08.md
「後半：実装担当向け技術付録」4節。`run/plugins/common/word_production.py` の型を踏襲。

【役割】読むだけ（`run/plugins/base.py` の規約）。太郎も環境も変えない。
  読むのは ctx.last_world_pred（run/trainer.py._world_predictor_step が
  on_step_late より前に置く辞書、または cfg.world_predictor 無効時は None）。

【出力先】実験ファイルの `plugins.world_predictor_log.events_out` に
  `<出力先>/世界の予測器.csv` の形で指定する（word_productionと同じ相対パス規約）。

【列】step, t_sec, present, visible, vanished, parent_spoke, parent_text,
  err_state, err_vec, err_parent, err_slow, err_total, baseline, z
  （err_slowは追記2026-09-08。仕様書末尾「追記」節）
"""
import csv
import os

from run.plugins.base import Plugin

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))

_COLUMNS = ["step", "t_sec", "present", "visible", "vanished", "parent_spoke",
            "parent_text", "err_state", "err_vec", "err_parent", "err_slow",
            "err_total", "baseline", "z"]


def _abs_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(_REPO_ROOT, p)


class WorldPredictorLog(Plugin):
    name = "world_predictor_log"

    def setup(self, ctx):
        self.events_out = self.config.get("events_out")
        self.rows = []

    def on_step_late(self, ctx):
        ev = getattr(ctx, "last_world_pred", None)
        if not ev:
            return
        self.rows.append({k: ev.get(k) for k in _COLUMNS})

    def metrics(self, ctx):
        if not self.rows:
            return None
        errs = [r["err_total"] for r in self.rows if r.get("err_total") is not None]
        if not errs:
            return None
        return {"world_predictor.err_total_mean": sum(errs) / len(errs)}

    def report(self, ctx):
        if self.rows and self.events_out:
            path = _abs_path(self.events_out)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(_COLUMNS)
                for r in self.rows:
                    w.writerow([r.get(c, "") for c in _COLUMNS])
        return {"世界の予測器_行数": len(self.rows)}
