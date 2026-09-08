# -*- coding: utf-8 -*-
"""測る道具：世界の予測器（M7a／M7b-1）の誤差をCSVに書く。

【仕様】F/docs/二語文/仕様_M7a_世界の予測器_測るだけ_2026-09-08.md
「後半：実装担当向け技術付録」4節。`run/plugins/common/word_production.py` の型を踏襲。
M7b-1（物ごとの予測器と驚きの配線）追記：F/docs/二語文/仕様_M7b-1_物ごとの予測器と
驚きの配線_2026-09-09.md「後半」5節。

【役割】読むだけ（`run/plugins/base.py` の規約）。太郎も環境も変えない。
  読むのは ctx.last_world_pred・ctx.world_pred_by_file（run/trainer.py.
  _world_predictor_step が on_step_late より前に置く。cfg.world_predictor
  無効時は last_world_pred=None・world_pred_by_file は置かれない/空）。

【出力先】実験ファイルの `plugins.world_predictor_log.events_out` に
  `<出力先>/世界の予測器.csv` の形で指定する（word_productionと同じ相対パス規約）。
  物ごとCSV（`世界の予測器_物ごと.csv`）は同じフォルダに自動で置く（events_outの
  ファイル名を置き換えるだけ、新しいconfigキーは増やさない）。

【列】step, t_sec, present, visible, vanished, parent_spoke, parent_text,
  err_state, err_vec, err_parent, err_slow, err_total, baseline, z,
  n_files, z_max, z_max_id, ne_level
  （err_slowは追記2026-09-08。n_files以降はM7b-1で追記。multi_object無効時は
  空文字。仕様書末尾「追記」節）

【世界の予測器_物ごと.csv の列】
  step, t_sec, file_id, attended, visible, vanished, err_state, z_state
  （ctx.world_pred_by_file から。毎tick・物ごとに1行。multi_object無効時は
  この辞書が置かれない/空のため1行も出ない＝ファイル自体を作らない）
"""
import csv
import os

from run.plugins.base import Plugin

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))

_COLUMNS = ["step", "t_sec", "present", "visible", "vanished", "parent_spoke",
            "parent_text", "err_state", "err_vec", "err_parent", "err_slow",
            "err_total", "baseline", "z",
            "n_files", "z_max", "z_max_id", "ne_level"]

_BY_FILE_COLUMNS = ["step", "t_sec", "file_id", "attended", "visible",
                     "vanished", "err_state", "z_state"]


def _abs_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(_REPO_ROOT, p)


def _by_file_path(events_path):
    """<dir>/世界の予測器.csv → <dir>/世界の予測器_物ごと.csv（拡張子の直前に挿入）。"""
    base, ext = os.path.splitext(events_path)
    return base + "_物ごと" + ext


class WorldPredictorLog(Plugin):
    name = "world_predictor_log"

    def setup(self, ctx):
        self.events_out = self.config.get("events_out")
        self.rows = []
        self.by_file_rows = []

    def on_step_late(self, ctx):
        ev = getattr(ctx, "last_world_pred", None)
        if ev:
            self.rows.append({k: ev.get(k) for k in _COLUMNS})
        # 【M7b-1・2026-09-09】ctx.world_pred_by_file は multi_object=True の
        #   ときだけ trainer.py が置く（辞書 file_id -> {...}）。既定（無効）では
        #   getattrがNoneを返し、by_file_rowsは1行も増えない＝新CSVは作られない。
        by_file = getattr(ctx, "world_pred_by_file", None)
        if by_file:
            step = ev.get("step") if ev else ctx.step
            t_sec = ev.get("t_sec") if ev else round(ctx.sim_sec, 3)
            for file_id, d in by_file.items():
                self.by_file_rows.append({
                    "step": step, "t_sec": t_sec, "file_id": file_id,
                    "attended": d.get("attended"), "visible": d.get("visible"),
                    "vanished": d.get("vanished"), "err_state": d.get("err_state"),
                    "z_state": d.get("z_state"),
                })

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
        if self.by_file_rows and self.events_out:
            path = _by_file_path(_abs_path(self.events_out))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(_BY_FILE_COLUMNS)
                for r in self.by_file_rows:
                    w.writerow([r.get(c, "") for c in _BY_FILE_COLUMNS])
        return {"世界の予測器_行数": len(self.rows),
                "世界の予測器_物ごと_行数": len(self.by_file_rows)}
