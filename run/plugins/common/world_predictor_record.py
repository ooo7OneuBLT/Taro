# -*- coding: utf-8 -*-
"""記録する道具：世界の予測器（ポート型）が毎tick受け取った入力を1本のpklに残す。

【仕様】F/docs/二語文/仕様_M7b-1改3_切り分け実験_入力記録とオフライン再生_2026-09-10.md
「後半」1節。

【役割】読むだけ（`run/plugins/base.py` の規約）。太郎も環境も変えない。
  読むのは ctx.world_pred_inputs（`run/trainer.py` の _world_predictor_step_ports が
  wp.predict_all(...) の直後に置く。cfg.world_predictor.ports が真のときだけ置かれる）。

【出力先】実験ファイルの `plugins.world_predictor_record.out` に
  pkl の保存先パス（相対なら repo ルートからの相対）を書く。

【中身】pickle.dump したリスト（1要素=1tick）。各要素は dict：
  step, t_sec, vision（{file_id: {"obj_state": list, "obj_vec": np.float32 array or None}}）,
  hearing（{"parent_spoke", "chunk_id_plus1", "time_since_parent"}）,
  body（{"act": list or None}）, attended_id, visible（{file_id: bool}）,
  vanished（{file_id: bool}）。
  obj_vec は np.float32 にキャストして持つ（仕様「後半」1節「obj_vecはnp.float32に、
  obj_stateはlistのまま」）。
"""
import os
import pickle

import numpy as np

from run.plugins.base import Plugin

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))


def _abs_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(_REPO_ROOT, p)


def _cast_vision(vision):
    """{file_id: {"obj_state":..., "obj_vec":...}} を複製し、obj_vecだけ
    np.float32配列にする（None はNoneのまま）。obj_stateはlistのまま
    （仕様「後半」1節）。
    """
    out = {}
    for fid, d in (vision or {}).items():
        obj_state = list(d.get("obj_state")) if d.get("obj_state") is not None else None
        obj_vec_raw = d.get("obj_vec")
        obj_vec = None if obj_vec_raw is None else np.asarray(obj_vec_raw, dtype=np.float32)
        out[fid] = {"obj_state": obj_state, "obj_vec": obj_vec}
    return out


class WorldPredictorRecord(Plugin):
    name = "world_predictor_record"

    def setup(self, ctx):
        self.out = self.config.get("out")
        self.rows = []

    def on_step_late(self, ctx):
        inp = getattr(ctx, "world_pred_inputs", None)
        if not inp:
            return
        hearing = inp.get("hearing") or {}
        body = inp.get("body") or {}
        act = body.get("act")
        self.rows.append({
            "step": inp.get("step"),
            "t_sec": inp.get("t_sec"),
            "vision": _cast_vision(inp.get("vision")),
            "hearing": {
                "parent_spoke": hearing.get("parent_spoke"),
                "chunk_id_plus1": hearing.get("chunk_id_plus1"),
                "time_since_parent": hearing.get("time_since_parent"),
            },
            "body": {"act": list(act) if act is not None else None},
            "attended_id": inp.get("attended_id"),
            "visible": dict(inp.get("visible") or {}),
            "vanished": dict(inp.get("vanished") or {}),
        })

    def report(self, ctx):
        if not self.rows or not self.out:
            return {"世界の予測器_入力記録_行数": len(self.rows)}
        path = _abs_path(self.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fp:
            pickle.dump(self.rows, fp)
        return {"世界の予測器_入力記録_行数": len(self.rows), "世界の予測器_入力記録_path": path}
