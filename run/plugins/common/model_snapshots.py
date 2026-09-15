# -*- coding: utf-8 -*-
"""checkpointごとに脳の途中保存を残す（F2-33②・2026-09-01新設）。

【なぜ要るか】「追加学習のどこから接地が悪化するか」を知るには途中の脳が要る。
走行中にその場で測る案は、2026-07-30に「測定を入れると同一シードが一致しなく
なる」と実証済み（run/tools/check_divergence.py の --probe 注記）。
⇒ 走行中は**保存だけ**（乱数を消費せず・環境に触れず・決定性を壊さない）、
  測定は走行後にオフラインで行う。

保存する中身は taro_setup の保存形式の部分集合（brain / visual_projection /
brain_vocab）＝オフライン測定（接地の行列）に必要な最小限。

実験ファイルでの書き方（run.checkpoint の間隔ごとに1個保存される）:
    "plugins": {"model_snapshots": {"out_dir": "F/logs/.../snapshots"}}
"""
import os

from run.plugins.base import Plugin


class ModelSnapshots(Plugin):
    name = "model_snapshots"

    def setup(self, ctx):
        """ctx を受け取り、out_dirの設定を読み込んでディレクトリを作成し、保存件数を初期化する。戻り値は無い。"""
        self.out_dir = self.config.get("out_dir")
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
        self.saved = 0

    def on_checkpoint(self, ctx):
        """ctx を受け取り、out_dirがあり太郎の脳があれば、脳と（あれば）視覚投影・発話語彙の対応表をまとめてstep番号つきのファイルに保存する。戻り値は無い。"""
        if not self.out_dir:
            return
        taro = getattr(ctx, "taro", None)
        if taro is None or getattr(taro, "brain", None) is None:
            return
        import torch
        blob = {"brain": taro.brain.state_dict()}
        vp = getattr(taro, "_visual_projection", None)
        if vp is not None:
            blob["visual_projection"] = vp.state_dict()
        pv = getattr(taro, "produce_vocab", None)
        if pv is not None:
            blob["brain_vocab"] = {"char2idx": dict(pv.char2idx)}
        torch.save(blob, os.path.join(self.out_dir, "step%06d.pt" % ctx.step))
        self.saved += 1

    def report(self, ctx):
        """ctx を受け取り、out_dir未指定ならNoneを返し、指定されていれば保存した件数と出力先を辞書で返す。"""
        if not self.out_dir:
            return None
        return {"保存した途中の脳": self.saved, "出力": self.out_dir}
