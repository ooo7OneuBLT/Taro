"""感覚ブロックごとのprogress（学習進度）を可視化するための、読むだけのプローブ。

【なぜ、2026-08-05】現状の`progress`報酬は、固有感覚・視覚・触覚・前庭感覚など
複数の感覚ブロックの誤差を`t.block_pe()`で1個のMSEに混ぜてから、唯一の
`LearningProgress`（`t.lp`）に渡している（`run/trainer.py`720-721行）。
「今どのブロックが学びやすいかで報酬の出どころが決まってしまうのでは」という
疑問に対し、まず実測データでブロックごとの動きを見たい。

このプラグインは太郎の挙動を1ビットも変えない**測定器**。報酬・学習には
一切書き込まない（`t.block_pe()`自身は呼ばない。呼ぶとブロックごとの内訳が
取れず、1個に混ぜたMSEしか返らないため）。太郎の本能・報酬を変えないので
`doc/人間模倣からの逸脱リスト.md`への追記は不要
（設計：作業記録（非公開）
2節のとおり）。

置き場所の判定（落とし穴チェックリスト項28）：これは「外から測るもの」（測定器・
プローブ）であり太郎の中身ではない。よってtaro_coreではなくrun/plugins/commonに置く。

実験ファイルでの書き方：

    "plugins": {"block_progress_probe": {"out_path": "E/logs/xxx/block_progress.csv"}}

out_pathを指定しなければCSVは書き出されない（例外では落とさない。設定を渡し忘れた
だけの利用者に配慮。仕様の明記どおり）。

出力列: step, block_name, pe, pe_fast, pe_slow, progress
"""

import os

import torch

from run.plugins.base import Plugin
# 【なぜこのimportで解決できるか】run.taro_setup を先にimportした時点で
#   sys.pathにtaro_core/src/brain（learning_progress.pyの置き場所）が追加されている
#   （run/taro_setup.py 冒頭・contact_reward.pyが同じ経路でsomatosensory_cortexを
#   importしているのと同じパターン）。
from run.taro_setup import to_tensor  # noqa: F401  （importの副作用でsys.pathを通すため）
from learning_progress import LearningProgress


class BlockProgressProbe(Plugin):
    """t.block_pe()が1個に混ぜる直前の、ブロックごとの誤差・progressを記録する。"""

    name = "block_progress_probe"

    def setup(self, ctx):
        """ctx を受け取り、out_path設定・ブロックごとのLearningProgress辞書・記録行リスト・記録tick数を初期化する。戻り値は無い。"""
        self.out_path = self.config.get("out_path")
        # ブロック名ごとに別々のLearningProgressインスタンスを持つ（設計3節）。
        #   最初に見た時のtau_fast/tau_slowで作る（既存のprogress報酬と同じ時定数、
        #   比較可能にするため）。initは指定しない＝既定1.0でよい（設計7節Q1）。
        self._lps = {}
        self.rows = []
        self.steps_recorded = 0

    def on_step(self, ctx):
        # 【仕様の明記どおり】last/blocksのいずれかが無い/空なら、何もせず早期return。
        """ctx を受け取り、太郎の各感覚ブロックについて予測と実際の区間MSEを計算し、ブロックごとのLearningProgressを更新して誤差・進度を1行ずつ記録する。戻り値は無い。
        """
        last = getattr(ctx, "last", None)
        taro = getattr(ctx, "taro", None)
        blocks = getattr(taro, "blocks", None) if taro is not None else None
        if not last or not blocks:
            return
        pred = last.get("pred")
        nlp = last.get("nlp")
        if pred is None or nlp is None:
            return
        for (s, e, block_name) in blocks:
            pe = torch.nn.functional.mse_loss(pred[..., s:e], nlp[..., s:e])
            lp = self._lps.get(block_name)
            if lp is None:
                # 初回だけ、既存のprogress報酬と同じ時定数で生成する（設計3節）。
                lp = LearningProgress(tau_fast=taro.lp.tau_fast, tau_slow=taro.lp.tau_slow)
                self._lps[block_name] = lp
            progress = lp.update(pe.item())
            self.rows.append({
                "step": ctx.step,
                "block_name": block_name,
                "pe": pe.item(),
                "pe_fast": lp.pe_fast,
                "pe_slow": lp.pe_slow,
                "progress": progress,
            })
        self.steps_recorded += 1

    def report(self, ctx):
        """ctx を受け取り、記録が無ければNoneを返し、それ以外は記録tick数・ブロック名一覧・行数をまとめ、out_path指定時はCSVに書き出してパスを加えて返す。
        """
        if not self.rows:
            return None
        out = {
            "記録tick数": self.steps_recorded,
            "ブロック名": sorted(self._lps.keys()),
            "行数": len(self.rows),
        }
        if not self.out_path:
            out["注意"] = (
                "out_pathが指定されていないため、CSVは書き出していません。"
                "実験ファイルでplugins.block_progress_probe.out_pathを指定してください。")
            return out
        # 【なぜこの解決の仕方か】contact_reward.pyのself_trace_out書き出し部分と
        #   同じロジック（相対パスならリポジトリルート基準に直す）。
        path = self.out_path if os.path.isabs(self.out_path) else os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, os.pardir, os.pardir)), self.out_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cols = ["step", "block_name", "pe", "pe_fast", "pe_slow", "progress"]
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(cols)
            for r in self.rows:
                w.writerow([r.get(c, "") for c in cols])
        out["出力パス"] = path
        return out
