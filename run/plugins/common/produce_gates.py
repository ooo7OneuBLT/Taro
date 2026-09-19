# -*- coding: utf-8 -*-
"""測る道具：太郎が喋ろうとして**どこで止まったか**を1歩ごとに記録する。

【なぜ要るか・2026-09-16】白紙の太郎に語を教える走行が、3,000歩×10本で
**発話0回**だった。原因の在り処（2026-09-10のコミット `69651bb`＝眼球が
止まっていたのを直した回）までは二分探索で分かったが、**なぜ黙るのかは
一度も測っていなかった**。発話は `taro_core/src/brain/brain_tick.py` の
`_apply_word_production` にある門を**全部**通らないと出ない。どの門で
止まっているかが分かれば、直す場所が決まる。

【門（上から順に。1つでも閉じていれば黙る）】
    注意している物が無い … 消失発話（vanish_input）のとき、注意の対象が無い
    語が決まらない       … いま見ているものに対して言う語を選べなかった
    自信不足             … 選べたが、確信（sim）が閾値（produce.threshold）未満
    注視していない       … サッケード（目をぱっと動かす動作）の最中か、保持中でない
    クールダウン中       … 直前に喋ったばかり（既定2秒）
    通過                 … ここまで来たら声が出る

【この道具がやらないこと】率を出さない・判断しない（`run/plugins/base_取説.md`
「道具は列を出す。数えるのはあと」）。report では0/1列の**合計**だけ返す。

【読むだけ】太郎にも環境にも触らない。乱数を消費しない。読むのは ctx だけ。

【実験ファイルでの書き方】
    "plugins": {"produce_gates": {"out": "F/logs/<実験>/発話の門.csv"}}

【出る列】
    step / sim_sec / 止まった門 / 語が決まった / sim / 閾値
    サッケード残り秒 / 撃った回数 / 保持してよい / 前の発話からの秒
    親の的 / 親の状態 / 視線の先 / 発話した
"""
import csv
import os

from run.plugins.base import Plugin

# 上から順に「早く止まるほど手前」。report の並びをここで決める。
門の順 = ["注意している物が無い", "視線誘導反射が無効", "語が決まらない",
          "自信不足", "注視していない", "クールダウン中", "通過"]

列 = ["step", "sim_sec", "止まった門", "語が決まった", "sim", "閾値",
      "サッケード残り秒", "撃った回数", "保持してよい", "前の発話からの秒",
      "親の的", "親の状態", "視線の先", "発話した"]


def _abs_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)), p)


class ProduceGates(Plugin):
    """発話の門の通過記録（読むだけ）。"""

    name = "produce_gates"
    intervenes = None

    def setup(self, ctx):
        """出力先を読み、行を貯める入れ物を用意する。戻り値は無い。"""
        self.out = self.config.get("out")
        self.rows = []

    def on_step(self, ctx):
        """毎ステップ。太郎が控えた門の記録を1行にして積む。戻り値は無い。

        `ctx.produce_gate` は `_apply_word_production` がこのtickで置いた辞書。
        産出の配線が無い実験（喃語だけ・語の産出を切った走行）では None のまま
        なので、その場合は「門に入っていない」として空欄で1行だけ残す
        （行数と歩数が合わないと、あとで数えるときに分母を取り違える）。
        """
        g = getattr(ctx, "produce_gate", None) or {}
        行 = {"step": ctx.step, "sim_sec": round(ctx.sim_sec, 2),
              "親の的": ctx.親の的, "親の状態": ctx.親の状態,
              "視線の先": ctx.視線の先, "発話した": int(ctx.発話した)}
        for k in ("止まった門", "語が決まった", "sim", "閾値", "サッケード残り秒",
                  "撃った回数", "保持してよい", "前の発話からの秒"):
            行[k] = g.get(k, "")
        self.rows.append(行)

    def metrics(self, ctx):
        """区切りごとの値。直近の歩でどの門にいたかだけ run.csv へ混ぜる。"""
        if not self.rows:
            return None
        return {"produce_gates.止まった門": self.rows[-1].get("止まった門", "")}

    def report(self, ctx):
        """out があれば列をCSVへ書き、門ごとの**歩数**（率ではない）を返す。"""
        if self.rows and self.out:
            p = _abs_path(self.out)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(列)
                for r in self.rows:
                    w.writerow([r.get(c, "") for c in 列])
        数 = {}
        for r in self.rows:
            k = r.get("止まった門") or "（門に入っていない）"
            数[k] = 数.get(k, 0) + 1
        並び = [k for k in 門の順 if k in 数] + sorted(k for k in 数 if k not in 門の順)
        out = {"記録した歩数": len(self.rows)}
        for k in 並び:
            out["門_" + k] = 数[k]
        out["声を出した歩"] = sum(r.get("発話した", 0) for r in self.rows)
        if self.out:
            out["出力"] = self.out
        return out
