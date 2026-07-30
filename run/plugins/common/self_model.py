"""自己モデルの質を測る。★太郎の脳が要るので、run.type=train のときだけ使える。

測るものの意味（4つ）:
    classify  自分の行動と他人の行動を★言い当てられるか（50%=偶然）
    margin    正解と不正解の予測誤差の★差。大きいほど「自分の体を分かっている」
    corr      「こう動くはず」の予測と実際の変化の相関
    persist   ★「何もしない」と予測した場合と比べた誤差の比
              100未満＝勝ち／100超＝★何もしない予測より下手
              ⚠️統計の分野では naive forecast に負けることを
                "illusion of skill"（見かけの上手さ）と呼ぶ。MASE と同じ発想

判定の実体は `E/scripts/e_probes.py` の evaluate / agency_probe。
ここは**包むだけ**。⚠️判定を2箇所に書かない。

⚠️★agency（行為主体感）は**再現性が無い**（2026-07-30 実測）。
  同じコード・同じシードで 48.0% → 46.0% と変わる。
  主指標（classify/margin/corr/persist）は完全一致するので、
  **agency を根拠にした主張はできない**。既定では測らない。

実験ファイルでの書き方:
    "plugins": {"self_model": true}
    "plugins": {"self_model": {"agency": true}}   ← ★再現性が無いことを承知の上で
"""
import os
import sys

from run.plugins.base import Plugin

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))
for _p in (os.path.join(_ROOT, "E", "scripts"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class SelfModel(Plugin):
    name = "self_model"

    def setup(self, ctx):
        if ctx.brain is None:
            raise ValueError(
                "self_model は★太郎の脳が要る（run.type=train で使う）。\n"
                "  measure（脳を通さず環境だけ進める）では測れない")
        self.want_agency = bool(self.config.get("agency", False))
        if self.want_agency:
            print("⚠️[self_model] agency を測るよう指定された。★再現性が無い指標なので"
                  "（同条件で48.0%→46.0%）、これを根拠に主張しないこと", flush=True)
        self.rows = []
        self._last = None

    def on_checkpoint(self, ctx):
        import e_probes
        probe_ctx = getattr(ctx, "probe_ctx", None)
        if probe_ctx is None:
            # run/trainer.py が渡す。measure（脳を通さない）では測れない
            raise RuntimeError("self_model は★太郎の脳を通す実行（run.type=train）が要る")
        cl, mg, co, pr = e_probes.evaluate(probe_ctx)
        row = {"step": ctx.step, "classify": cl, "margin": mg,
               "corr": co, "persist": pr}
        if self.want_agency:
            ag, magr = e_probes.agency_probe(probe_ctx)
            row["agency"] = ag
            row["mag_ratio"] = magr
        self.rows.append(row)
        self._last = row
        ctx.log(row)

    def line(self, ctx):
        r = self._last
        if not r:
            return None
        s = (f"classify={r['classify']:.1f}% margin={r['margin']:+.1f}% "
             f"corr={r['corr']:.3f} persist={r['persist']:.1f}%")
        if "agency" in r:
            s += f" agency={r['agency']:.1f}%(★再現性なし)"
        return s

    def report(self, ctx):
        if not self.rows:
            return None
        import statistics as st
        k = min(8, len(self.rows))
        tail = self.rows[-k:]
        out = {"終盤の平均": {}, "測った回数": len(self.rows), "終盤の点数": k}
        for key in ("classify", "margin", "corr", "persist"):
            out["終盤の平均"][key] = round(st.mean(r[key] for r in tail), 3)
        if self.want_agency:
            out["終盤の平均"]["agency"] = round(
                st.mean(r["agency"] for r in tail), 2)
            out["⚠️注意"] = "agency は再現性が無い（同条件で48.0→46.0）"
        # ★persist が100を超えていたら黙って通さない
        if out["終盤の平均"]["persist"] > 100.0:
            out["★警告"] = ("persist が100超＝「何もしない」と予測した場合より下手。"
                          "モデルが壊れている疑い（2026-07-30 に Goal Babbling で"
                          "1000%になった）")
        return out
