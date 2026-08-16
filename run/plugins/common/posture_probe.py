"""頭の高さと報酬を記録する。

【なぜ要るか、2026-08-15】`reward="posture_height"`（頭の高さを報酬にする）で
15,000step 学習を1本回したところ、**学習曲線がどこにも残らなかった**。
`run/trainer.py` の `_record()` は報酬の列を持たず、CSV に列を足せるのは
プラグインの `metrics()` だけなので、記録するプラグインが無い実験では
「頭が上がったのか」を後から確かめる手段が無い（E/logs/head_lift_supine/seed0.csv）。

姿勢の学習は「頭の高さが上がったか」が結果そのものなので、それを測る道具を置く。

実験ファイルでの書き方:
    "plugins": {"posture_probe": true}

出る列:
    head_z_mean   区間の頭の高さの平均 [cm]
    head_z_max    区間の頭の高さの最大 [cm]
    rew_mean      区間の報酬の平均（trainer が ctx.last_reward に入れている値）

注意：`head_z` は**世界座標の絶対の高さ**で、`posture_height` 報酬が使う
  「開始時からの相対差」とは別物。絶対値で出すのは、相対差だと基準が
  リセットのたびに変わって区間どうしを比べられなくなるため。
  月齢や体格を変えた実験どうしを比べるときは、絶対値のままでは
  比較できない点に注意すること（そのときは開始時の値で引くこと）。
"""
from run.plugins.base import Plugin


class PostureProbe(Plugin):
    name = "posture_probe"

    def setup(self, ctx):
        self._seg = []            # 区間ぶんの頭の高さ [m]
        self._seg_rew = []        # 区間ぶんの報酬
        self.all_n = 0
        self.all_sum = 0.0
        self.all_max = None
        self.start_z = None

    def on_step_late(self, ctx):
        # 注意：報酬は `on_step` の時点ではまだ入っていない。trainer.py:926 で
        #   `ctx.last_reward` が組み立てられた**あと**に呼ばれる `on_step_late` で
        #   読むこと（run/plugins/base.py:51 の指示）。頭の高さも同じ tick の値に
        #   揃えたいので、両方まとめてここで取る。
        z = float(ctx.data.body("head").xpos[2])
        if self.start_z is None:
            self.start_z = z
        self._seg.append(z)
        self.all_n += 1
        self.all_sum += z
        self.all_max = z if self.all_max is None else max(self.all_max, z)
        r = getattr(ctx, "last_reward", None)
        if isinstance(r, dict):
            r = r.get("rew", r.get("reward"))
        if r is not None:
            try:
                self._seg_rew.append(float(r))
            except (TypeError, ValueError):
                pass

    def metrics(self, ctx):
        if not self._seg:
            return None
        out = {"head_z_mean": round(100.0 * sum(self._seg) / len(self._seg), 3),
               "head_z_max": round(100.0 * max(self._seg), 3)}
        if self._seg_rew:
            out["rew_mean"] = round(sum(self._seg_rew) / len(self._seg_rew), 6)
        self._seg, self._seg_rew = [], []
        return out

    def line(self, ctx):
        if not self.all_n:
            return None
        return (f"頭の高さ 平均{100.0 * self.all_sum / self.all_n:.2f}cm "
                f"最大{100.0 * self.all_max:.2f}cm")

    def report(self, ctx):
        if not self.all_n:
            return None
        return {"頭の高さの平均cm": round(100.0 * self.all_sum / self.all_n, 3),
                "頭の高さの最大cm": round(100.0 * self.all_max, 3),
                "開始時の頭の高さcm": round(100.0 * self.start_z, 3),
                "測ったステップ数": self.all_n}
