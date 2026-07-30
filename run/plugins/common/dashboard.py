"""★学習の様子の絵（HTML）を、記録の区切りごとに自動で作り直す。

【なぜプラグインにしたか、2026-07-30】ユーザーの要望：

> 視覚化の部分も run システムに組み込んで完全に自動化できないかな？

図を作るのは「外から見る道具」なので、測る道具と同じ枠（プラグイン）に収める。
学習ループは図のことを知らないままでよい（責務を混ぜない）。

⚠️★ctx を**読むだけ**。太郎も環境も変えない。
  実際に読むのは CSV（同じフォルダに書かれたもの）で、絵を作るのは
  `run/tools/dashboard.py`。ここは**呼ぶだけ**。

【自動でONになる】`run.csv` を指定した学習では、実験ファイルに書かなくても
`run/main.py` が自動で足す。切りたいときは実験ファイルに
    "plugins": {"dashboard": false}
と明示する。

実験ファイルでの書き方（明示する場合）:
    "plugins": {"dashboard": true}
    "plugins": {"dashboard": {"dir": "E/logs/別のフォルダ", "title": "見出し"}}
"""
import os

from run.plugins.base import Plugin


class Dashboard(Plugin):
    name = "dashboard"

    def setup(self, ctx):
        # 出力先は「CSVを置いたフォルダ」が既定（同じ実験の仲間が並ぶ場所）
        d = self.config.get("dir")
        if not d:
            csv_path = (ctx.spec.get("run") or {}).get("csv")
            d = os.path.dirname(csv_path) if csv_path else None
        self.dir = d
        self.title = self.config.get("title")
        self.made = None
        if not self.dir:
            print("⚠️[dashboard] 出力先が決まらないので絵を作りません"
                  "（run.csv か plugins.dashboard.dir を指定してください）", flush=True)

    def _make(self):
        if not self.dir:
            return
        from run.tools import dashboard as dash
        try:
            self.made = dash.make(self.dir, self.title)
        except Exception as e:      # noqa: BLE001
            # ⚠️握りつぶすが黙らない。★絵作りの失敗で**学習を落とさない**のが大事。
            #   （2時間の学習が図の不具合で消えるのは本末転倒）
            print(f"⚠️[dashboard] 絵を作れませんでした: {type(e).__name__}: {e}",
                  flush=True)

    def on_checkpoint(self, ctx):
        # ⚠️★CSV はこのあと書かれるので、ここで作る絵は**1つ前**の区切りまでを映す。
        #   （記録の順番＝測る→CSVに書く→次の学習。1区切り遅れるだけなので許容）
        self._make()

    def report(self, ctx):
        self._make()            # 最後にもう一度（最新の行まで入った絵にする）
        if not self.made:
            return None
        return {"絵": os.path.relpath(self.made, os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir, os.pardir)))}
